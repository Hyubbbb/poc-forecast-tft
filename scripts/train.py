"""TFT 학습 단독 실행 스크립트 (Phase 1 — 노트북에서 분리).

cell 11(split + TimeSeriesDataSet)과 cell 17(TFT 학습 + 산출물 저장)의 SoT.
노트북은 train_tft()를 import하거나, 학습된 best.ckpt를 로드하는 모드로 동작.

사용:
    python scripts/train.py --config configs/tft.yaml --run-name bg_test
    python scripts/train.py --config configs/tft.yaml --dry-run
    nohup python scripts/train.py --config configs/tft.yaml --run-name bg_test > train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainConfig:
    cfg: dict
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path) as f:
            return cls(cfg=yaml.safe_load(f))

    def apply_dry_run(self) -> "TrainConfig":
        dry = self.cfg.get("dry_run", {})
        if "max_epochs" in dry:
            self.cfg["trainer"]["max_epochs"] = dry["max_epochs"]
        if "batch_size" in dry:
            self.cfg["trainer"]["batch_size"] = dry["batch_size"]
        if "data_path" in dry:
            self.cfg["data"]["parquet_path"] = dry["data_path"]
        if "out_dir" in dry:
            self.cfg["artifacts"]["out_dir"] = dry["out_dir"]
        # tiny fixture 길이가 짧아 long encoder/decoder cfg가 학습 불가능할 때
        # dry_run에서 dataset hp를 축소 가능 (Phase A 용품 e104_d16용).
        for key in ("encoder_len", "decoder_len", "min_encoder_len"):
            if key in dry:
                self.cfg["dataset"][key] = dry[key]
        self.cfg.setdefault("mlflow", {})["source_tag"] = "dry_run"
        return self


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(
    parquet_path: str | Path,
    zero_ratio_threshold: float = 0.30,
    target_col: str = "WEEKLY_SALE_QTY_CNS",
) -> pd.DataFrame:
    """parquet 캐시를 로드하고 노트북 cell 4와 동일한 ACTUAL_END 컷을 적용한다.

    zero_ratio_threshold: 주별 SC×WEEK 중 qty=0 비율이 이 값 미만인 주를 'normal'로 본다.
        의류 default=0.30. 모자(HEA) supplies 처럼 sparse한 경우 0.99 이상으로 완화 필요.
    target_col: zero-ratio 컷 산정에 쓸 타깃 컬럼. supplies cp6677 처럼 컬럼명이 `WEEKLY_SALE_QTY`
        인 경우 호출자가 명시 (config: data.target).
    """
    df = pd.read_parquet(parquet_path)
    if "WEEK_START" in df.columns:
        df["WEEK_START"] = pd.to_datetime(df["WEEK_START"])

    weekly = df.groupby("WEEK_START").agg(
        qty_sum=(target_col, "sum"),
        zero_ratio=(target_col, lambda s: (s == 0).mean()),
    )
    normal_weeks = weekly[(weekly["qty_sum"] > 0) & (weekly["zero_ratio"] < zero_ratio_threshold)].index
    actual_end = normal_weeks.max()
    df = df[df["WEEK_START"] <= actual_end].copy()
    df.attrs["actual_end"] = actual_end
    print(f"[load_data] zero_ratio_threshold={zero_ratio_threshold}, target={target_col}, actual_end={actual_end}, rows={len(df)}")
    return df


def prepare_features(df: pd.DataFrame, ds_cfg: dict) -> pd.DataFrame:
    """cell 11의 시즌 필터 + WEEK_OF_YEAR + time_idx + dtype 정리.

    in_season_filter: True (의류 default)면 SSN_START~SSN_END 구간으로 row를 제한한다.
    False (용품)면 SC당 전체 weekly history를 보존해 long encoder_len 학습에 활용한다.
    """
    df = df.copy()
    df["WEEK_OF_YEAR"] = df["WEEK_START"].dt.isocalendar().week.astype(int)
    # cyclic encoding — 모자 계절성을 phase-aware로 표현 (52주 주기)
    df["WEEK_SIN"] = np.sin(2 * np.pi * df["WEEK_OF_YEAR"] / 52.0)
    df["WEEK_COS"] = np.cos(2 * np.pi * df["WEEK_OF_YEAR"] / 52.0)
    df["MONTH"] = df["WEEK_START"].dt.month.astype(int)
    df["QUARTER"] = df["WEEK_START"].dt.quarter.astype(int)

    in_season_filter = ds_cfg.get("in_season_filter", True)
    if in_season_filter and "SSN_START" in df.columns and "SSN_END" in df.columns:
        df = df[(df["WEEK_START"] >= df["SSN_START"]) & (df["WEEK_START"] <= df["SSN_END"])].copy()

    group_key = ds_cfg["group_key"] if "group_key" in ds_cfg else "SC_CD"
    df = df.sort_values([group_key, "WEEK_START"]).reset_index(drop=True)
    df["time_idx"] = df.groupby(group_key).cumcount()
    # SC별 첫 판매 주차로부터 경과 (cold-start 핵심 신호). time_idx와 의미 같지만
    # in_season_filter 등으로 일부 주가 빠져도 실 경과 주차를 보존.
    first_week = df.groupby(group_key)["WEEK_START"].transform("min")
    df["WEEKS_SINCE_FIRST_SALE"] = ((df["WEEK_START"] - first_week).dt.days // 7).astype(int)

    static_cats = [c for c in ds_cfg.get("static_categoricals", []) if c in df.columns]
    static_reals = [c for c in ds_cfg.get("static_reals", []) if c in df.columns]
    tv_known = [c for c in ds_cfg.get("time_varying_known_reals", []) if c in df.columns]
    tv_unknown = [c for c in ds_cfg.get("time_varying_unknown_reals", []) if c in df.columns]

    for c in static_cats:
        df[c] = df[c].fillna("__missing__").astype(str)
    target = ds_cfg["target"]
    real_cols = list(set(static_reals + tv_known + tv_unknown + [target]))
    real_cols = [c for c in real_cols if c in df.columns]
    df[real_cols] = df[real_cols].astype(float)

    # SC별 forward-fill — known_real 중 미래 결측을 마지막 known 값으로 채움.
    # WEEKLY_DISC_RAT 처럼 inference 시점에서 미래 값이 실측 불가능한데 known으로 모델에
    # 노출해야 하는 경우 사용. 잔여 NaN(history 시작 시점)은 다음 fillna(0.0) 가 처리.
    ffill_cols = [c for c in ds_cfg.get("forward_fill_reals", []) if c in df.columns]
    if ffill_cols:
        df[ffill_cols] = df.groupby(group_key)[ffill_cols].ffill()

    df[real_cols] = df[real_cols].fillna(0.0)

    df.attrs["static_categoricals"] = static_cats
    df.attrs["static_reals"] = static_reals
    df.attrs["time_varying_known_reals"] = tv_known
    df.attrs["time_varying_unknown_reals"] = tv_unknown
    return df


def build_datasets(df: pd.DataFrame, cfg: dict):
    """cell 11의 walk-forward 3-split + TimeSeriesDataSet 빌드."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import EncoderNormalizer, GroupNormalizer, NaNLabelEncoder

    ds_cfg = cfg["dataset"]
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]

    target = data_cfg["target"]
    group_key = data_cfg["group_key"]

    train_decoder_end = pd.Timestamp(split_cfg["train_decoder_end"])
    val_cutoff = pd.Timestamp(split_cfg["val_cutoff"])

    static_cats = df.attrs["static_categoricals"]
    static_reals = df.attrs["static_reals"]
    tv_known = df.attrs["time_varying_known_reals"]
    tv_unknown = df.attrs["time_varying_unknown_reals"]

    categorical_encoders = {col: NaNLabelEncoder(add_nan=True, warn=False) for col in static_cats}

    # target_normalizer 토글 — cfg에 키 없으면 기존 EncoderNormalizer 유지 (회귀 안전).
    norm_kind = ds_cfg.get("target_normalizer", "encoder")
    if norm_kind == "group_log1p":
        target_normalizer = GroupNormalizer(groups=[group_key], transformation="log1p")
    else:
        target_normalizer = EncoderNormalizer(transformation="softplus")

    training = TimeSeriesDataSet(
        df[df["WEEK_START"] <= train_decoder_end],
        time_idx="time_idx",
        target=target,
        group_ids=[group_key],
        min_encoder_length=ds_cfg["min_encoder_len"],
        max_encoder_length=ds_cfg["encoder_len"],
        min_prediction_length=1,
        max_prediction_length=ds_cfg["decoder_len"],
        static_categoricals=static_cats,
        static_reals=static_reals,
        time_varying_known_reals=["time_idx"] + tv_known,
        time_varying_unknown_reals=[target] + tv_unknown,
        target_normalizer=target_normalizer,
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df[df["WEEK_START"] <= val_cutoff], predict=True, stop_randomization=True
    )
    test = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    return training, validation, test


def _build_monitoring_callbacks():
    """MLflow 학습 모니터링용 콜백 — lr / epoch_time_sec / grad_norm_l2 / val_mae_norm.

    val_mae_norm 은 정규화 공간 trend 보조 지표. 정확한 horizon×WAPE 는 학습 종료 후
    _enrich_mlflow_run() 이 wape_h{N} metric + wape_by_horizon.json 으로 적재한다.
    """
    import time

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import LearningRateMonitor

    class EpochTimer(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self._t0 = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module):
            pl_module.log(
                "epoch_time_sec", time.perf_counter() - self._t0,
                on_epoch=True, on_step=False, prog_bar=False,
            )

    class GradNormLogger(pl.Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            sq = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().data.norm(2).item()) ** 2
            pl_module.log(
                "grad_norm_l2", sq ** 0.5,
                on_step=True, on_epoch=False, prog_bar=False,
            )

    class ValMetricLogger(pl.Callback):
        def __init__(self):
            self._abs_err_sum = 0.0
            self._n = 0

        def on_validation_epoch_start(self, trainer, pl_module):
            self._abs_err_sum = 0.0
            self._n = 0

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            x, y = batch
            with torch.no_grad():
                out = pl_module(x)
            pred = out["prediction"] if isinstance(out, dict) else getattr(out, "prediction", None)
            if pred is None:
                return
            q = list(getattr(pl_module.loss, "quantiles", []))
            p50_idx = q.index(0.5) if 0.5 in q else len(q) // 2
            p = pred[..., p50_idx]
            t = y[0] if isinstance(y, (tuple, list)) else y
            self._abs_err_sum += float((p - t).abs().sum().item())
            self._n += int(t.numel())

        def on_validation_epoch_end(self, trainer, pl_module):
            if self._n > 0:
                pl_module.log(
                    "val_mae_norm", self._abs_err_sum / self._n,
                    on_epoch=True, prog_bar=False,
                )

    return [
        LearningRateMonitor(logging_interval="epoch"),
        EpochTimer(),
        GradNormLogger(),
        ValMetricLogger(),
    ]


def train_tft(cfg: dict, run_name: str | None = None) -> dict:
    """TFT 학습. notebook에서 import 가능. dict(best_path, training, model, config) 반환."""
    import lightning.pytorch as pl
    import mlflow
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    set_seed(42)

    df = load_data(
        cfg["data"]["parquet_path"],
        zero_ratio_threshold=cfg["data"].get("actual_end_zero_ratio_threshold", 0.30),
        target_col=cfg["data"]["target"],
    )
    df = prepare_features(df, {**cfg["dataset"], "target": cfg["data"]["target"], "group_key": cfg["data"]["group_key"]})

    training, validation, test = build_datasets(df, cfg)

    trainer_cfg = cfg["trainer"]
    train_dl = training.to_dataloader(
        train=True, batch_size=trainer_cfg["batch_size"], num_workers=trainer_cfg["num_workers"]
    )
    val_dl = validation.to_dataloader(
        train=False, batch_size=trainer_cfg["batch_size"], num_workers=trainer_cfg["num_workers"]
    )

    model_cfg = cfg["model"]
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=model_cfg["learning_rate"],
        hidden_size=model_cfg["hidden_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        hidden_continuous_size=min(model_cfg["hidden_continuous_size"], model_cfg["hidden_size"]),
        loss=QuantileLoss(quantiles=list(model_cfg["quantiles"])),
        log_interval=trainer_cfg.get("log_interval", 10),
        reduce_on_plateau_patience=model_cfg.get("reduce_on_plateau_patience", 4),
    )

    mlflow_cfg = cfg.get("mlflow", {})
    logger = None
    if mlflow_cfg.get("tracking_uri"):
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        logger = MLFlowLogger(
            experiment_name=mlflow_cfg["experiment"],
            tracking_uri=mlflow_cfg["tracking_uri"],
            run_name=run_name,
            tags={"source": mlflow_cfg.get("source_tag", "bg")},
        )

    out_dir = Path(cfg["artifacts"]["out_dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        accelerator=trainer_cfg.get("accelerator", "auto"),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0.1),
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=trainer_cfg.get("early_stopping_patience", 5), mode="min"),
            ModelCheckpoint(
                dirpath=str(out_dir),
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                filename="tft-{epoch:02d}-{val_loss:.4f}",
            ),
            *_build_monitoring_callbacks(),
        ],
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    best_path = trainer.checkpoint_callback.best_model_path

    save_artifacts(best_path, tft, training, out_dir, cfg=cfg)

    # Phase A: MLflow run에 artifact + flatten params + 자동 horizon × WAPE 평가 적재.
    # Lightning MLFlowLogger가 만든 run에 mlflow.start_run(run_id=...)으로 재attach.
    if logger is not None:
        _enrich_mlflow_run(logger, mlflow_cfg, cfg, tft, training, df, out_dir)

    return {
        "best_path": best_path,
        "out_dir": str(out_dir),
        "training": training,
        "validation": validation,
        "test": test,
        "model": tft,
        "config": cfg,
        "df": df,
    }


def _enrich_mlflow_run(
    logger,
    mlflow_cfg: dict,
    cfg: dict,
    model,
    training,
    df,
    out_dir: Path,
    *,
    parent_run_id: str | None = None,
) -> None:
    """학습 직후 MLflow run 보강. spec: mlflow-experiment-tracking v0.2.0.

    적재 내용 (모두 자동):
    - tags: make_run_tags 표준 8개 (source, experiment_intent, data_scope,
            backtest_fold, git_sha, model_architecture, random_seed, is_baseline)
    - params: cfg.* (flatten_cfg) + dataset audit (style_list/sha256/sanity) + env
    - metrics: 4계층 (overall + horizon + bin + STYLE×bin) + baseline lift
    - artifacts: config.yaml, training_dataset.pkl, forecast.parquet,
                 viz/{interpretability, overlay}, sanity/dataset_audit.json,
                 best_ckpt_path.txt (ckpt 자체는 용량 사유 직접 업로드 X)

    parent_run_id 제공 시 MLflow native nested run 으로 연결 (mlflow.parentRunId tag 자동 부착).
    """
    import json
    import mlflow
    import pandas as pd

    from scripts.eval_utils import (
        compute_full_metrics,
        make_naive_cohort_mean,
        make_seasonal_naive,
        wape,
    )
    from scripts.forecast_utils import flatten_cfg, predict_dataframe
    from scripts.mlflow_logging import (
        compute_dataset_audit,
        infer_model_architecture,
        log_full_metrics,
        log_params_safe,
        make_run_tags,
        set_run_tags,
        system_env_params,
    )

    tracking_uri = mlflow_cfg.get("tracking_uri")
    run_id = getattr(logger, "run_id", None)
    if not tracking_uri or not run_id:
        return

    mlflow.set_tracking_uri(tracking_uri)

    # parent_run_id → MLflow native parentRunId tag (UI 트리 구조)
    if parent_run_id:
        try:
            set_run_tags(run_id, {"mlflow.parentRunId": parent_run_id}, tracking_uri=tracking_uri)
        except Exception as e:
            print(f"⚠️  parent_run_id tag skipped: {e}")

    # ── 1. Tags — 표준 8개 ──
    try:
        intent = mlflow_cfg.get("experiment_intent", "unspecified")
        data_scope = mlflow_cfg.get("data_scope") or Path(cfg["data"]["parquet_path"]).stem
        fold = mlflow_cfg.get("backtest_fold", "single")
        seed = cfg.get("train", {}).get("seed")
        is_baseline = bool(mlflow_cfg.get("is_baseline", False))
        arch = infer_model_architecture(cfg)
        tags = make_run_tags(
            experiment_intent=intent,
            data_scope=data_scope,
            backtest_fold=fold,
            model_architecture=arch,
            random_seed=seed,
            is_baseline=is_baseline,
            source=mlflow_cfg.get("source_tag"),
        )
        set_run_tags(run_id, tags, tracking_uri=tracking_uri)
    except Exception as e:
        print(f"⚠️  set_run_tags skipped: {e}")

    # ── 2. Params — cfg + dataset audit + env ──
    train_end = pd.Timestamp(cfg["split"]["train_decoder_end"])
    val_end = pd.Timestamp(cfg["split"]["val_cutoff"])
    decoder_len = cfg["dataset"]["decoder_len"]
    group_key = cfg["data"]["group_key"]
    target = cfg["data"]["target"]
    parquet_path = cfg["data"]["parquet_path"]

    try:
        log_params_safe(run_id, flatten_cfg(cfg, prefix="cfg"), tracking_uri=tracking_uri)
    except Exception as e:
        print(f"⚠️  flatten_cfg log_params skipped: {e}")

    sanity_dir = out_dir / "sanity"
    sanity_dir.mkdir(exist_ok=True)
    try:
        audit = compute_dataset_audit(
            df, parquet_path=parquet_path, cutoff=train_end, val_end=val_end,
            group_key=group_key, style_col="STYLE_CD", target=target,
        )
        log_params_safe(run_id, audit, tracking_uri=tracking_uri)
        (sanity_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"⚠️  dataset_audit skipped: {e}")

    try:
        log_params_safe(run_id, system_env_params(), tracking_uri=tracking_uri)
    except Exception as e:
        print(f"⚠️  env params skipped: {e}")

    # ── 3. best_ckpt_path.txt (ckpt 직접 업로드 X) ──
    ckpt_path = out_dir / "best.ckpt"
    if ckpt_path.exists():
        (out_dir / "best_ckpt_path.txt").write_text(str(ckpt_path.resolve()))

    # ── 4. 4계층 auto eval (validation 윈도우: train_end ~ val_end) ──
    forecast = None
    try:
        forecast = predict_dataframe(
            model, training, df, train_end,
            decoder_len=decoder_len, group_key=group_key,
        )
        df_ts = df.copy()
        df_ts["WEEK_START"] = pd.to_datetime(df_ts["WEEK_START"])
        val_window = df_ts[
            (df_ts["WEEK_START"] > train_end)
            & (df_ts["WEEK_START"] <= train_end + pd.Timedelta(weeks=decoder_len))
        ]
        actual_df = val_window[[group_key, "WEEK_START", target]].rename(
            columns={"WEEK_START": "forecast_week", target: "actual"}
        )

        style_map = None
        if "STYLE_CD" in df_ts.columns:
            style_map = df_ts.drop_duplicates(group_key).set_index(group_key)["STYLE_CD"]

        # baseline WAPE — overall 단위
        baseline_wapes: dict = {}
        try:
            ncm = make_naive_cohort_mean(df_ts, group_key=group_key, target=target)
            sn = make_seasonal_naive(df_ts, group_key=group_key, target=target)
            actuals, ncm_preds, sn_preds = [], [], []
            for _, row in actual_df.iterrows():
                sc = row[group_key]
                h = int((row["forecast_week"] - train_end).days // 7)
                actuals.append(row["actual"])
                ncm_preds.append(ncm(df_ts, sc, train_end, h))
                sn_preds.append(sn(df_ts, sc, train_end, h))
            actuals_s = pd.Series(actuals)
            for name, preds in [("naive_cohort_mean", pd.Series(ncm_preds)),
                                ("seasonal_naive", pd.Series(sn_preds))]:
                m = preds.notna()
                if m.any():
                    baseline_wapes[name] = wape(actuals_s[m].values, preds[m].values)
        except Exception as e:
            print(f"⚠️  baseline wape skipped: {e}")

        metrics = compute_full_metrics(
            forecast, actual_df, model, group_key=group_key,
            style_map=style_map, baseline_wapes=baseline_wapes or None,
        )
        n = log_full_metrics(run_id, metrics, tracking_uri=tracking_uri)
        print(f"  logged {n}/{len(metrics)} metrics (4계층)")

        # forecast.parquet
        forecast_path = out_dir / "forecast.parquet"
        forecast.to_parquet(forecast_path, index=False)

    except Exception as e:
        print(f"⚠️  4계층 auto eval skipped: {e}")
        actual_df = None
        style_map = None

    # ── 5. Visualization (interpretability + overlay) ──
    viz_dir = out_dir / "viz"
    if forecast is not None and actual_df is not None:
        try:
            from scripts.visualize import (
                make_all_sc_overlay_pdf,
                make_attention_plots,
                make_interactive_dashboard,
                make_vsn_plots,
            )
            interp_dir = viz_dir / "interpretability"
            overlay_dir = viz_dir / "overlay"
            batch_size = cfg["trainer"].get("batch_size", 128)
            try:
                make_vsn_plots(model, training, df, train_end,
                               decoder_len=decoder_len, out_dir=interp_dir, batch_size=batch_size)
            except Exception as e:
                print(f"⚠️  vsn plots skipped: {e}")
            try:
                make_attention_plots(model, training, df, train_end,
                                     decoder_len=decoder_len, group_key=group_key, target=target,
                                     out_dir=interp_dir, top_n_sc=3, batch_size=batch_size)
            except Exception as e:
                print(f"⚠️  attention plots skipped: {e}")
            try:
                df_ts = df.copy()
                df_ts["WEEK_START"] = pd.to_datetime(df_ts["WEEK_START"])
                make_all_sc_overlay_pdf(
                    forecast, actual_df, group_key=group_key,
                    out_pdf=overlay_dir / "all_sc_overlay.pdf",
                    style_map=style_map, history_df=df_ts, history_target=target,
                )
                make_interactive_dashboard(
                    forecast, actual_df, group_key=group_key,
                    out_html=overlay_dir / "interactive_dashboard.html",
                    style_map=style_map, history_df=df_ts, history_target=target,
                )
            except Exception as e:
                print(f"⚠️  overlay viz skipped: {e}")
        except Exception as e:
            print(f"⚠️  visualize module skipped: {e}")

    # ── 6. Artifacts — config + training_dataset + best_ckpt_path + sanity/ + viz/ + forecast ──
    with mlflow.start_run(run_id=run_id):
        for fname in ("training_dataset.pkl", "config.yaml", "best_ckpt_path.txt", "forecast.parquet"):
            fpath = out_dir / fname
            if fpath.exists():
                try:
                    mlflow.log_artifact(str(fpath))
                except Exception as e:
                    print(f"⚠️  log_artifact({fname}) skipped: {e}")
        for sub in ("sanity", "viz"):
            sub_dir = out_dir / sub
            if sub_dir.exists() and any(sub_dir.iterdir()):
                try:
                    mlflow.log_artifacts(str(sub_dir), artifact_path=sub)
                except Exception as e:
                    print(f"⚠️  log_artifacts({sub}) skipped: {e}")


def save_artifacts(best_path: str, model, training, out_dir: Path, cfg: dict | None = None) -> None:
    """cell 17의 산출물 저장 + cfg yaml 보존 (Phase A: predict.py에서 재사용)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_path and Path(best_path).exists() and Path(best_path).resolve() != (out_dir / "best.ckpt").resolve():
        shutil.copy(best_path, out_dir / "best.ckpt")
    torch.save(model.state_dict(), out_dir / "tft_state_dict.pt")
    with open(out_dir / "training_dataset.pkl", "wb") as f:
        pickle.dump(training, f)
    if cfg is not None:
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TFT 학습 단독 실행")
    p.add_argument("--config", type=str, required=True, help="configs/tft.yaml 경로")
    p.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    p.add_argument("--max-epochs", type=int, default=None, help="config 덮어쓰기")
    p.add_argument("--data", type=str, default=None, help="data parquet 경로 덮어쓰기")
    p.add_argument("--train-decoder-end", type=str, default=None,
                   help="walk-forward train 윈도우 끝 (YYYY-MM-DD). config split.train_decoder_end 덮어쓰기")
    p.add_argument("--val-cutoff", type=str, default=None,
                   help="walk-forward val 윈도우 끝 (YYYY-MM-DD). config split.val_cutoff 덮어쓰기")
    p.add_argument("--out-dir", type=str, default=None,
                   help="산출물 디렉토리. config artifacts.out_dir 덮어쓰기")
    p.add_argument("--dry-run", action="store_true", help="tiny fixture + 1 epoch (e2e smoke)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tc = TrainConfig.from_yaml(args.config)
    if args.dry_run:
        tc = tc.apply_dry_run()
    if args.max_epochs is not None:
        tc.cfg["trainer"]["max_epochs"] = args.max_epochs
    if args.data is not None:
        tc.cfg["data"]["parquet_path"] = args.data
    if args.train_decoder_end is not None:
        tc.cfg["split"]["train_decoder_end"] = args.train_decoder_end
    if args.val_cutoff is not None:
        tc.cfg["split"]["val_cutoff"] = args.val_cutoff
    if args.out_dir is not None:
        tc.cfg["artifacts"]["out_dir"] = args.out_dir

    print(f"=== TFT train ===")
    print(f"  config             = {args.config}")
    print(f"  data               = {tc.cfg['data']['parquet_path']}")
    print(f"  train_decoder_end  = {tc.cfg['split']['train_decoder_end']}")
    print(f"  val_cutoff         = {tc.cfg['split']['val_cutoff']}")
    print(f"  max_epochs         = {tc.cfg['trainer']['max_epochs']}")
    print(f"  out_dir            = {tc.cfg['artifacts']['out_dir']}")
    print(f"  dry_run            = {args.dry_run}")
    print(f"  source_tag         = {tc.cfg.get('mlflow', {}).get('source_tag', 'bg')}")

    result = train_tft(tc.cfg, run_name=args.run_name)
    print(f"\nbest ckpt: {result['best_path']}")
    print(f"out_dir:   {result['out_dir']}")


if __name__ == "__main__":
    main()
