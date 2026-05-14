"""학습 artifact 재사용 + 추론·평가 SoT (cp6677 baseline).

`scripts/eval_utils.py`의 저수준 함수(`wape`, `make_naive_cohort_mean`, `predict_with_tft`)
위에 노트북·CLI·`scripts/train.py` 자동 평가가 공유할 고수준 wrapper를 제공한다.

핵심:
- `load_artifact(artifact_dir)`: best.ckpt + training_dataset.pkl + config.yaml 로드
- `predict_dataframe(...)`: `predict_with_tft` dict → SC×h DataFrame 정리 (q-prefix 컬럼)
- `evaluate_horizons(...)`: horizon × WAPE/n_sc 표 (BRAND 슬라이스 옵션)
- `_bin_horizon(...)`: cold/mid/far bin 분류 (eval_utils.compute_bin_metrics 가 lazy import)
- `flatten_cfg(...)`: nested → flat (MLflow log_params 입력용)

Stage 0 (MLflow 인프라) 이후, 학습 후 metric 적재는 `scripts.mlflow_logging.log_full_metrics`
가 SoT. 본 모듈의 metric 함수는 cp6677 notebook 의 inline 분석용 + train.py 의 fallback 용.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.eval_utils import (
    make_naive_cohort_mean,
    predict_with_tft,
    quantile_col_name,
    resolve_quantile_cols,
    wape,
)


def load_artifact(artifact_dir: str | Path) -> dict:
    """학습 결과물 디렉토리에서 model + training_dataset (+ config) 로드.

    필수: best.ckpt, training_dataset.pkl
    선택: config.yaml (없으면 None 반환)
    """
    from pytorch_forecasting import TemporalFusionTransformer

    artifact_dir = Path(artifact_dir)
    ckpt = artifact_dir / "best.ckpt"
    ts_pkl = artifact_dir / "training_dataset.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(f"best.ckpt 없음: {ckpt}")
    if not ts_pkl.exists():
        raise FileNotFoundError(f"training_dataset.pkl 없음: {ts_pkl}")

    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
    with open(ts_pkl, "rb") as f:
        training_dataset = pickle.load(f)

    config = None
    cfg_path = artifact_dir / "config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

    return {"model": model, "training_dataset": training_dataset, "config": config}


def predict_dataframe(
    model,
    training_dataset,
    df: pd.DataFrame,
    cutoff,
    *,
    decoder_len: int = 8,
    batch_size: int = 128,
    group_key: str = "SC_CD",
) -> pd.DataFrame:
    """`predict_with_tft` dict 반환을 SC×h long DataFrame으로 정리.

    columns: [group_key, h, forecast_week, q{NN}...]
    q-컬럼명은 `model.loss.quantiles` 그대로 라벨링.
    예: quantiles=[0.25, 0.5, 0.75] → q25, q50, q75
    """
    cutoff = pd.Timestamp(cutoff)
    out = predict_with_tft(model, training_dataset, df, cutoff, decoder_len=decoder_len, batch_size=batch_size)
    idx_df = out["index"].copy() if isinstance(out["index"], pd.DataFrame) else pd.DataFrame(out["index"])
    forecast_weeks = pd.DatetimeIndex(out["forecast_weeks"])
    quantiles = out["quantiles"]
    qcol_names = [quantile_col_name(q) for q in quantiles]
    preds = out["preds"]  # (n, decoder_len, n_q)

    rows = []
    for i, sc in enumerate(idx_df[group_key].values):
        for h in range(decoder_len):
            row = {group_key: sc, "h": h + 1, "forecast_week": forecast_weeks[h]}
            for j, name in enumerate(qcol_names):
                row[name] = float(preds[i, h, j])
            rows.append(row)
    return pd.DataFrame(rows, columns=[group_key, "h", "forecast_week"] + qcol_names)


def evaluate_horizons(
    df: pd.DataFrame,
    model,
    training_dataset,
    cutoff,
    *,
    decoder_len: int = 8,
    baselines: tuple[str, ...] = ("naive_cohort_mean",),
    brand_slice: bool = False,
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
) -> pd.DataFrame:
    """horizon별 WAPE/n_sc 표. (model, h, [BRAND], wape, n_sc) row.

    baselines: 추가로 비교할 baseline 이름 — `naive_cohort_mean`만 지원 (추후 확장).
    brand_slice: True면 BRAND_CD 슬라이스 row도 포함 (df에 BRAND_CD 컬럼 필요).
    """
    cutoff = pd.Timestamp(cutoff)
    forecast = predict_dataframe(model, training_dataset, df, cutoff, decoder_len=decoder_len, group_key=group_key)
    mid_col = resolve_quantile_cols(forecast, model)["mid"]
    tft_label = f"tft_{mid_col}"

    df = df.copy()
    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"])

    actuals_long = []
    for h in range(1, decoder_len + 1):
        wk = cutoff + pd.Timedelta(weeks=h)
        sub = df.loc[df["WEEK_START"] == wk, [group_key, target]].rename(columns={target: "actual"})
        sub["h"] = h
        actuals_long.append(sub)
    actuals = pd.concat(actuals_long, ignore_index=True)

    cm_predict = make_naive_cohort_mean(df) if "naive_cohort_mean" in baselines else None
    sc_brand = (
        df.drop_duplicates(group_key).set_index(group_key)["BRAND_CD"]
        if brand_slice and "BRAND_CD" in df.columns
        else None
    )

    rows = []
    for h in range(1, decoder_len + 1):
        actual_h = actuals[actuals["h"] == h].set_index(group_key)["actual"]
        if actual_h.empty:
            continue
        f_h = forecast[forecast["h"] == h].set_index(group_key)
        common = actual_h.index.intersection(f_h.index)
        if len(common) == 0:
            continue

        rows.append({
            "model": tft_label, "h": h,
            "wape": wape(actual_h.loc[common].values, f_h.loc[common, mid_col].values),
            "n_sc": int(len(common)),
        })
        if brand_slice and sc_brand is not None:
            for brand in sorted(sc_brand.unique()):
                brand_sc = sc_brand[sc_brand == brand].index
                cmn_b = brand_sc.intersection(common)
                if len(cmn_b) == 0:
                    continue
                rows.append({
                    "model": tft_label, "BRAND": brand, "h": h,
                    "wape": wape(actual_h.loc[cmn_b].values, f_h.loc[cmn_b, mid_col].values),
                    "n_sc": int(len(cmn_b)),
                })

        if cm_predict is not None:
            cm_h = pd.Series({sc: cm_predict(df, sc, cutoff, h) for sc in common}).dropna()
            if not cm_h.empty:
                cmn_cm = cm_h.index
                rows.append({
                    "model": "naive_cohort_mean", "h": h,
                    "wape": wape(actual_h.loc[cmn_cm].values, cm_h.values),
                    "n_sc": int(len(cmn_cm)),
                })

    cols = ["model", "h", "wape", "n_sc"]
    if brand_slice:
        cols = ["model", "h", "BRAND", "wape", "n_sc"]
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return pd.DataFrame(columns=cols)
    if brand_slice and "BRAND" not in df_out.columns:
        df_out["BRAND"] = pd.NA
    return df_out[[c for c in cols if c in df_out.columns]]


def _bin_horizon(h: int) -> str:
    """horizon → cold(t+1..4) / mid(t+5..12) / far(t+13..). plan v0.3.0 R8."""
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if h <= 4:
        return "cold"
    if h <= 12:
        return "mid"
    return "far"


def flatten_cfg(cfg: dict, *, sep: str = ".", prefix: str = "") -> dict[str, Any]:
    """nested dict → flat. list/tuple은 str로 변환 (MLflow log_params 호환)."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_cfg(v, sep=sep, prefix=key))
        elif isinstance(v, (list, tuple)):
            out[key] = str(list(v))
        else:
            out[key] = v
    return out


