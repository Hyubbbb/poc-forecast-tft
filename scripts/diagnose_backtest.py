"""Validation 진단 + 6개월 backtest 실측-예측 시각화.

이슈 진단:
- MLflow `val_MAPE` 가 1e8+ 인 이유 (sparse target + max 24k skew → 분모 0/근접에서 폭발).
- 실제 모델 품질 = WAPE/MAE/p50-residual + horizon별 / STYLE별 trend.

산출물 (artifacts/<out_dir>/diagnose/):
- summary.json           — 전역 metric (WAPE, MAE, MAPE-safe, SMAPE, coverage)
- horizon_metrics.csv    — horizon × {WAPE, MAE, bias, n}
- horizon_metrics.png    — horizon × WAPE line plot
- style_weekly.csv       — STYLE_CD × forecast_week × {actual, q_mid, q_low, q_high, wape}
- style_weekly.png       — STYLE × forecast_week actual vs median line plot
- sample_sc.png          — 샘플 SC fan-chart (low-high quantile band + actual)
- val_zero_diagnosis.png — actual=0 vs actual>0 row 분포 (MAPE 폭발 원인 시각화)

⚠️ quantile 컬럼명은 학습 config 의 `model.quantiles` 그대로 사용 (`q25/q50/q75` 등).
   과거 코드의 `p10/p90` 라벨은 실제 학습 quantile 과 무관한 fake label 이었음 — 제거.

사용:
    .venv/bin/python scripts/diagnose_backtest.py \
        --artifact-dir artifacts/tft_supplies_cp_e52_d26 \
        --data data/sc_weekly_cp6677_model_ready.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps_frac: float = 0.01) -> float:
    """MAPE-safe: 분모를 max(|y|, eps_frac * mean|y|) 로 floor → sparse 폭발 차단."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    floor = max(eps_frac * float(np.abs(y_true).mean()), 1.0)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.abs(y_true).sum())
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)[mask] / denom[mask]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out-name", default="diagnose")
    p.add_argument("--n-sample-sc", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()

    from scripts.eval_utils import resolve_quantile_cols
    from scripts.forecast_utils import load_artifact, predict_dataframe
    from scripts.train import prepare_features

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir

    loaded = load_artifact(artifact_dir)
    cfg = loaded["config"]
    model = loaded["model"]
    training_ds = loaded["training_dataset"]

    target = cfg["data"]["target"]
    group_key = cfg["data"]["group_key"]
    decoder_len = cfg["dataset"]["decoder_len"]
    train_decoder_end = pd.Timestamp(cfg["split"]["train_decoder_end"])  # 학습 마지막 주 = forecast cutoff

    out_dir = artifact_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== diagnose backtest ===")
    print(f"  artifact_dir = {artifact_dir}")
    print(f"  data         = {args.data}")
    print(f"  target       = {target}")
    print(f"  group_key    = {group_key}")
    print(f"  cutoff       = {train_decoder_end}")
    print(f"  decoder_len  = {decoder_len}")
    print(f"  out_dir      = {out_dir}")

    df = pd.read_parquet(args.data)
    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"])
    ds_cfg = {**cfg["dataset"], "group_key": group_key, "target": target}
    df_feat = prepare_features(df, ds_cfg)

    forecast = predict_dataframe(
        model, training_ds, df_feat, train_decoder_end,
        decoder_len=decoder_len, group_key=group_key,
    )
    qmap = resolve_quantile_cols(forecast, model)
    LOW, MID, HIGH = qmap["low"], qmap["mid"], qmap["high"]
    # 학습 시 사용한 quantile 그대로. low/high 간격(예: q25-q75 = IQR=50%CI, q10-q90 = 80%CI)에
    # 따라 coverage 의 이론적 목표가 달라짐 — summary 출력 시 명시.
    coverage_target = float(qmap["quantiles"][-1] - qmap["quantiles"][0])
    print(f"  quantiles    = {qmap['quantiles']}  → cols low/mid/high = {LOW}/{MID}/{HIGH}")
    print(f"  coverage 이론 목표 = {coverage_target:.0%} ({LOW}~{HIGH})")

    val = df_feat[(df_feat["WEEK_START"] > train_decoder_end)
                  & (df_feat["WEEK_START"] <= train_decoder_end + pd.Timedelta(weeks=decoder_len))]
    val = val[[group_key, "WEEK_START", target]].rename(
        columns={"WEEK_START": "forecast_week", target: "actual"}
    )
    # ⚠️ inner-join 으로 실제 살아있는 (series, week) 만 평가.
    # left-join + fillna(0) 시 라이프사이클 종료된 series 가 fake-zero 로 들어가 MAPE/WAPE 가 폭발.
    join = forecast.merge(val, on=[group_key, "forecast_week"], how="inner")
    n_predicted = forecast[group_key].nunique()
    n_alive = join[group_key].nunique()
    print(f"\n  forecasted series = {n_predicted}, alive in val window = {n_alive} (drop {n_predicted - n_alive})")
    print(f"  joined rows       = {len(join):,} ({n_alive} SC × {join['forecast_week'].nunique()} weeks)")

    y = join["actual"].to_numpy()
    y_mid = join[MID].to_numpy()
    y_low = join[LOW].to_numpy()
    y_high = join[HIGH].to_numpy()
    summary = {
        "n_rows": int(len(join)),
        "n_sc": int(join[group_key].nunique()),
        "n_weeks": int(join["forecast_week"].nunique()),
        "quantiles": qmap["quantiles"],
        "actual_sum": float(y.sum()),
        "actual_mean": float(y.mean()),
        "actual_zero_ratio": float((y == 0).mean()),
        f"{MID}_sum": float(y_mid.sum()),
        f"{MID}_mean": float(y_mid.mean()),
        f"wape_{MID}": wape(y, y_mid),
        f"mae_{MID}": float(np.abs(y - y_mid).mean()),
        f"rmse_{MID}": float(np.sqrt(np.mean((y - y_mid) ** 2))),
        f"bias_{MID}": float((y_mid - y).mean()),
        f"smape_{MID}": smape(y, y_mid),
        f"mape_safe_{MID}": safe_mape(y, y_mid),
        f"{LOW}_{HIGH}_coverage": float(((y >= y_low) & (y <= y_high)).mean()),
        f"{LOW}_{HIGH}_coverage_target": coverage_target,
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:25s} = {v:,}")
        else:
            print(f"  {k:25s} = {v}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    rows = []
    for h, sub in join.groupby("h"):
        y_h, p_h = sub["actual"].to_numpy(), sub[MID].to_numpy()
        rows.append({
            "h": int(h),
            "forecast_week": sub["forecast_week"].iloc[0],
            "n": len(sub),
            "actual_sum": float(y_h.sum()),
            f"{MID}_sum": float(p_h.sum()),
            "wape": wape(y_h, p_h),
            "mae": float(np.abs(y_h - p_h).mean()),
            "rmse": float(np.sqrt(np.mean((y_h - p_h) ** 2))),
            "mape_safe": safe_mape(y_h, p_h),
            "bias": float((p_h - y_h).mean()),
            "coverage": float(((sub["actual"] >= sub[LOW]) & (sub["actual"] <= sub[HIGH])).mean()),
        })
    horizon_df = pd.DataFrame(rows).sort_values("h")
    horizon_df.to_csv(out_dir / "horizon_metrics.csv", index=False)
    print("\n=== horizon × metric (head 6) ===")
    print(horizon_df.head(6).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(horizon_df["h"], horizon_df["wape"], marker="o", label="WAPE")
    axes[0].plot(horizon_df["h"], horizon_df["mape_safe"], marker="s", label="MAPE-safe")
    axes[0].set_xlabel("horizon (weeks)"); axes[0].set_ylabel("error rate")
    axes[0].set_title("horizon × WAPE / MAPE-safe"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(horizon_df["h"], horizon_df["actual_sum"], "o-", label="actual sum")
    axes[1].plot(horizon_df["h"], horizon_df[f"{MID}_sum"], "x-", label=f"{MID} sum")
    axes[1].set_xlabel("horizon (weeks)"); axes[1].set_ylabel("qty sum")
    axes[1].set_title(f"horizon × actual vs {MID} sum"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Validation backtest — {train_decoder_end.date()} + {decoder_len}w")
    plt.tight_layout()
    fig.savefig(out_dir / "horizon_metrics.png", dpi=120)
    plt.close(fig)

    sc_meta_cols = [c for c in ["STYLE_CD", "TEAM_CD"] if c in df_feat.columns]
    if sc_meta_cols:
        meta = df_feat[[group_key] + sc_meta_cols].drop_duplicates(group_key)
        join = join.merge(meta, on=group_key, how="left")

    if "STYLE_CD" in join.columns:
        style_weekly = (
            join.groupby(["STYLE_CD", "forecast_week"])
            .agg(**{
                "actual": ("actual", "sum"),
                MID: (MID, "sum"),
                LOW: (LOW, "sum"),
                HIGH: (HIGH, "sum"),
                "n_sc": (group_key, "nunique"),
            })
            .reset_index()
        )
        style_weekly["wape"] = (style_weekly["actual"] - style_weekly[MID]).abs() / style_weekly["actual"].clip(lower=1)
        style_weekly.to_csv(out_dir / "style_weekly.csv", index=False)

        band_label = f"{LOW}-{HIGH} ({coverage_target:.0%} CI)"
        styles = sorted(style_weekly["STYLE_CD"].unique())
        fig, axes = plt.subplots(len(styles), 1, figsize=(11, 3.5 * len(styles)), squeeze=False)
        for ax, style in zip(axes[:, 0], styles):
            sub = style_weekly[style_weekly["STYLE_CD"] == style].sort_values("forecast_week")
            ax.fill_between(sub["forecast_week"], sub[LOW], sub[HIGH], alpha=0.2, label=band_label)
            ax.plot(sub["forecast_week"], sub[MID], "x-", label=f"{MID} (predicted)")
            ax.plot(sub["forecast_week"], sub["actual"], "o-", label="actual")
            ax.set_title(f"{style}  (n_sc≈{int(sub['n_sc'].mean())})")
            ax.set_ylabel("weekly qty sum"); ax.legend(); ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        fig.savefig(out_dir / "style_weekly.png", dpi=120)
        plt.close(fig)

    actual_total_by_sc = join.groupby(group_key)["actual"].sum().sort_values(ascending=False)
    sample_scs = actual_total_by_sc.head(args.n_sample_sc).index.tolist()
    cols = 2
    rows_n = (len(sample_scs) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(13, 3.2 * rows_n))
    axes = np.atleast_2d(axes)
    for ax, sc in zip(axes.flat, sample_scs):
        sub = join[join[group_key] == sc].sort_values("forecast_week")
        hist = df_feat[(df_feat[group_key] == sc)
                       & (df_feat["WEEK_START"] >= train_decoder_end - pd.Timedelta(weeks=26))
                       & (df_feat["WEEK_START"] <= train_decoder_end + pd.Timedelta(weeks=decoder_len))]
        ax.plot(hist["WEEK_START"], hist[target], "o-", ms=3, color="0.5", label="actual (hist+val)")
        ax.fill_between(sub["forecast_week"], sub[LOW], sub[HIGH], alpha=0.3,
                        label=f"{LOW}-{HIGH} ({coverage_target:.0%} CI)")
        ax.plot(sub["forecast_week"], sub[MID], "x-", label=MID)
        ax.axvline(train_decoder_end, color="red", lw=0.8, ls="--", label="cutoff")
        ax.set_title(str(sc)[:32]); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.flat[len(sample_scs):]:
        ax.set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "sample_sc.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    is_zero = join["actual"] == 0
    axes[0].hist(join.loc[is_zero, MID], bins=60, alpha=0.6, label=f"actual=0  (n={is_zero.sum()})")
    axes[0].hist(join.loc[~is_zero, MID], bins=60, alpha=0.6, label=f"actual>0 (n={(~is_zero).sum()})")
    axes[0].set_xlabel(f"{MID} prediction"); axes[0].set_ylabel("count")
    axes[0].set_title(f"{MID} by actual=0 vs >0  — MAPE 폭발 원인 진단")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    nz = ~is_zero
    if nz.any():
        per_row_mape = (join.loc[nz, "actual"] - join.loc[nz, MID]).abs() / join.loc[nz, "actual"].abs()
        axes[1].hist(np.log10(per_row_mape.clip(lower=1e-3, upper=1e9)), bins=60)
        axes[1].set_xlabel("log10(|y-p|/|y|)  for actual>0")
        axes[1].set_title(f"per-row MAPE (actual>0) — max={per_row_mape.max():.2g}")
        axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "val_zero_diagnosis.png", dpi=120)
    plt.close(fig)

    print(f"\n✅ saved → {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
