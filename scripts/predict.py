"""학습된 TFT artifact로 신규 row 예측 — Phase A.

학습은 수 시간이 소요되므로, 학습 끝난 ckpt + training_dataset.pkl + config.yaml을
재사용해 임의 cutoff에서 horizon 예측을 산출한다.

사용 예:
    .venv/bin/python scripts/predict.py \\
        --artifact-dir artifacts/dry_run_cp_steady_d8 \\
        --input data/sc_weekly_supplies.parquet \\
        --cutoff 2026-04-01 \\
        --out forecasts/supplies_cp_steady_d8_2026-04-01.parquet

옵션:
    --decoder-len <INT>          # config["dataset"]["decoder_len"] 덮어쓰기
    --log-mlflow-run-id <RUN_ID> # 학습 run에 inference artifact 추가 적재
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="학습된 TFT artifact로 신규 row 예측")
    p.add_argument("--artifact-dir", type=str, required=True,
                   help="best.ckpt + training_dataset.pkl + config.yaml이 든 디렉토리")
    p.add_argument("--input", type=str, required=True,
                   help="입력 parquet (학습 데이터와 동일 schema). cutoff 이전 history 포함 필요")
    p.add_argument("--cutoff", type=str, required=True,
                   help="예측 기준 주차 (YYYY-MM-DD). t+1..t+decoder_len 예측 산출")
    p.add_argument("--out", type=str, required=True,
                   help="forecast parquet 저장 경로")
    p.add_argument("--decoder-len", type=int, default=None,
                   help="config의 decoder_len 덮어쓰기 (옵션)")
    p.add_argument("--log-mlflow-run-id", type=str, default=None,
                   help="기존 학습 run에 inference artifact 추가 적재 (옵션)")
    p.add_argument("--mlflow-tracking-uri", type=str, default=None,
                   help="--log-mlflow-run-id 사용 시 tracking URI. 미지정 시 config.yaml의 mlflow.tracking_uri 사용")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from scripts.forecast_utils import load_artifact, predict_dataframe
    from scripts.train import prepare_features

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir

    print("=== predict ===")
    print(f"  artifact_dir = {artifact_dir}")
    print(f"  input        = {args.input}")
    print(f"  cutoff       = {args.cutoff}")
    print(f"  out          = {args.out}")

    loaded = load_artifact(artifact_dir)
    cfg = loaded["config"] or {}

    decoder_len = args.decoder_len or cfg.get("dataset", {}).get("decoder_len", 8)
    print(f"  decoder_len  = {decoder_len}")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    df = pd.read_parquet(input_path)
    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"])

    # 학습 시와 동일한 feature 가공 (in_season_filter 분기 보존)
    ds_cfg = cfg.get("dataset", {})
    ds_cfg = {
        **ds_cfg,
        "group_key": cfg.get("data", {}).get("group_key", "SC_CD"),
        "target": cfg.get("data", {}).get("target", "WEEKLY_SALE_QTY_CNS"),
    }
    df = prepare_features(df, ds_cfg)

    cutoff = pd.Timestamp(args.cutoff)
    forecast = predict_dataframe(
        loaded["model"], loaded["training_dataset"], df, cutoff,
        decoder_len=decoder_len,
    )

    # SC별 시점-불변 metadata를 forecast에 left join (STYLE/COLOR 단위 사후분석용).
    # input df에 존재하는 컬럼만 자동 선택 → 스키마 변경에 안전.
    META_COLS = ["STYLE_CD", "COLOR_CD", "PROD_CD", "COLOR_BASE_CD", "TEAM_CD", "SESN_SUB", "SSN_CD"]
    present_meta = [c for c in META_COLS if c in df.columns]
    if present_meta:
        meta = df[["SC_CD"] + present_meta].drop_duplicates(subset=["SC_CD"])
        forecast = forecast.merge(meta, on="SC_CD", how="left")
        print(f"  meta cols joined = {present_meta}")

    # 음수 0 clip — 운영 안전장치 (qty < 0 은 의미상 불가능).
    # quantile 컬럼은 `q\d{2}` 형식 (학습 config 의 model.quantiles 그대로).
    quant_cols = [c for c in forecast.columns if len(c) == 3 and c.startswith("q") and c[1:].isdigit()]
    if quant_cols:
        before_neg = int((forecast[quant_cols] < 0).any(axis=1).sum())
        forecast[quant_cols] = forecast[quant_cols].clip(lower=0)
        if before_neg:
            print(f"  clipped {before_neg} rows with negative quantile predictions ({quant_cols})")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(out_path, index=False)
    print(f"\n✅ forecast saved: {out_path} ({len(forecast):,} rows, {forecast['SC_CD'].nunique():,} SC × h={decoder_len})")

    if args.log_mlflow_run_id:
        import mlflow
        tracking_uri = args.mlflow_tracking_uri or cfg.get("mlflow", {}).get("tracking_uri")
        if not tracking_uri:
            print("⚠️  MLflow tracking URI 미지정 — log skip")
            return
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=args.log_mlflow_run_id):
            mlflow.log_artifact(str(out_path), artifact_path="inference")
            mlflow.set_tag(f"inference_cutoff_{args.cutoff}", "logged")
        print(f"✅ logged to MLflow run {args.log_mlflow_run_id}")


if __name__ == "__main__":
    main()
