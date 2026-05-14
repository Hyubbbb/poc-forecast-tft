"""MLflow 적재 SoT — tags / metrics / nested run / dataset audit.

spec: `specs/mlflow-experiment-tracking.md` v0.2.0

주요 함수:
- `make_run_tags(...)`: 8개 표준 tag dict 생성
- `log_full_metrics(run_id, metrics_dict, ...)`: flat dot-prefix dict → MLflow log_metric
- `compute_dataset_audit(df, ...)`: style_list/sha256/sanity params
- `safe_git_sha()`: 자동 캡쳐
- `infer_model_architecture(cfg)`: cfg → "tft"/"lgbm"/...

설계 원칙:
- MLflow 호출은 모두 본 모듈에서만. 다른 코드는 본 모듈만 import.
- mlflow.parentRunId 는 MLflow native 가 자동 부착 → 별도 tag 없음.
- random_seed 미설정 시 명시적 경고 + tag seed_not_set=true.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_git_sha() -> str:
    """현재 HEAD 의 짧은 sha. uncommitted change 있으면 `<sha>-dirty`. 실패 시 'unknown'."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def infer_model_architecture(cfg: dict) -> str:
    """cfg 구조에서 모델 architecture 식별. TFT 외 모델 추가 시 확장."""
    model_cfg = cfg.get("model", {}) or {}
    if "quantiles" in model_cfg and "hidden_size" in model_cfg:
        return "tft"
    if "n_estimators" in model_cfg or "boosting_type" in model_cfg:
        return "lgbm"
    return cfg.get("model_architecture", "tft")  # explicit override


def make_run_tags(
    *,
    experiment_intent: str,
    data_scope: str,
    backtest_fold: str = "single",
    model_architecture: str = "tft",
    random_seed: int | None = None,
    is_baseline: bool = False,
    source: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """spec v0.2.0 의 8개 표준 tag dict 생성.

    - source: 기존 호환 (`cfg.mlflow.source_tag`). None 이면 'unset' 으로 적재.
    - experiment_intent: snake_case 사람-읽기 (예: `baseline_v1`)
    - data_scope: 데이터 식별 (예: `cp6677_only`)
    - backtest_fold: 'single' / '2024' / '2024-Q3'
    - model_architecture: 'tft'/'lgbm'/'nbeats'/... (cross-architecture 비교용)
    - random_seed: int. None 이면 `seed_not_set=true` 부착 + 경고.
    - is_baseline: comparison report 의 default baseline 식별 (`data_scope` 별 1개)

    Returns 모든 value 는 str (MLflow tag 계약).
    """
    tags: dict[str, str] = {
        "source": source or "unset",
        "experiment_intent": experiment_intent,
        "data_scope": data_scope,
        "backtest_fold": backtest_fold,
        "git_sha": safe_git_sha(),
        "model_architecture": model_architecture,
        "is_baseline": "true" if is_baseline else "false",
    }
    if random_seed is not None:
        tags["random_seed"] = str(int(random_seed))
    else:
        tags["random_seed"] = "unset"
        tags["seed_not_set"] = "true"
        warnings.warn(
            "random_seed 미설정 — 재현성 보장 안됨. cfg.train.seed 추가 권장.",
            UserWarning,
            stacklevel=2,
        )
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags


def file_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """파일 sha256 hex digest. 동일 데이터 검증용."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_audit(
    df: pd.DataFrame,
    *,
    parquet_path: str | Path,
    cutoff,
    group_key: str = "SC_CD",
    style_col: str = "STYLE_CD",
    target: str = "WEEKLY_SALE_QTY",
    val_end=None,
) -> dict[str, Any]:
    """학습 데이터 sanity audit. spec v0.2.0 의 params 표 그대로.

    Returns dict (MLflow log_params 호환):
        style_list: json str ['CP66','CP77']
        style_count: int
        n_series_per_style: json dict {'CP66': 31, 'CP77': 45}
        input_parquet_path: str
        input_parquet_sha256: str
        n_series_train, n_rows_train: int
        n_series_val_alive, n_rows_val: int  (val_end 제공 시)
        target_zero_ratio_train: float

    Args:
        df: 학습 입력 DataFrame (WEEK_START 컬럼 필요)
        parquet_path: 학습 입력 parquet 경로 (sha256 계산용)
        cutoff: train_decoder_end (train 종료 timestamp)
        val_end: val_cutoff (val 종료 timestamp). None 이면 val sanity 생략.
    """
    cutoff = pd.Timestamp(cutoff)
    df = df.copy()
    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"])
    train_mask = df["WEEK_START"] <= cutoff
    train = df[train_mask]

    audit: dict[str, Any] = {
        "input_parquet_path": str(parquet_path),
        "input_parquet_sha256": file_sha256(parquet_path),
        "n_series_train": int(train[group_key].nunique()),
        "n_rows_train": int(len(train)),
    }
    if target in train.columns:
        audit["target_zero_ratio_train"] = float((train[target] == 0).mean()) if len(train) else 0.0
    if style_col in df.columns:
        by_style = (
            df.drop_duplicates(group_key)[[group_key, style_col]]
            .groupby(style_col)
            .size()
            .to_dict()
        )
        audit["style_list"] = json.dumps(sorted({str(s) for s in df[style_col].dropna().unique()}))
        audit["style_count"] = len(by_style)
        audit["n_series_per_style"] = json.dumps({str(k): int(v) for k, v in by_style.items()})
    if val_end is not None:
        val_end = pd.Timestamp(val_end)
        val = df[(df["WEEK_START"] > cutoff) & (df["WEEK_START"] <= val_end)]
        audit["n_series_val_alive"] = int(val[group_key].nunique())
        audit["n_rows_val"] = int(len(val))
    return audit


def system_env_params() -> dict[str, str]:
    """torch/PF 버전 + device 자동 캡쳐. params 적재용."""
    out: dict[str, str] = {}
    try:
        import torch
        out["torch_version"] = str(torch.__version__)
        if torch.backends.mps.is_available():
            out["device"] = "mps"
        elif torch.cuda.is_available():
            out["device"] = "cuda"
        else:
            out["device"] = "cpu"
    except Exception:
        out["torch_version"] = "unknown"
        out["device"] = "unknown"
    try:
        import pytorch_forecasting as pf
        out["pytorch_forecasting_version"] = str(pf.__version__)
    except Exception:
        out["pytorch_forecasting_version"] = "unknown"
    return out


def log_full_metrics(
    run_id: str,
    metrics: dict[str, float],
    *,
    tracking_uri: str | None = None,
    step: int | None = None,
) -> int:
    """flat dot-prefix metric dict → MLflow log_metric. 적재된 개수 반환.

    NaN/inf metric 은 skip (MLflow 가 거부함).
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    n_logged = 0
    with mlflow.start_run(run_id=run_id):
        for key, value in metrics.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            if step is not None:
                mlflow.log_metric(key, v, step=step)
            else:
                mlflow.log_metric(key, v)
            n_logged += 1
    return n_logged


def log_params_safe(
    run_id: str,
    params: dict[str, Any],
    *,
    tracking_uri: str | None = None,
    max_value_len: int = 500,
) -> int:
    """params dict → mlflow.log_param. MLflow 의 250자 제한 회피 위해 long string truncate.

    None 은 skip. dict/list 는 json.dumps. 적재된 개수 반환.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    n_logged = 0
    with mlflow.start_run(run_id=run_id):
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                v_str = json.dumps(value)
            else:
                v_str = str(value)
            if len(v_str) > max_value_len:
                v_str = v_str[: max_value_len - 14] + "...[TRUNCATED]"
            try:
                mlflow.log_param(key, v_str)
                n_logged += 1
            except Exception as e:  # MLflow rejects duplicate param key
                warnings.warn(f"log_param({key}) skipped: {e}", stacklevel=2)
    return n_logged


def set_run_tags(
    run_id: str,
    tags: dict[str, str],
    *,
    tracking_uri: str | None = None,
) -> int:
    """tag dict → mlflow.set_tag. 적재 개수 반환."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    n_logged = 0
    with mlflow.start_run(run_id=run_id):
        for key, value in tags.items():
            mlflow.set_tag(key, str(value))
            n_logged += 1
    return n_logged


def log_aggregate_metrics_to_parent(
    parent_run_id: str,
    child_metrics_list: list[dict[str, float]],
    *,
    tracking_uri: str | None = None,
    namespace: str = "aggregate",
) -> int:
    """child metric dict list → parent run 에 mean/median/std 집계 적재.

    Returns 적재 개수.

    예: child 가 wape_q50 각각 [0.55, 0.48, 0.62] → parent 에:
        aggregate.mean.wape_q50 = 0.55, aggregate.std.wape_q50 = 0.057, ...
    """
    if not child_metrics_list:
        return 0
    # 모든 child 가 공유하는 key 만 집계
    common_keys = set(child_metrics_list[0].keys())
    for m in child_metrics_list[1:]:
        common_keys &= set(m.keys())

    aggregates: dict[str, float] = {}
    for key in common_keys:
        values = [m[key] for m in child_metrics_list]
        finite = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        if not finite:
            continue
        arr = np.asarray(finite, dtype=float)
        aggregates[f"{namespace}.mean.{key}"] = float(arr.mean())
        aggregates[f"{namespace}.median.{key}"] = float(np.median(arr))
        if len(arr) > 1:
            aggregates[f"{namespace}.std.{key}"] = float(arr.std(ddof=0))
    return log_full_metrics(parent_run_id, aggregates, tracking_uri=tracking_uri)


def start_parent_run(
    experiment: str,
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> str:
    """parent run 시작 → run_id 반환. child run 들은 이 id 를 `parent_run_id` 로 사용.

    tags 에 `is_parent=true` + `n_children` 자동 부착.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    parent_tags = {"is_parent": "true"}
    if tags:
        parent_tags.update({k: str(v) for k, v in tags.items()})
    with mlflow.start_run(run_name=run_name, tags=parent_tags) as run:
        return run.info.run_id


def update_parent_children_count(
    parent_run_id: str,
    n_children: int,
    *,
    tracking_uri: str | None = None,
) -> None:
    """parent run 의 `n_children` tag 갱신."""
    set_run_tags(parent_run_id, {"n_children": str(int(n_children))}, tracking_uri=tracking_uri)
