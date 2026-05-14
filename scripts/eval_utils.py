"""평가용 순수 유틸 — 노트북과 tests 양쪽에서 import.

cell 13(wape, naive baselines), cell 15(mini_lgb_t1), cell 19(predict_with_tft)의 SoT.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error. denom=0 또는 모두 NaN이면 NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.size == 0:
        return float("nan")
    denom = float(np.nansum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    err = float(np.nansum(np.abs(y_true - y_pred)))
    return err / denom


def _recent_mean(df: pd.DataFrame, sc, cutoff, n: int = 4, *, group_key: str = "SC_CD", target: str = "WEEKLY_SALE_QTY_CNS") -> float:
    recent = (
        df[(df[group_key] == sc) & (df["WEEK_START"] <= cutoff)]
        .sort_values("WEEK_START")
        .tail(n)
    )
    return float(recent[target].mean()) if len(recent) else float("nan")


def naive_persistence(df: pd.DataFrame, sc, cutoff, h: int, *, group_key: str = "SC_CD", target: str = "WEEKLY_SALE_QTY_CNS") -> float:
    """y[t+h] = y[t]. h는 무시(모든 horizon에 마지막 관측치)."""
    last = (
        df[(df[group_key] == sc) & (df["WEEK_START"] <= cutoff)]
        .sort_values("WEEK_START")
        .tail(1)
    )
    return float(last[target].iloc[0]) if len(last) else float("nan")


def build_cohort_lookup(
    df_full: pd.DataFrame,
    cutoff,
    *,
    cohort_keys: tuple[str, ...] = ("SESN_SUB", "PRDT_KIND_CD", "ITEM"),
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
) -> dict:
    d = df_full[df_full["WEEK_START"] <= cutoff].copy()
    first = d.groupby(group_key)["WEEK_START"].transform("min")
    d["_rel"] = ((d["WEEK_START"] - first).dt.days // 7).astype(int)
    return d.groupby([*cohort_keys, "_rel"])[target].mean().to_dict()


def make_naive_cohort_mean(
    df: pd.DataFrame,
    *,
    cohort_keys: tuple[str, ...] = ("SESN_SUB", "PRDT_KIND_CD", "ITEM"),
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
):
    """cutoff별 cohort lookup을 캐시하는 클로저 반환. miss 시 _recent_mean으로 채움.

    이 채움 동작은 cell 13의 기존 의도된 설계 (cohort에 historical row가 없을 때 SC 자체 최근 4주 평균).
    """
    cohort_cache: dict = {}
    stats = {"hits": 0, "misses": 0}

    def _predict(d: pd.DataFrame, sc, cutoff, h: int) -> float:
        if cutoff not in cohort_cache:
            cohort_cache[cutoff] = build_cohort_lookup(
                d, cutoff, cohort_keys=cohort_keys, group_key=group_key, target=target
            )
        rec = (
            d[(d[group_key] == sc) & (d["WEEK_START"] <= cutoff)]
            .sort_values("WEEK_START")
        )
        if not len(rec):
            return float("nan")
        r0 = rec.iloc[0]
        first_w = r0["WEEK_START"]
        cohort_vals = tuple(r0[k] for k in cohort_keys)
        rel = (cutoff + pd.Timedelta(weeks=h) - first_w).days // 7
        pred = cohort_cache[cutoff].get((*cohort_vals, rel))
        if pred is not None:
            stats["hits"] += 1
            return float(pred)
        stats["misses"] += 1
        return _recent_mean(d, sc, cutoff, group_key=group_key, target=target)

    _predict.stats = stats  # type: ignore[attr-defined]
    return _predict


def make_seasonal_naive(
    df: pd.DataFrame,
    *,
    weeks_per_season: int = 52,
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
):
    """y[t+h] = y[t+h-weeks_per_season] lookup closure.

    같은 캘린더 시점의 작년 값으로 예측. history < weeks_per_season인 SC는
    NaN 반환 — 다른 baseline으로 치환하지 않고, lift 표 단계에서 cell drop.
    """
    stats = {"hits": 0, "misses": 0}

    def _predict(d: pd.DataFrame, sc, cutoff, h: int) -> float:
        cutoff = pd.Timestamp(cutoff)
        target_week = cutoff + pd.Timedelta(weeks=h)
        lookup_week = target_week - pd.Timedelta(weeks=weeks_per_season)
        row = d[(d[group_key] == sc) & (d["WEEK_START"] == lookup_week)]
        if row.empty:
            stats["misses"] += 1
            return float("nan")
        stats["hits"] += 1
        return float(row[target].iloc[0])

    _predict.stats = stats  # type: ignore[attr-defined]
    return _predict


def mini_lgb_t1(
    df: pd.DataFrame,
    cutoff,
    *,
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
    static_categoricals: list[str] | None = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
) -> pd.Series:
    """cell 15: lag_1 + WEEK_OF_YEAR + categorical로 학습한 t+1 LightGBM. 신상품(lag_1=NaN)도 평가에 포함."""
    import lightgbm as lgb

    static_categoricals = static_categoricals or [
        "SEX", "FIT_INFO1", "FAB_TYPE", "ITEM",
        "SESN_SUB", "PRDT_KIND_CD", "BRAND_CD", "SSN_CD", "COLOR_CD",
    ]
    cats_present = [c for c in static_categoricals if c in df.columns]

    train = (
        df[df["WEEK_START"] <= cutoff]
        .sort_values([group_key, "WEEK_START"])
        .copy()
    )
    train["lag_1"] = train.groupby(group_key)[target].shift(1)
    last_y = train.groupby(group_key)[target].last().rename("lag_1")
    target_w = cutoff + pd.Timedelta(weeks=1)
    test = df[df["WEEK_START"] == target_w].set_index(group_key).join(last_y, how="left")

    feats = ["lag_1", "WEEK_OF_YEAR", *cats_present]
    required = ["WEEK_OF_YEAR", *cats_present]
    train_clean = train.dropna(subset=required + [target])
    test_clean = test.dropna(subset=required)
    if len(test_clean) == 0:
        return pd.Series(dtype=float)

    train_clean = train_clean.copy()
    test_clean = test_clean.copy()
    for c in cats_present:
        train_clean[c] = train_clean[c].astype("category")
        test_clean[c] = test_clean[c].astype("category")

    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(train_clean[feats], train_clean[target], categorical_feature=cats_present)
    return pd.Series(model.predict(test_clean[feats]).clip(min=0), index=test_clean.index)


def quantile_col_name(q: float) -> str:
    """0.25 → 'q25', 0.5 → 'q50', 0.95 → 'q95'."""
    return f"q{int(round(float(q) * 100)):02d}"


def quantile_columns(quantiles) -> dict:
    """quantile list → {'names': [...], 'low', 'mid', 'high'} 컬럼명 매핑.

    - low  = 최소 quantile, high = 최대, mid = 가장 0.5 에 가까운 quantile.
    - 학습 시 사용한 quantile 그대로 라벨링 (p10/p90 처럼 가짜 라벨 금지).
    """
    qs = sorted(float(q) for q in quantiles)
    if not qs:
        raise ValueError("quantiles list is empty")
    names = [quantile_col_name(q) for q in qs]
    mid_idx = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
    return {"names": names, "quantiles": qs, "low": names[0], "mid": names[mid_idx], "high": names[-1]}


def resolve_quantile_cols(forecast: pd.DataFrame, model=None) -> dict:
    """`model.loss.quantiles` 또는 forecast df 컬럼에서 quantile 매핑 추출.

    우선순위: model.loss.quantiles → forecast df 의 `q\\d{2}` 컬럼.
    """
    if model is not None and hasattr(model, "loss") and hasattr(model.loss, "quantiles"):
        return quantile_columns(model.loss.quantiles)
    qcols = sorted(c for c in forecast.columns if len(c) == 3 and c.startswith("q") and c[1:].isdigit())
    if not qcols:
        raise ValueError(f"No q-prefixed quantile columns in forecast: {list(forecast.columns)}")
    qs = [int(c[1:]) / 100.0 for c in qcols]
    return quantile_columns(qs)


def predict_with_tft(
    model,
    base_dataset,
    df: pd.DataFrame,
    cutoff,
    *,
    decoder_len: int = 8,
    batch_size: int = 128,
) -> dict:
    """학습된 TFT로 임의 cutoff에서 multi-horizon 예측.

    Returns dict:
      - index: pred.index (series_id metadata)
      - forecast_weeks: DatetimeIndex (cutoff+1 .. cutoff+decoder_len)
      - quantiles: list[float]  ← model.loss.quantiles (SoT)
      - preds: ndarray (n_series, decoder_len, n_quantiles)
      - q{NN}: ndarray (n_series, decoder_len)  ← 각 quantile 별 prediction
                예: quantiles=[0.25, 0.5, 0.75] → keys = q25, q50, q75
    """
    from pytorch_forecasting import TimeSeriesDataSet

    cutoff = pd.Timestamp(cutoff)
    sub = df[df["WEEK_START"] <= cutoff + pd.Timedelta(weeks=decoder_len)].copy()
    ds = TimeSeriesDataSet.from_dataset(base_dataset, sub, predict=True, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    preds = model.predict(dl, mode="quantiles", return_x=True, return_y=True, return_index=True)
    out = preds.output.cpu().numpy()  # (n, decoder_len, n_quantiles)
    weeks = pd.date_range(cutoff + pd.Timedelta(weeks=1), periods=decoder_len, freq="W-MON")

    quantiles = [float(q) for q in model.loss.quantiles]
    if out.shape[-1] != len(quantiles):
        raise ValueError(
            f"prediction quantile axis ({out.shape[-1]}) != model.loss.quantiles ({len(quantiles)})"
        )
    result: dict = {
        "index": preds.index,
        "forecast_weeks": weeks,
        "quantiles": quantiles,
        "preds": out,
    }
    for i, q in enumerate(quantiles):
        result[quantile_col_name(q)] = out[..., i]
    return result


# ---------------------------------------------------------------------------
# 4계층 metric 헬퍼 (overall + horizon + bin + STYLE×bin)
# spec: specs/mlflow-experiment-tracking.md v0.2.0
# ---------------------------------------------------------------------------
def safe_mape(y_true, y_pred, eps_frac: float = 0.01) -> float:
    """MAPE-safe: 분모를 max(|y|, eps_frac · mean|y|, 1.0) 로 floor → sparse 폭발 차단.

    sparse target 에서 일반 MAPE 가 1e8+ 로 폭발하는 문제 회피용.
    [[feedback_metric_for_sparse_target]] — overall 1개만 적재 권장.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    floor = max(eps_frac * float(np.abs(y_true).mean()), 1.0)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE — 분모 (|y|+|p|)/2 가 0 인 row 제외."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)[mask] / denom[mask]))


def _inner_join(forecast_df: pd.DataFrame, actual_df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """forecast × actual inner-join on (group_key, forecast_week).

    inner-join 강제 — left-join+fillna(0) 은 라이프사이클 종료 series 가 fake-zero 로
    들어가 모든 metric 인플레이트. ([[project_cp6677_e52_d26_diagnosis]])
    """
    return forecast_df.merge(actual_df, on=[group_key, "forecast_week"], how="inner")


def compute_overall_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model,
    *,
    group_key: str = "SC_CD",
) -> dict[str, float]:
    """전역 metric flat dict. keys: `overall.{wape|mae|rmse|bias|mape_safe|smape}_{MID}` + coverage 3종.

    quantile-aware: 컬럼명은 `model.loss.quantiles` 그대로 사용 (`q25/q50/q75` 등).
    """
    qmap = resolve_quantile_cols(forecast_df, model)
    MID, LOW, HIGH = qmap["mid"], qmap["low"], qmap["high"]
    join = _inner_join(forecast_df, actual_df, group_key)
    if join.empty:
        return {}
    y = join["actual"].to_numpy()
    y_mid = join[MID].to_numpy()
    y_low = join[LOW].to_numpy()
    y_high = join[HIGH].to_numpy()
    coverage = float(((y >= y_low) & (y <= y_high)).mean())
    coverage_target = float(qmap["quantiles"][-1] - qmap["quantiles"][0])
    return {
        f"overall.wape_{MID}": wape(y, y_mid),
        f"overall.mae_{MID}": float(np.abs(y - y_mid).mean()),
        f"overall.rmse_{MID}": float(np.sqrt(np.mean((y - y_mid) ** 2))),
        f"overall.bias_{MID}": float((y_mid - y).mean()),
        f"overall.mape_safe_{MID}": safe_mape(y, y_mid),
        f"overall.smape_{MID}": smape(y, y_mid),
        f"overall.{LOW}_{HIGH}_coverage": coverage,
        f"overall.{LOW}_{HIGH}_coverage_gap": coverage - coverage_target,
        f"overall.{LOW}_{HIGH}_coverage_target": coverage_target,
    }


def compute_horizon_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model,
    *,
    group_key: str = "SC_CD",
) -> dict[str, float]:
    """horizon별 flat dict. keys: `horizon.{wape|mae|bias|coverage}_h{N}`."""
    qmap = resolve_quantile_cols(forecast_df, model)
    MID, LOW, HIGH = qmap["mid"], qmap["low"], qmap["high"]
    join = _inner_join(forecast_df, actual_df, group_key)
    out: dict[str, float] = {}
    for h, sub in join.groupby("h"):
        if sub.empty:
            continue
        y = sub["actual"].to_numpy()
        p = sub[MID].to_numpy()
        y_low = sub[LOW].to_numpy()
        y_high = sub[HIGH].to_numpy()
        h_int = int(h)
        out[f"horizon.wape_h{h_int}"] = wape(y, p)
        out[f"horizon.mae_h{h_int}"] = float(np.abs(y - p).mean())
        out[f"horizon.bias_h{h_int}"] = float((p - y).mean())
        out[f"horizon.coverage_h{h_int}"] = float(((y >= y_low) & (y <= y_high)).mean())
    return out


def compute_bin_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model,
    *,
    group_key: str = "SC_CD",
) -> dict[str, float]:
    """horizon-bin 집계 (cold/mid/far). keys: `bin.{cold|mid|far}.{wape_{MID}|mae_{MID}|coverage}`."""
    from scripts.forecast_utils import _bin_horizon  # lazy import — 순환 회피

    qmap = resolve_quantile_cols(forecast_df, model)
    MID, LOW, HIGH = qmap["mid"], qmap["low"], qmap["high"]
    join = _inner_join(forecast_df, actual_df, group_key)
    if join.empty:
        return {}
    join = join.copy()
    join["_bin"] = join["h"].apply(_bin_horizon)
    out: dict[str, float] = {}
    for bin_name, sub in join.groupby("_bin"):
        y = sub["actual"].to_numpy()
        p = sub[MID].to_numpy()
        y_low = sub[LOW].to_numpy()
        y_high = sub[HIGH].to_numpy()
        out[f"bin.{bin_name}.wape_{MID}"] = wape(y, p)
        out[f"bin.{bin_name}.mae_{MID}"] = float(np.abs(y - p).mean())
        out[f"bin.{bin_name}.coverage"] = float(((y >= y_low) & (y <= y_high)).mean())
    return out


def compute_style_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model,
    *,
    group_key: str = "SC_CD",
    style_map: pd.Series | None = None,
) -> dict[str, float]:
    """STYLE × horizon-bin flat dict. style_map=None 이면 빈 dict.

    keys:
      style.{STYLE}.{cold|mid|far}.{wape_{MID}|mae_{MID}}
      style.{STYLE}.overall.wape_{MID}
    """
    if style_map is None:
        return {}
    from scripts.forecast_utils import _bin_horizon

    qmap = resolve_quantile_cols(forecast_df, model)
    MID = qmap["mid"]
    join = _inner_join(forecast_df, actual_df, group_key)
    if join.empty:
        return {}
    style_series = style_map.rename("STYLE_CD") if style_map.name != "STYLE_CD" else style_map
    join = join.join(style_series, on=group_key).dropna(subset=["STYLE_CD"])
    if join.empty:
        return {}
    join = join.copy()
    join["_bin"] = join["h"].apply(_bin_horizon)
    out: dict[str, float] = {}
    for (style, bin_name), sub in join.groupby(["STYLE_CD", "_bin"]):
        y = sub["actual"].to_numpy()
        p = sub[MID].to_numpy()
        out[f"style.{style}.{bin_name}.wape_{MID}"] = wape(y, p)
        out[f"style.{style}.{bin_name}.mae_{MID}"] = float(np.abs(y - p).mean())
    for style, sub in join.groupby("STYLE_CD"):
        y = sub["actual"].to_numpy()
        p = sub[MID].to_numpy()
        out[f"style.{style}.overall.wape_{MID}"] = wape(y, p)
    return out


def compute_lift_metrics(tft_wape: float, baseline_wapes: dict[str, float]) -> dict[str, float]:
    """baseline 대비 lift. 양수 = TFT 가 baseline 보다 잘함.

    Returns:
      overall.{baseline_name}.wape — baseline 자체 WAPE
      overall.lift_{baseline_name} — (baseline - tft) / baseline
      overall.lift_best — max(baseline) 기준
    """
    out: dict[str, float] = {}
    valid: list[float] = []
    for name, w in baseline_wapes.items():
        if np.isnan(w):
            continue
        out[f"overall.{name}.wape"] = float(w)
        if w > 0 and not np.isnan(tft_wape):
            out[f"overall.lift_{name}"] = float((w - tft_wape) / w)
            valid.append(float(w))
    if valid and not np.isnan(tft_wape):
        best = max(valid)
        out["overall.lift_best"] = float((best - tft_wape) / best)
    return out


def compute_full_metrics(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    model,
    *,
    group_key: str = "SC_CD",
    style_map: pd.Series | None = None,
    baseline_wapes: dict[str, float] | None = None,
) -> dict[str, float]:
    """4계층 metric 통합 flat dict. MLflow `log_metric` 에 그대로 박을 수 있음.

    Args:
        forecast_df: `predict_dataframe` 산출. columns [group_key, h, forecast_week, q...]
        actual_df:   columns [group_key, forecast_week, actual]
        model:       TFT model (`model.loss.quantiles` 가 SoT)
        style_map:   Series (SC → STYLE). None 이면 style.* 계층 skip.
        baseline_wapes: {'naive_cohort_mean': 0.45, 'seasonal_naive': 0.52} — 있으면 lift 계산.

    Returns flat dict with dot-prefix keys:
        overall.wape_q50, overall.q25_q75_coverage, ...
        horizon.wape_h1, horizon.coverage_h1, ...
        bin.cold.wape_q50, bin.cold.coverage, ...
        style.CP66.cold.wape_q50, style.CP66.overall.wape_q50, ...
        overall.naive_cohort_mean.wape, overall.lift_best, ...
    """
    metrics: dict[str, float] = {}
    metrics.update(compute_overall_metrics(forecast_df, actual_df, model, group_key=group_key))
    metrics.update(compute_horizon_metrics(forecast_df, actual_df, model, group_key=group_key))
    metrics.update(compute_bin_metrics(forecast_df, actual_df, model, group_key=group_key))
    metrics.update(
        compute_style_metrics(forecast_df, actual_df, model, group_key=group_key, style_map=style_map)
    )
    if baseline_wapes:
        qmap = resolve_quantile_cols(forecast_df, model)
        tft_wape = metrics.get(f"overall.wape_{qmap['mid']}", float("nan"))
        metrics.update(compute_lift_metrics(tft_wape, baseline_wapes))
    return metrics
