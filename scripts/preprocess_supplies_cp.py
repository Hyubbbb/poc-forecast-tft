"""모자(CP) PoC 전용 전처리 함수 — 노트북/배치 공용 SoT.

기존 7종 함수 + CSV→parquet 어댑터(TFT 모델링 직전 단계):
- filter_styles: STYLE_CD 리스트 기준 필터 (회사 통용 기준)
- sc_history_summary: SC별 history_weeks/total_qty/zero_ratio/first_sale_week/last_sale_week
- filter_short_history_low_volume: history/판매량 임계 미달 SC 제거 + drop 사유 리포트
- fillna_audit: prepare_features fillna 전후 결측 점검 (0-fill 사각지대 탐지)
- detect_qty_spikes: SC별 IQR×k 초과 spike 식별 (drop 안 함, 시각화/검증용)
- zero_fill_audit: SQL spine vs actual 판매주 정량 비교 (전역 cutoff 정책 진단용)
- dormancy_gap_summary: SC별 연속 0판매 주 길이 분포 (spine cutoff 정책 결정 근거)
- drop_target_leakage_and_redundant: 채널 분해 컬럼(타깃 합산 leakage) + 단일값 컬럼 제거
- clip_negative_target: 음수 타깃 0으로 clip (log1p 정규화 호환)
- build_series_id: STYLE_CD + COLOR_CD_NORM 으로 시계열 고유 ID 부여
- build_model_ready_from_cp_csv: sc_weekly_cp6677.csv → TFT 입력용 parquet 어댑터
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def filter_styles(
    df: pd.DataFrame,
    styles: list[str] | tuple[str, ...],
    style_col: str = "STYLE_CD",
) -> pd.DataFrame:
    """STYLE_CD ∈ styles 인 row만 보존. 회사 통용 기준.

    모자 한 종류를 추가하려면 styles 리스트에 그 STYLE_CD를 추가하면 됨.
    예: styles = ["M19FCP66", "M19SCP77", "M22NCP001"]
    """
    if style_col not in df.columns:
        raise ValueError(
            f"'{style_col}' 컬럼이 데이터에 없습니다. "
            f"SQL(03b)의 SELECT 절에 STYLE_CD를 포함시켜 새로 추출하세요."
        )
    return df[df[style_col].isin(list(styles))].copy()


def sc_history_summary(
    df: pd.DataFrame,
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
    week_col: str = "WEEK_START",
) -> pd.DataFrame:
    """SC별 history_weeks / total_qty / zero_ratio / first_sale_week / last_sale_week 요약."""
    g = df.groupby(group_key)
    base = pd.DataFrame({
        "history_weeks": g[week_col].nunique(),
        "total_qty": g[target].sum(),
        "zero_ratio": g[target].apply(lambda s: float((s == 0).mean())),
    })
    positive = df[df[target] > 0]
    pg = positive.groupby(group_key)
    base["first_sale_week"] = pg[week_col].min()
    base["last_sale_week"] = pg[week_col].max()
    return base.reset_index()


@dataclass
class FilterReport:
    kept: int
    dropped: int
    drop_short_history: int
    drop_low_volume: int
    drop_both: int


def filter_short_history_low_volume(
    df: pd.DataFrame,
    min_weeks: int = 52,
    min_total_qty: float = 50.0,
    group_key: str = "SC_CD",
    target: str = "WEEKLY_SALE_QTY_CNS",
    week_col: str = "WEEK_START",
) -> tuple[pd.DataFrame, FilterReport]:
    """history >= min_weeks AND total_qty >= min_total_qty 인 SC만 보존.

    Returns: (filtered df, FilterReport with drop counts by reason)
    """
    summary = sc_history_summary(df, group_key=group_key, target=target, week_col=week_col)
    short = summary["history_weeks"] < min_weeks
    low_vol = summary["total_qty"] < min_total_qty
    drop_mask = short | low_vol
    drop_short_history = int((short & ~low_vol).sum())
    drop_low_volume = int((low_vol & ~short).sum())
    drop_both = int((short & low_vol).sum())

    keep_sc = summary.loc[~drop_mask, group_key].tolist()
    report = FilterReport(
        kept=len(keep_sc),
        dropped=int(drop_mask.sum()),
        drop_short_history=drop_short_history,
        drop_low_volume=drop_low_volume,
        drop_both=drop_both,
    )
    return df[df[group_key].isin(keep_sc)].copy(), report


def fillna_audit(
    df: pd.DataFrame,
    real_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """real_cols (fillna 0.0 대상) + cat_cols ("__missing__" 대상) 의 결측 점검.

    각 컬럼별로 누락 비율과 prepare_features 처리 정책을 한 표에 정리.
    `unhandled=True` 인 컬럼이 있으면 fillna 사각지대 (대응 정책 없는 컬럼).
    """
    rows = []
    handled = set(real_cols) | set(cat_cols)
    for c in df.columns:
        n_null = int(df[c].isna().sum())
        null_ratio = n_null / max(len(df), 1)
        if c in real_cols:
            policy = "fillna(0.0)"
        elif c in cat_cols:
            policy = 'fillna("__missing__")'
        else:
            policy = "—"
        rows.append({
            "column": c,
            "null_count": n_null,
            "null_ratio": null_ratio,
            "fillna_policy": policy,
            "unhandled": n_null > 0 and c not in handled,
        })
    return pd.DataFrame(rows).sort_values(["unhandled", "null_count"], ascending=[False, False]).reset_index(drop=True)


def detect_qty_spikes(
    df: pd.DataFrame,
    target: str = "WEEKLY_SALE_QTY_CNS",
    group_key: str = "SC_CD",
    k: float = 3.0,
) -> pd.DataFrame:
    """SC별 IQR×k 초과 spike 주를 식별. drop 안 하고 (SC_CD, WEEK_START, qty, q3_plus_k_iqr) 반환.

    사용자 정책: 보존 + 시각화만. PoC 단계 데이터 파괴 X.
    """
    rows = []
    for sc, sub in df.groupby(group_key):
        q1, q3 = sub[target].quantile([0.25, 0.75]).values
        iqr = q3 - q1
        threshold = q3 + k * iqr
        spikes = sub[sub[target] > threshold]
        for _, r in spikes.iterrows():
            rows.append({
                group_key: sc,
                "WEEK_START": r.get("WEEK_START"),
                "qty": float(r[target]),
                "threshold": float(threshold),
            })
    return pd.DataFrame(rows)


def zero_fill_audit(
    df: pd.DataFrame,
    target: str = "WEEKLY_SALE_QTY_CNS",
    group_key: str = "SC_CD",
    week_col: str = "WEEK_START",
) -> pd.DataFrame:
    """SQL spine vs actual 판매주 비교 — 0-fill 양/비율 SC별 정량.

    컬럼:
    - actual_weeks: target>0 인 주 수
    - first_sale / last_sale: target>0 의 min / max
    - total_qty: target 합
    - spine_full: first_sale ~ data_max_week (현재 03b 정책)
    - spine_active: first_sale ~ last_sale (옵션 A: last_sale로 자름)
    - zfill_full / zfill_active: 각 spine 대비 0-fill 주 수
    - zratio_full / zratio_active: 각 spine 대비 0-fill 비율
    """
    data_max = df[week_col].max()
    pos = df[df[target] > 0]
    pg = pos.groupby(group_key)
    out = pd.DataFrame({
        "actual_weeks": pg[week_col].nunique(),
        "first_sale": pg[week_col].min(),
        "last_sale": pg[week_col].max(),
        "total_qty": df.groupby(group_key)[target].sum(),
    })
    out["spine_full"] = ((data_max - out["first_sale"]).dt.days // 7) + 1
    out["spine_active"] = ((out["last_sale"] - out["first_sale"]).dt.days // 7) + 1
    out["zfill_full"] = out["spine_full"] - out["actual_weeks"]
    out["zfill_active"] = out["spine_active"] - out["actual_weeks"]
    out["zratio_full"] = out["zfill_full"] / out["spine_full"]
    out["zratio_active"] = np.where(
        out["spine_active"] > 0,
        out["zfill_active"] / out["spine_active"].replace(0, np.nan),
        0.0,
    )
    return out.reset_index()


def dormancy_gap_summary(
    df: pd.DataFrame,
    target: str = "WEEKLY_SALE_QTY_CNS",
    group_key: str = "SC_CD",
    week_col: str = "WEEK_START",
) -> pd.DataFrame:
    """SC별 연속 0판매 주(gap) 길이 — spine cutoff 정책 결정의 근거 데이터.

    leading 0(첫 판매 전)은 SQL spine 정의상 없음.
    trailing 0(마지막 판매 후 ~ spine 끝)도 포함 → "permanent dormancy" 후보.
    내부 0(판매 사이 휴면)과 trailing 0 모두 한 행씩 출력.
    """
    rows = []
    for sc, sub in df.sort_values([group_key, week_col]).groupby(group_key, sort=False):
        cur = 0
        for q in sub[target].values:
            if q == 0:
                cur += 1
            else:
                if cur > 0:
                    rows.append({group_key: sc, "gap_weeks": cur})
                cur = 0
        if cur > 0:
            rows.append({group_key: sc, "gap_weeks": cur})
    if not rows:
        return pd.DataFrame(columns=[group_key, "gap_weeks"])
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TFT 모델링 직전 단계: CSV → 모델 입력용 parquet 어댑터
# ─────────────────────────────────────────────────────────────────────────────

LEAKAGE_CHANNEL_COLS = (
    "WEEKLY_SALE_QTY_RTL",
    "WEEKLY_SALE_QTY_RF",
    "WEEKLY_SALE_QTY_DOME",
    "WEEKLY_SALE_QTY_NOTAX",
    "WEEKLY_SALE_AMT",
)


def drop_target_leakage_and_redundant(
    df: pd.DataFrame,
    leakage_cols: tuple[str, ...] = LEAKAGE_CHANNEL_COLS,
    constant_cols: tuple[str, ...] = ("BRAND_CD", "PRDT_KIND_CD"),
    assert_constant: bool = True,
) -> pd.DataFrame:
    """채널 분해(타깃 합산 leakage) + cp6677 내 단일값 컬럼 제거.

    RTL+RF+DOME+NOTAX ≡ WEEKLY_SALE_QTY 이므로 4개 채널은 절대 모델 입력 X.
    SALE_AMT = qty × price 라 qty 정보 누설.
    BRAND_CD/PRDT_KIND_CD 는 cp6677 모집단에서 단일값이라 정보량 0.
    """
    if assert_constant:
        for c in constant_cols:
            if c in df.columns:
                n = df[c].nunique(dropna=False)
                if n > 1:
                    raise ValueError(
                        f"'{c}' 가 단일값이 아님 (n_unique={n}). "
                        "drop 정책 재검토 필요. assert_constant=False 로 강제 진행 가능."
                    )
    to_drop = [c for c in (*leakage_cols, *constant_cols) if c in df.columns]
    return df.drop(columns=to_drop)


def clip_negative_target(
    df: pd.DataFrame,
    target_col: str = "WEEKLY_SALE_QTY",
) -> pd.DataFrame:
    """반품 우세로 음수가 된 주는 0으로 clip (TFT log1p 정규화 호환)."""
    out = df.copy()
    out[target_col] = out[target_col].clip(lower=0)
    return out


def build_series_id(
    df: pd.DataFrame,
    style_col: str = "STYLE_CD",
    color_col: str = "COLOR_CD_NORM",
    out_col: str = "series_id",
) -> pd.DataFrame:
    """series_id = STYLE_CD + '_' + COLOR_CD_NORM. TFT group_ids 단일 키."""
    out = df.copy()
    out[out_col] = out[style_col].astype(str) + "_" + out[color_col].astype(str)
    return out


def build_model_ready_from_cp_csv(
    csv_path: str | Path,
    parquet_path: str | Path | None = None,
    *,
    target_col: str = "WEEKLY_SALE_QTY",
    min_weeks: int = 78,
    min_total_qty: float = 50.0,
) -> tuple[pd.DataFrame, dict]:
    """sc_weekly_cp6677.csv → TFT 모델 입력용 DataFrame.

    파이프라인 (모든 단계가 명시적):
      1) CSV 로드 + START_DT→WEEK_START rename + dtype 정규화
      2) leakage 채널 컬럼 + 단일값 컬럼 drop
      3) 음수 타깃 → 0 clip
      4) series_id = STYLE_CD + '_' + COLOR_CD_NORM 부여
      5) history < min_weeks 또는 total_qty < min_total_qty 시리즈 제거
      6) (옵션) parquet 저장

    Returns
    -------
    df : 모델 입력 준비 완료 DataFrame
    report : {n_rows, n_series_before, n_series_after, dropped_short, dropped_low_volume, dropped_both}

    target_col 은 train.py 와 일치해야 함 (config: data.target).
    """
    df = pd.read_csv(csv_path)
    if "START_DT" in df.columns:
        df = df.rename(columns={"START_DT": "WEEK_START"})
    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"])
    if "END_DT" in df.columns:
        df["END_DT"] = pd.to_datetime(df["END_DT"])

    df = drop_target_leakage_and_redundant(df)
    df = clip_negative_target(df, target_col=target_col)
    df = build_series_id(df)

    n_series_before = df["series_id"].nunique()

    df_filt, report = filter_short_history_low_volume(
        df,
        min_weeks=min_weeks,
        min_total_qty=min_total_qty,
        group_key="series_id",
        target=target_col,
        week_col="WEEK_START",
    )

    summary = {
        "n_rows": int(len(df_filt)),
        "n_series_before": int(n_series_before),
        "n_series_after": int(report.kept),
        "dropped_short": int(report.drop_short_history),
        "dropped_low_volume": int(report.drop_low_volume),
        "dropped_both": int(report.drop_both),
        "min_weeks": int(min_weeks),
        "min_total_qty": float(min_total_qty),
    }

    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_filt.to_parquet(parquet_path, index=False)
        summary["parquet_path"] = str(parquet_path)

    return df_filt, summary
