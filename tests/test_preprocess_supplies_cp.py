"""scripts/preprocess_supplies_cp 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.preprocess_supplies_cp import (
    LEAKAGE_CHANNEL_COLS,
    build_model_ready_from_cp_csv,
    build_series_id,
    clip_negative_target,
    detect_qty_spikes,
    dormancy_gap_summary,
    drop_target_leakage_and_redundant,
    filter_short_history_low_volume,
    filter_styles,
    fillna_audit,
    sc_history_summary,
    zero_fill_audit,
)


@pytest.fixture
def synthetic_df():
    """4 SC × 60주 합성 데이터.
    - sc1: CP66, history 60w, total_qty 600 (통과)
    - sc2: CP77, history 60w, total_qty 30 (low_volume drop)
    - sc3: CP66, history 30w (short_history drop)
    - sc4: AZ (CP 아님)
    """
    weeks = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    rows = []
    for w in weeks:
        rows.append({"SC_CD": "M_3ACP6601N_07BLK", "PROD_CD": "3ACP6601N", "STYLE_CD": "M22NCP66001", "WEEK_START": w, "WEEKLY_SALE_QTY_CNS": 10})
        rows.append({"SC_CD": "M_3ACP7701N_50WHT", "PROD_CD": "3ACP7701N", "STYLE_CD": "M22NCP77001", "WEEK_START": w, "WEEKLY_SALE_QTY_CNS": 0 if w.month < 11 else 5})
    weeks30 = weeks[:30]
    for w in weeks30:
        rows.append({"SC_CD": "M_3ACP6602N_07BLK", "PROD_CD": "3ACP6602N", "STYLE_CD": "M22NCP66002", "WEEK_START": w, "WEEKLY_SALE_QTY_CNS": 20})
    for w in weeks:
        rows.append({"SC_CD": "M_DKAZ11054_BKS", "PROD_CD": "DKAZ11054", "STYLE_CD": "M25FAZ11054", "WEEK_START": w, "WEEKLY_SALE_QTY_CNS": 100})
    return pd.DataFrame(rows)


def test_filter_styles_basic(synthetic_df):
    out = filter_styles(synthetic_df, styles=["M22NCP66001", "M22NCP77001"])
    sc_kept = set(out["SC_CD"].unique())
    assert sc_kept == {"M_3ACP6601N_07BLK", "M_3ACP7701N_50WHT"}
    assert "M_3ACP6602N_07BLK" not in sc_kept  # 다른 STYLE
    assert "M_DKAZ11054_BKS" not in sc_kept


def test_filter_styles_missing_column_raises(synthetic_df):
    df = synthetic_df.drop(columns=["STYLE_CD"])
    with pytest.raises(ValueError, match="STYLE_CD"):
        filter_styles(df, styles=["M22NCP66001"])


def test_sc_history_summary(synthetic_df):
    summary = sc_history_summary(synthetic_df)
    sc1 = summary[summary["SC_CD"] == "M_3ACP6601N_07BLK"].iloc[0]
    assert sc1["history_weeks"] == 60
    assert sc1["total_qty"] == 600
    sc3 = summary[summary["SC_CD"] == "M_3ACP6602N_07BLK"].iloc[0]
    assert sc3["history_weeks"] == 30


def test_filter_short_history_low_volume(synthetic_df):
    cp = filter_styles(synthetic_df, styles=["M22NCP66001", "M22NCP77001", "M22NCP66002"])
    filtered, report = filter_short_history_low_volume(cp, min_weeks=52, min_total_qty=50)
    kept_sc = set(filtered["SC_CD"].unique())
    # sc1 (CP66, 60w, 600 qty) only — sc2 fails low_volume, sc3 fails short_history
    assert kept_sc == {"M_3ACP6601N_07BLK"}
    assert report.kept == 1
    assert report.drop_short_history == 1  # sc3
    assert report.drop_low_volume == 1     # sc2


def test_fillna_audit_flags_unhandled():
    df = pd.DataFrame({
        "WEEKLY_SALE_QTY_CNS": [1.0, 2.0, np.nan],
        "STOCK_RATIO": [0.5, np.nan, 0.7],
        "ORPHAN": [np.nan, 1.0, 2.0],
        "PROD_CD": ["a", None, "b"],
    })
    audit = fillna_audit(
        df,
        real_cols=["WEEKLY_SALE_QTY_CNS", "STOCK_RATIO"],
        cat_cols=["PROD_CD"],
    )
    orphan_row = audit[audit["column"] == "ORPHAN"].iloc[0]
    assert orphan_row["unhandled"]
    stock_row = audit[audit["column"] == "STOCK_RATIO"].iloc[0]
    assert not stock_row["unhandled"]


def test_detect_qty_spikes(synthetic_df):
    df = synthetic_df.copy()
    # 인위적 spike 1건
    sc1_mask = df["SC_CD"] == "M_3ACP6601N_07BLK"
    df.loc[df[sc1_mask].index[0], "WEEKLY_SALE_QTY_CNS"] = 9999
    spikes = detect_qty_spikes(df, k=3.0)
    assert (spikes["SC_CD"] == "M_3ACP6601N_07BLK").any()
    assert spikes.loc[spikes["SC_CD"] == "M_3ACP6601N_07BLK", "qty"].max() == 9999


@pytest.fixture
def dormancy_df():
    """10주 spine, 2 SC — zero_fill_audit / dormancy_gap_summary 검증용.

    - M_DEAD: week 0 에 1개 판매, 이후 9주 0 (trailing dormancy)
    - M_GAPPY: week 0, week 4 에 1개씩, 나머지 0 (내부 gap=3 + trailing=5)
    """
    weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    rows = []
    for i, w in enumerate(weeks):
        rows.append({"SC_CD": "M_DEAD", "WEEK_START": w, "WEEKLY_SALE_QTY_CNS": 1 if i == 0 else 0})
    for i, w in enumerate(weeks):
        qty = 1 if i in (0, 4) else 0
        rows.append({"SC_CD": "M_GAPPY", "WEEK_START": w, "WEEKLY_SALE_QTY_CNS": qty})
    return pd.DataFrame(rows)


def test_zero_fill_audit_dead_sc(dormancy_df):
    """1주만 판매한 SC: spine_full 대비 zfill 비율이 부풀려지고, spine_active로 자르면 0."""
    audit = zero_fill_audit(dormancy_df)
    dead = audit[audit["SC_CD"] == "M_DEAD"].iloc[0]
    assert dead["actual_weeks"] == 1
    assert dead["spine_full"] == 10           # first(0) ~ data_max(9) = 10주
    assert dead["spine_active"] == 1          # last == first → 1주
    assert dead["zfill_full"] == 9
    assert dead["zfill_active"] == 0
    assert dead["zratio_full"] == pytest.approx(0.9)
    assert dead["zratio_active"] == 0.0


def test_zero_fill_audit_gappy_sc(dormancy_df):
    audit = zero_fill_audit(dormancy_df)
    gappy = audit[audit["SC_CD"] == "M_GAPPY"].iloc[0]
    assert gappy["actual_weeks"] == 2          # week 0, 4
    assert gappy["spine_full"] == 10
    assert gappy["spine_active"] == 5          # first(0) ~ last(4) = 5주
    assert gappy["zfill_full"] == 8
    assert gappy["zfill_active"] == 3
    assert gappy["total_qty"] == 2


def test_dormancy_gap_summary(dormancy_df):
    gaps = dormancy_gap_summary(dormancy_df)
    # M_DEAD: trailing gap=9
    dead_gaps = sorted(gaps[gaps["SC_CD"] == "M_DEAD"]["gap_weeks"].tolist())
    assert dead_gaps == [9]
    # M_GAPPY: 1,0,0,0,1,0,0,0,0,0 → 내부 gap=3 + trailing=5
    gappy_gaps = sorted(gaps[gaps["SC_CD"] == "M_GAPPY"]["gap_weeks"].tolist())
    assert gappy_gaps == [3, 5]


def test_dormancy_gap_summary_empty():
    """모든 SC가 매주 판매 → gap 없음 → 빈 DataFrame."""
    weeks = pd.date_range("2024-01-01", periods=5, freq="W-MON")
    df = pd.DataFrame({
        "SC_CD": ["M_A"] * 5,
        "WEEK_START": weeks,
        "WEEKLY_SALE_QTY_CNS": [1, 2, 3, 4, 5],
    })
    gaps = dormancy_gap_summary(df)
    assert gaps.empty
    assert list(gaps.columns) == ["SC_CD", "gap_weeks"]


# ─────────────────────────────────────────────────────────────────────────────
# TFT 모델링 직전 단계 어댑터 테스트
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def cp6677_like_df():
    """sc_weekly_cp6677.csv 와 동일한 스키마의 합성 데이터.

    - s1 (CP66_07BKS): 80주, qty=10/wk → kept
    - s2 (CP77_50BGD): 80주, qty=10/wk, 1주 음수 (-5) 포함 → kept (clip 후)
    - s3 (CP66_45SAS): 40주만 history → short_history drop
    """
    weeks = pd.date_range("2022-01-03", periods=80, freq="W-MON")
    short_weeks = weeks[:40]
    rows = []
    for w in weeks:
        rows.append({
            "BRAND_CD": "M", "STYLE_CD": "M19FCP66", "COLOR_CD_NORM": "07BKS",
            "TEAM_CD": "07", "COLOR_BASE_CD": "BKS",
            "START_DT": w, "END_DT": w + pd.Timedelta(days=6),
            "PRDT_KIND_CD": "HEA",
            "WEEKLY_SALE_QTY": 10, "WEEKLY_SALE_AMT": 350000, "WEEKLY_DISC_RAT": 0.02,
            "WEEKLY_SALE_QTY_RTL": 5, "WEEKLY_SALE_QTY_RF": 2,
            "WEEKLY_SALE_QTY_DOME": 0, "WEEKLY_SALE_QTY_NOTAX": 3,
            "BOW_STOCK": 100, "STOCK_RATIO": 0.1, "CUM_INTAKE": 1000,
            "FCST_AVG_MIN_TEMP": 10.0, "FCST_AVG_MAX_TEMP": 20.0, "FCST_TOTAL_PCP": 5.0,
            "FCST_MIN_MIN_TEMP": 5.0, "FCST_MAX_MAX_TEMP": 25.0, "FCST_TEMP_RANGE": 20.0,
        })
    for i, w in enumerate(weeks):
        rows.append({
            "BRAND_CD": "M", "STYLE_CD": "M19SCP77", "COLOR_CD_NORM": "50BGD",
            "TEAM_CD": "50", "COLOR_BASE_CD": "BGD",
            "START_DT": w, "END_DT": w + pd.Timedelta(days=6),
            "PRDT_KIND_CD": "HEA",
            "WEEKLY_SALE_QTY": -5 if i == 10 else 10,  # 음수 1건
            "WEEKLY_SALE_AMT": 350000, "WEEKLY_DISC_RAT": 0.02,
            "WEEKLY_SALE_QTY_RTL": 5, "WEEKLY_SALE_QTY_RF": 2,
            "WEEKLY_SALE_QTY_DOME": 0, "WEEKLY_SALE_QTY_NOTAX": 3,
            "BOW_STOCK": 100, "STOCK_RATIO": 0.1, "CUM_INTAKE": 1000,
            "FCST_AVG_MIN_TEMP": 10.0, "FCST_AVG_MAX_TEMP": 20.0, "FCST_TOTAL_PCP": 5.0,
            "FCST_MIN_MIN_TEMP": 5.0, "FCST_MAX_MAX_TEMP": 25.0, "FCST_TEMP_RANGE": 20.0,
        })
    for w in short_weeks:
        rows.append({
            "BRAND_CD": "M", "STYLE_CD": "M19FCP66", "COLOR_CD_NORM": "45SAS",
            "TEAM_CD": "45", "COLOR_BASE_CD": "SAS",
            "START_DT": w, "END_DT": w + pd.Timedelta(days=6),
            "PRDT_KIND_CD": "HEA",
            "WEEKLY_SALE_QTY": 10, "WEEKLY_SALE_AMT": 350000, "WEEKLY_DISC_RAT": 0.02,
            "WEEKLY_SALE_QTY_RTL": 5, "WEEKLY_SALE_QTY_RF": 2,
            "WEEKLY_SALE_QTY_DOME": 0, "WEEKLY_SALE_QTY_NOTAX": 3,
            "BOW_STOCK": 100, "STOCK_RATIO": 0.1, "CUM_INTAKE": 1000,
            "FCST_AVG_MIN_TEMP": 10.0, "FCST_AVG_MAX_TEMP": 20.0, "FCST_TOTAL_PCP": 5.0,
            "FCST_MIN_MIN_TEMP": 5.0, "FCST_MAX_MAX_TEMP": 25.0, "FCST_TEMP_RANGE": 20.0,
        })
    return pd.DataFrame(rows)


def test_drop_target_leakage_removes_channel_and_amt(cp6677_like_df):
    out = drop_target_leakage_and_redundant(cp6677_like_df)
    for col in LEAKAGE_CHANNEL_COLS:
        assert col not in out.columns, f"{col} 가 leakage drop 후에도 남음"
    assert "BRAND_CD" not in out.columns
    assert "PRDT_KIND_CD" not in out.columns
    assert "WEEKLY_SALE_QTY" in out.columns  # 타깃은 보존


def test_drop_target_leakage_raises_on_multivalued_constant(cp6677_like_df):
    df = cp6677_like_df.copy()
    df.loc[df.index[:5], "BRAND_CD"] = "X"  # 인위적 다중값
    with pytest.raises(ValueError, match="BRAND_CD"):
        drop_target_leakage_and_redundant(df, assert_constant=True)


def test_clip_negative_target_no_negative(cp6677_like_df):
    out = clip_negative_target(cp6677_like_df, target_col="WEEKLY_SALE_QTY")
    assert out["WEEKLY_SALE_QTY"].min() >= 0
    # 음수가 정확히 0으로 갔는지 (s2 i=10 위치)
    assert (cp6677_like_df["WEEKLY_SALE_QTY"] < 0).sum() == 1
    assert (out["WEEKLY_SALE_QTY"] == 0).sum() >= 1


def test_build_series_id_unique_per_style_color(cp6677_like_df):
    out = build_series_id(cp6677_like_df)
    assert "series_id" in out.columns
    # series_id 가 (STYLE_CD, COLOR_CD_NORM) 1:1 매핑
    pairs = out.groupby("series_id")[["STYLE_CD", "COLOR_CD_NORM"]].nunique()
    assert (pairs["STYLE_CD"] == 1).all()
    assert (pairs["COLOR_CD_NORM"] == 1).all()
    expected = {"M19FCP66_07BKS", "M19SCP77_50BGD", "M19FCP66_45SAS"}
    assert set(out["series_id"].unique()) == expected


def test_build_model_ready_filters_short_history(cp6677_like_df, tmp_path):
    csv_path = tmp_path / "cp_like.csv"
    cp6677_like_df.to_csv(csv_path, index=False)
    out_path = tmp_path / "cp_model_ready.parquet"
    df, report = build_model_ready_from_cp_csv(
        csv_path, parquet_path=out_path, min_weeks=78, min_total_qty=50.0
    )
    # 40주짜리 시리즈는 drop
    series_after = set(df["series_id"].unique())
    assert "M19FCP66_45SAS" not in series_after
    assert series_after == {"M19FCP66_07BKS", "M19SCP77_50BGD"}
    assert report["n_series_before"] == 3
    assert report["n_series_after"] == 2
    assert report["dropped_short"] == 1
    # leakage 컬럼 부재 + 음수 부재 + WEEK_START 존재
    for col in LEAKAGE_CHANNEL_COLS:
        assert col not in df.columns
    assert df["WEEKLY_SALE_QTY"].min() >= 0
    assert "WEEK_START" in df.columns
    assert out_path.exists()
