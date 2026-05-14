"""seasonal_naive baseline 계약 — y[t+h] = y[t+h-weeks_per_season].

history가 weeks_per_season(기본 52w) 이상인 SC만 정상 lookup, 미만이면 NaN.
다른 baseline으로 치환하지 않음 — lift 표에서 row drop 가능하도록 NaN 반환.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.eval_utils import make_seasonal_naive


def _df_with_history(weeks: int, sc: str = "A", start: str = "2024-01-01") -> pd.DataFrame:
    """sc 1개에 대해 weeks 길이의 weekly 시계열 생성. WEEKLY_SALE_QTY_CNS는 1..weeks."""
    return pd.DataFrame({
        "SC_CD": [sc] * weeks,
        "WEEK_START": pd.date_range(start, periods=weeks, freq="W-MON"),
        "WEEKLY_SALE_QTY_CNS": list(range(1, weeks + 1)),
    })


class TestSeasonalNaiveLookup:
    @pytest.fixture
    def df_60w(self) -> pd.DataFrame:
        return _df_with_history(60)

    def test_returns_y_at_t_minus_52_for_h1(self, df_60w):
        predict = make_seasonal_naive(df_60w)
        cutoff = df_60w["WEEK_START"].iloc[55]
        target_week = cutoff + pd.Timedelta(weeks=1)
        lookup_week = target_week - pd.Timedelta(weeks=52)
        expected = float(df_60w.loc[df_60w["WEEK_START"] == lookup_week, "WEEKLY_SALE_QTY_CNS"].iloc[0])
        assert predict(df_60w, "A", cutoff, h=1) == pytest.approx(expected)

    def test_returns_y_at_t_minus_52_for_h8(self, df_60w):
        predict = make_seasonal_naive(df_60w)
        cutoff = df_60w["WEEK_START"].iloc[55]
        target_week = cutoff + pd.Timedelta(weeks=8)
        lookup_week = target_week - pd.Timedelta(weeks=52)
        row = df_60w.loc[df_60w["WEEK_START"] == lookup_week]
        if row.empty:
            assert math.isnan(predict(df_60w, "A", cutoff, h=8))
        else:
            assert predict(df_60w, "A", cutoff, h=8) == pytest.approx(float(row["WEEKLY_SALE_QTY_CNS"].iloc[0]))

    def test_history_under_52w_returns_nan(self):
        df = _df_with_history(weeks=30)
        predict = make_seasonal_naive(df)
        cutoff = df["WEEK_START"].iloc[-1]
        assert math.isnan(predict(df, "A", cutoff, h=1))

    def test_unknown_sc_returns_nan(self, df_60w):
        predict = make_seasonal_naive(df_60w)
        cutoff = df_60w["WEEK_START"].iloc[55]
        assert math.isnan(predict(df_60w, "UNKNOWN", cutoff, h=1))

    def test_empty_df_returns_nan(self):
        empty = pd.DataFrame({"SC_CD": [], "WEEK_START": pd.to_datetime([]), "WEEKLY_SALE_QTY_CNS": []})
        predict = make_seasonal_naive(empty)
        assert math.isnan(predict(empty, "A", pd.Timestamp("2025-01-01"), h=1))


class TestSeasonalNaiveCustomConfig:
    def test_weeks_per_season_26_returns_26w_lag(self):
        df = _df_with_history(weeks=40)
        predict = make_seasonal_naive(df, weeks_per_season=26)
        cutoff = df["WEEK_START"].iloc[30]
        target_week = cutoff + pd.Timedelta(weeks=1)
        lookup_week = target_week - pd.Timedelta(weeks=26)
        expected = float(df.loc[df["WEEK_START"] == lookup_week, "WEEKLY_SALE_QTY_CNS"].iloc[0])
        assert predict(df, "A", cutoff, h=1) == pytest.approx(expected)

    def test_custom_target_column(self):
        df = _df_with_history(60).rename(columns={"WEEKLY_SALE_QTY_CNS": "y"})
        predict = make_seasonal_naive(df, target="y")
        cutoff = df["WEEK_START"].iloc[55]
        target_week = cutoff + pd.Timedelta(weeks=1)
        lookup_week = target_week - pd.Timedelta(weeks=52)
        expected = float(df.loc[df["WEEK_START"] == lookup_week, "y"].iloc[0])
        assert predict(df, "A", cutoff, h=1) == pytest.approx(expected)


class TestSeasonalNaiveStats:
    def test_stats_counter_tracks_hits_and_misses(self):
        # LONG: 2024-01-01 시작 60주. cutoff=iloc[55], h=1, lookup=2024-02-05 → in LONG → hit.
        # LATE_START: 2025-01-01 시작 20주. 같은 cutoff/h에서 lookup=2024-02-05 → 데이터 시작 이전 → miss.
        df = pd.concat([
            _df_with_history(60, sc="LONG", start="2024-01-01"),
            _df_with_history(20, sc="LATE_START", start="2025-01-01"),
        ], ignore_index=True)
        predict = make_seasonal_naive(df)
        cutoff = pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=55)
        predict(df, "LONG", cutoff, h=1)        # hit
        predict(df, "LATE_START", cutoff, h=1)  # miss (lookup before SC's first week)
        predict(df, "UNKNOWN", cutoff, h=1)     # miss (unknown SC)
        assert predict.stats["hits"] == 1
        assert predict.stats["misses"] == 2

    def test_multiple_scs_independent(self):
        df = pd.concat([
            _df_with_history(60, sc="A", start="2024-01-01"),
            _df_with_history(60, sc="B", start="2024-01-01"),
        ], ignore_index=True)
        df.loc[df["SC_CD"] == "B", "WEEKLY_SALE_QTY_CNS"] *= 10  # B는 A의 10배
        predict = make_seasonal_naive(df)
        cutoff = df["WEEK_START"].iloc[55]
        assert predict(df, "B", cutoff, h=1) == pytest.approx(10 * predict(df, "A", cutoff, h=1))
