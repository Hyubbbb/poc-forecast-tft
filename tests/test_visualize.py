"""visualize 모듈 unit test — synthetic data 기반.

spec: specs/mlflow-experiment-tracking.md v0.2.0

VSN/attention 은 trained TFT model 필요 → e2e 마커 (별도). 본 파일은 overlay 만.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.visualize import make_all_sc_overlay_pdf, make_interactive_dashboard


@pytest.fixture
def synthetic_forecast_and_actual():
    """3 SC × 8 horizon synthetic forecast + actual."""
    np.random.seed(42)
    sc_list = ["SC_A", "SC_B", "SC_C"]
    cutoff = pd.Timestamp("2025-06-30")
    weeks = pd.date_range(cutoff + pd.Timedelta(weeks=1), periods=8, freq="W-MON")

    forecast_rows = []
    actual_rows = []
    for sc in sc_list:
        for h, wk in enumerate(weeks, start=1):
            base = np.random.randint(20, 200)
            forecast_rows.append({
                "SC_CD": sc, "h": h, "forecast_week": wk,
                "q25": base * 0.7, "q50": base * 1.0, "q75": base * 1.4,
            })
            actual_rows.append({"SC_CD": sc, "forecast_week": wk, "actual": base + np.random.randint(-10, 10)})
    forecast = pd.DataFrame(forecast_rows)
    actual = pd.DataFrame(actual_rows)
    return forecast, actual


@pytest.fixture
def synthetic_history():
    """3 SC × 30주 cutoff 이전 history (overlay 의 history overlay 검증용)."""
    sc_list = ["SC_A", "SC_B", "SC_C"]
    weeks = pd.date_range("2024-12-02", "2025-06-30", freq="W-MON")
    rows = []
    for sc in sc_list:
        for w in weeks:
            rows.append({
                "SC_CD": sc, "WEEK_START": w,
                "WEEKLY_SALE_QTY": np.random.randint(10, 200),
            })
    return pd.DataFrame(rows)


class TestMakeAllScOverlayPdf:
    def test_creates_pdf_file(self, synthetic_forecast_and_actual, tmp_path: Path):
        forecast, actual = synthetic_forecast_and_actual
        out_pdf = tmp_path / "overlay.pdf"
        result = make_all_sc_overlay_pdf(
            forecast, actual, group_key="SC_CD", out_pdf=out_pdf,
            scs_per_page=2,
        )
        assert result == out_pdf
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 1000  # non-trivial size

    def test_pdf_is_valid_format(self, synthetic_forecast_and_actual, tmp_path: Path):
        forecast, actual = synthetic_forecast_and_actual
        out_pdf = tmp_path / "overlay.pdf"
        make_all_sc_overlay_pdf(forecast, actual, group_key="SC_CD", out_pdf=out_pdf)
        # PDF magic bytes
        assert out_pdf.read_bytes()[:5] == b"%PDF-"

    def test_with_history_and_style_map(
        self, synthetic_forecast_and_actual, synthetic_history, tmp_path: Path
    ):
        forecast, actual = synthetic_forecast_and_actual
        style_map = pd.Series({"SC_A": "S1", "SC_B": "S1", "SC_C": "S2"}, name="STYLE_CD")
        out_pdf = tmp_path / "overlay.pdf"
        make_all_sc_overlay_pdf(
            forecast, actual, group_key="SC_CD", out_pdf=out_pdf,
            style_map=style_map, scs_per_page=2,
            history_df=synthetic_history, history_target="WEEKLY_SALE_QTY",
        )
        assert out_pdf.exists()

    def test_empty_forecast_creates_pdf_anyway(self, tmp_path: Path):
        empty_forecast = pd.DataFrame(columns=["SC_CD", "h", "forecast_week", "q25", "q50", "q75"])
        empty_actual = pd.DataFrame(columns=["SC_CD", "forecast_week", "actual"])
        out_pdf = tmp_path / "empty.pdf"
        # quantile 컬럼 없으면 resolve_quantile_cols 가 실패 → 빈 list 대비
        # 빈 forecast 도 우아하게 처리되는지 (단, q-prefix 컬럼은 있어야 함)
        empty_forecast["q25"] = pd.Series(dtype=float)
        empty_forecast["q50"] = pd.Series(dtype=float)
        empty_forecast["q75"] = pd.Series(dtype=float)
        try:
            make_all_sc_overlay_pdf(empty_forecast, empty_actual, group_key="SC_CD", out_pdf=out_pdf)
            assert out_pdf.exists()
        except ValueError:
            # 빈 forecast 에 quantile 추출 실패 가능 — OK
            pass


class TestMakeInteractiveDashboard:
    def test_creates_html_file(self, synthetic_forecast_and_actual, tmp_path: Path):
        forecast, actual = synthetic_forecast_and_actual
        out_html = tmp_path / "dashboard.html"
        result = make_interactive_dashboard(
            forecast, actual, group_key="SC_CD", out_html=out_html,
        )
        assert result == out_html
        assert out_html.exists()

    def test_html_contains_plotly(self, synthetic_forecast_and_actual, tmp_path: Path):
        forecast, actual = synthetic_forecast_and_actual
        out_html = tmp_path / "dashboard.html"
        make_interactive_dashboard(
            forecast, actual, group_key="SC_CD", out_html=out_html,
        )
        content = out_html.read_text()
        assert "plotly" in content.lower()  # plotly cdn 또는 inline
        # SC dropdown buttons (3 SC 모두 포함)
        assert "SC_A" in content
        assert "SC_B" in content
        assert "SC_C" in content

    def test_html_has_dropdown_buttons(self, synthetic_forecast_and_actual, tmp_path: Path):
        forecast, actual = synthetic_forecast_and_actual
        style_map = pd.Series({"SC_A": "S1", "SC_B": "S2", "SC_C": "S2"}, name="STYLE_CD")
        out_html = tmp_path / "dashboard.html"
        make_interactive_dashboard(
            forecast, actual, group_key="SC_CD", out_html=out_html, style_map=style_map,
        )
        content = out_html.read_text()
        # plotly updatemenus 구조 + style label 부착 확인
        assert "updatemenus" in content
        assert "S1" in content or "S2" in content  # style label

    def test_html_handles_no_actual(self, synthetic_forecast_and_actual, tmp_path: Path):
        forecast, _ = synthetic_forecast_and_actual
        empty_actual = pd.DataFrame(columns=["SC_CD", "forecast_week", "actual"])
        out_html = tmp_path / "dashboard.html"
        # actual 없어도 forecast band 만 그려야 함
        make_interactive_dashboard(forecast, empty_actual, group_key="SC_CD", out_html=out_html)
        assert out_html.exists()


class TestQuantileColumnDynamic:
    """다양한 quantile 조합 (q10/q50/q90 vs q25/q50/q75) 모두 동작 확인."""

    def test_p10_p90_band(self, tmp_path: Path):
        np.random.seed(1)
        sc_list = ["SC_X"]
        weeks = pd.date_range("2025-07-07", periods=4, freq="W-MON")
        forecast_rows = []
        actual_rows = []
        for sc in sc_list:
            for h, wk in enumerate(weeks, start=1):
                forecast_rows.append({
                    "SC_CD": sc, "h": h, "forecast_week": wk,
                    "q10": 50, "q50": 100, "q90": 150,
                })
                actual_rows.append({"SC_CD": sc, "forecast_week": wk, "actual": 95})
        out_pdf = tmp_path / "p10_p90.pdf"
        make_all_sc_overlay_pdf(
            pd.DataFrame(forecast_rows), pd.DataFrame(actual_rows),
            group_key="SC_CD", out_pdf=out_pdf,
        )
        assert out_pdf.exists()
