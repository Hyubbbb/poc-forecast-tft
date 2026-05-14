"""학습 결과 시각화 SoT — interpretability (VSN, attention) + overlay (PDF, plotly HTML).

spec: `specs/mlflow-experiment-tracking.md` v0.2.0 Phase 2

함수:
- `make_vsn_plots(...)`: 정적/encoder/decoder 변수 중요도 bar plot 3종
- `make_attention_plots(...)`: 매출 상위 N SC 의 attention heatmap (decoder × encoder)
- `make_all_sc_overlay_pdf(...)`: 모든 SC actual vs q50+CI 시계열, 페이지당 N SC PDF
- `make_interactive_dashboard(...)`: plotly HTML — SC selector + STYLE 필터 + week slider

interpretability 부분은 `notebooks/tft_poc.py:984-1015` 의 코드를 이식 (재구현 X).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# ---------------------------------------------------------------------------
# Internal: raw prediction (attention + interpretation 양쪽에서 사용)
# ---------------------------------------------------------------------------
def _build_predict_dataloader(model, training_dataset, df: pd.DataFrame, cutoff, decoder_len: int, batch_size: int):
    """`predict_with_tft` 의 dataloader 빌드 부분을 재사용 (mode='raw' 용)."""
    from pytorch_forecasting import TimeSeriesDataSet

    cutoff = pd.Timestamp(cutoff)
    sub = df[df["WEEK_START"] <= cutoff + pd.Timedelta(weeks=decoder_len)].copy()
    ds = TimeSeriesDataSet.from_dataset(training_dataset, sub, predict=True, stop_randomization=True)
    return ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


def _get_raw_predictions(model, training_dataset, df: pd.DataFrame, cutoff, decoder_len: int, batch_size: int = 128):
    """model.predict(mode='raw', return_x, return_index) 결과 반환.

    VSN interpret_output + attention 양쪽에서 공유.
    """
    dl = _build_predict_dataloader(model, training_dataset, df, cutoff, decoder_len, batch_size)
    return model.predict(dl, mode="raw", return_x=True, return_index=True)


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------
def make_vsn_plots(
    model,
    training_dataset,
    df: pd.DataFrame,
    cutoff,
    *,
    decoder_len: int,
    out_dir: str | Path,
    batch_size: int = 128,
) -> list[Path]:
    """Variable Selection Network 가중치 bar plot 3종 (static / encoder / decoder) PNG.

    notebooks/tft_poc.py:988-996 이식. `best_tft.interpret_output(reduction='sum')` +
    `best_tft.plot_interpretation()` 산출을 dict 별로 PNG 저장.

    Returns 저장된 파일 경로 list.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _get_raw_predictions(model, training_dataset, df, cutoff, decoder_len, batch_size)
    interpretation = model.interpret_output(raw.output, reduction="sum")
    figs = model.plot_interpretation(interpretation)

    saved: list[Path] = []
    if isinstance(figs, dict):
        for name, fig in figs.items():
            fpath = out_dir / f"vsn_{name}.png"
            fig.savefig(fpath, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved.append(fpath)
    return saved


def make_attention_plots(
    model,
    training_dataset,
    df: pd.DataFrame,
    cutoff,
    *,
    decoder_len: int,
    group_key: str,
    target: str,
    out_dir: str | Path,
    top_n_sc: int = 3,
    batch_size: int = 128,
) -> Path | None:
    """매출 상위 N SC 의 attention heatmap (decoder × encoder). 1 PNG, n subplot.

    notebooks/tft_poc.py:998-1015 이식 + 매출 상위 N SC 선택 로직 추가.

    Returns 저장된 파일 경로 (또는 attention 없으면 None).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _get_raw_predictions(model, training_dataset, df, cutoff, decoder_len, batch_size)
    attn = raw.output.get("encoder_attention", None)
    if attn is None:
        return None

    # 매출 상위 SC 선택 (cutoff 이전 N주 누적 매출 기준)
    cutoff = pd.Timestamp(cutoff)
    recent_window = df[(df["WEEK_START"] > cutoff - pd.Timedelta(weeks=52)) & (df["WEEK_START"] <= cutoff)]
    if recent_window.empty or target not in recent_window.columns:
        # fallback: 전체 누적
        sc_volume = df.groupby(group_key)[target].sum() if target in df.columns else None
    else:
        sc_volume = recent_window.groupby(group_key)[target].sum()

    idx_df = raw.index.copy() if isinstance(raw.index, pd.DataFrame) else pd.DataFrame(raw.index)
    sc_in_attn = idx_df[group_key].astype(str).values

    if sc_volume is not None and not sc_volume.empty:
        top_sc = sc_volume.sort_values(ascending=False).head(top_n_sc).index.astype(str).tolist()
        case_idx = [i for i, sc in enumerate(sc_in_attn) if sc in top_sc]
        case_idx = case_idx[:top_n_sc] or list(range(min(top_n_sc, len(sc_in_attn))))
    else:
        # fallback: 균등 분포 sample
        n = len(sc_in_attn)
        case_idx = sorted({0, n // 2, n - 1})

    n_cases = len(case_idx)
    fig, axes = plt.subplots(n_cases, 1, figsize=(11, 3 * n_cases), squeeze=False)
    for ax, idx in zip(axes[:, 0], case_idx):
        attn_arr = attn[idx]
        if hasattr(attn_arr, "cpu"):
            attn_avg = attn_arr.mean(dim=1).cpu().numpy()  # (decoder_len, encoder_len)
        else:
            attn_avg = np.asarray(attn_arr).mean(axis=1)  # numpy fallback
        im = ax.imshow(attn_avg, aspect="auto", cmap="viridis")
        ax.set_xlabel("encoder lookback (오래된 ← → 최근)")
        ax.set_ylabel("forecast horizon h")
        ax.set_title(f"{group_key}={sc_in_attn[idx]}  (idx={idx})")
        plt.colorbar(im, ax=ax, label="attention")
    fig.suptitle(f"Attention heatmap — top {n_cases} SC by recent volume")
    plt.tight_layout()
    out_path = out_dir / "attention_top3.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Overlay PDF — All SC actual vs q50 + CI band
# ---------------------------------------------------------------------------
def make_all_sc_overlay_pdf(
    forecast: pd.DataFrame,
    actual_df: pd.DataFrame,
    *,
    group_key: str,
    out_pdf: str | Path,
    style_map: pd.Series | None = None,
    scs_per_page: int = 4,
    history_df: pd.DataFrame | None = None,
    history_target: str | None = None,
    history_weeks: int = 26,
) -> Path:
    """모든 SC actual vs q50+CI band PDF. 페이지당 `scs_per_page` SC.

    Args:
        forecast: predict_dataframe 산출 (q-prefix 컬럼)
        actual_df: columns [group_key, forecast_week, actual]
        group_key: SC 식별 컬럼
        out_pdf: 저장 경로
        style_map: SC → STYLE Series (subplot 제목에 STYLE 표시용, optional)
        scs_per_page: 페이지당 panel 개수 (2×2 권장)
        history_df: cutoff 이전 history 도 보여주려면 전체 df + history_target 컬럼 제공
        history_target: history 의 target 컬럼명 (예: WEEKLY_SALE_QTY)
        history_weeks: history 표시 주 수

    Returns out_pdf Path.
    """
    from scripts.eval_utils import resolve_quantile_cols

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    qmap = resolve_quantile_cols(forecast, None)  # forecast 컬럼에서 유추
    LOW, MID, HIGH = qmap["low"], qmap["mid"], qmap["high"]
    coverage_target = float(qmap["quantiles"][-1] - qmap["quantiles"][0])
    band_label = f"{LOW}-{HIGH} ({coverage_target:.0%} CI)"

    sc_list = sorted(forecast[group_key].astype(str).unique())
    if not sc_list:
        # 빈 PDF 라도 생성
        with PdfPages(out_pdf) as pdf:
            fig = plt.figure(); plt.text(0.5, 0.5, "no SC", ha="center"); pdf.savefig(fig); plt.close(fig)
        return out_pdf

    # history_df 가 string 화 안 됐을 수도 있으니 보정
    actual_df = actual_df.copy()
    actual_df[group_key] = actual_df[group_key].astype(str)
    actual_df["forecast_week"] = pd.to_datetime(actual_df["forecast_week"])
    forecast = forecast.copy()
    forecast[group_key] = forecast[group_key].astype(str)
    forecast["forecast_week"] = pd.to_datetime(forecast["forecast_week"])

    cutoff = forecast["forecast_week"].min() - pd.Timedelta(weeks=1)
    hist_start = cutoff - pd.Timedelta(weeks=history_weeks)

    n_pages = (len(sc_list) + scs_per_page - 1) // scs_per_page
    rows = int(np.ceil(scs_per_page / 2))
    cols = 2 if scs_per_page > 1 else 1

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            page_scs = sc_list[page * scs_per_page : (page + 1) * scs_per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows), squeeze=False)
            for ax, sc in zip(axes.flat, page_scs):
                fc_sc = forecast[forecast[group_key] == sc].sort_values("forecast_week")
                act_sc = actual_df[actual_df[group_key] == sc].sort_values("forecast_week")

                if history_df is not None and history_target is not None:
                    h_df = history_df[
                        (history_df[group_key].astype(str) == sc)
                        & (history_df["WEEK_START"] >= hist_start)
                        & (history_df["WEEK_START"] <= forecast["forecast_week"].max())
                    ]
                    if not h_df.empty:
                        ax.plot(h_df["WEEK_START"], h_df[history_target], "o-", ms=2,
                                color="0.5", alpha=0.7, label="history+actual")

                if not act_sc.empty:
                    ax.plot(act_sc["forecast_week"], act_sc["actual"], "o-", ms=4,
                            color="black", label="actual (val)")

                ax.fill_between(fc_sc["forecast_week"], fc_sc[LOW], fc_sc[HIGH],
                                alpha=0.25, color="C0", label=band_label)
                ax.plot(fc_sc["forecast_week"], fc_sc[MID], "x-", ms=4,
                        color="C0", label=f"{MID} (predicted)")
                ax.axvline(cutoff, color="red", lw=0.8, ls="--", alpha=0.6)

                style_label = ""
                if style_map is not None:
                    s = style_map.get(sc) if hasattr(style_map, "get") else None
                    style_label = f" [{s}]" if s is not None else ""
                ax.set_title(f"{sc[:32]}{style_label}", fontsize=10)
                ax.legend(fontsize=7, loc="upper left")
                ax.grid(alpha=0.3)
                ax.tick_params(axis="x", rotation=30, labelsize=8)

            # remaining panels 비우기
            for ax in axes.flat[len(page_scs):]:
                ax.set_visible(False)

            fig.suptitle(f"All SC overlay — page {page + 1}/{n_pages}", fontsize=11)
            plt.tight_layout()
            pdf.savefig(fig, dpi=110)
            plt.close(fig)

    return out_pdf


# ---------------------------------------------------------------------------
# Interactive HTML — plotly
# ---------------------------------------------------------------------------
def make_interactive_dashboard(
    forecast: pd.DataFrame,
    actual_df: pd.DataFrame,
    *,
    group_key: str,
    out_html: str | Path,
    style_map: pd.Series | None = None,
    history_df: pd.DataFrame | None = None,
    history_target: str | None = None,
    history_weeks: int = 26,
) -> Path:
    """plotly interactive dashboard — SC dropdown + STYLE 필터.

    초기 화면: 첫 SC. 우측 상단 dropdown 으로 다른 SC 선택. STYLE 필터 dropdown 도 옵션.

    Returns out_html Path.
    """
    import plotly.graph_objects as go
    from scripts.eval_utils import resolve_quantile_cols

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    qmap = resolve_quantile_cols(forecast, None)
    LOW, MID, HIGH = qmap["low"], qmap["mid"], qmap["high"]
    coverage_target = float(qmap["quantiles"][-1] - qmap["quantiles"][0])
    band_label = f"{LOW}-{HIGH} ({coverage_target:.0%} CI)"

    forecast = forecast.copy()
    forecast[group_key] = forecast[group_key].astype(str)
    forecast["forecast_week"] = pd.to_datetime(forecast["forecast_week"])
    actual_df = actual_df.copy()
    actual_df[group_key] = actual_df[group_key].astype(str)
    actual_df["forecast_week"] = pd.to_datetime(actual_df["forecast_week"])

    sc_list = sorted(forecast[group_key].unique())
    if not sc_list:
        out_html.write_text("<html><body>No SC in forecast.</body></html>")
        return out_html

    cutoff = forecast["forecast_week"].min() - pd.Timedelta(weeks=1)
    hist_start = cutoff - pd.Timedelta(weeks=history_weeks)

    # SC 별 trace 4종 (low, mid, high band, actual). visible=False 로 초기엔 첫 SC 만.
    fig = go.Figure()
    sc_trace_idx: dict[str, list[int]] = {}
    for sc in sc_list:
        fc_sc = forecast[forecast[group_key] == sc].sort_values("forecast_week")
        act_sc = actual_df[actual_df[group_key] == sc].sort_values("forecast_week")
        traces_added: list[int] = []

        # CI band — high upper line + low lower with fill
        fig.add_trace(go.Scatter(
            x=fc_sc["forecast_week"], y=fc_sc[HIGH],
            mode="lines", line=dict(width=0), showlegend=False,
            name=f"{HIGH}", visible=False, hoverinfo="skip",
        )); traces_added.append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(
            x=fc_sc["forecast_week"], y=fc_sc[LOW],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(31,119,180,0.2)",
            name=band_label, visible=False, hoverinfo="skip",
        )); traces_added.append(len(fig.data) - 1)
        # MID line
        fig.add_trace(go.Scatter(
            x=fc_sc["forecast_week"], y=fc_sc[MID],
            mode="lines+markers", line=dict(color="rgb(31,119,180)"),
            name=f"{MID} predicted", visible=False,
        )); traces_added.append(len(fig.data) - 1)
        # actual
        if not act_sc.empty:
            fig.add_trace(go.Scatter(
                x=act_sc["forecast_week"], y=act_sc["actual"],
                mode="lines+markers", line=dict(color="black", width=2),
                marker=dict(size=8), name="actual", visible=False,
            )); traces_added.append(len(fig.data) - 1)
        # history (optional)
        if history_df is not None and history_target is not None:
            h_df = history_df[
                (history_df[group_key].astype(str) == sc)
                & (history_df["WEEK_START"] >= hist_start)
                & (history_df["WEEK_START"] <= forecast["forecast_week"].max())
            ]
            if not h_df.empty:
                fig.add_trace(go.Scatter(
                    x=h_df["WEEK_START"], y=h_df[history_target],
                    mode="lines+markers", line=dict(color="gray", width=1),
                    marker=dict(size=4), opacity=0.6, name="history", visible=False,
                )); traces_added.append(len(fig.data) - 1)

        sc_trace_idx[sc] = traces_added

    # 첫 SC 만 visible=True
    first_sc = sc_list[0]
    for i in sc_trace_idx[first_sc]:
        fig.data[i].visible = True

    # SC dropdown buttons
    buttons = []
    for sc in sc_list:
        style_label = ""
        if style_map is not None:
            s = style_map.get(sc) if hasattr(style_map, "get") else None
            style_label = f" [{s}]" if s is not None else ""
        vis = [False] * len(fig.data)
        for i in sc_trace_idx[sc]:
            vis[i] = True
        buttons.append(dict(
            label=f"{sc}{style_label}",
            method="update",
            args=[{"visible": vis}, {"title": f"SC: {sc}{style_label}"}],
        ))

    fig.update_layout(
        title=f"SC: {first_sc}",
        xaxis_title="forecast_week",
        yaxis_title="weekly qty",
        hovermode="x unified",
        updatemenus=[
            dict(
                type="dropdown",
                showactive=True,
                buttons=buttons,
                x=1.02, xanchor="left",
                y=1.0, yanchor="top",
            )
        ],
        shapes=[
            dict(
                type="line", x0=cutoff, x1=cutoff, y0=0, y1=1, yref="paper",
                line=dict(color="red", width=1, dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=cutoff, y=1, yref="paper", showarrow=False,
                text="cutoff", font=dict(color="red", size=10),
                xanchor="left", yanchor="bottom",
            )
        ],
        width=1100, height=500,
    )

    fig.write_html(str(out_html), include_plotlyjs="cdn")
    return out_html
