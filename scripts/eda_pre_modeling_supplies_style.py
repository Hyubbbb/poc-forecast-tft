"""Pre-modeling EDA for SUPPLIES (CP HEA) — STYLE×TEAM×COLOR_BASE grain.

Reads the user-preprocessed FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES export and emits
a self-contained HTML report. Independent from `eda_pre_modeling.py` (which uses
the SC_CD/PROD_CD grain on the SQL output).

Series key here is `SC_KEY = STYLE_CD/TEAM_CD/COLOR_BASE_CD` (모자 SKU 단위).
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def section(title: str, body_html: str, intro: str = "") -> str:
    intro_html = f"<p class='intro'>{intro}</p>" if intro else ""
    return f"<section><h2>{title}</h2>{intro_html}{body_html}</section>"


def img(fig) -> str:
    return f"<img src='data:image/png;base64,{fig_to_b64(fig)}'/>"


def df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(classes="tbl", border=0, float_format=lambda x: f"{x:,.3f}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load(path: Path, max_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # SQL raw 컬럼명 복원:
    #   - CNS suffix = 위탁 통합 (SALE_NML_*_CNS + SALE_RET_*_CNS).
    #   - AC_ prefix = "Accumulated" 누적 (raw fact `DB_SCS_W` 의 누적 컬럼).
    #     CUM_INTAKE 는 SQL alias, raw 는 `AC_STOR_QTY_KOR` (한국 누적 입고 수량).
    df = df.rename(columns={
        "WEEKLY_SALE_QTY": "WEEKLY_SALE_QTY_CNS",
        "WEEKLY_SALE_AMT": "WEEKLY_SALE_AMT_CNS",
        "CUM_INTAKE": "AC_STOR_QTY_KOR",
    })
    df["WEEK_START"] = pd.to_datetime(df["START_DT"])
    if max_date is not None:
        before = len(df)
        df = df[df["WEEK_START"] <= max_date].copy()
        print(f"cap {max_date.date()}: {before:,} -> {len(df):,} rows")
    df["YEAR"] = df["WEEK_START"].dt.year
    df["MONTH"] = df["WEEK_START"].dt.month
    df["ISO_WEEK"] = df["WEEK_START"].dt.isocalendar().week.astype(int)
    df["SC_KEY"] = (
        df["STYLE_CD"].astype(str) + "/"
        + df["TEAM_CD"].astype(str) + "/"
        + df["COLOR_BASE_CD"].astype(str)
    )
    return df


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def sec_overview(df: pd.DataFrame) -> str:
    n_rows = len(df)
    n_sc = df["SC_KEY"].nunique()
    n_style = df["STYLE_CD"].nunique()
    n_team = df["TEAM_CD"].nunique()
    n_color_base = df["COLOR_BASE_CD"].nunique()
    n_color_norm = df["COLOR_CD_NORM"].nunique()
    period = (df["WEEK_START"].min(), df["WEEK_START"].max())
    n_weeks = df["WEEK_START"].nunique()
    zero_qty = (df["WEEKLY_SALE_QTY_CNS"] == 0).mean()
    neg_qty = (df["WEEKLY_SALE_QTY_CNS"] < 0).mean()
    neg_stock = (df["BOW_STOCK"] < 0).mean()
    brand_str = ", ".join(sorted(df["BRAND_CD"].astype(str).unique()))
    kind_str = ", ".join(sorted(df["PRDT_KIND_CD"].astype(str).unique()))

    tbl = pd.DataFrame({
        "metric": [
            "총 row",
            "BRAND_CD",
            "PRDT_KIND_CD",
            "SC_KEY 수 (STYLE×TEAM×COLOR_BASE)",
            "STYLE_CD 수",
            "TEAM_CD 수",
            "COLOR_BASE_CD 수",
            "COLOR_CD_NORM 수 (참고)",
            "고유 WEEK_START 수",
            "기간 시작",
            "기간 끝",
            "WEEKLY_SALE_QTY_CNS == 0 비율 (zero-fill)",
            "WEEKLY_SALE_QTY_CNS < 0 비율 (반품 우세 주)",
            "BOW_STOCK < 0 비율",
        ],
        "value": [
            f"{n_rows:,}",
            brand_str,
            kind_str,
            f"{n_sc:,}",
            f"{n_style:,}",
            f"{n_team:,}",
            f"{n_color_base:,}",
            f"{n_color_norm:,}",
            f"{n_weeks:,}",
            period[0].date(),
            period[1].date(),
            f"{zero_qty:.1%}",
            f"{neg_qty:.1%}",
            f"{neg_stock:.1%}",
        ],
    })
    return section(
        "1. 데이터 개요",
        df_to_html(tbl),
        intro="모델링 직전 데이터 contract 점검. 사용자 사전 전처리(STYLE 집계, 활성 기간 cut)가 반영된 fact. zero-fill·음수 비중으로 후속 손실/clip 정책의 신호.",
    )


def sec_history_length(df: pd.DataFrame) -> str:
    hist = df.groupby("SC_KEY")["WEEK_START"].agg(["min", "max", "count"]).rename(
        columns={"count": "weeks_observed"})
    hist["span_weeks"] = ((hist["max"] - hist["min"]).dt.days // 7) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(hist["span_weeks"], bins=50, color="#4a90d9", edgecolor="white")
    axes[0].axvline(52, color="red", linestyle="--", label="52w (1년)")
    axes[0].axvline(104, color="orange", linestyle="--", label="104w (2년)")
    axes[0].set_title("SC_KEY 별 시계열 길이 분포 (spine 기간)")
    axes[0].set_xlabel("weeks (max - min)"); axes[0].set_ylabel("# SC_KEY")
    axes[0].legend()

    axes[1].hist(hist["weeks_observed"], bins=50, color="#7bbd5b", edgecolor="white")
    axes[1].set_title("SC_KEY 별 실측 주 수 분포")
    axes[1].set_xlabel("# rows in fact"); axes[1].set_ylabel("# SC_KEY")
    plt.tight_layout()

    summary = hist[["span_weeks", "weeks_observed"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T

    return section(
        "2. SC_KEY 별 시계열 길이 (TFT encoder length 결정 근거)",
        img(fig) + df_to_html(summary.round(1)),
        intro="min_encoder_length / max_encoder_length 결정에 사용. p10 부근에서 컷하면 short-history SC가 잘려나가는 비중을 가늠. cold-start 평가용으로는 의도적으로 short-history도 포함하는 게 유리.",
    )


def sec_timeseries(df: pd.DataFrame) -> str:
    weekly_total = df.groupby("WEEK_START")["WEEKLY_SALE_QTY_CNS"].sum()
    by_style = df.groupby(["WEEK_START", "STYLE_CD"])["WEEKLY_SALE_QTY_CNS"].sum().unstack()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(weekly_total.index, weekly_total.values, color="#333")
    axes[0].set_title("전체 SUM(WEEKLY_SALE_QTY_CNS) — 주간")
    axes[0].set_ylabel("qty")

    for sty in by_style.columns:
        axes[1].plot(by_style.index, by_style[sty], label=sty, alpha=0.85)
    axes[1].set_title("STYLE 별 SUM(WEEKLY_SALE_QTY_CNS) — 주간")
    axes[1].set_ylabel("qty"); axes[1].legend()
    plt.tight_layout()

    top_sc = (df.groupby("SC_KEY")["WEEKLY_SALE_QTY_CNS"].sum()
              .sort_values(ascending=False).head(6).index.tolist())
    fig2, axes2 = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
    for ax, sc in zip(axes2.flat, top_sc):
        sub = df[df["SC_KEY"] == sc].sort_values("WEEK_START")
        ax.plot(sub["WEEK_START"], sub["WEEKLY_SALE_QTY_CNS"], color="#4a90d9")
        ax.set_title(sc, fontsize=9)
    fig2.suptitle("판매 합계 상위 6 SC_KEY 시계열", y=1.01)
    plt.tight_layout()

    return section(
        "3. 시계열 패턴 (steady-state 가정 검증)",
        img(fig) + img(fig2),
        intro="모자는 steady-state 가정이지만 실제 trend/structural break가 있으면 TFT의 static feature만으론 부족. 상위 SC_KEY 시계열에서 cohort effect도 함께 확인.",
    )


def sec_target_distribution(df: pd.DataFrame) -> str:
    pos = df.loc[df["WEEKLY_SALE_QTY_CNS"] > 0, "WEEKLY_SALE_QTY_CNS"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(df["WEEKLY_SALE_QTY_CNS"].clip(upper=df["WEEKLY_SALE_QTY_CNS"].quantile(0.99)),
                 bins=60, color="#4a90d9", edgecolor="white")
    axes[0].set_title("WEEKLY_SALE_QTY_CNS 분포 (clip @ p99)")
    axes[0].set_xlabel("qty"); axes[0].set_ylabel("# rows")

    axes[1].hist(np.log1p(pos), bins=60, color="#7bbd5b", edgecolor="white")
    axes[1].set_title("log1p(qty>0) 분포")
    axes[1].set_xlabel("log1p(qty)")
    plt.tight_layout()

    pct = df["WEEKLY_SALE_QTY_CNS"].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).to_frame("WEEKLY_SALE_QTY_CNS")

    return section(
        "4. Target (WEEKLY_SALE_QTY_CNS) 분포",
        img(fig) + df_to_html(pct.round(2)),
        intro="long-tail / zero-inflation 정도. log1p 변환 또는 quantile loss 고려 신호.",
    )


def sec_zero_sparsity(df: pd.DataFrame) -> str:
    by_sc = df.groupby("SC_KEY")["WEEKLY_SALE_QTY_CNS"].agg(
        zero_ratio=lambda x: (x == 0).mean(),
        total=lambda x: len(x))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(by_sc["zero_ratio"], bins=40, color="#a368c8", edgecolor="white")
    ax.set_title("SC_KEY 별 zero-week 비율 분포 (= sparsity)")
    ax.set_xlabel("zero ratio"); ax.set_ylabel("# SC_KEY")

    over_50 = (by_sc["zero_ratio"] >= 0.5).sum()
    over_80 = (by_sc["zero_ratio"] >= 0.8).sum()
    total = len(by_sc)
    note = f"<p>zero-week ≥ 50% : {over_50:,} SC_KEY ({over_50/total:.1%}) · ≥ 80% : {over_80:,} SC_KEY ({over_80/total:.1%})</p>"

    return section(
        "5. Zero-fill sparsity (SC_KEY 별 무판매 주 비중)",
        img(fig) + note,
        intro="zero ratio가 높은 SC_KEY는 학습에서 cohort 단위 down-weight 또는 ZIP/Tweedie loss 고려. 너무 sparse한 SC_KEY는 학습 모집단에서 제외도 검토.",
    )


def sec_seasonality(df: pd.DataFrame) -> str:
    monthly = df.groupby("MONTH")["WEEKLY_SALE_QTY_CNS"].mean()
    yearly = df.groupby("YEAR")["WEEKLY_SALE_QTY_CNS"].mean()
    iso = df.groupby("ISO_WEEK")["WEEKLY_SALE_QTY_CNS"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(monthly.index, monthly.values, color="#4a90d9")
    axes[0].set_title("월별 평균 SALE_QTY (per row)"); axes[0].set_xticks(range(1, 13))
    axes[1].bar(yearly.index.astype(str), yearly.values, color="#7bbd5b")
    axes[1].set_title("연도별 평균 SALE_QTY")
    axes[2].plot(iso.index, iso.values, color="#a368c8")
    axes[2].set_title("ISO_WEEK 별 평균 SALE_QTY")
    axes[2].set_xlabel("week of year")
    plt.tight_layout()

    return section(
        "6. 계절성",
        img(fig),
        intro="모자=steady-state 가정이라도 봄/가을 peak, 야구 시즌(4~10월) 효과 등 잠재 신호. TFT의 time index encoding으로 학습 가능하지만 fourier-like calendar feature 추가도 검토.",
    )


def sec_team_breakdown(df: pd.DataFrame) -> str:
    by_team = df.groupby("TEAM_CD").agg(
        n_sc_keys=("SC_KEY", "nunique"),
        rows=("WEEKLY_SALE_QTY_CNS", "size"),
        total_qty=("WEEKLY_SALE_QTY_CNS", "sum"),
        mean_qty=("WEEKLY_SALE_QTY_CNS", "mean"),
        median_qty=("WEEKLY_SALE_QTY_CNS", "median"),
    )
    zero_by_team = (df.assign(_zero=(df["WEEKLY_SALE_QTY_CNS"] == 0))
                      .groupby("TEAM_CD")["_zero"].mean()
                      .rename("zero_ratio"))
    by_team = by_team.join(zero_by_team).sort_values("total_qty", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].bar(by_team.index.astype(str), by_team["n_sc_keys"], color="#4a90d9")
    axes[0].set_title("TEAM_CD 별 SC_KEY 수")
    axes[0].set_xlabel("TEAM_CD"); axes[0].set_ylabel("# SC_KEY")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(by_team.index.astype(str), by_team["total_qty"], color="#7bbd5b")
    axes[1].set_title("TEAM_CD 별 총 SALE_QTY")
    axes[1].set_xlabel("TEAM_CD"); axes[1].set_ylabel("SUM(qty)")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()

    top_teams = by_team.head(8).index.tolist()
    weekly_team = (df[df["TEAM_CD"].isin(top_teams)]
                   .groupby(["WEEK_START", "TEAM_CD"])["WEEKLY_SALE_QTY_CNS"].sum()
                   .unstack(fill_value=0))
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(weekly_team))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(top_teams):
        if t not in weekly_team.columns:
            continue
        vals = weekly_team[t].values
        ax2.fill_between(weekly_team.index, bottom, bottom + vals,
                         label=str(t), color=cmap(i % 10), alpha=0.75)
        bottom = bottom + vals
    ax2.set_title("주간 SUM(SALE_QTY) — 상위 8 TEAM_CD (stacked)")
    ax2.set_ylabel("qty"); ax2.legend(loc="upper left", ncol=4, fontsize=8)
    plt.tight_layout()

    return section(
        "7. TEAM_CD 단면 (모자 SKU 1차 분석축)",
        img(fig) + img(fig2) + df_to_html(by_team.round(2)),
        intro="새 grain의 핵심 단면. TEAM별 SC 수 / 판매 비중 / sparsity 차이가 cohort-aware loss·feature 가중치 신호. 상위 TEAM이 시계열 SUM의 대부분을 차지하는지(80/20)도 점검.",
    )


def sec_channel_breakdown(df: pd.DataFrame) -> str:
    chans = ["WEEKLY_SALE_QTY_RTL", "WEEKLY_SALE_QTY_RF",
             "WEEKLY_SALE_QTY_DOME", "WEEKLY_SALE_QTY_NOTAX"]
    chans = [c for c in chans if c in df.columns]
    sums = df[chans].sum()
    total = sums.sum()
    share = (sums / total).rename("share")
    null_ratio = df[chans].isna().mean().rename("null_ratio")
    zero_ratio = (df[chans] == 0).mean().rename("zero_ratio")

    tbl = pd.concat([sums.rename("sum"), share, null_ratio, zero_ratio], axis=1)

    # 잔차: WEEKLY_SALE_QTY_CNS - sum(channels)
    residual = df["WEEKLY_SALE_QTY_CNS"] - df[chans].sum(axis=1)
    res_summary = pd.DataFrame({
        "metric": [
            "잔차 row 수 (≠ 0)",
            "잔차 비율",
            "잔차 min",
            "잔차 max",
            "잔차 mean",
        ],
        "value": [
            f"{(residual != 0).sum():,}",
            f"{(residual != 0).mean():.2%}",
            f"{residual.min():,.1f}",
            f"{residual.max():,.1f}",
            f"{residual.mean():,.3f}",
        ],
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    short = [c.replace("WEEKLY_SALE_QTY_", "") for c in chans]
    ax.bar(short, sums.values, color=["#4a90d9", "#7bbd5b", "#a368c8", "#d99"])
    ax.set_title("채널별 총 SUM(SALE_QTY)"); ax.set_ylabel("qty")
    for i, v in enumerate(sums.values):
        ax.text(i, v, f"{v/total:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    weekly_chan = (df.groupby("WEEK_START")[chans].sum())
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    bottom = np.zeros(len(weekly_chan))
    cmap = ["#4a90d9", "#7bbd5b", "#a368c8", "#d99"]
    for i, c in enumerate(chans):
        vals = weekly_chan[c].values
        ax2.fill_between(weekly_chan.index, bottom, bottom + vals,
                         label=c.replace("WEEKLY_SALE_QTY_", ""),
                         color=cmap[i % 4], alpha=0.75)
        bottom = bottom + vals
    ax2.set_title("주간 채널별 SUM(SALE_QTY) — stacked")
    ax2.set_ylabel("qty"); ax2.legend()
    plt.tight_layout()

    return section(
        "8. 채널 분해 (RTL / RF / DOME / NOTAX)",
        img(fig) + img(fig2) + df_to_html(tbl.round(4)) +
        "<h3>WEEKLY_SALE_QTY_CNS = sum(채널)? 잔차 검증</h3>" + df_to_html(res_summary),
        intro="WEEKLY_SALE_QTY_CNS가 채널 합과 일치해야 fact가 정합. 잔차가 있으면 채널 분류 누락 또는 추가 채널 존재. 채널 비중은 학습 시 채널별 모델 vs 통합 모델 결정 신호. (CNS=NML+RET 위탁 통합, 채널 컬럼은 각 채널별 NML+RET 합산이라 CNS suffix 없음)",
    )


def sec_features(df: pd.DataFrame) -> str:
    feats = ["BOW_STOCK", "STOCK_RATIO", "AC_STOR_QTY_KOR",
             "FCST_AVG_MIN_TEMP", "FCST_AVG_MAX_TEMP", "FCST_TOTAL_PCP",
             "FCST_MIN_MIN_TEMP", "FCST_MAX_MAX_TEMP",
             "FCST_TEMP_RANGE", "WEEKLY_DISC_RAT"]
    feats = [f for f in feats if f in df.columns]

    n = len(feats)
    rows = (n + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(18, 3.3 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, f in zip(axes_flat, feats):
        s = pd.to_numeric(df[f], errors="coerce").dropna()
        if s.empty:
            ax.set_title(f"{f} (모두 NULL)"); continue
        lo, hi = s.quantile([0.005, 0.995])
        ax.hist(s.clip(lo, hi), bins=50, color="#4a90d9", edgecolor="white")
        ax.set_title(f"{f}\nclip [{lo:.1f},{hi:.1f}]", fontsize=9)
    for ax in list(axes_flat)[len(feats):]:
        ax.axis("off")
    plt.tight_layout()

    null = df[feats].isna().mean().to_frame("null_ratio")
    return section(
        "9. 외생 변수 분포 (재고 / 날씨 / 할인율)",
        img(fig) + df_to_html(null.round(3)),
        intro="continuous covariate의 scale 격차 큼 → TFT 기본 normalizer 또는 standardize 필요. NULL 비율 높으면 imputation 전략 결정. 신규 FCST_MIN_MIN_TEMP/MAX_MAX_TEMP는 이전 EDA에 없던 변수.",
    )


def sec_correlation(df: pd.DataFrame) -> str:
    cols = ["WEEKLY_SALE_QTY_CNS", "BOW_STOCK", "STOCK_RATIO", "AC_STOR_QTY_KOR",
            "FCST_AVG_MIN_TEMP", "FCST_AVG_MAX_TEMP", "FCST_TOTAL_PCP",
            "FCST_MIN_MIN_TEMP", "FCST_MAX_MAX_TEMP",
            "FCST_TEMP_RANGE", "WEEKLY_DISC_RAT"]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Spearman 상관 (target ↔ features)")
    plt.tight_layout()

    return section(
        "10. Target ↔ Feature 상관 (Spearman)",
        img(fig),
        intro="모델 입력 전 상관 부호와 크기 점검. BOW_STOCK / TEMP / DISC와의 관계 신호. 강한 다중공선성은 TFT variable selection에 영향.",
    )


def sec_cold_start(df: pd.DataFrame) -> str:
    first = df.groupby("SC_KEY")["WEEK_START"].min().rename("first_week").to_frame()
    first["year"] = first["first_week"].dt.year
    by_year = first.groupby("year").size().rename("# new SC_KEY").to_frame()

    fig, ax = plt.subplots(figsize=(8, 4))
    by_year.plot(kind="bar", ax=ax, color="#a368c8", legend=False)
    ax.set_title("SC_KEY 신규 등장(첫 판매 주) 연도별 분포 — cold-start 모집단")
    plt.tight_layout()
    return section(
        "11. Cold-start 모집단 (SC_KEY 첫 등장 분포)",
        img(fig) + df_to_html(by_year),
        intro="신규 SC_KEY의 cold-start 평가 신호. STYLE×TEAM×COLOR_BASE 단위에서 신규 진입 비중 확인. 평가 분위 설정 시 활용.",
    )


def sec_top_sc(df: pd.DataFrame) -> str:
    by_sc = df.groupby("SC_KEY")["WEEKLY_SALE_QTY_CNS"].agg(
        total="sum", mean="mean", median="median",
        zero_ratio=lambda x: (x == 0).mean(),
        weeks="count")
    top = by_sc.sort_values("total", ascending=False).head(20)
    bot_active = (by_sc.query("weeks >= 26 and zero_ratio < 0.5")
                       .sort_values("total").head(20))

    return section(
        "12. SC_KEY 분포 — 상위·하위 (장기·활성) 20",
        f"<h3>판매 합계 상위 20</h3>{df_to_html(top.round(2))}"
        f"<h3>26주 이상 활성 + zero_ratio<50% 인 SC_KEY 중 합계 하위 20</h3>{df_to_html(bot_active.round(2))}",
        intro="head-tail 분포가 학습 손실에 미치는 영향 가늠. weighted loss 또는 cohort-wise split 검토.",
    )


def sec_negative_diagnostic(df: pd.DataFrame) -> str:
    neg_qty = df[df["WEEKLY_SALE_QTY_CNS"] < 0]
    neg_stock = df[df["BOW_STOCK"] < 0]

    n_total = len(df)
    n_neg_qty = len(neg_qty)
    n_neg_stock = len(neg_stock)

    summary = pd.DataFrame({
        "metric": [
            "WEEKLY_SALE_QTY_CNS < 0 row 수",
            "WEEKLY_SALE_QTY_CNS < 0 비율",
            "WEEKLY_SALE_QTY_CNS < 0 합계",
            "BOW_STOCK < 0 row 수",
            "BOW_STOCK < 0 비율",
            "BOW_STOCK 최소값",
        ],
        "value": [
            f"{n_neg_qty:,}",
            f"{n_neg_qty/n_total:.3%}",
            f"{neg_qty['WEEKLY_SALE_QTY_CNS'].sum():,.0f}" if n_neg_qty else "0",
            f"{n_neg_stock:,}",
            f"{n_neg_stock/n_total:.3%}",
            f"{df['BOW_STOCK'].min():,.0f}",
        ],
    })

    body = df_to_html(summary)

    if n_neg_qty:
        top_neg = (neg_qty.groupby("SC_KEY")["WEEKLY_SALE_QTY_CNS"]
                          .agg(neg_count="size", neg_sum="sum")
                          .sort_values("neg_sum").head(10))
        body += "<h3>음수 SALE_QTY 발생 상위 10 SC_KEY</h3>" + df_to_html(top_neg.round(1))

    if n_neg_stock:
        # 분포 + top SC_KEY
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        s = neg_stock["BOW_STOCK"]
        lo = max(s.min(), s.quantile(0.01))
        axes[0].hist(s.clip(lower=lo), bins=40, color="#d99", edgecolor="white")
        axes[0].set_title(f"음수 BOW_STOCK 분포 (n={n_neg_stock:,})")
        axes[0].set_xlabel("BOW_STOCK"); axes[0].set_ylabel("# rows")

        top_neg_stock = (neg_stock.groupby("SC_KEY")["BOW_STOCK"]
                                   .agg(neg_count="size", neg_min="min", neg_sum="sum")
                                   .sort_values("neg_sum").head(10))
        axes[1].axis("off")
        plt.tight_layout()
        body += img(fig)
        body += "<h3>음수 BOW_STOCK 발생 상위 10 SC_KEY</h3>" + df_to_html(top_neg_stock.round(1))

    return section(
        "13. 음수값 진단 (반품/재고조정)",
        body,
        intro="사용자 사전 cut에도 반품·조정 음수가 남을 수 있음. 모델 입력 시 abs/clip/zero-replace 정책 결정 근거. 음수 발생이 특정 SC_KEY에 몰려있는지 확인.",
    )


def sec_active_period(df: pd.DataFrame) -> str:
    nz = df[df["WEEKLY_SALE_QTY_CNS"] > 0]
    if nz.empty:
        return section("14. SC_KEY 활성 기간", "<p>non-zero row 없음.</p>")

    agg = nz.groupby("SC_KEY")["WEEK_START"].agg(["min", "max", "count"]).rename(
        columns={"min": "first_sale", "max": "last_sale", "count": "active_weeks_obs"})
    agg["span_weeks"] = ((agg["last_sale"] - agg["first_sale"]).dt.days // 7) + 1

    global_max = df["WEEK_START"].max()
    agg["is_active"] = agg["last_sale"] >= (global_max - pd.Timedelta(weeks=4))

    n_total = len(agg)
    n_active = int(agg["is_active"].sum())
    n_dead = n_total - n_active

    top_sc = (nz.groupby("SC_KEY")["WEEKLY_SALE_QTY_CNS"].sum()
              .sort_values(ascending=False).head(30).index.tolist())
    g = agg.loc[top_sc].sort_values("first_sale")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for i, (sc, row) in enumerate(g.iterrows()):
        color = "#4a90d9" if row["is_active"] else "#bbb"
        ax.barh(i, (row["last_sale"] - row["first_sale"]).days,
                left=row["first_sale"], height=0.7, color=color)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=7)
    ax.set_title("판매 합계 상위 30 SC_KEY 의 첫판매~마지막판매 (회색=비활성)")
    ax.set_xlabel("주차"); ax.tick_params(axis="x", rotation=45)

    axes[1].hist(agg["span_weeks"], bins=40, color="#7bbd5b", edgecolor="white")
    axes[1].axvline(52, color="red", linestyle="--", label="52w")
    axes[1].axvline(104, color="orange", linestyle="--", label="104w")
    axes[1].set_title("SC_KEY 별 활성 span 분포\n(첫 non-zero ~ 마지막 non-zero)")
    axes[1].set_xlabel("weeks"); axes[1].set_ylabel("# SC_KEY"); axes[1].legend()
    plt.tight_layout()

    summary_tbl = agg[["span_weeks", "active_weeks_obs"]].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.round(1)

    status_tbl = pd.DataFrame({
        "metric": ["전체 SC_KEY (non-zero 1회 이상)", "활성 SC_KEY (마지막 판매 ≤ 4주 전)",
                   "비활성 SC_KEY (마지막 판매 > 4주 전)", "활성 비율"],
        "value": [f"{n_total:,}", f"{n_active:,}", f"{n_dead:,}",
                  f"{n_active/n_total:.1%}"]
    })

    return section(
        "14. SC_KEY 활성 기간 (첫 판매 → 마지막 판매)",
        img(fig) + df_to_html(status_tbl) +
        "<h3>span_weeks / active_weeks_obs 분포</h3>" + df_to_html(summary_tbl),
        intro="사용자가 사전 cut을 적용했어도 단종 SC_KEY가 남았는지 점검. 마지막 판매 > 4주 전이면 학습 모집단 재검토 신호.",
    )


def sec_nonzero_distribution(df: pd.DataFrame) -> str:
    nz = df.loc[df["WEEKLY_SALE_QTY_CNS"] > 0, "WEEKLY_SALE_QTY_CNS"]
    if nz.empty:
        return section("15. Non-zero 판매량 분포", "<p>non-zero 없음.</p>")

    p99 = nz.quantile(0.99)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(nz.clip(upper=p99), bins=60, color="#4a90d9", edgecolor="white")
    axes[0].set_title(f"qty>0 분포 (clip @ p99={p99:.0f})")
    axes[0].set_xlabel("qty"); axes[0].set_ylabel("# rows")

    axes[1].hist(np.log1p(nz), bins=60, color="#7bbd5b", edgecolor="white")
    axes[1].set_title("log1p(qty>0)")
    axes[1].set_xlabel("log1p(qty)")

    axes[2].boxplot([nz.values], vert=True, showfliers=False,
                     patch_artist=True,
                     boxprops=dict(facecolor="#a368c8", alpha=0.6))
    axes[2].set_xticks([1]); axes[2].set_xticklabels(["qty>0"])
    axes[2].set_title("Boxplot (outlier 제거)")
    plt.tight_layout()

    pct = nz.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).to_frame("qty>0")

    nonzero_ratio = (df["WEEKLY_SALE_QTY_CNS"] > 0).mean()
    note = f"<p>전체 row 중 qty>0 비율: <b>{nonzero_ratio:.1%}</b> ({len(nz):,} / {len(df):,})</p>"

    return section(
        "15. Non-zero 판매량 분포",
        note + img(fig) + df_to_html(pct.round(2)),
        intro="zero-fill 영향 빼고 진짜 판매량 분포. log1p 후 정규성에 가까워지면 Gaussian/quantile loss의 normalizer만 잘 잡으면 됨. boxplot의 IQR 범위로 typical 주차 판매량 가늠.",
    )


def sec_missing(df: pd.DataFrame) -> str:
    null = df.isna().mean().sort_values(ascending=False).to_frame("null_ratio")
    null = null[null["null_ratio"] > 0]
    if null.empty:
        return section("16. Missing", "<p>결측치 없음 (모든 컬럼 fully populated).</p>")
    return section(
        "16. Missing values",
        df_to_html(null.round(4)),
        intro="모델 입력 단계에서 imputation/제거 정책 확정 근거.",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_html(df: pd.DataFrame, src: Path) -> str:
    css = """
    body { font-family: -apple-system, sans-serif; max-width: 1280px; margin: 24px auto; padding: 0 24px; color: #222; }
    h1 { border-bottom: 2px solid #4a90d9; padding-bottom: 6px; }
    h2 { color: #4a90d9; margin-top: 48px; }
    h3 { color: #555; margin-top: 24px; font-size: 1em; }
    section { margin-bottom: 28px; }
    .intro { color: #555; font-size: 0.95em; font-style: italic; margin: 6px 0 18px; }
    img { max-width: 100%; display: block; margin: 12px 0; }
    .tbl { border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }
    .tbl th, .tbl td { padding: 4px 10px; text-align: right; border-bottom: 1px solid #eee; }
    .tbl th { background: #f4f8fc; }
    .nav { background: #f4f8fc; padding: 12px 18px; border-radius: 6px; margin-bottom: 24px; }
    .nav a { margin-right: 14px; color: #4a90d9; text-decoration: none; }
    """
    sections = [
        sec_overview(df),
        sec_history_length(df),
        sec_timeseries(df),
        sec_target_distribution(df),
        sec_zero_sparsity(df),
        sec_seasonality(df),
        sec_team_breakdown(df),
        sec_channel_breakdown(df),
        sec_features(df),
        sec_correlation(df),
        sec_cold_start(df),
        sec_top_sc(df),
        sec_negative_diagnostic(df),
        sec_active_period(df),
        sec_nonzero_distribution(df),
        sec_missing(df),
    ]
    nav_links = " · ".join(
        f"<a href='#s{i+1}'>{i+1}</a>" for i in range(len(sections)))
    body = "".join(
        s.replace("<section>", f"<section id='s{i+1}'>", 1)
        for i, s in enumerate(sections))

    return f"""<!doctype html><html lang='ko'><head>
<meta charset='utf-8'><title>Supplies (CP HEA) Weekly STYLE×TEAM×COLOR_BASE Pre-modeling EDA</title>
<style>{css}</style></head><body>
<h1>Supplies(CP HEA) 주간 STYLE×TEAM×COLOR_BASE — 수요예측 모델링 전 EDA</h1>
<p><b>Source</b>: {src}<br/>
<b>Rows</b>: {len(df):,} · <b>SC_KEY</b>: {df['SC_KEY'].nunique():,} · <b>Period</b>: {df['WEEK_START'].min().date()} ~ {df['WEEK_START'].max().date()}<br/>
<b>Grain</b>: STYLE_CD × TEAM_CD × COLOR_BASE_CD (사용자 사전 전처리: PART_CD→STYLE 집계, 판매 기간 cut 적용)</p>
<div class='nav'>{nav_links}</div>
{body}
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="/Users/kimhyub/Downloads/FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES.csv")
    ap.add_argument("--output",
                    default="artifacts/eda_pre_modeling_supplies_style.html")
    ap.add_argument("--max-date", default="none",
                    help="WEEK_START 상한. 기본 'none' (사용자 사전 cut 적용 가정)")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    cap = None if args.max_date.lower() == "none" else pd.Timestamp(args.max_date)
    df = load(src, max_date=cap)
    print(f"loaded {len(df):,} rows · SC_KEY={df['SC_KEY'].nunique():,} · "
          f"period={df['WEEK_START'].min().date()}~{df['WEEK_START'].max().date()}")
    html = build_html(df, src)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
