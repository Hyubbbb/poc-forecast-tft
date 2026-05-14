# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-jupytext.text_representation.format_version
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TFT PoC — 모자(CP66/CP77) 베이스라인
#
# **목표**: CP66/CP77 모자 시리즈의 SC 단위 주간 판매량을 TFT로 예측 (2022~2024 학습 → 2025 52주).
#
# **모집단**: CP66/CP77 시리즈 우선 → 추후 점진 확대.
#
# **흐름**:
# 1. Setup (MLflow, seed, device)
# 2. SQL 흐름 안내 (markdown)
# 3. Data load + 전체 EDA
# 4. CP66/CP77 추출
# 5. 부적합 SC 필터링 (history≥52w & total_qty≥50)
# 6. 이상치 시각화 (보존, drop 안 함)
# 7. Feature engineering 미리보기
# 8. TimeSeriesDataSet 구성
# 9. Naive / LightGBM baseline
# 10. TFT 학습 (or load best.ckpt)
# 11. 평가: horizon × WAPE, STYLE × horizon, history-bin × horizon
# 12. Forecast 시각화 + 모델 비교
#
# 변수 한 줄로 모집단 토글: `MODEL_POPULATION = "CP66_CP77" | "CP_ALL" | "HEA_ALL"`.

# %% [markdown]
# ## 1. Setup

# %%
# scripts/*.py 수정 시 노트북 재시작 없이 자동 reload
# %load_ext autoreload
# %autoreload 2

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 학습/평가 대상 STYLE_CD 리스트 (회사 통용 기준).
# 모자를 추가하려면 STYLE_CD 를 이 리스트에 append.
TARGET_STYLES = ["M19FCP66", "M19SCP77"]

CONFIG_PATH = PROJECT_ROOT / "configs/tft_supplies_cp_baseline.yaml"
DATA_PATH = PROJECT_ROOT / "data/sc_weekly_cp6677.csv"  # [Phase A-3] 03b 출력 CSV (CP66·CP77 전체 모수, COLOR_CD_NORM 포함)
MLFLOW_URI = "http://10.90.8.125:5000/"

mlflow.set_tracking_uri(MLFLOW_URI)

import random  # noqa: E402
import torch  # noqa: E402

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 한글 폰트 설정 (matplotlib 한글 깨짐 방지)
import platform  # noqa: E402
_KFONT = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["font.family"] = _KFONT
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지

print(f"target_styles={TARGET_STYLES}, device={DEVICE}, font={_KFONT}")

# %% [markdown]
# ## 2. SQL 흐름 (참고)
#
# ```
# 01_FACT_SALES_DAILY_SUPPLIES.sql           (모자 daily fact, 의류 미접촉)
#         ↓
# 03a_FACT_SALES_WEEKLY_SC_SHOP_SUPPLIES.sql (SC × SHOP 주간 집계)
#         ↓
# 03b_FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES.sql  ← ★ 최종 학습 데이터셋 SoT
#         ↓ (DataGrip Fetch all → CSV → parquet)
# data/sc_weekly_cp_hea_steady.parquet
# ```
#
# **TAG_PRICE / SALE_DT_1ST / 공휴일** 추가 가이드 (사용자 SQL 작업):
# - TAG_PRICE: 01 product_master CTE에서 이미 join. 03b BASE/SC_TOTAL_KEYS의 `ANY_VALUE(TAG_PRICE)` 추가 + 최종 SELECT 절 노출.
# - SALE_DT_1ST: 01 product_master에 `ANY_VALUE(SALE_DT_1ST)` 추가 → 03a/03b 전파.
# - 공휴일: 사내 캘린더 테이블이 있으면 03b WEEK_SPINE에 LEFT JOIN, 없으면 python 단에서 `add_kr_holidays(df)` 추가.

# %% [markdown]
# ## 3. Data load

# %%
df_raw = pd.read_csv(
    DATA_PATH,
    parse_dates=["WEEK_START"],
    # zero-padded 코드 필드는 정수 파싱 방지 (TEAM_CD '07' → 7 되는 문제)
    dtype={"TEAM_CD": str, "COLOR_CD": str, "COLOR_CD_NORM": str, "COLOR_BASE_CD": str, "SSN_CD": str},
)
# [Phase A-3] SCK_CD = STYLE_CD + TEAM_CD + COLOR_BASE_CD — 시즌 경계 통합 후보 그레인
df_raw["SCK_CD"] = df_raw["STYLE_CD"] + "_" + df_raw["TEAM_CD"] + "_" + df_raw["COLOR_BASE_CD"]
print(f"raw rows={len(df_raw):,}, cols={df_raw.shape[1]}")
print(f"WEEK_START: {df_raw['WEEK_START'].min()} → {df_raw['WEEK_START'].max()}")
print(f"unique SC_CD: {df_raw['SC_CD'].nunique():,}")
print(f"columns: {list(df_raw.columns)}")
df_raw.head(2)

# %% [markdown]
# ## 4. EDA — 전체 데이터 베이스라인

# %%
# Stage 1 정리 후 inline (구 scripts.eda_utils.{nullity_report, zero_qty_summary} 대체).
print("=== nullity (top 10) ===")
_null_rows = []
for _col in df_raw.columns:
    _mask = df_raw[_col].isna()
    _null_rows.append({
        "col": _col,
        "n_null": int(_mask.sum()),
        "ratio": float(_mask.mean()),
        "n_sc_with_null": int(df_raw.loc[_mask, "SC_CD"].nunique()) if "SC_CD" in df_raw.columns else 0,
    })
display_null = pd.DataFrame(_null_rows).sort_values("ratio", ascending=False).head(10)
print(display_null)
print()

print("=== zero qty summary ===")
_qty = df_raw["WEEKLY_SALE_QTY"].fillna(0)
_overall_zero = float((_qty == 0).mean()) if len(_qty) else 0.0
print(f"overall zero ratio: {_overall_zero:.3f}")
print(f"per-SC zero ratio (mean): {df_raw.assign(_z=(_qty == 0)).groupby('SC_CD')['_z'].mean().describe()}")

# %% [markdown]
# ## 4-A3. EDA — Phase A-3 (legacy + N 통합 모수)
#
# **컨텍스트**: `01_FACT_SALES_DAILY_SUPPLIES.sql` 의 `SESN_SUB='N'` 필터를 해제하여
# legacy(N 도입 이전) PART_CD 까지 끌어왔다. COLOR_CD 는 `PRCS.DB_PRDT_COLOR_MAP`
# `(STYLE_CD, COLOR_CD) → COLOR_CD_AFT` 매핑을 통해 `COLOR_CD_NORM` 으로 propagate.
#
# 본 EDA 블록은 학습 재실행 전에 다음 3가지 질문에 답한다:
# 1. legacy vs N 시기별 비중 — 언제, 얼마나 들어왔나
# 2. 매핑 커버리지·정합성 — DB_PRDT_COLOR_MAP 이 어디까지 덮나
# 3. 시계열 연속성 — legacy→N 전환 시기에 SUM(SALE_QTY)/AMT 가 단절·중복되는가
#
# 출력 → 셀 4-A3.5 의 마크다운 결정 노트로 종합.

# %% [markdown]
# ### 4-A3.0  스키마·컬럼 존재성 검증

# %%
_REQ_A3 = ["COLOR_CD", "COLOR_CD_NORM", "STYLE_CD", "SESN_SUB", "WEEK_START", "WEEKLY_SALE_QTY_CNS", "WEEKLY_SALE_AMT_CNS"]
_missing_a3 = [c for c in _REQ_A3 if c not in df_raw.columns]
assert not _missing_a3, f"[Phase A-3] 누락 컬럼: {_missing_a3}. SQL 01/03a/03b/extract 재실행 필요."

# CP66/CP77 시리즈 = STYLE_CD 가 'CP66' 또는 'CP77' 을 포함. 시즌 프리픽스(M19F·M21S 등) 변동 흡수.
df_cp_full = df_raw[df_raw["STYLE_CD"].str.contains("CP66|CP77", na=False, regex=True)].copy()
df_cp_full["legacy_flag"] = (df_cp_full["SESN_SUB"] != "N")
df_cp_full["color_mapped"] = (df_cp_full["COLOR_CD"] != df_cp_full["COLOR_CD_NORM"])

print(f"전체 모자 row: {len(df_raw):,} / CP66·CP77 row: {len(df_cp_full):,}")
print(f"  SESN_SUB 분포:\n{df_cp_full['SESN_SUB'].value_counts(dropna=False).to_string()}")
print(f"  legacy(SESN_SUB!='N') rows: {df_cp_full['legacy_flag'].sum():,} ({df_cp_full['legacy_flag'].mean():.1%})")
print(f"  COLOR_CD != COLOR_CD_NORM rows: {df_cp_full['color_mapped'].sum():,} ({df_cp_full['color_mapped'].mean():.1%})")

# %% [markdown]
# ### 4-A3.1  legacy vs N 시기별 비중
#
# 연도별 SUM(SALE_QTY) 기준 legacy vs N 비중. row count 가 아니라 판매 수량 비중이
# 실제 학습 영향력에 가깝다.

# %%
df_cp_full["year"] = df_cp_full["WEEK_START"].dt.year
df_cp_full["yw"] = df_cp_full["WEEK_START"].dt.to_period("W").dt.start_time

# 연도 × (legacy/N) SUM(SALE_QTY)
yr_pivot = (
    df_cp_full.assign(bucket=np.where(df_cp_full["legacy_flag"], "legacy", "N"))
    .groupby(["year", "bucket"], observed=True)["WEEKLY_SALE_QTY_CNS"]
    .sum()
    .unstack(fill_value=0)
    .reindex(columns=["legacy", "N"], fill_value=0)
)
yr_pivot["legacy_share"] = yr_pivot["legacy"] / (yr_pivot["legacy"] + yr_pivot["N"]).replace(0, np.nan)
print("=== 연도별 SUM(SALE_QTY) (legacy vs N) ===")
print(yr_pivot.to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
yr_pivot[["legacy", "N"]].plot(kind="bar", stacked=True, ax=axes[0], color=["#d99", "#9c9"])
axes[0].set_title("연도별 SUM(SALE_QTY) (legacy 적색 / N 녹색)"); axes[0].set_ylabel("SUM(SALE_QTY)"); axes[0].tick_params(axis="x", rotation=0)
axes[1].plot(yr_pivot.index, yr_pivot["legacy_share"], "o-")
axes[1].set_ylim(0, 1); axes[1].set_title("legacy 비중 (SALE_QTY 기준)"); axes[1].set_ylabel("legacy share")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 4-A3.2  매핑 커버리지·정합성
#
# `COLOR_CD_NORM` 은 미매핑 시 `COLOR_CD` 로 fallback. 따라서 NULL 은 없고
# `COLOR_CD != COLOR_CD_NORM` 비율이 실제 매핑 적용분.
#
# - legacy row 중 매핑이 실제로 바뀐 비율 → 매핑 커버리지
# - (STYLE_CD, COLOR_CD) → (STYLE_CD, COLOR_CD_NORM) 카디널리티 축소 (collapse) 표

# %%
legacy_only = df_cp_full[df_cp_full["legacy_flag"]]
if len(legacy_only):
    legacy_mapped = legacy_only["color_mapped"].mean()
    print(f"legacy row 중 매핑 적용 비율: {legacy_mapped:.1%} "
          f"({legacy_only['color_mapped'].sum():,} / {len(legacy_only):,})")
else:
    print("legacy row 없음 — DB_PRDT_COLOR_MAP 매핑 효과 확인 불가")

cardinality = (
    df_cp_full.groupby("STYLE_CD")
    .agg(n_color_raw=("COLOR_CD", "nunique"), n_color_norm=("COLOR_CD_NORM", "nunique"))
)
cardinality["collapse_ratio"] = 1 - cardinality["n_color_norm"] / cardinality["n_color_raw"]
print("\n=== STYLE_CD × COLOR_CD 카디널리티 축소 ===")
print(cardinality.sort_values("collapse_ratio", ascending=False).to_string())

# 미매핑 잔여 (legacy 인데 매핑 안 됨) — 상위 (STYLE_CD, COLOR_CD) 보고
unmapped = (
    legacy_only[~legacy_only["color_mapped"]]
    .groupby(["STYLE_CD", "COLOR_CD"]).size().rename("rows").sort_values(ascending=False)
)
print(f"\n=== legacy 미매핑 (STYLE_CD, COLOR_CD) top 10  ===  총 {len(unmapped)} 종")
print(unmapped.head(10).to_string())

# %% [markdown]
# ### 4-A3.3  시계열 연속성 — legacy→N 전환 단절·중복 여부
#
# STYLE_CD × WEEK_START 단위 SUM(SALE_QTY), SUM(SALE_AMT) 를 (a) raw COLOR_CD 기준,
# (b) COLOR_CD_NORM 기준으로 각각 합쳐 line plot. 둘이 일치하면 grain 만 다르고
# 총량은 보존된다는 sanity. 또한 legacy 시기 row 의 시작·종료 시점을 vertical
# line 으로 표시해 전환 구간 단절을 시각화.

# %%
ts_raw = (
    df_cp_full.groupby(["STYLE_CD", "WEEK_START"], observed=True)
    .agg(qty=("WEEKLY_SALE_QTY_CNS", "sum"), amt=("WEEKLY_SALE_AMT_CNS", "sum"))
    .reset_index()
)

legacy_range = (
    df_cp_full[df_cp_full["legacy_flag"]].groupby("STYLE_CD")["WEEK_START"]
    .agg(["min", "max"])
)
print("=== STYLE 별 legacy WEEK_START 범위 ===")
print(legacy_range.to_string())

styles = sorted(ts_raw["STYLE_CD"].unique())
n = len(styles)
ncols = 2; nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), squeeze=False)
for ax, sty in zip(axes.flat, styles):
    sub = ts_raw[ts_raw["STYLE_CD"] == sty].sort_values("WEEK_START")
    ax.plot(sub["WEEK_START"], sub["qty"], lw=0.9)
    if sty in legacy_range.index:
        ax.axvspan(legacy_range.loc[sty, "min"], legacy_range.loc[sty, "max"], color="orange", alpha=0.15, label="legacy 구간")
    ax.set_title(f"{sty}  weekly SUM(SALE_QTY)"); ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=8)
for ax in axes.flat[n:]:
    ax.set_visible(False)
plt.tight_layout(); plt.show()

# 총량 보존 sanity — raw vs norm 합산 차이 (legacy→N 매핑이 grain 만 바꾸면 총합 동일해야)
sum_raw = df_cp_full.groupby(["STYLE_CD", "WEEK_START", "COLOR_CD"], observed=True)["WEEKLY_SALE_QTY_CNS"].sum().sum()
sum_norm = df_cp_full.groupby(["STYLE_CD", "WEEK_START", "COLOR_CD_NORM"], observed=True)["WEEKLY_SALE_QTY_CNS"].sum().sum()
print(f"\nSUM(QTY)  raw grain={sum_raw:,.0f}  /  norm grain={sum_norm:,.0f}  /  diff={sum_raw - sum_norm:+,.0f}")

# %% [markdown]
# ### 4-A3.4  SC history 영향 — `color_cd_norm` 으로 grain 교체 시 모집단 변화
#
# Phase A-2 의 `history ≥ 52w` 게이트가 SC 정의를 어떻게 바꾸는지.

# %%
def _history_weeks(df: pd.DataFrame, color_col: str) -> pd.Series:
    g = df.groupby(["BRAND_CD", "PROD_CD", color_col], observed=True)["WEEK_START"]
    return (g.max() - g.min()).dt.days // 7 + 1

hist_raw = _history_weeks(df_cp_full, "COLOR_CD")
hist_norm = _history_weeks(df_cp_full, "COLOR_CD_NORM")

print(f"raw COLOR_CD  SC 수: {len(hist_raw):,} / history≥52w: {(hist_raw >= 52).sum():,}")
print(f"norm COLOR_CD SC 수: {len(hist_norm):,} / history≥52w: {(hist_norm >= 52).sum():,}")
print(f"  Δ history≥52w: {(hist_norm >= 52).sum() - (hist_raw >= 52).sum():+}")

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(hist_raw, bins=40, alpha=0.55, label=f"raw COLOR_CD (n={len(hist_raw):,})")
ax.hist(hist_norm, bins=40, alpha=0.55, label=f"norm COLOR_CD (n={len(hist_norm):,})")
ax.axvline(52, color="r", ls="--", label="52w 게이트")
ax.set_xlabel("history weeks (max-min+1)"); ax.set_ylabel("SC count")
ax.set_title("SC history 분포 — raw vs norm grain"); ax.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 4-A3.5  결정 노트 (사용자 검수 영역)
#
# 위 4 개 셀 결과를 토대로 다음 3 안 중 택일:
#
# **(i) 현 spec(R11 단일 모집단·`SESN_SUB='N'` 기반) 유지** — `COLOR_CD_NORM` 은
# EDA 참고용으로만 저장. 학습 데이터 변경 없음.
# - 적용 조건: legacy 비중이 작거나(예: < 5%), 매핑 커버리지가 낮아(<50%) 노이즈가
#   더 큰 경우.
#
# **(ii) 학습 grain 을 `COLOR_CD_NORM` 으로 교체** — Phase A-2 parquet 재추출.
# - 적용 조건: 매핑 커버리지가 높고(>80%), 4-A3.3 의 SUM(QTY) 시계열이 legacy→N
#   전환 구간에서 단절·중복 없이 매끄럽게 이어짐.
# - 후속: SQL 03b grain 을 `COLOR_CD_NORM` 으로 교체, spec `train-sc-tft-supplies-cp.md`
#   R11 갱신 + v0.4.0 bump, parquet 재추출, Stage A re-baseline.
#
# **(iii) 부분 흡수** — legacy 중 매핑된 행만 흡수, 미매핑·이상 시계열 SC 는 제외.
# - 적용 조건: 매핑 커버리지가 중간(50~80%)이거나 일부 SC 만 단절이 보일 때.
# - 후속: SQL 01 에 화이트리스트 CTE 추가 + spec 갱신.
#
# 위 결정은 본 노트북 다음 라운드(`MODEL_POPULATION` 토글 또는 신규 spec)에서 실행.

# %% [markdown]
# ## 4-A4. EDA — SCK 그레인 라이프사이클
#
# **컨텍스트**: 기존 `SC_CD = BRAND + PROD_CD + COLOR_CD` 는 SSN_CD 마다 PROD_CD 가 바뀌어
# 동일 상품의 시계열이 끊긴다. 대안 그레인 **`SCK_CD = STYLE_CD + TEAM_CD + COLOR_BASE_CD`**
# 를 정의하여 시즌 경계를 건너 같은 상품의 판매 이력을 연결한다.
#
# 본 블록은 `data/sc_weekly_cp6677.csv` (03b 출력) 를 직접 로드해 다음을 시각화한다:
# 1. SCK ↔ SC 통합 효과 (몇 SC → 몇 SCK)
# 2. SCK 라이프사이클 (first/last non-zero sale week, trailing zero 분포)
# 3. SCK × WEEK 판매량 히트맵 (시작·종료 시점 한눈에)
# 4. 색상 카탈로그 진화 — yearly 신규/소멸
# 5. 색상 split/merge + 미매핑 잔여
#
# 결정 노트(4-A4.7) 에서 SCK 그레인 채택 여부 판단.

# %% [markdown]
# ### 4-A4.0  SCK_CD 확인 (df_raw 재사용)
#
# `DATA_PATH` 가 이미 `sc_weekly_cp6677.csv` 로 설정되어 cell 3 에서 `df_raw` 로 로드됨.
# 또한 cell 3 에서 `SCK_CD` 컬럼이 함께 추가됨 — 본 셀은 sanity check 만.

# %%
df = df_raw   # alias — 아래 A4 셀들에서 df 로 짧게 참조
print(f"rows={len(df):,}  cols={df.shape[1]}")
print(f"unique SC_CD={df['SC_CD'].nunique():,}  unique SCK_CD={df['SCK_CD'].nunique():,}")
print(f"WEEK_START: {df['WEEK_START'].min().date()} → {df['WEEK_START'].max().date()}")
df.head(2)

# %% [markdown]
# ### 4-A4.1  STYLE 별 SCK 인벤토리

# %%
print("=== STYLE_CD 별 SCK 개수 ===")
print(df.groupby("STYLE_CD")["SCK_CD"].nunique().to_string())

print("\n=== STYLE_CD 별 SCK 샘플 ===")
for sty, grp in df.groupby("STYLE_CD"):
    sample = sorted(grp["SCK_CD"].unique())[:5]
    print(f"  {sty}: {sample}")

print("\n=== 한 SCK 가 걸친 시즌 (SSN_CD 수) 분포 ===")
print(df.groupby("SCK_CD")["SSN_CD"].nunique().describe().to_string())

# %% [markdown]
# ### 4-A4.2  SC_CD ↔ SCK_CD 통합 효과
#
# 한 SCK 가 몇 SC / 몇 PROD_CD 를 흡수했는지 — 시즌 경계 통합 효과.

# %%
collapse = df.groupby("SCK_CD").agg(
    n_sc=("SC_CD", "nunique"),
    n_prod=("PROD_CD", "nunique"),
    n_ssn=("SSN_CD", "nunique"),
)
print("=== SCK 당 흡수 SC / PROD_CD / SSN_CD 분포 ===")
print(collapse.describe().to_string())

share_solo = (collapse["n_sc"] == 1).mean()
print(f"\n단일 SC 만 갖는 SCK 비율: {share_solo:.1%}  (← 1.0 이면 통합 효과 없음)")

print("\n=== 가장 많이 통합된 SCK Top 10 (n_prod desc) ===")
print(collapse.sort_values("n_prod", ascending=False).head(10).to_string())

# %% [markdown]
# ### 4-A4.3  SCK 라이프사이클 테이블
#
# `sc_history_summary` 의 `first_sale_week` / `last_sale_week` 는 이미 양수 판매 기준.
# data cutoff 와 `last_sale_week` 의 차이 = trailing zero 길이 (실질 종료 후 dead 기간).

# %%
from scripts.preprocess_supplies_cp import sc_history_summary  # noqa: E402

sck_summary = sc_history_summary(df, group_key="SCK_CD")
print("=== SCK 라이프사이클 요약 ===")
print(sck_summary[["history_weeks", "total_qty", "zero_ratio", "first_sale_week", "last_sale_week"]].describe().to_string())

data_cutoff = df["WEEK_START"].max()
sck_summary["trailing_zero_weeks"] = ((data_cutoff - sck_summary["last_sale_week"]).dt.days // 7).astype("Int64")
print(f"\n=== trailing zero weeks 분포 (last non-zero ~ data cutoff {data_cutoff.date()}) ===")
print(sck_summary["trailing_zero_weeks"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())

print(f"\nhistory >= 52w 통과 SCK: {(sck_summary['history_weeks'] >= 52).sum()} / {len(sck_summary)}")

# %% [markdown]
# ### 4-A4.4  SCK × WEEK 판매량 히트맵 (STYLE 별)
#
# 행 = SCK, 열 = 주차. log1p 스케일. SCK 정렬 = first_sale_week 오름차순 → 신규 등장 시점이 시각적으로 위→아래 흐름.

# %%
sck_first = sck_summary.set_index("SCK_CD")["first_sale_week"]

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
for ax, sty in zip(axes, sorted(df["STYLE_CD"].unique())):
    sub = df[df["STYLE_CD"] == sty]
    pivot = sub.pivot_table(index="SCK_CD", columns="WEEK_START", values="WEEKLY_SALE_QTY_CNS", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[sck_first.loc[pivot.index].sort_values().index]
    im = ax.imshow(np.log1p(pivot.values), aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_title(f"{sty}  ({pivot.shape[0]} SCK × {pivot.shape[1]} weeks, log1p scale)")
    ax.set_ylabel("SCK (first sale week 순)")
    # x tick: 연 단위
    weeks = pivot.columns
    year_starts = [i for i, w in enumerate(weeks) if i == 0 or w.year != weeks[i - 1].year]
    ax.set_xticks(year_starts)
    ax.set_xticklabels([weeks[i].strftime("%Y") for i in year_starts])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.02)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 4-A4.5  색상 카탈로그 진화 — 연도별 신규 / 소멸 / 유지

# %%
df["year"] = df["WEEK_START"].dt.year
active = df[df["WEEKLY_SALE_QTY_CNS"] > 0].groupby(["STYLE_CD", "year"])["SCK_CD"].apply(set)

rows = []
for sty in sorted(df["STYLE_CD"].unique()):
    years = sorted(y for (s, y) in active.index if s == sty)
    prev: set[str] = set()
    for y in years:
        cur = active[(sty, y)]
        rows.append({
            "STYLE_CD": sty, "year": y,
            "continuing": len(cur & prev),
            "new": len(cur - prev),
            "gone": len(prev - cur),
            "total_active": len(cur),
        })
        prev = cur
catalog = pd.DataFrame(rows)
print("=== 연도별 SCK 카탈로그 변화 ===")
print(catalog.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, sty in zip(axes, sorted(df["STYLE_CD"].unique())):
    sub = catalog[catalog["STYLE_CD"] == sty].set_index("year")
    ax.bar(sub.index, sub["continuing"], label="continuing", color="#88c")
    ax.bar(sub.index, sub["new"], bottom=sub["continuing"], label="new", color="#9c9")
    ax.bar(sub.index, -sub["gone"], label="gone", color="#d99")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(sty); ax.set_xlabel("year"); ax.set_ylabel("SCK count")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 4-A4.6  색상 split / merge / 미매핑 잔여
#
# - **split**: legacy COLOR_CD 1 개가 N체계 여러 COLOR_CD_NORM 으로 분기
# - **merge**: 여러 legacy COLOR_CD 가 N체계 같은 COLOR_CD_NORM 으로 통합
# - **unmapped**: legacy 시기인데 매핑 안 됨 (COLOR_CD == COLOR_CD_NORM, COLOR_BASE_CD 가 1-char 깨짐)

# %%
legacy_only = df[df["COLOR_CD"] != df["COLOR_CD_NORM"]]
print(f"매핑 적용된 row: {len(legacy_only):,} / {len(df):,} ({len(legacy_only) / len(df):.1%})")

splits = legacy_only.groupby(["STYLE_CD", "COLOR_CD"])["COLOR_CD_NORM"].nunique()
splits = splits[splits > 1]
print(f"\n=== 1→N split: {len(splits)} 건 ===")
print(splits.head(20).to_string() if len(splits) else "  없음")

merges = legacy_only.groupby(["STYLE_CD", "COLOR_CD_NORM"])["COLOR_CD"].nunique()
merges = merges[merges > 1]
print(f"\n=== N→1 merge: {len(merges)} 건 ===")
print(merges.head(20).to_string() if len(merges) else "  없음")

# 미매핑 legacy: SESN_SUB != 'N' 이고 COLOR_CD == COLOR_CD_NORM
# (color_base_cd 길이로 깨진 매핑 추가 식별)
df["color_base_len"] = df["COLOR_BASE_CD"].astype(str).str.len()
maybe_unmapped = df[(df["SESN_SUB"] != "N") & (df["COLOR_CD"] == df["COLOR_CD_NORM"])]
print(f"\n=== 미매핑 legacy 후보 rows: {len(maybe_unmapped):,} ===")
print(maybe_unmapped.groupby(["STYLE_CD", "COLOR_CD"]).size().sort_values(ascending=False).head(10).to_string()
      if len(maybe_unmapped) else "  없음")
print(f"\nCOLOR_BASE_CD 길이 분포 (3 = N체계 정상, 1 = 깨진 매핑):")
print(df["color_base_len"].value_counts().sort_index().to_string())

# %% [markdown]
# ### 4-A4.7  결정 노트 (사용자 검수)
#
# 위 6 셀 결과 종합 → 3 안 중 택일:
#
# **(i) SCK_CD 그레인 채택 + 활동 종료 컷오프**
# - 적용 조건: A4.2 통합 효과 큼 (단일 SC SCK 비율 < 50%), A4.4 히트맵에서 시즌 경계가 부드럽게 이어짐, A4.6 split/merge 미미.
# - 후속: SQL 03b 의 spine 을 SCK 단위로 재정의, `last_sale_week + buffer` 이후 row 제외.
#
# **(ii) SC_CD 유지 + COLOR_CD_NORM 만 보조**
# - 적용 조건: 통합 효과 작음 (단일 SC SCK 비율 ≥ 80%), 또는 split/merge 가 많아 SCK 일관성 부족.
# - 후속: spec R11 유지, 본 EDA 만 참고용 보존.
#
# **(iii) 혼합** — STYLE × TEAM × COLOR_BASE 가 안정적인 SCK 만 채택, split/merge/미매핑 영향 큰 SCK 는 SC_CD 단위 유지.
# - 적용 조건: 일부 SCK 만 문제, 대부분 통합 가능.
# - 후속: SQL 01 에 화이트리스트 CTE 추가.
#
# 다음 라운드 (plan 외): SQL 03b zero-fill 재정의 / spec v0.4.0 / 메모리 SKU 분류 갱신.

# %% [markdown]
# ## 5. STYLE_CD 기준 모집단 추출
#
# `TARGET_STYLES` 리스트에 정의된 STYLE_CD 의 row 만 보존. STYLE_CD 컬럼이 없으면 에러
# (SQL 03b의 SELECT 절에 STYLE_CD가 포함되어야 함).

# %%
from scripts.preprocess_supplies_cp import filter_styles  # noqa: E402

df_cp = filter_styles(df_raw, styles=TARGET_STYLES)
print(f"before filter: {df_raw['SC_CD'].nunique():,} SC / {len(df_raw):,} rows")
print(f"after  filter: {df_cp['SC_CD'].nunique():,} SC / {len(df_cp):,} rows")

print("\nSTYLE_CD별 SC 수:")
print(df_cp.groupby("STYLE_CD")["SC_CD"].nunique().sort_values(ascending=False))

# %% [markdown]
# ## 5.5 0-fill 필요량 점검 (CP66/CP77)
#
# SQL 03b spine = `FIRST_SALE_WEEK ~ DATA_MAX_WEEK` (전역 cutoff). Dead SC가 spine에 들어가면 4년치 0-fill로 부풀려져 EDA/학습 노이즈.
#
# **점검 항목**:
# - `zratio_full`: 현재 SQL 정책(전역 cutoff)에서 SC별 0-fill 비율
# - `zratio_active`: spine을 `last_sale_week` 까지로 자른 가상 정책의 비율 (옵션 A)
# - dormancy gap 분포: 연속 0판매 길이 — buffer 길이 결정의 도메인 근거
# - dead SC 후보: `last_sale < DATA_MAX - 26w` 인 SC 비중

# %%
from scripts.preprocess_supplies_cp import dormancy_gap_summary, zero_fill_audit  # noqa: E402

zf = zero_fill_audit(df_cp)
print("=== zero-fill audit (CP66/CP77, n SC = {:,}) ===".format(len(zf)))
print(zf[["actual_weeks", "spine_full", "spine_active",
          "zfill_full", "zfill_active",
          "zratio_full", "zratio_active"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

n_total = len(df_cp)
n_zero = int((df_cp["WEEKLY_SALE_QTY_CNS"] == 0).sum())
print(f"\n전체 row: {n_total:,}")
print(f"actual(qty>0): {n_total - n_zero:,} ({(n_total - n_zero)/n_total:.1%})")
print(f"0-fill row   : {n_zero:,} ({n_zero/n_total:.1%})")

# dead SC 후보 (last_sale < DATA_MAX - 26w)
data_max = df_cp["WEEK_START"].max()
zf["dead_candidate"] = zf["last_sale"] < (data_max - pd.Timedelta(weeks=26))
dead_sc = set(zf[zf["dead_candidate"]]["SC_CD"])
dead_rows = df_cp["SC_CD"].isin(dead_sc).sum()
print(f"\ndead candidate (last_sale < DATA_MAX-26w): {len(dead_sc)} / {len(zf)} SC")
print(f"  이들이 차지하는 row: {dead_rows:,} ({dead_rows/n_total:.1%})")

# dormancy gap
gaps = dormancy_gap_summary(df_cp)
if not gaps.empty:
    pct = gaps["gap_weeks"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    print(f"\ndormancy gap (n={len(gaps):,}): "
          f"p50={pct.loc[0.5]:.0f}w, p75={pct.loc[0.75]:.0f}w, "
          f"p90={pct.loc[0.9]:.0f}w, p95={pct.loc[0.95]:.0f}w, "
          f"p99={pct.loc[0.99]:.0f}w, max={gaps['gap_weeks'].max():.0f}w")

# 시각화: zratio 분포 + gap 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(zf["zratio_full"], bins=30, alpha=0.55, label="spine_full (현재 SQL)")
axes[0].hist(zf["zratio_active"], bins=30, alpha=0.55, label="spine_active (last_sale 종료)")
axes[0].set_xlabel("zero-fill ratio"); axes[0].set_ylabel("SC count")
axes[0].set_title("SC별 0-fill 비율 분포"); axes[0].legend()
if not gaps.empty:
    axes[1].hist(gaps["gap_weeks"], bins=30)
    axes[1].axvline(26, color="r", ls="--", label="26w (buffer 후보)")
    axes[1].set_xlabel("연속 0판매 주 길이"); axes[1].set_ylabel("gap count")
    axes[1].set_title("dormancy gap 분포"); axes[1].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. 부적합 SC 필터링
#
# 기준: `history_weeks ≥ 52` AND `total_qty ≥ 50`.

# %%
from scripts.preprocess_supplies_cp import filter_short_history_low_volume, sc_history_summary  # noqa: E402

summary = sc_history_summary(df_cp)
print("=== SC summary (전 모집단) ===")
print(summary[["history_weeks", "total_qty", "zero_ratio"]].describe())

df_filt, report = filter_short_history_low_volume(df_cp, min_weeks=52, min_total_qty=50)
print(f"\nkept={report.kept}, dropped={report.dropped}")
print(f"  drop_short_history(only)={report.drop_short_history}")
print(f"  drop_low_volume(only)   ={report.drop_low_volume}")
print(f"  drop_both              ={report.drop_both}")

# scatter: history × total_qty
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(summary["history_weeks"], summary["total_qty"], alpha=0.4)
ax.axvline(52, color="r", ls="--", label="min_weeks=52")
ax.axhline(50, color="g", ls="--", label="min_total_qty=50")
ax.set_xlabel("history_weeks"); ax.set_ylabel("total_qty"); ax.set_yscale("log")
ax.set_title("SC distribution (history × total_qty)"); ax.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. 이상치 시각화 (보존)
#
# `WEEKLY_SALE_QTY_CNS` IQR×3 초과 spike — drop 안 하고 식별만.

# %%
from scripts.preprocess_supplies_cp import detect_qty_spikes  # noqa: E402

spikes = detect_qty_spikes(df_filt, k=3.0)
print(f"total spike rows: {len(spikes):,} ({spikes['SC_CD'].nunique() if len(spikes) else 0} SC)")
if len(spikes):
    print("\nspike top 5 SC by qty:")
    print(spikes.sort_values("qty", ascending=False).head(5).to_string(index=False))

# boxplot
fig, ax = plt.subplots(figsize=(10, 4))
sample = df_filt.groupby("SC_CD")["WEEKLY_SALE_QTY_CNS"].agg(list).head(20)
ax.boxplot(sample.values, labels=[s[:12] for s in sample.index], showfliers=True)
ax.set_yscale("symlog"); ax.set_title("WEEKLY_SALE_QTY_CNS box (sample 20 SC)")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. 결측 0-fill 사각지대 점검

# %%
import yaml  # noqa: E402

from scripts.preprocess_supplies_cp import fillna_audit  # noqa: E402

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
ds = cfg["dataset"]
target = cfg["data"]["target"]
real_cols = list(set(ds.get("static_reals", []) + ds.get("time_varying_known_reals", []) + ds.get("time_varying_unknown_reals", []) + [target]))
cat_cols = ds.get("static_categoricals", [])

audit = fillna_audit(df_filt, real_cols=real_cols, cat_cols=cat_cols)
print("=== fillna audit ===")
print(audit.to_string(index=False))
unhandled = audit[audit["unhandled"]]
if not unhandled.empty:
    print(f"\n⚠️  unhandled null columns: {unhandled['column'].tolist()}")
else:
    print("\n✅ 모든 결측 컬럼이 fillna 정책 대상.")

# %% [markdown]
# ## 9. Feature engineering 미리보기
#
# `prepare_features` 가 자동 생성하는 파생: WEEK_OF_YEAR, WEEK_SIN, WEEK_COS, WEEKS_SINCE_FIRST_SALE, time_idx.

# %%
from scripts.train import prepare_features  # noqa: E402

ds_cfg = {**cfg["dataset"], "target": target, "group_key": cfg["data"]["group_key"]}
df_feat = prepare_features(df_filt, ds_cfg)
print("derived columns sample:")
print(df_feat[["SC_CD", "WEEK_START", "WEEK_OF_YEAR", "WEEK_SIN", "WEEK_COS", "WEEKS_SINCE_FIRST_SALE", "time_idx"]].head(3))

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].hist(df_feat["WEEKS_SINCE_FIRST_SALE"], bins=50); axes[0].set_title("WEEKS_SINCE_FIRST_SALE")
axes[1].scatter(df_feat["WEEK_OF_YEAR"], df_feat["WEEK_SIN"], s=4); axes[1].set_title("WEEK_SIN")
axes[2].scatter(df_feat["WEEK_OF_YEAR"], df_feat["WEEK_COS"], s=4); axes[2].set_title("WEEK_COS")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 10. TimeSeriesDataSet 구성 (학습 전 검증용)

# %%
from scripts.train import build_datasets  # noqa: E402

cfg_run = {**cfg}
cfg_run["data"]["parquet_path"] = str(DATA_PATH)  # 노트북 inline df 사용 시 train_tft 안에서 reload하지 않도록 별도 처리 필요

training, validation, test = build_datasets(df_feat, cfg_run)
print(f"train sample={len(training)}, val sample={len(validation)}, test sample={len(test)}")

# %% [markdown]
# ## 11. Naive / LightGBM baseline

# %%
from scripts.eval_utils import build_cohort_lookup, make_naive_cohort_mean, mini_lgb_t1, naive_persistence, wape  # noqa: E402

VAL_CUTOFF = pd.Timestamp(cfg["split"]["val_cutoff"])
DECODER_LEN = cfg["dataset"]["decoder_len"]
test_df = df_feat[(df_feat["WEEK_START"] > VAL_CUTOFF - pd.Timedelta(weeks=DECODER_LEN)) & (df_feat["WEEK_START"] <= VAL_CUTOFF)]

if len(test_df) and (test_df["WEEKLY_SALE_QTY_CNS"].sum() > 0):
    pers_pred = test_df.groupby("SC_CD")["WEEKLY_SALE_QTY_CNS"].shift(1).fillna(0).values
    pers_actual = test_df["WEEKLY_SALE_QTY_CNS"].values
    print(f"naive persistence t+1 WAPE = {wape(pers_actual, pers_pred):.3f}")
else:
    print("⚠️  test_df empty — split/data 확인 필요")

# LightGBM은 실행 시간 길 수 있어 옵션 셀로 분리
RUN_LGB = False
if RUN_LGB:
    lgb_pred, lgb_actual = mini_lgb_t1(df_feat, cutoff=VAL_CUTOFF)
    print(f"LightGBM t+1 WAPE = {wape(lgb_actual, lgb_pred):.3f}")

# %% [markdown]
# ## 12. TFT 학습 (or load best.ckpt)
#
# 학습은 수십 분 ~ 수 시간 소요 (1189 SC × encoder=104 × decoder=52 기준). 이미 학습된 best.ckpt가 있으면 로드, 없으면 새로 학습.

# %%
from scripts.forecast_utils import load_artifact  # noqa: E402
from scripts.train import train_tft  # noqa: E402

OUT_DIR = PROJECT_ROOT / cfg["artifacts"]["out_dir"]
BEST_CKPT = OUT_DIR / "best.ckpt"

FORCE_RETRAIN = False
if BEST_CKPT.exists() and not FORCE_RETRAIN:
    print(f"loading existing artifact: {OUT_DIR}")
    loaded = load_artifact(OUT_DIR)
    model = loaded["model"]
    training_ds = loaded["training_dataset"]
else:
    print(f"training new model → {OUT_DIR}")
    result = train_tft(cfg_run, run_name="supplies_cp_baseline_d52_notebook")
    model = result["model"]
    training_ds = result["training"]

# %% [markdown]
# ## 13. 평가 — horizon × WAPE / STYLE × horizon / history-bin × horizon

# %%
from scripts.forecast_utils import evaluate_horizons, evaluate_horizons_styled, plot_horizon_wape  # noqa: E402

eval_h = evaluate_horizons(df_feat, model, training_ds, VAL_CUTOFF, decoder_len=DECODER_LEN, brand_slice=False)
print("=== horizon × WAPE ===")
print(eval_h.head())

fig = plot_horizon_wape(eval_h)
plt.tight_layout(); plt.show()

# STYLE_CD가 있으면 STYLE 단위 평가
if "STYLE_CD" in df_feat.columns:
    eval_styled = evaluate_horizons_styled(df_feat, model, training_ds, VAL_CUTOFF, decoder_len=DECODER_LEN)
    print("\n=== STYLE × horizon WAPE (head) ===")
    print(eval_styled.head(10))

# %% [markdown]
# ## 14. Forecast 시각화 — 샘플 SC × 52w

# %%
from scripts.eval_utils import resolve_quantile_cols  # noqa: E402
from scripts.forecast_utils import predict_dataframe  # noqa: E402

forecast = predict_dataframe(model, training_ds, df_feat, VAL_CUTOFF - pd.Timedelta(weeks=DECODER_LEN), decoder_len=DECODER_LEN)
print(f"forecast shape: {forecast.shape}, columns: {list(forecast.columns)}")

# quantile 컬럼명은 학습 config 의 model.quantiles 그대로 (`q25/q50/q75` 등).
qmap = resolve_quantile_cols(forecast, model)
LOW, MID, HIGH = qmap["low"], qmap["mid"], qmap["high"]
CI_TARGET = float(qmap["quantiles"][-1] - qmap["quantiles"][0])
print(f"quantiles = {qmap['quantiles']} → cols low/mid/high = {LOW}/{MID}/{HIGH}  (CI 목표 {CI_TARGET:.0%})")

sample_sc = forecast["SC_CD"].drop_duplicates().head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, sc in zip(axes.flat, sample_sc):
    fc_sc = forecast[forecast["SC_CD"] == sc].sort_values("forecast_week")
    actual_sc = df_feat[(df_feat["SC_CD"] == sc) & (df_feat["WEEK_START"] >= fc_sc["forecast_week"].min()) & (df_feat["WEEK_START"] <= fc_sc["forecast_week"].max())]
    ax.fill_between(fc_sc["forecast_week"], fc_sc[LOW], fc_sc[HIGH], alpha=0.3, label=f"{LOW}-{HIGH}")
    ax.plot(fc_sc["forecast_week"], fc_sc[MID], label=MID)
    ax.plot(actual_sc["WEEK_START"], actual_sc["WEEKLY_SALE_QTY_CNS"], "o-", ms=3, label="actual")
    ax.set_title(sc[:24]); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 15. STYLE_CD × 주차 단위 metric
#
# 모델 예측 결과를 STYLE_CD 별 + forecast_week 별로 집계해 WAPE / MAE / Bias 산출.

# %%
# forecast ↔ actual join (forecast_week 기준)
fc_with_actual = forecast.merge(
    df_feat[["SC_CD", "WEEK_START", "WEEKLY_SALE_QTY_CNS"]].rename(
        columns={"WEEK_START": "forecast_week", "WEEKLY_SALE_QTY_CNS": "actual"}
    ),
    on=["SC_CD", "forecast_week"],
    how="left",
)

# STYLE_CD attach (SC → STYLE 1:1 매핑)
sc_to_style = df_feat[["SC_CD", "STYLE_CD"]].drop_duplicates(subset=["SC_CD"])
fc_with_actual = fc_with_actual.merge(sc_to_style, on="SC_CD", how="left")
print(f"STYLE_CD attach: {fc_with_actual['STYLE_CD'].nunique()} 개")

target = fc_with_actual[fc_with_actual["STYLE_CD"].isin(TARGET_STYLES)].copy()
target["actual"] = target["actual"].fillna(0.0)
print(
    f"target rows = {len(target):,}  "
    f"({target['SC_CD'].nunique()} SC × {target['forecast_week'].nunique()} weeks × {target['STYLE_CD'].nunique()} styles)"
)


def _metrics(g: pd.DataFrame) -> pd.Series:
    y = g["actual"].astype(float).to_numpy()
    p = g[MID].astype(float).to_numpy()
    abs_sum = np.abs(y - p).sum()
    return pd.Series({
        "n": len(g),
        "actual_sum": float(y.sum()),
        f"{MID}_sum": float(p.sum()),
        "wape": float(abs_sum / max(y.sum(), 1e-9)),
        "mae": float(np.abs(y - p).mean()),
        "bias": float((p - y).mean()),
    })


by_week_style = (
    target.groupby(["STYLE_CD", "forecast_week"], observed=True)
    .apply(_metrics, include_groups=False)
    .reset_index()
)
print("\n=== STYLE × week (head 15) ===")
print(by_week_style.head(15).to_string(index=False))

# WAPE 주차별 line plot
fig, ax = plt.subplots(figsize=(11, 4))
for style in TARGET_STYLES:
    sub = by_week_style[by_week_style["STYLE_CD"] == style]
    if len(sub):
        ax.plot(sub["forecast_week"], sub["wape"], marker="o", ms=3, label=style)
ax.set_xlabel("forecast_week"); ax.set_ylabel("WAPE")
ax.set_title("주차별 WAPE × STYLE_CD"); ax.legend()
plt.xticks(rotation=30); plt.tight_layout(); plt.show()

# 실측 vs 예측 합계 (STYLE × week) line plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
for ax, style in zip(axes, TARGET_STYLES):
    sub = by_week_style[by_week_style["STYLE_CD"] == style]
    if len(sub):
        ax.plot(sub["forecast_week"], sub["actual_sum"], "o-", ms=3, label="actual")
        ax.plot(sub["forecast_week"], sub[f"{MID}_sum"], "x-", ms=4, label=MID)
        ax.set_title(style); ax.legend(); ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()

# STYLE 총합 metric
total_style = target.groupby("STYLE_CD").apply(_metrics, include_groups=False).reset_index()
print("\n=== STYLE 총합 metric ===")
print(total_style.to_string(index=False))

# %% [markdown]
# ## 16. 시점-의존성 sanity (v5 보완 2)
#
# 동일 SC의 decoder 52w 안에서 p50이 시점에 따라 충분히 변하는지. sparse target이라 작을 수 있지만
# median이 0에 가까우면 v1처럼 시점-무관 평탄 예측 회귀 의심.

# %%
sc_std = forecast.groupby("SC_CD")[MID].std()
print(f"SC당 {MID} std — median: {sc_std.median():.3f}, mean: {sc_std.mean():.3f}, max: {sc_std.max():.3f}")
print(f"std > 0 인 SC 비율: {(sc_std > 0).mean():.1%}")

fig, ax = plt.subplots(figsize=(8, 3))
sc_std.hist(bins=50, ax=ax); ax.set_xlabel(f"SC별 {MID} std (52w decoder 안)"); ax.set_ylabel("SC count")
ax.set_title("시점-의존성 분포 — 0 근접이 많으면 평탄 예측 회귀 의심")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 17. 신뢰구간 calibration (v5 보완 3)
#
# `[LOW, HIGH]` 적중률. 학습 quantile 에 따라 이론 목표가 달라짐
# (예: [0.25, 0.5, 0.75] → 50% CI, [0.1, 0.5, 0.9] → 80% CI).
# 격차 -10%p 이상이면 over-confident, +10%p 이상이면 under-confident.

# %%
fc_with_actual = fc_with_actual.copy()
fc_with_actual["covered"] = (
    (fc_with_actual["actual"] >= fc_with_actual[LOW]) & (fc_with_actual["actual"] <= fc_with_actual[HIGH])
)
print(f"전체 {LOW}-{HIGH} coverage: {fc_with_actual['covered'].mean():.1%} (목표 {CI_TARGET:.0%})")

fc_with_actual["h_bin"] = pd.cut(fc_with_actual["h"], bins=[0, 4, 12, 52], labels=["cold(1-4w)", "mid(5-12w)", "far(13-52w)"])
cal = (
    fc_with_actual.groupby(["STYLE_CD", "h_bin"], observed=True)["covered"]
    .agg(["mean", "count"]).rename(columns={"mean": "coverage_pct"})
    .reset_index()
)
cal["coverage_pct"] = (cal["coverage_pct"] * 100).round(1)
print("\n=== STYLE × horizon-bin coverage ===")
print(cal.to_string(index=False))

# %% [markdown]
# ## 18. (선택) 모델 비교 / Yearly backtest
#
# 후속 라운드 자리:
# - LightGBM 실행 (`RUN_LGB=True` 셀 11)
# - Naive/LGB/TFT 통합 WAPE 비교 표 → MLflow 적재
# - `scripts/backtest_yearly.py --stage A` 호출 안내

# %%
print("\n다음 라운드: LightGBM 실행 + 모델 비교 표 + Yearly backtest.")
