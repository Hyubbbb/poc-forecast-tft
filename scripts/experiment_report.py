"""다중 MLflow run 비교 보고 CLI.

spec: `specs/mlflow-experiment-tracking.md` v0.2.0 Phase 4

사용법:
    # intent 기준 비교
    python scripts/experiment_report.py \\
        --experiment SC_Total_TFT_supplies_cp \\
        --intents baseline_v1 feature_eng_weather_lag \\
        --baseline-intent baseline_v1 \\
        --out-dir artifacts/reports/feature_eng_check

    # data_scope 기준 비교 (STYLE 추가 영향)
    python scripts/experiment_report.py \\
        --experiment SC_Total_TFT_supplies_cp \\
        --data-scopes cp6677_only cp6677_plus_hz01 \\
        --baseline-data-scope cp6677_only

산출:
- comparison.md         (사용자-facing 4계층 metric pivot)
- cross_run_metrics.csv (raw, pandas 재로드 가능)
- lift_by_intent.png    (intent별 핵심 metric bar)
- horizon_curve_by_intent.png (intent별 horizon × WAPE line)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_filter(intents: list[str] | None, data_scopes: list[str] | None) -> str:
    """MLflow filter_string. 비어있으면 빈 문자열 (전체).

    MLflow REST 가 IN(...) 미지원이라 OR 체인 사용. 각 group 은 괄호로 묶음.
    """
    parts = []
    if intents:
        chain = " OR ".join(f"tags.experiment_intent = '{i}'" for i in intents)
        parts.append(f"({chain})" if len(intents) > 1 else chain)
    if data_scopes:
        chain = " OR ".join(f"tags.data_scope = '{d}'" for d in data_scopes)
        parts.append(f"({chain})" if len(data_scopes) > 1 else chain)
    return " AND ".join(parts)


def _select_baseline_runs(
    runs: pd.DataFrame,
    baseline_intent: str | None,
    baseline_data_scope: str | None,
) -> pd.DataFrame:
    """baseline 기준 row 추출. is_baseline=true 또는 명시 intent/scope 매칭."""
    if baseline_intent:
        return runs[runs["tags.experiment_intent"] == baseline_intent]
    if baseline_data_scope:
        return runs[runs["tags.data_scope"] == baseline_data_scope]
    # is_baseline=true 가 있으면 그것
    if "tags.is_baseline" in runs.columns:
        return runs[runs["tags.is_baseline"] == "true"]
    return runs.iloc[:0]


def _aggregate_metric_columns(runs: pd.DataFrame, key: str | None) -> pd.DataFrame:
    """metric columns 추출 + key 별 평균. is_parent=true 인 row 는 child 가 따로 있으므로 우선 제외."""
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    if not metric_cols:
        return pd.DataFrame()
    df = runs[metric_cols + ([key] if key else [])].copy()
    df.columns = [c.replace("metrics.", "") if c.startswith("metrics.") else c for c in df.columns]
    if key:
        return df.groupby(key).mean(numeric_only=True).reset_index()
    return df.mean(numeric_only=True).to_frame().T


def _compute_lift_columns(pivot: pd.DataFrame, baseline_key: str, key_col: str) -> pd.DataFrame:
    """baseline_key 행 대비 다른 행의 lift 계산. lift = (baseline - other) / baseline. 양수=other 가 좋음."""
    if baseline_key not in pivot[key_col].values:
        return pivot
    baseline_row = pivot[pivot[key_col] == baseline_key].iloc[0]
    metric_cols = [c for c in pivot.columns if c != key_col]
    out = pivot.copy()
    for col in metric_cols:
        # WAPE/MAE/RMSE/MAPE-safe: 낮을수록 좋음 → lift = (baseline - other) / baseline
        # coverage: 높을수록 좋음 (목표 근접) → lift_coverage 는 별도
        if "wape" in col or "mae" in col or "rmse" in col or "mape_safe" in col or "bias" in col:
            b = float(baseline_row[col]) if col in baseline_row else np.nan
            if not np.isnan(b) and b != 0:
                out[f"lift_vs_baseline.{col}"] = (b - pivot[col]) / b
    return out


def _make_horizon_curve(runs: pd.DataFrame, group_col: str, out_path: Path) -> Path | None:
    """horizon × WAPE line plot per group."""
    horizon_cols = [c for c in runs.columns if c.startswith("metrics.horizon.wape_h")]
    if not horizon_cols:
        return None
    rows = []
    for _, r in runs.iterrows():
        for c in horizon_cols:
            h = int(c.replace("metrics.horizon.wape_h", ""))
            v = r[c]
            if pd.isna(v):
                continue
            rows.append({"group": r[group_col], "h": h, "wape": float(v)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 4))
    for grp, sub in df.groupby("group"):
        agg = sub.groupby("h")["wape"].mean().reset_index()
        ax.plot(agg["h"], agg["wape"], "o-", ms=4, label=str(grp))
    ax.set_xlabel("horizon h")
    ax.set_ylabel("WAPE")
    ax.set_title(f"horizon × WAPE × {group_col}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _make_lift_bar(pivot: pd.DataFrame, key_col: str, baseline_key: str, out_path: Path) -> Path | None:
    """핵심 metric bar plot — overall.wape + bin.{cold|mid|far}.wape."""
    core = ["overall.wape_q50", "bin.cold.wape_q50", "bin.mid.wape_q50", "bin.far.wape_q50"]
    present = [c for c in core if c in pivot.columns]
    if not present:
        return None
    df = pivot[[key_col] + present].copy()
    df = df.set_index(key_col)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(present))
    width = 0.8 / max(1, len(df.index))
    for i, (key, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width - 0.4, row[present].values, width,
               label=f"{key}{' (baseline)' if key == baseline_key else ''}")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_q50", "").replace("overall.", "") for p in present], rotation=15)
    ax.set_ylabel("WAPE")
    ax.set_title(f"Core metric × {key_col}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _format_md_table(df: pd.DataFrame, *, float_fmt: str = "{:.4f}") -> str:
    """markdown 표 — 숫자는 .4f, 다른 타입은 그대로."""
    if df.empty:
        return "(empty)"
    headers = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in headers:
            v = r[c]
            if isinstance(v, float):
                cells.append(float_fmt.format(v) if not np.isnan(v) else "—")
            elif v is None or (isinstance(v, str) and v == ""):
                cells.append("—")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "---|" * len(headers)
    return "\n".join([head, sep] + rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-run comparison report via MLflow search.")
    p.add_argument("--experiment", required=True, help="MLflow experiment name (예: SC_Total_TFT_supplies_cp)")
    p.add_argument("--tracking-uri", default="http://10.90.8.125:5000/")
    p.add_argument("--intents", nargs="*", default=None, help="experiment_intent 필터 (다중 가능)")
    p.add_argument("--data-scopes", nargs="*", default=None, help="data_scope 필터 (다중 가능)")
    p.add_argument("--baseline-intent", default=None, help="baseline 기준 intent. 미지정 시 is_baseline=true 사용")
    p.add_argument("--baseline-data-scope", default=None, help="baseline 기준 data_scope")
    p.add_argument("--include-parents", action="store_true",
                   help="is_parent=true run 도 포함 (default: child only)")
    p.add_argument("--out-dir", required=True, help="보고서 출력 디렉토리")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp is None:
        print(f"❌ Experiment 없음: {args.experiment}")
        sys.exit(1)

    filter_string = _build_filter(args.intents, args.data_scopes)
    print(f"=== experiment_report ===")
    print(f"  experiment : {args.experiment}  (id={exp.experiment_id})")
    print(f"  filter     : {filter_string or '(none)'}")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], filter_string=filter_string)
    if runs.empty:
        print("❌ 일치하는 run 없음")
        sys.exit(1)

    # parent 제외 (default)
    if not args.include_parents and "tags.is_parent" in runs.columns:
        runs = runs[runs["tags.is_parent"] != "true"]
    print(f"  matched    : {len(runs)} runs")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # raw csv
    raw_path = out_dir / "cross_run_metrics.csv"
    runs.to_csv(raw_path, index=False)
    print(f"  → {raw_path}")

    # 그룹 키 결정
    if args.intents or args.baseline_intent:
        group_col = "tags.experiment_intent"
        baseline_key = args.baseline_intent
    elif args.data_scopes or args.baseline_data_scope:
        group_col = "tags.data_scope"
        baseline_key = args.baseline_data_scope
    else:
        group_col = "tags.experiment_intent"
        baseline_key = None

    if group_col not in runs.columns:
        print(f"⚠️  {group_col} tag 없음 — group 미적용")
        group_col = None

    # pivot — group 별 metric 평균
    if group_col:
        pivot = _aggregate_metric_columns(runs, group_col)
        if baseline_key and baseline_key in pivot[group_col].values:
            pivot = _compute_lift_columns(pivot, baseline_key, group_col)
    else:
        pivot = _aggregate_metric_columns(runs, None)

    pivot_path = out_dir / "metric_pivot.csv"
    pivot.to_csv(pivot_path, index=False)
    print(f"  → {pivot_path}")

    # plots
    lift_plot = _make_lift_bar(pivot, group_col or "summary", baseline_key or "(none)", out_dir / "lift_by_intent.png") if group_col else None
    horizon_plot = _make_horizon_curve(runs, group_col, out_dir / "horizon_curve_by_intent.png") if group_col else None
    if lift_plot:
        print(f"  → {lift_plot}")
    if horizon_plot:
        print(f"  → {horizon_plot}")

    # comparison.md
    md_lines = [
        f"# Experiment comparison — {args.experiment}",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Filter: `{filter_string or '(none)'}`",
        f"- Matched runs: **{len(runs)}**",
        f"- Group: `{group_col}`",
        f"- Baseline: `{baseline_key or '(none)'}`",
        "",
        "## Overall metrics (pivot)",
        "",
    ]
    overall_cols = [c for c in pivot.columns
                    if (c == group_col or c.startswith("overall.") or c.startswith("lift_vs_baseline.overall."))]
    if overall_cols:
        md_lines.append(_format_md_table(pivot[overall_cols]))
    else:
        md_lines.append("(no overall metrics)")

    md_lines += ["", "## Horizon-bin metrics", ""]
    bin_cols = [c for c in pivot.columns
                if (c == group_col or c.startswith("bin.") or c.startswith("lift_vs_baseline.bin."))]
    if len(bin_cols) > 1:
        md_lines.append(_format_md_table(pivot[bin_cols]))
    else:
        md_lines.append("(no bin metrics)")

    md_lines += ["", "## Run-level detail", ""]
    detail_cols = [c for c in ["run_id", "tags.mlflow.runName", "tags.experiment_intent",
                                "tags.data_scope", "tags.backtest_fold", "tags.git_sha",
                                "tags.is_baseline", "metrics.overall.wape_q50",
                                "metrics.overall.q25_q75_coverage"]
                   if c in runs.columns]
    if detail_cols:
        detail = runs[detail_cols].copy()
        detail.columns = [c.replace("tags.", "").replace("metrics.", "") for c in detail.columns]
        md_lines.append(_format_md_table(detail))

    md_lines += ["", "## Artifacts", "",
                 f"- `{raw_path.name}` — raw search_runs output (pandas re-loadable)",
                 f"- `{pivot_path.name}` — metric pivot (group averages + lift)"]
    if lift_plot:
        md_lines.append(f"- `{lift_plot.name}` — core metric bar")
    if horizon_plot:
        md_lines.append(f"- `{horizon_plot.name}` — horizon × WAPE line")

    md_path = out_dir / "comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  → {md_path}")
    print(f"\n✅ Report saved to {out_dir}")


if __name__ == "__main__":
    main()
