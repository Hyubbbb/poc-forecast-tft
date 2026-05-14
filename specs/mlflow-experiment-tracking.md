---
module: mlflow-experiment-tracking
status: draft
kind: module
version: "0.2.0"
owner: kimhyub
tags: [mlflow, experiment-tracking, infrastructure, comparison, visualization, interpretability, stage0]
---

# MLflow 실험 트래킹 + 시각화 강화 인프라 (Stage 0)

## Purpose

피처 추가·학습 데이터 보강·백테스팅 다양화 등 모델 고도화 실험을 **유의미한 비교 단위**로 묶어 트래킹하고, **사업부 설명용 + 사용자 검증용 시각화**까지 자동 생성하는 인프라를 마련한다. 결과로 다음 4가지를 보장한다.

1. **재현성** — 같은 run을 다시 돌릴 수 있도록 config + training_dataset + forecast + diagnose + viz 일체를 MLflow 에 적재. random_seed + input_parquet_sha256 강제.
2. **비교성** — 같은 `experiment_intent` 의 N child runs 를 nested 로 묶고, parent run 에 집계 metric 박아 MLflow UI 한 줄에서 비교. cross-architecture 비교를 위해 `model_architecture` tag 도입.
3. **다층 metric** — overall · per-horizon · per-bin · per-STYLE×bin 의 4계층 metric 을 일관 dot-prefix 명명 규칙으로 적재해 사후 슬라이싱 자유도 확보.
4. **시각화 자동화** — TFT interpretability (VSN + attention) + per-SC overlay (PDF + plotly HTML) 를 학습 끝난 직후 자동 생성, MLflow artifact 적재.

본 spec 은 **인프라**만 다룬다. 실제 개선 실험(피처/데이터/BT)은 본 인프라 *위에서* 별도 spec.

## TODO

### Phase 1 — Helpers + spec
- [x] 본 spec v0.2.0
- [x] `specs/_index.yaml` v0.2.0 갱신
- [x] `scripts/eval_utils.py` 에 4계층 metric 헬퍼 추가:
  - `compute_coverage_metrics(forecast, actual, model) → dict` (overall + horizon)
  - `compute_lift_metrics(tft_wape, baseline_wape_dict) → dict` (vs naive_cohort_mean·seasonal_naive)
  - `compute_full_metrics(forecast, actual, model, style_map=None, baselines_predict=None) → dict` (4계층 통합)
- [x] `scripts/mlflow_logging.py` 신규 모듈 — 트래킹 SoT
  - `make_run_tags(intent, data_scope, fold, model_arch, seed, is_baseline) → dict` (8 tag)
  - `log_full_metrics(run_id, metrics_dict, tracking_uri=None)` (nested dict → flat dot-prefix)
  - `compute_dataset_audit(df, style_col, parquet_path) → dict` (style_list, n_series_per_style, sha256, sanity params)
  - `start_nested_runs(parent_intent, child_specs)` context manager
  - `safe_git_sha()` (자동 캡쳐, 실패 시 'unknown')

### Phase 2 — Visualization
- [x] `requirements.txt` 에 `plotly>=5.18` 추가
- [x] `scripts/visualize.py` 신규 — 4 함수:
  - `make_vsn_plots(model, training_ds, df_feat, cutoff, out_dir) → list[Path]` (3 PNG)
  - `make_attention_plots(model, training_ds, df_feat, cutoff, group_key, top_n_sc, out_dir) → Path` (1 PNG)
  - `make_all_sc_overlay_pdf(forecast, actual_df, group_key, qmap, out_pdf, scs_per_page=4) → Path`
  - `make_interactive_dashboard(forecast, actual_df, group_key, style_col, qmap, out_html) → Path`
- [x] `notebooks/tft_poc.py:990-1015` 의 VSN+attention 코드 이식 (재구현 X)

### Phase 3 — Integration
- [x] `scripts/train.py:_enrich_mlflow_run` 보강:
  - 시그니처 `_enrich_mlflow_run(..., parent_run_id: str | None = None)`
  - parent 있을 때 `mlflow.start_run(nested=True)` 사용
  - `compute_dataset_audit` 호출 → sanity params 적재
  - `compute_full_metrics` 호출 → `log_full_metrics`
  - `make_vsn_plots`, `make_attention_plots`, `make_all_sc_overlay_pdf`, `make_interactive_dashboard` 자동 호출 → `viz/` artifact
  - `make_run_tags` 호출 → 8 tag 적재
- [ ] (deferred) `scripts/forecast_utils.log_eval_to_mlflow` 4계층 확장 (호환 유지)
- [x] `configs/tft_supplies_cp_e52_d26.yaml` 보강:
  - `mlflow.experiment` 정정: `SC_Total_TFT` → `SC_Total_TFT_supplies_cp`
  - `mlflow.experiment_intent`: `baseline_v1`
  - `mlflow.is_baseline`: `true`
  - `train.seed`: `42` (재현성)

### Phase 4 — Comparison + Baseline + Tests
- [x] `scripts/experiment_report.py` 신규 (mlflow.search_runs + pandas pivot + plot)
- [x] `tests/test_mlflow_logging.py` (mock client unit)
- [x] `tests/test_visualize.py` (smoke: PDF/HTML 생성 검증)
- [x] 기존 `tft_supplies_cp_e52_d26` MLflow run 에 4계층 metric + viz artifact 추가 적재 → `experiment_intent=baseline_v1`, `is_baseline=true`
- [x] `comparison.md` self-test (baseline 1개 → lift=0)
- [ ] (deferred) `scripts/backtest_yearly.py` 의 7 flat run → 1 parent + 7 child 리팩토링

## Inputs

| name | type | ref | description |
|------|------|-----|-------------|
| MLflow tracking server | service | `http://10.90.8.125:5000/` | 사내 MLflow |
| 기존 로깅 자산 | python | `scripts/train.py:377-428` `_enrich_mlflow_run`, `scripts/forecast_utils.py:309` `log_eval_to_mlflow`, `:282` `flatten_cfg` | 보강 대상 |
| 평가 SoT | python | `scripts/forecast_utils.py` `evaluate_horizons`, `evaluate_horizons_styled`, `_bin_horizon` | metric 계산 위임 |
| quantile SoT | python | `scripts/eval_utils.py` `resolve_quantile_cols`, `quantile_columns` | quantile-aware 컬럼 |
| Interpretability 참조 | python | `notebooks/tft_poc.py:990-1015` (VSN, attention nested run) | `scripts/visualize.py` 로 이식 |
| Phase A-2 spec | spec | `specs/train-sc-tft-supplies-cp.md` | 본 인프라 첫 소비자 (Stage 2 일부) |
| Baseline 후보 run | mlflow | `d46f3d4eec02451abd3444d0888afadd` (`SC_Total_TFT/cp6677_e52_d26_iqr_20260513_1706`) | 재로깅 대상 |

## Outputs

### 신규 코드
| name | path | description |
|------|------|-------------|
| 로깅 헬퍼 | `scripts/mlflow_logging.py` | tags/metrics/nested/audit SoT |
| 시각화 헬퍼 | `scripts/visualize.py` | interpretability + overlay |
| 비교 보고 CLI | `scripts/experiment_report.py` | 다중 run pivot + 시각화 |
| 테스트 | `tests/test_mlflow_logging.py`, `tests/test_visualize.py` | mock + smoke |

### MLflow 적재 표준 — Per child run

**Tags (8개, 최종 확정)**:

| Tag | Source | 예시 |
|---|---|---|
| `source` | 기존 `cfg.mlflow.source_tag` | `supplies_cp_e52_d26_h2_2025_noStock_iqr` |
| `experiment_intent` | `cfg.mlflow.experiment_intent` | `baseline_v1`, `feature_eng_weather_lag` |
| `data_scope` | `cfg.mlflow.data_scope` 또는 parquet stem 자동 | `cp6677_only`, `cp_hea_steady_v1` |
| `backtest_fold` | CLI 또는 `cfg.mlflow.backtest_fold` | `single`, `2024`, `2024-Q3` |
| `git_sha` | 자동 (`git rev-parse --short HEAD`) | `6fbbd92` |
| `model_architecture` | 자동 (`cfg.model` 의 구조에서 유도) | `tft`, `lgbm`, `nbeats` |
| `random_seed` | `cfg.train.seed` | `42` |
| `is_baseline` | `cfg.mlflow.is_baseline` | `"true"` / `"false"` |

⚠️ MLflow native tags 활용:
- `mlflow.parentRunId` — nested run 시 자동 부착 (별도 tag 필요 X)
- `mlflow.runName` — `run_name` 인자로 제어

**제거된 tag (v0.1.0 대비)**:
- ~~`parent_run_id`~~ — `mlflow.parentRunId` native 와 중복
- ~~`quantiles`~~ — `cfg.model.quantiles` params 와 중복

**Params (기존 + 신규)**:

```
# 기존 (flatten_cfg)
cfg.* — yaml 전체 flatten

# 신규: dataset audit
style_list                  # json list (예: ["CP66", "CP77"])
style_count                 # int
n_series_per_style          # json dict (예: {"CP66": 31, "CP77": 45})
input_parquet_path          # 학습 입력 경로
input_parquet_sha256        # 동일 데이터 검증용 해시
n_series_train, n_rows_train
n_series_val_alive, n_rows_val
target_zero_ratio_train     # sparsity sanity

# 신규: 환경
torch_version, pytorch_forecasting_version
device                      # "mps" / "cuda" / "cpu"
```

**Metrics — 4계층 (dot prefix)**:

```
# (1) overall — 학습 끝 1번 적재
overall.wape_q50, overall.mae_q50, overall.rmse_q50, overall.bias_q50
overall.mape_safe_q50, overall.smape_q50
overall.{LOW}_{HIGH}_coverage          # 예: overall.q25_q75_coverage
overall.{LOW}_{HIGH}_coverage_gap      # actual coverage - target
overall.{LOW}_{HIGH}_coverage_target   # 이론 목표 (참고용)
overall.naive_cohort_mean.wape
overall.seasonal_naive.wape
overall.lift_naive_cohort_mean         # (baseline_wape - tft_wape) / baseline_wape, 양수=TFT 승
overall.lift_seasonal_naive
overall.lift_best                      # max(naive, seasonal) 기준

# (2) horizon (h=1..decoder_len)
horizon.wape_h{N}, horizon.mae_h{N}, horizon.bias_h{N}, horizon.coverage_h{N}

# (3) horizon-bin (cold/mid/far, _bin_horizon 활용)
bin.cold.wape_q50, bin.cold.mae_q50, bin.cold.coverage
bin.mid.wape_q50, ...
bin.far.wape_q50, ...

# (4) STYLE × horizon-bin (style_map 제공 시)
style.{STYLE_CD}.cold.wape_q50, style.{STYLE_CD}.cold.mae_q50
style.{STYLE_CD}.mid.wape_q50, ...
style.{STYLE_CD}.far.wape_q50, ...
style.{STYLE_CD}.overall.wape_q50      # 해당 STYLE 의 전 horizon 평균
```

**MAPE 박지 않음** — sparse target 폭발 (`[[feedback_metric_for_sparse_target]]`).
**q-prefix 강제** — `p50` 하드코딩 금지 (`[[feedback_quantile_labeling]]`).

**Artifacts**:

```
config.yaml                              # 학습 입력 (재현용)
training_dataset.pkl                     # 재학습/추론용 (2 MB / run)
forecast.parquet                         # 예측 결과 q-prefix 컬럼
best_ckpt_path.txt                       # 로컬 ckpt 절대 경로 (용량 사유 직접 업로드 X)

diagnose/                                # 기존 진단 산출물 (그대로)
├── summary.json
├── horizon_metrics.csv + .png
├── style_weekly.csv + .png
├── sample_sc.png                        # top-6 SC fan-chart (legacy 유지)
└── val_zero_diagnosis.png

viz/                                     # 신규 — 사업부·사용자 시각화
├── interpretability/                    # train.py 자동
│   ├── vsn_static.png                  # 정적 변수 중요도 bar
│   ├── vsn_encoder.png                 # encoder 변수 중요도 bar
│   ├── vsn_decoder.png                 # decoder 변수 중요도 bar
│   └── attention_top3.png              # 매출 상위 3 SC attention heatmap
└── overlay/                             # train.py 자동
    ├── all_sc_overlay.pdf              # 모든 SC actual vs q50+CI band, 페이지당 4 SC
    └── interactive_dashboard.html      # plotly: SC selector + STYLE 필터 + week slider

sanity/
└── dataset_audit.json                   # style_list, n_series_per_style, sha256 등 사람-읽기
```

### MLflow 적재 표준 — Per parent run

| 항목 | 값 |
|---|---|
| `tags.is_parent` | `"true"` |
| `tags.experiment_intent` | child 와 동일 |
| `tags.n_children` | child run 개수 |
| metrics | child 평균/median 의 `aggregate.*` namespace |
| artifacts | `comparison.md` + `cross_fold_metrics.csv` |

```
aggregate.mean.wape_q50, aggregate.median.wape_q50
aggregate.std.wape_q50, aggregate.std.coverage
aggregate.mean.lift_best
```

## Rules

### 1. MLflow run 구조 — nested (사용자 결정 B)
- 1 cutoff/fold 학습 = 1 child run
- 같은 `experiment_intent` 의 N child = 1 parent (parent metric = child 집계)
- 단일 cutoff 학습이면 parent 없이 child 만 (intent 태그로 후속 비교 가능)
- 헬퍼: `scripts/mlflow_logging.start_nested_runs(parent_intent, child_specs)`

### 2. MLflow experiment 명명 — 도메인별 분리 (사용자 결정)
```
SC_Total_TFT_supplies_cp     ← 현 작업 (CP66/CP77 모자)
SC_Total_TFT_supplies_dsc    ← 다른 용품 카테고리 (향후)
SC_Total_TFT_apparel         ← 의류 (향후)
SC_Total_LGBM_supplies_cp    ← LGBM head-to-head (Phase B)
```

run 명명: `{intent}_{config_short}_{fold}_{ts}` (예: `baseline_v1_e52d26_single_20260513_1706`)

### 3. Metric 범위 — 4계층 전부 (사용자 결정)
- overall + horizon + horizon-bin + STYLE×bin 자동 적재
- WAPE 외 MAE/RMSE/bias/coverage/lift 함께
- **MAPE 박지 않음** (sparse 폭발). MAPE-safe 만 overall 1개.
- quantile-aware: 컬럼명은 학습 `model.loss.quantiles` 그대로

### 4. Artifact 정책 — 풀패키지 (사용자 결정)
- config + training_dataset + forecast + diagnose + viz + sanity 모두 MLflow artifact
- `best.ckpt` 만 용량 사유 (5.9 MB) MLflow 직접 업로드 X — `best_ckpt_path.txt` 에 로컬 절대 경로 박음
- 매 run 의 training_dataset.pkl = 2 MB (cp6677). 100 run = 200 MB (합리적). 운영 단계에서 정리 정책 별도

### 5. 태깅 컨벤션
- `experiment_intent` 는 사람-읽기 가능 snake_case. 자동 생성 X (의도가 목적)
- `data_scope` 는 입력 parquet stem 또는 사람-읽기 식별자 (예: `cp6677_only`)
- `backtest_fold` = `single` (단일 cutoff) 또는 `YYYY` / `YYYY-QN`
- `git_sha` = 자동 캡쳐. uncommitted change 있어도 fail 하지 않고 `<sha>-dirty` 로 적재
- `random_seed` = `cfg.train.seed`. 미설정 시 명시적 경고 + run tag `seed_not_set=true`
- `is_baseline=true` 는 `data_scope` 별 1개만 허용 (comparison report 의 default baseline)

### 6. 시각화 강화 (사용자 결정)
- **인터프리터빌리티 자동**: 학습 끝난 직후 `_enrich_mlflow_run` 이 `scripts/visualize.py` 호출
- **All SC overlay PDF**: 모든 SC 의 actual vs q50+CI 시계열 plot, 페이지당 4 SC. matplotlib PdfPages 사용
- **Plotly interactive HTML**: SC selector + STYLE 필터 + week range slider. 사업부가 zoom/SC 선택 가능
- 기존 `diagnose/sample_sc.png` (top-6) 는 legacy 로 유지 (overview 용)

### 7. data_scope 변화 추적 (사용자 결정)
- `data_scope` tag = 사람-읽기 식별자 (예: `cp6677_only` → `cp6677_plus_hz01`)
- `style_list`, `style_count`, `n_series_per_style` params 자동 적재
- `experiment_report.py --data-scopes A B --baseline-data-scope A` 로 자동 lift 비교
- `input_parquet_sha256` 으로 동일 데이터 검증 가능

### 8. Comparison report 계약
- Input 모드 2종:
  - `--intents I1 I2 ... --baseline-intent I1` (intent 비교)
  - `--data-scopes D1 D2 ... --baseline-data-scope D1` (data 비교)
- 추가 옵션: `--experiment NAME`, `--out-dir PATH`
- Output:
  - `comparison.md` (사용자-facing 4계층 metric pivot 표)
  - `cross_run_metrics.csv` (raw, pandas 재로드)
  - `lift_by_intent.png`, `horizon_curve_by_intent.png`
- baseline lift 자동 계산 (양수 = baseline 대비 개선)

### 9. 회귀 안전망
- 기존 `_enrich_mlflow_run` 호출자 (`scripts/train.py:363`) 인터페이스 호환 유지 — `parent_run_id` optional
- 의류 PoC (`specs/train-sc-tft-multistep.md`) 및 기존 supplies (`specs/train-sc-tft-supplies.md`) 학습 흐름 무변경
- 기존 `log_eval_to_mlflow` 호출자 호환 인터페이스 유지

### 10. 테스트 정책 (사용자 결정)
- `tests/test_mlflow_logging.py` 는 `mlflow.tracking.MlflowClient` mock — 실제 서버 hit X
- `pytest.mark.mlflow` 마커 + skip-if-unreachable (사외 CI 통과 보장)
- 시각화 테스트는 임시 디렉토리에서 PDF/HTML 생성 → 파일 존재 + 손상 안 됨 확인

## Acceptance Criteria

### 인프라
- [x] `scripts/mlflow_logging.py` import smoke + 헬퍼 unit test green
- [x] `compute_full_metrics` 가 4계층 metric dict 반환 — shape/key 계약 테스트
- [x] `make_run_tags` 8개 필수 키 모두 포함 — 계약 테스트
- [x] `start_nested_runs` parent-child 구조 mock 검증
- [x] `compute_dataset_audit` sha256 + style_list 검증

### 시각화
- [x] `scripts/visualize.py` import smoke
- [x] `make_vsn_plots` 가 3 PNG 파일 생성 (단위 테스트)
- [x] `make_attention_plots` 가 1 PNG 생성
- [x] `make_all_sc_overlay_pdf` 가 76 SC 모두 포함한 PDF 생성 (cp6677 fixture 기반)
- [x] `make_interactive_dashboard` 가 valid HTML 파일 생성 (BeautifulSoup 검증)

### 보강된 train 흐름
- [x] `_enrich_mlflow_run(..., parent_run_id=None)` 시그니처 추가, 기존 호출자 무변경
- [x] `parent_run_id` 전달 시 MLflow nested run 으로 등록
- [x] dry-run 1회 (cp6677 e52_d26): MLflow run 에 8 tag + 4계층 metric + sanity params + artifact 디렉토리 일체 적재
- [x] `viz/interpretability/` 4 PNG + `viz/overlay/` PDF+HTML 적재 확인

### 비교 보고
- [x] `scripts/experiment_report.py --intents baseline_v1 --baseline-intent baseline_v1` → self-comparison report 생성
- [x] `comparison.md` 가 4계층 metric pivot 표 포함
- [x] `cross_run_metrics.csv` 가 pandas 로 재로드 가능
- [x] lift 모두 0 (self-comparison)

### 회귀
- [x] `pytest tests/ -q` 전부 green (현 85 + 신규 ≥10)
- [x] 기존 `_enrich_mlflow_run` 호출 (의류 dry-run) 무변경 확인
- [x] `scripts/backtest_yearly.py` Stage A tiny smoke 통과 (parent run 1 + child run 1 dry-run-fold)

### Baseline 고정
- [x] 현 `tft_supplies_cp_e52_d26` MLflow run 에 4계층 metric + viz/* artifact 추가 적재
- [x] tag 변경: `experiment_intent=baseline_v1`, `is_baseline=true`
- [x] MLflow UI 에서 4계층 metric/artifact 확인 가능
- [x] 본 baseline 기준으로 후속 실험 lift 자동 계산되는지 보고 스크립트로 확인

## Phase B (별도 spec, 비대상)

본 인프라를 사용하는 **실제 개선 실험**들. 각각 별도 spec 으로 분기:
- `train-sc-tft-supplies-cp.md` (Phase A-2) — 2-stage decoder sweep × 4 fold (이미 작성)
- `feature-eng-v1` — promo/holiday/stock 재투입, weather lag 등
- `data-scope-steady-hea` — Phase A-2 plan 의 steady-HEA 모집단 추가
- `business-report` — 사업부 1-pager PDF 자동화
- 운영 배포용 MLflow Model Registry — 합격선 정해진 후
- Optuna 자동 튜닝 — nested run 활용 가능하나 별도 spec
- error heatmap / volume tier / calibration plot — 시각화 보강 후속 spec
- MLflow 서버 다운 시 local-only fallback — 운영 정책

## References

### 기존 자산
- `scripts/train.py:377-428` `_enrich_mlflow_run` (보강 대상)
- `scripts/forecast_utils.py:309-335` `log_eval_to_mlflow` (보강 대상)
- `scripts/forecast_utils.py:282-294` `flatten_cfg` (재사용)
- `notebooks/tft_poc.py:990-1015` VSN+attention nested run (이식 대상)

### 평가 SoT
- `scripts/forecast_utils.py:91` `evaluate_horizons`
- `scripts/forecast_utils.py:198` `evaluate_horizons_styled`
- `scripts/forecast_utils.py:188` `_bin_horizon`
- `scripts/eval_utils.py` `resolve_quantile_cols`, `quantile_columns`, `make_naive_cohort_mean`, `make_seasonal_naive`
- `scripts/diagnose_backtest.py` (산출물 디렉토리 구조 참조)

### 사용자 결정 (본 세션, 2026-05-14)
- Run 구조: B (cutoff=run, intent=parent nested)
- Metric 범위: 전부 (4계층)
- Artifact: 풀패키지 (ckpt 는 path tag만)
- Metric naming: dot prefix
- Experiment 명명: 도메인별 분리
- Overlay 시각화: All SC PDF + Plotly HTML 둘 다
- Interpretability: train.py 자동 적재
- Style 추적: data_scope tag + style_list params
- Baseline: 기존 run 재로깅 (intent=baseline_v1)
- 테스트: mock unit + `pytest.mark.mlflow` 통합

### 관련 spec
- `specs/train-sc-tft-supplies-cp.md` (Phase A-2, 본 인프라 첫 소비자)
- `specs/train-sc-tft-multistep.md` (의류 PoC, 회귀 보호 대상)
- `specs/train-sc-tft-supplies.md` (Phase A, deprecated)

### 메모리
- [[feedback_metric_for_sparse_target]] — MAPE 박지 말 것
- [[feedback_quantile_labeling]] — q-prefix SoT
- [[reference_fnf_mlflow]] — 사내 MLflow URL
- [[project_cp6677_e52_d26_diagnosis]] — baseline 후보

### Plan
- `~/.claude/plans/nested-noodling-shamir.md` (Stage 0 plan, 본 spec 의 모태)

## Changelog

- 2026-05-14 v0.2.0 — 사용자 결정 4건 추가 반영 + plan 의 모든 결정 반영:
  - tag set 재구성 (제거 2: parent_run_id/quantiles, 신규 3: model_architecture/random_seed/is_baseline)
  - 시각화 강화 섹션 신설 (interpretability auto + All SC PDF + plotly HTML)
  - experiment 명명 도메인별 분리
  - data_scope 변화 추적 메커니즘 명시
  - 재현성 강제 (random_seed, parquet sha256)
  - Phase A-2 정합 (backtest_yearly 리팩토링 포함)
- 2026-05-14 v0.1.0 — 초안. 사용자 결정 4건 반영. 미합의 항목 5개 spec 본문에 명시.
