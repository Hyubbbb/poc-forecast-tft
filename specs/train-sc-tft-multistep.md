---
module: train-sc-tft-multistep
status: draft
kind: module
version: "0.1.0"
owner: kimhyub
tags: [poc, ml, sc-total, tft, multi-horizon, interpretability, notebook]
---

# SC-Total TFT 다중 호라이즌(t+1~t+8) 예측 PoC

## Purpose

SC-Total grain의 주간 판매량을 **t+1 ~ t+8 (8주) 동시 예측**하는 TFT 모델을 PoC로 구축한다. 기존 `serp-distSupplementAI/train-sc-total` (LightGBM, t+1) 의 **대체 후보**로서 동일 데이터·평가에서 head-to-head 비교가 가능해야 하며, 해석가능성은 TFT 고유 산출물(VSN 가중치 + 시점별 attention heatmap) 의 case study로 입증한다. PoC 단계는 **단일 노트북** 흐름으로 진행하며, `src/` 모듈화·운영 통합·Champion alias 등록은 비대상.

## TODO

- [ ] requirements.txt + 노트북 골격 생성
- [ ] Snowflake → `data/sc_weekly.parquet` 1회 추출 (`FACT_SALES_WEEKLY_SC_TOTAL`)
- [ ] EDA 셀 (SC 수, 주차 분포, 결측, lag-1/52 autocorr) — encoder length 결정 근거
- [ ] 피처 정의 (lag_1/2, WEEK_OF_YEAR, temp/precip, TAG_PRICE, delivery, SALE_QTY_23F, 범주형)
- [ ] Naive-seasonal baseline (`y[t+h]=y[t+h-52]` + 다년 평균 비교)
- [ ] LightGBM 베이스라인 회수 (사내 MLflow `SC_Total_LightGBM` run 다운로드 또는 셀 내 재학습)
- [ ] TFT 학습 (encoder_len=52 또는 104, decoder_len=8, quantiles=[0.1,0.5,0.9])
- [ ] Walk-forward 평가 → horizon별 WAPE 표 (t+1..t+8 × {TFT, LightGBM, Naive})
- [ ] VSN 가중치 시각화 (static/encoder/decoder)
- [ ] Attention heatmap (대표 SC 2~3개 case study)
- [ ] 종합 비교 리포트 셀 + AC 체크
- [ ] MLflow artifact 등록 검증 (메트릭 + plot + 체크포인트)

## Inputs

| name | type | ref | description |
|------|------|-----|-------------|
| FACT_SALES_WEEKLY_SC_TOTAL | snowflake_table | FACT_SALES_WEEKLY_SC_TOTAL | 주간 SC-Total 판매 base 테이블 (1회 추출 → parquet 캐시) |
| 외부 변수 컬럼 풀 | snowflake_columns | (위 테이블) | temp_mean/max/min, precipitation_sum, TAG_PRICE, WEEKLY_STYLE_COLOR_DELV_QTY, SALE_QTY_23F, WEEK_OF_YEAR, 범주형(SESN_SUB, FIT_INFO1, FAB_TYPE, PRDT_KIND_CD, SEX, ITEM, COLOR_CD) |
| LightGBM 베이스라인 | mlflow_run | http://10.90.8.125:5000/ experiment `SC_Total_LightGBM` | t+1 예측치 회수 (artifact 다운로드 또는 셀 내 재학습) |

## Outputs

| name | type | ref | description |
|------|------|-----|-------------|
| MLflow run | mlflow_run | http://10.90.8.125:5000/ experiment `SC_Total_TFT` | TFT 메트릭(WAPE/MAE/RMSE/Pinball/RMSLE × horizon), 체크포인트, plots |
| Horizon별 WAPE 표 | mlflow_artifact | `wape_by_horizon.json` + `wape_by_horizon.png` | t+1..t+8 × {TFT, LightGBM, Naive} 비교표 |
| VSN 가중치 | mlflow_artifact | `vsn_static.png`, `vsn_encoder.png`, `vsn_decoder.png` | Variable Selection Network 가중치 bar plots |
| Attention heatmap | mlflow_artifact | `attention_<SC>.png` | 대표 SC case study (시점 × encoder lookback) |
| 메인 노트북 | notebook | `notebooks/tft_poc.ipynb` | 셀 단위 재현 가능한 단일 노트북 |

## Rules

- **Grain·Split**: SC × 주간(MON–SUN). Train/Val/Test 시간 기반 분할 (test_week = `snap_to_week_start(cutoff_date)`). 기존 `train-sc-total` 컨벤션 준수.
- **Walk-forward**: 노트북 셀 내 `for cutoff in cutoffs:` 루프. `--cutoff-dates` 다중 주차 컨셉 차용 (CLI 만들지 않음).
- **MLflow URI**: `http://10.90.8.125:5000/` 강제. localhost 사용 금지.
- **노트북 우선**: `src/` 패키지화 금지. 효율보다 단계 가독성·검증 가능성 우선.
- **하드웨어**: Apple Silicon MPS 우선 시도, 부족하면 Colab Pro로 동일 노트북 이전.
- **PoC 비대상**: 운영 배포·Champion alias 자동 승급·Docker 통합·다른 모델 비교(N-BEATS/DeepAR 등).

## Implementation

- `notebooks/tft_poc.ipynb` (메인, 셀 11개 + 헤더)
- `requirements.txt` (torch, pytorch-forecasting, pytorch-lightning, mlflow, lightgbm, polars/pandas, snowflake-connector-python)
- `data/` (parquet 캐시, gitignore)
- `.omc/specs/deep-interview-tft-poc.md` (요구사항 인터뷰 결과 — 본 spec의 상위 컨텍스트)

## Acceptance Criteria

- [ ] 노트북 Restart & Run All 시 끝까지 통과
- [ ] horizon별 WAPE 표(t+1..t+8 × {TFT, LightGBM, Naive}) 셀 출력
- [ ] **t+1 WAPE: LightGBM 대비 ±2%p 이내**
- [ ] **t+2 ~ t+8 horizon별 WAPE: naive-seasonal baseline 대비 각 horizon별로 ≤ baseline**
- [ ] **시즌 시작 직후 cold cutoff (`SESN_SUB`별 typical_start + ~2주)에서의 horizon별 WAPE도 naive baseline 대비 개선** (TFT cold-start 차별화 입증)
- [ ] **cutoff × horizon WAPE 분리 표** 산출 (cold cutoff 2개 + warm cutoff 2개 권장) + MLflow log
- [ ] **PRDT_KIND_CD × horizon WAPE 분리 표** 산출 (의류 카테고리 INN/OUT/BOT/WTC 별 TFT 성능 비교) + MLflow log
- [ ] **SC history 길이 bin × horizon WAPE 분리 표** 산출 (COLD 4-12주 / WARM 13-25주 / MATURE 26+주) — 사용자 비즈니스 가치 = COLD bin 성능
- [ ] **Multi-horizon 일관성 검증**: t+1..t+8 P50 예측 시계열의 `|Δ|` 분포 plot + horizon 간 단조 위반 비율 출력 (LightGBM Direct 대비 TFT의 native multi-horizon 가치 입증)
- [ ] VSN 가중치 3종(static/encoder/decoder) bar plot 노트북 + MLflow artifact 양쪽 존재
- [ ] Attention heatmap 대표 SC ≥2개 노트북 + MLflow artifact 양쪽 존재
- [ ] MLflow UI에서 experiment `SC_Total_TFT` run에 메트릭 + artifact + 체크포인트(`.ckpt`) 등록 확인
- [ ] 종합 비교 리포트 셀이 표 + 한 줄 해석 출력
- [ ] `_index.yaml` status가 적절히 갱신 (PoC 종료 시 `done` 또는 `archived`)

## References

- 기존 SSOT: `/Users/kimhyub/Downloads/hyub/FnF/serp-distSupplementAI/specs/train-sc-total.md`
- 인터뷰 spec: `.omc/specs/deep-interview-tft-poc.md`
- TFT 논문: Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)
- pytorch-forecasting docs: https://pytorch-forecasting.readthedocs.io/

## Changelog

- 2026-04-30 v0.1.0 — 초기 draft (deep-interview 5라운드, 최종 ambiguity 8.8%)
