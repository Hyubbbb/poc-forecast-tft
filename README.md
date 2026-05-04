# poc-forecast-tft

SC-Total 주간 판매량을 **TFT(Temporal Fusion Transformer)** 로 **t+1 ~ t+8** 동시 예측하는 PoC. 기존 LightGBM 단일-단계(t+1) 모델의 대체 후보로서 head-to-head 비교 + 해석가능성(VSN, attention) 검증이 목적.

- **Spec**: `specs/train-sc-tft-multistep.md` (FnF SDD)
- **인터뷰 spec**: `.omc/specs/deep-interview-tft-poc.md`
- **메인 노트북**: `notebooks/tft_poc.ipynb`
- **MLflow**: http://10.90.8.125:5000/ — experiment `SC_Total_TFT`
- **베이스라인**: LightGBM (`serp-distSupplementAI/specs/train-sc-total`, t+1) + Naive-seasonal (t+1..t+8)

## 실행

```bash
pip install -r requirements.txt
jupyter lab notebooks/tft_poc.ipynb
```

노트북은 위에서 아래로 셀 단위(셋업 → 데이터 → EDA → 피처 → 베이스라인 → TFT → 평가 → 해석 → 리포트) 가시화 흐름. 각 셀이 결과를 직접 출력하므로 단계별로 검토하며 진행한다. 학습 시간/메모리 부족 시 동일 노트북을 **Colab Pro** 에 그대로 올려 실행 가능.

## Acceptance Criteria

`specs/train-sc-tft-multistep.md` 의 `## Acceptance Criteria` 체크박스를 직접 검증.
