-- ============================================================================
--  PoC TFT — SC-Total 주간 판매량 raw snapshot 추출 쿼리
--
--  목적: TFT 다중 호라이즌(t+1..t+8) 예측 PoC용 정적 데이터셋 1회 추출
--  Grain: SC_CD × WEEK_START (운영 SSOT FACT_SALES_WEEKLY_SC_TOTAL과 동일)
--  대상 기간: WEEK_START >= '2022-01-03' (3년+, 시즌 사이클 2회 이상 커버)
--
--  사용:
--    1) DataGrip 등에서 본 쿼리 실행
--    2) Result 탭 'Fetch all data' 클릭 (페이지 한계 우회)
--    3) Export Data → CSV → ../data/sc_weekly.csv 로 저장
--    4) notebooks/tft_poc.ipynb 셀 2 실행 → CSV → parquet 자동 변환
--
--  운영 참조:
--    - serp-distSupplementAI/sql/query/03b_FACT_SALES_WEEKLY_SC_TOTAL.sql
--    - serp-distSupplementAI/src/worker/feature_builder.py: _load_weekly_base, _load_product_master
-- ============================================================================
SELECT
    -- ===== Grain / Keys =====
    f.SC_CD,
    f.BRAND_CD,
    f.SSN_CD,
    f.PROD_CD,
    f.COLOR_CD,
    f.WEEK_START,

    -- ===== Target & 판매 메트릭 =====
    f.WEEKLY_SALE_QTY,        -- raw target (운영 LightGBM은 shift(-1)→TARGET_SALE_QTY 사용. PoC TFT는 raw 사용)
    f.WEEKLY_SALE_AMT,
    f.N_SALE_DAYS,
    f.WEEKLY_DISC_RAT,

    -- ===== Base 카테고리 =====
    f.PRDT_KIND_CD,

    -- ===== 재고 / SKU (DB_SCS_W 기반, base 테이블에 LEFT JOIN 됨) =====
    f.BOW_STOCK,              -- 주초 가용재고
    f.STOCK_RATIO,            -- 재고 비율 0~1
    f.CUM_INTAKE,             -- 누적입고
    f.SCS_W_DISC_RATE,        -- DB_SCS_W 기반 주간 할인율
    f.NUM_SIZES,              -- SC당 활성 사이즈 수 (정적)

    -- ===== 타겟 주(WEEK_START+7..+13) 서울 날씨 forecast (실측 또는 예보) =====
    f.FCST_AVG_MIN_TEMP,
    f.FCST_AVG_MAX_TEMP,
    f.FCST_TOTAL_PCP,
    f.FCST_MIN_MIN_TEMP,
    f.FCST_MAX_MAX_TEMP,
    f.FCST_TEMP_RANGE,

    -- ===== 시즌 메타 =====
    f.SSN_START,
    f.SSN_END,

    -- ===== 상품 마스터 (PRCS.DB_PRDT 집계) — static categoricals =====
    p.TAG_PRICE,
    p.SEX,
    p.FIT_INFO1,
    p.FAB_TYPE,
    p.ITEM,
    p.SESN_SUB                -- 1=Spring, 2=Pre-Summer, 3=Summer, 4=Fall, 5=Pre-Winter, 6=Winter

FROM ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL f
LEFT JOIN (
    SELECT
        PART_CD AS PROD_CD,
        ANY_VALUE(TAG_PRICE)  AS TAG_PRICE,
        ANY_VALUE(SEX)        AS SEX,
        ANY_VALUE(FIT_INFO1)  AS FIT_INFO1,
        ANY_VALUE(FAB_TYPE)   AS FAB_TYPE,
        ANY_VALUE(ITEM)       AS ITEM,
        ANY_VALUE(SESN_SUB)   AS SESN_SUB
    FROM PRCS.DB_PRDT
    GROUP BY PART_CD
) p
  ON p.PROD_CD = f.PROD_CD

WHERE f.WEEK_START >= '2022-01-03'   -- 3년+ 스냅샷 (encoder_len=52 + 작년/재작년 시즌 transfer 위해)

ORDER BY f.SC_CD, f.WEEK_START;
