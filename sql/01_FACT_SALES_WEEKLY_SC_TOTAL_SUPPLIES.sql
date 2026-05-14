/*
================================================================================
    FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES — 모자(CP) PRCS-based (Phase A-5)

    Source (PRCS):
      - DB_SCS_W           : weekly 판매/재고 fact (CNS 타겟 + RTL/RF/DOME/NOTAX 모니터링)
      - DB_PRDT            : PART_CD → STYLE_CD/PRDT_KIND_CD (cfg whitelist + static)
      - DB_PRDT_COLOR_MAP  : legacy COLOR_CD → N COLOR_CD 매핑

    Outputs (두 테이블 한 파일):
      - ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES : 학습/평가용 메인 fact
      - ML_DIST.DEAD_SC_SUPPLIES                    : 한 번도 CNS 판매 없던 SC 진단

    Grain: (BRAND, STYLE_CD, COLOR_CD_NORM, WEEK_START)
      - 학습 단위는 (BRAND_CD, STYLE_CD, COLOR_CD_NORM) 으로 식별
      - START_DT = DB_SCS_W.START_DT (월요일 보장, DDL 명시)
      - 같은 STYLE×COLOR_NORM 의 여러 PART_CD (계보) 합산
      - SIZE_CD 통합 (모자 free size)

    Spine / cut (Phase A-5 후속 결정, 2026-05-13):
      - DB_SCS_W 는 PART×COLOR×SIZE 별 row 가 이미 연속 weekly (sentinel 2035/2122 row 까지 포함).
        → 별도 spine 불필요.
      - 단, raw 의 미래 sentinel row 가 trailing zero 로 잡혀서 학습 데이터를 오염시키므로
        SC 별 FIRST_SALE_WEEK ~ LAST_SALE_WEEK (non-zero 첫/마지막 주) 으로 cut.
      - 한 번도 판매 없는 SC 23개는 메인 fact 제외 → DEAD_SC_SUPPLIES 로 별도 출력.
      - **진행 중 incomplete week 제외**: TEMP 단계에서 `END_DT < CURRENT_DATE()` 필터로
        일요일까지 완결된 주만 통과. 수요일 실행 시 그 주의 부분 집계가 학습에 들어가는 위험 차단.

    Target & Monitoring:
      - 타겟: SALE_NML_QTY_CNS + SALE_RET_QTY_CNS (위탁 통합)
      - 4채널 분해: RTL / RF / DOME / NOTAX (각 NML+RET) → 평가 진단용

    Features:
      - 재고: BOW_STOCK / STOCK_RATIO / AC_STOR_QTY_KOR (STYLE×COLOR_NORM grain LAG, cut 전 raw 기반)
      - 날씨: 서울 RGN_CD='01' 타겟 주(START_DT+7~+13) 평균
================================================================================
*/

-- ============================================================================
-- 0) TEMP TABLE — weekly raw aggregate (메인 fact + dead 진단 공유)
--    DB_SCS_W 풀스캔 + 매핑을 한 번만 수행.
-- ============================================================================
CREATE OR REPLACE TEMP TABLE ML_DIST._WEEKLY_RAW_SUPPLIES AS
WITH
cfg_brands AS (
    SELECT column1 AS brand_cd FROM VALUES ('M')
),
cfg_parent_kinds AS (
    SELECT column1 AS parent_prdt_kind_cd FROM VALUES ('A')
),
cfg_styles AS (
    SELECT column1 AS style_cd FROM VALUES ('M19SCP77'), ('M19FCP66')
),
-- 상품 마스터 (PART_CD 단위)
part_master AS (
    SELECT
        PART_CD,
        ANY_VALUE(PARENT_PRDT_KIND_CD) AS PARENT_PRDT_KIND_CD,
        ANY_VALUE(PRDT_KIND_CD)        AS PRDT_KIND_CD,
        ANY_VALUE(ITEM)                AS ITEM,
        ANY_VALUE(STYLE_CD)            AS STYLE_CD
    FROM PRCS.DB_PRDT
    GROUP BY PART_CD
),
-- legacy COLOR_CD → N COLOR_CD 매핑
color_map AS (
    SELECT STYLE_CD, COLOR_CD, ANY_VALUE(COLOR_CD_AFT) AS COLOR_CD_AFT
    FROM PRCS.DB_PRDT_COLOR_MAP
    GROUP BY STYLE_CD, COLOR_CD
)
SELECT
    sw.BRD_CD                                            AS BRAND_CD,
    pm.STYLE_CD                                           AS STYLE_CD,
    COALESCE(cm.COLOR_CD_AFT, sw.COLOR_CD)                AS COLOR_CD_NORM,
    ANY_VALUE(pm.PRDT_KIND_CD)                            AS PRDT_KIND_CD,
    sw.START_DT::date                                     AS WEEK_START,
    -- ===== Target (CNS = 위탁 통합) =====
    SUM(sw.SALE_NML_QTY_CNS      + sw.SALE_RET_QTY_CNS)      AS WEEKLY_SALE_QTY_CNS,
    SUM(sw.SALE_NML_SALE_AMT_CNS + sw.SALE_RET_SALE_AMT_CNS) AS WEEKLY_SALE_AMT_CNS,
    -- ===== Channel monitoring (CNS 분해) =====
    SUM(sw.SALE_NML_QTY_RTL      + sw.SALE_RET_QTY_RTL)      AS WEEKLY_SALE_QTY_RTL,
    SUM(sw.SALE_NML_QTY_RF       + sw.SALE_RET_QTY_RF)       AS WEEKLY_SALE_QTY_RF,
    SUM(sw.SALE_NML_QTY_DOME     + sw.SALE_RET_QTY_DOME)     AS WEEKLY_SALE_QTY_DOME,
    SUM(sw.SALE_NML_QTY_NOTAX    + sw.SALE_RET_QTY_NOTAX)    AS WEEKLY_SALE_QTY_NOTAX,
    -- ===== 할인율 계산용 =====
    SUM(sw.SALE_NML_SALE_AMT_CNS) AS WEEKLY_SALE_AMT_NML_CNS,
    SUM(sw.SALE_NML_TAG_AMT_CNS)  AS WEEKLY_TAG_AMT_NML_CNS,
    -- ===== 재고 누적 (LAG 용) =====
    SUM(sw.AC_STOR_QTY_KOR)                               AS AC_STOR_QTY_KOR,
    SUM(sw.AC_SALE_NML_QTY_CNS + sw.AC_SALE_RET_QTY_CNS)  AS AC_SALE_QTY_CNS
FROM PRCS.DB_SCS_W sw
    JOIN part_master pm
        ON  sw.PART_CD = pm.PART_CD
        AND pm.PARENT_PRDT_KIND_CD IN (SELECT parent_prdt_kind_cd FROM cfg_parent_kinds)
        AND pm.ITEM = 'CP'
        AND pm.STYLE_CD IN (SELECT style_cd FROM cfg_styles)
    LEFT JOIN color_map cm
        ON cm.STYLE_CD = pm.STYLE_CD
       AND cm.COLOR_CD = sw.COLOR_CD
WHERE sw.BRD_CD IN (SELECT brand_cd FROM cfg_brands)
  -- 일요일까지 완결된 주만 포함 (END_DT = START_DT + 6 < CURRENT_DATE).
  -- 진행 중인 주 (예: 수요일 실행 시 그 주의 월~수까지 부분 집계) 학습 오염 방지.
  AND DATEADD('day', 6, sw.START_DT::date) < CURRENT_DATE()
GROUP BY 1, 2, 3, 5;


-- ============================================================================
-- 1) MAIN FACT — dead 제외 + first/last non-zero week 으로 cut
-- ============================================================================
CREATE OR REPLACE TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES AS
WITH
-- raw 의 미래 sentinel cut (first/last non-zero week 사이만 통과)
weekly_capped AS (
    SELECT
        BRAND_CD,
        STYLE_CD,
        COLOR_CD_NORM,
        PRDT_KIND_CD,
        WEEK_START,
        WEEKLY_SALE_QTY_CNS,
        WEEKLY_SALE_AMT_CNS,
        WEEKLY_SALE_AMT_NML_CNS,
        WEEKLY_TAG_AMT_NML_CNS,
        WEEKLY_SALE_QTY_RTL,
        WEEKLY_SALE_QTY_RF,
        WEEKLY_SALE_QTY_DOME,
        WEEKLY_SALE_QTY_NOTAX,
        MIN(CASE WHEN WEEKLY_SALE_QTY_CNS != 0 THEN WEEK_START END)
            OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM) AS FIRST_SALE_WEEK,
        MAX(CASE WHEN WEEKLY_SALE_QTY_CNS != 0 THEN WEEK_START END)
            OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM) AS LAST_SALE_WEEK
    FROM ML_DIST._WEEKLY_RAW_SUPPLIES
    QUALIFY FIRST_SALE_WEEK IS NOT NULL                                 -- dead SC 제외
        AND WEEK_START BETWEEN FIRST_SALE_WEEK AND LAST_SALE_WEEK       -- trailing/leading cut
),
-- BOW Stock = 전주 누적입고 - 전주 누적판매 (cut 전 raw 위에서 LAG 해야 첫 주 정확)
STOCK_FEATURES AS (
    SELECT
        BRAND_CD,
        STYLE_CD,
        COLOR_CD_NORM,
        WEEK_START,
        AC_STOR_QTY_KOR,
        COALESCE(
            LAG(AC_STOR_QTY_KOR, 1) OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM ORDER BY WEEK_START),
            AC_STOR_QTY_KOR
        ) - COALESCE(
            LAG(AC_SALE_QTY_CNS, 1)  OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM ORDER BY WEEK_START),
            0
        ) AS BOW_STOCK,
        (COALESCE(
            LAG(AC_STOR_QTY_KOR, 1) OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM ORDER BY WEEK_START),
            AC_STOR_QTY_KOR
        ) - COALESCE(
            LAG(AC_SALE_QTY_CNS, 1)  OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM ORDER BY WEEK_START),
            0
        )) / NULLIF(COALESCE(
            LAG(AC_STOR_QTY_KOR, 1) OVER (PARTITION BY BRAND_CD, STYLE_CD, COLOR_CD_NORM ORDER BY WEEK_START),
            AC_STOR_QTY_KOR
        ), 0) AS STOCK_RATIO
    FROM ML_DIST._WEEKLY_RAW_SUPPLIES
),
-- 날씨 범위는 cut 된 weekly_capped 의 WEEK_START 범위만큼
WC_RANGE AS (
    SELECT MIN(WEEK_START) AS MIN_W, MAX(WEEK_START) AS MAX_W
    FROM weekly_capped
),
WEATHER_DAILY AS (
    SELECT DT, MIN_TEMP, MAX_TEMP, PCP_AMT
    FROM COMM.DM_WEATHER_D
    WHERE RGN_CD = '01'
      AND DT BETWEEN (SELECT MIN_W FROM WC_RANGE)::DATE + 7
                 AND (SELECT MAX_W FROM WC_RANGE)::DATE + 13
),
WEATHER_FORECAST_DAILY AS (
    SELECT DT, MIN_TEMP, MAX_TEMP, PCP_AMT
    FROM COMM.DM_WEATHER_FORECAST_D
    WHERE RGN_CD = '01'
      AND DT BETWEEN (SELECT MIN_W FROM WC_RANGE)::DATE + 7
                 AND (SELECT MAX_W FROM WC_RANGE)::DATE + 13
),
WEATHER_MERGED AS (
    SELECT
        COALESCE(A.DT, F.DT)             AS DT,
        COALESCE(A.MIN_TEMP, F.MIN_TEMP) AS MIN_TEMP,
        COALESCE(A.MAX_TEMP, F.MAX_TEMP) AS MAX_TEMP,
        COALESCE(A.PCP_AMT, F.PCP_AMT)   AS PCP_AMT
    FROM WEATHER_DAILY A
    FULL OUTER JOIN WEATHER_FORECAST_DAILY F ON A.DT = F.DT
),
TARGET_WEEK_WEATHER AS (
    SELECT
        DATEADD('day', -(DAYOFWEEKISO(DT::date) - 1)::int, DT::date) - INTERVAL '7 DAY' AS WEEK_START,
        AVG(MIN_TEMP::FLOAT)                          AS FCST_AVG_MIN_TEMP,
        AVG(MAX_TEMP::FLOAT)                          AS FCST_AVG_MAX_TEMP,
        SUM(PCP_AMT::FLOAT)                           AS FCST_TOTAL_PCP,
        MIN(MIN_TEMP::FLOAT)                          AS FCST_MIN_MIN_TEMP,
        MAX(MAX_TEMP::FLOAT)                          AS FCST_MAX_MAX_TEMP,
        MAX(MAX_TEMP::FLOAT) - MIN(MIN_TEMP::FLOAT)   AS FCST_TEMP_RANGE
    FROM WEATHER_MERGED
    GROUP BY 1
)
SELECT
    -- ===== Component keys =====
    WC.BRAND_CD,
    WC.STYLE_CD,
    WC.COLOR_CD_NORM,
    LEFT(WC.COLOR_CD_NORM, 2)         AS TEAM_CD,
    SUBSTRING(WC.COLOR_CD_NORM, 3, 3) AS COLOR_BASE_CD,
    -- ===== Time =====
    WC.WEEK_START                              AS START_DT,
    DATEADD('day', 6, WC.WEEK_START)           AS END_DT,
    -- ===== Product attrs =====
    WC.PRDT_KIND_CD,
    -- ===== Target =====
    WC.WEEKLY_SALE_QTY_CNS,
    WC.WEEKLY_SALE_AMT_CNS,
    CASE WHEN WC.WEEKLY_TAG_AMT_NML_CNS > 0
         THEN 1 - WC.WEEKLY_SALE_AMT_NML_CNS / NULLIF(WC.WEEKLY_TAG_AMT_NML_CNS, 0)
         ELSE NULL
    END AS WEEKLY_DISC_RAT,
    -- ===== Channel monitoring =====
    WC.WEEKLY_SALE_QTY_RTL,
    WC.WEEKLY_SALE_QTY_RF,
    WC.WEEKLY_SALE_QTY_DOME,
    WC.WEEKLY_SALE_QTY_NOTAX,
    -- ===== Stock features =====
    SF.BOW_STOCK,
    SF.STOCK_RATIO,
    SF.AC_STOR_QTY_KOR,
    -- ===== Weather (서울 타겟 주) =====
    TWW.FCST_AVG_MIN_TEMP,
    TWW.FCST_AVG_MAX_TEMP,
    TWW.FCST_TOTAL_PCP,
    TWW.FCST_MIN_MIN_TEMP,
    TWW.FCST_MAX_MAX_TEMP,
    TWW.FCST_TEMP_RANGE
FROM weekly_capped WC
LEFT JOIN STOCK_FEATURES SF
    ON  SF.BRAND_CD      = WC.BRAND_CD
    AND SF.STYLE_CD      = WC.STYLE_CD
    AND SF.COLOR_CD_NORM = WC.COLOR_CD_NORM
    AND SF.WEEK_START    = WC.WEEK_START
LEFT JOIN TARGET_WEEK_WEATHER TWW
    ON TWW.WEEK_START = WC.WEEK_START
ORDER BY WC.BRAND_CD, WC.STYLE_CD, WC.COLOR_CD_NORM, START_DT;


-- ============================================================================
-- 2) DEAD diagnostics — 한 번도 CNS 판매 없던 SC (= 메인 fact 에서 제외된 SC)
--    재고 입고만 됐는지 / 채널별 어디서 활동 있었는지 분기 진단용.
-- ============================================================================
CREATE OR REPLACE TABLE ML_DIST.DEAD_SC_SUPPLIES AS
SELECT
    BRAND_CD,
    STYLE_CD,
    COLOR_CD_NORM,
    LEFT(COLOR_CD_NORM, 2)         AS TEAM_CD,
    SUBSTRING(COLOR_CD_NORM, 3, 3) AS COLOR_BASE_CD,
    ANY_VALUE(PRDT_KIND_CD)        AS PRDT_KIND_CD,
    MIN(WEEK_START)                AS FIRST_ROW,
    MAX(WEEK_START)                AS LAST_ROW,
    COUNT(*)                       AS ROW_COUNT,
    MAX(AC_STOR_QTY_KOR)                AS MAX_AC_STOR_QTY_KOR,     -- 입고는 됐는지
    MAX(AC_SALE_QTY_CNS)                 AS MAX_AC_SALE_QTY_CNS,      -- (보통 0)
    SUM(WEEKLY_SALE_QTY_RTL)       AS TOTAL_RTL_QTY,
    SUM(WEEKLY_SALE_QTY_RF)        AS TOTAL_RF_QTY,
    SUM(WEEKLY_SALE_QTY_DOME)      AS TOTAL_DOME_QTY,
    SUM(WEEKLY_SALE_QTY_NOTAX)     AS TOTAL_NOTAX_QTY
FROM ML_DIST._WEEKLY_RAW_SUPPLIES
GROUP BY BRAND_CD, STYLE_CD, COLOR_CD_NORM
HAVING MAX(CASE WHEN WEEKLY_SALE_QTY_CNS != 0 THEN WEEK_START END) IS NULL
ORDER BY MAX_AC_STOR_QTY_KOR DESC, COLOR_CD_NORM;


-- ============================================================================
-- 코멘트
-- ============================================================================
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES SET COMMENT =
    'SC-Total 주간 판매 fact (모자 Phase A-5, PRCS.DB_SCS_W 기반, 01+03b 통합). grain=BRAND×STYLE×COLOR_CD_NORM×WEEK. SC 별 first/last non-zero 주 사이로 cut (raw sentinel 미래 row 제거). dead SC 23개는 DEAD_SC_SUPPLIES 로 분리. 타겟=NML_QTY_CNS+RET_QTY_CNS. RTL/RF/DOME/NOTAX 4채널 propagate. 서울 타겟 주 날씨 선적재.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN START_DT COMMENT '주 시작일 (월요일). DB_SCS_W.START_DT 그대로.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN END_DT   COMMENT '주 종료일 (일요일) = START_DT + 6 days.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_SALE_QTY_CNS COMMENT '주간 위탁(CNS) net = SUM(SALE_NML_QTY_CNS + SALE_RET_QTY_CNS).';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_SALE_AMT_CNS COMMENT '주간 위탁 판매금액 (CNS NML+RET).';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_DISC_RAT COMMENT '주간 할인율 = 1 - NML_SALE_AMT_CNS / NML_TAG_AMT_CNS. 무판매 주는 NULL.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_SALE_QTY_RTL   COMMENT 'CNS 분해 — 소매(RTL) net. 평가 진단용.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_SALE_QTY_RF    COMMENT 'CNS 분해 — RF net.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_SALE_QTY_DOME  COMMENT 'CNS 분해 — 도매(DOME) net.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN WEEKLY_SALE_QTY_NOTAX COMMENT 'CNS 분해 — 면세(NOTAX) net.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN BOW_STOCK   COMMENT '주초 가용재고 = 전주 누적입고 - 전주 누적판매 (cut 전 raw 위에서 LAG).';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN STOCK_RATIO COMMENT '재고 비율 = BOW_STOCK / 전주 누적입고.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN AC_STOR_QTY_KOR  COMMENT '해당 주까지 누적입고 수량 = SUM(PRCS.DB_SCS_W.AC_STOR_QTY_KOR). raw alias 그대로.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN FCST_AVG_MIN_TEMP COMMENT '타겟 주(START_DT+7~+13) 서울 평균 최저기온.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN FCST_AVG_MAX_TEMP COMMENT '타겟 주 서울 평균 최고기온.';
ALTER TABLE ML_DIST.FACT_SALES_WEEKLY_SC_TOTAL_SUPPLIES ALTER COLUMN FCST_TOTAL_PCP COMMENT '타겟 주 서울 총 강수량.';

ALTER TABLE ML_DIST.DEAD_SC_SUPPLIES SET COMMENT =
    '메인 fact 에서 제외된 SC (한 번도 CNS 판매 없음). 재고 사전 입고/매핑 오류/dead inventory 등 후속 진단용. SELECT * FROM ML_DIST.DEAD_SC_SUPPLIES ORDER BY MAX_AC_STOR_QTY_KOR DESC; 로 한 번 점검 권장.';
ALTER TABLE ML_DIST.DEAD_SC_SUPPLIES ALTER COLUMN MAX_AC_STOR_QTY_KOR COMMENT '누적입고 최댓값. >0 이면 사전 입고 후 미출시 가능성, =0 이면 raw 매핑 오류 가능성.';
ALTER TABLE ML_DIST.DEAD_SC_SUPPLIES ALTER COLUMN TOTAL_RTL_QTY   COMMENT '소매 채널 총 판매 (CNS=0 인데 RTL>0 이면 채널 코드 불일치 의심).';
ALTER TABLE ML_DIST.DEAD_SC_SUPPLIES ALTER COLUMN TOTAL_RF_QTY    COMMENT 'RF 채널 총 판매.';
ALTER TABLE ML_DIST.DEAD_SC_SUPPLIES ALTER COLUMN TOTAL_DOME_QTY  COMMENT '도매 채널 총 판매.';
ALTER TABLE ML_DIST.DEAD_SC_SUPPLIES ALTER COLUMN TOTAL_NOTAX_QTY COMMENT '면세 채널 총 판매.';
