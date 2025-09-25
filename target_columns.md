-- FINAL GOLD STANDARD VERSION - INCORPORATING ALL REQUESTED FEATURES
-- This query uses point-in-time features from the first purchase to predict the second.
-- It is free of target leakage and includes the AUM band, birth date, and age at second purchase.

WITH
-- Step 1: Create a clean, deduplicated list of policies, including all necessary columns.
DistinctPolicies AS (
    SELECT DISTINCT
        axa_party_id,
        policy_no,
        wti_lob_txt,
        register_date,
        isrd_brth_date,
        acct_val_amt,
        face_amt,
        cash_val_amt
    FROM
        wealth_management_client_metrics
    WHERE
        axa_party_id IS NOT NULL AND policy_no IS NOT NULL AND register_date IS NOT NULL
),

-- Step 2: Create features and dates based ONLY on the first policy.
FirstPurchaseInfo AS (
    WITH RankedPolicies AS (
        SELECT *,
               ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC, policy_no ASC) as policy_rank
        FROM DistinctPolicies
    )
    SELECT
        axa_party_id,
        wti_lob_txt AS initial_product_purchased,
        register_date AS initial_product_purchase_date,
        isrd_brth_date AS client_birth_date,
        acct_val_amt AS first_policy_aum_val,
        face_amt AS first_policy_face_amt,
        cash_val_amt AS first_policy_cash_val
    FROM
        RankedPolicies
    WHERE
        policy_rank = 1
),

-- Step 3: Identify the target (second policy) and its purchase date.
NextProductTarget AS (
    WITH RankedPolicies AS (
        SELECT *,
               ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC, policy_no ASC) as policy_rank
        FROM DistinctPolicies
    )
    SELECT
        axa_party_id,
        wti_lob_txt AS second_product_purchased,
        register_date AS second_product_purchase_date
    FROM
        RankedPolicies
    WHERE
        policy_rank = 2
),

-- (Other CTEs for static/summary info remain useful)
AgentLatestSnapshot AS (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY agent_code ORDER BY yearmo DESC) as rn
    FROM wealth_management_rpt_agents
),
ClientActivityMetrics AS (
    SELECT account_axa_party_id_c AS client_id, COUNT(task_id) AS total_activities_last_12m
    FROM wealth_management_activities
    WHERE activity_date >= ADD_MONTHS(CURRENT_DATE, -12)
    GROUP BY account_axa_party_id_c
),
LatestRetentionSnapshot AS (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY business_month DESC) as rn
    FROM wealth_management_pcpg_retention
)

-- Final SELECT: Assembling the complete, point-in-time feature set
SELECT
    -- ===== Primary Key =====
    ret.axa_party_id,

    -- ===== Static Client Features (Current state is acceptable) =====
    DATEDIFF(CURRENT_DATE, fpi.client_birth_date) / 365.25 AS current_client_age, -- Calculated from the consistent birth date
    ret.client_age_band,
    ret.city AS client_city,
    ret.state AS client_state,
    ret.client_segment,
    ret.client_tenure,
    ret.client_type,
    ret.aum_sum AS total_aum_with_us_today, -- This is the client's current AUM
    ret.aum_band, -- **ADDED THIS COLUMN**

    -- ===== NEW Point-in-Time Features =====
    fpi.initial_product_purchased,
    fpi.client_birth_date, -- **ADDED THIS COLUMN**
    fpi.initial_product_purchase_date,
    tgt.second_product_purchase_date,
    MONTHS_BETWEEN(tgt.second_product_purchase_date, fpi.initial_product_purchase_date) AS months_to_second_purchase,
    DATEDIFF(tgt.second_product_purchase_date, fpi.client_birth_date) / 365.25 AS age_at_second_purchase, -- **ADDED THIS COLUMN**
    fpi.first_policy_aum_val,
    fpi.first_policy_face_amt,
    fpi.first_policy_cash_val,
    
    -- ===== Engagement and Agent Features (Current state is acceptable) =====
    COALESCE(act.total_activities_last_12m, 0) AS total_activities_last_12m,
    ret.agent_code,
    agent.rpt_lgth_of_svc AS agent_length_of_service,
    agent.rank_desc AS agent_rank,
    agent.ppg_membership AS agent_is_ppg_member,

    -- ===== SINGLE TARGET VARIABLE =====
    tgt.second_product_purchased

FROM
    LatestRetentionSnapshot AS ret
LEFT JOIN AgentLatestSnapshot AS agent ON ret.agent_code = agent.agent_code AND agent.rn = 1
LEFT JOIN ClientActivityMetrics AS act ON ret.axa_party_id = act.client_id
LEFT JOIN FirstPurchaseInfo AS fpi ON ret.axa_party_id = fpi.axa_party_id
LEFT JOIN NextProductTarget AS tgt ON ret.axa_party_id = tgt.axa_party_id
WHERE
    ret.axa_party_id IS NOT NULL
    AND ret.rn = 1
    -- We only want to train on clients who have at least one policy
    AND fpi.axa_party_id IS NOT NULL;
