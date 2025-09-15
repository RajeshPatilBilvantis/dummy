-- THE ULTIMATE "TIME MACHINE" QUERY - POINT-IN-TIME FEATURES AND TARGET
-- This query builds a dataset that is logically sound for prediction by using features that were known
-- at the time of the client's FIRST purchase to predict their SECOND purchase.

WITH
-- Step 1: Rank all policies for each client chronologically. This is our single source of truth.
-- We must include all raw data points needed for our calculations here.
RankedPolicies AS (
    SELECT
        axa_party_id,
        policy_no,
        wti_lob_txt AS product_category,
        register_date,
        acct_val_amt AS policy_aum, -- AUM of each individual policy
        isrd_brth_date AS client_birth_date,
        ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC, policy_no ASC) as policy_rank
    FROM
        wealth_management_client_metrics
    WHERE
        axa_party_id IS NOT NULL AND policy_no IS NOT NULL AND register_date IS NOT NULL
),

-- Step 2: Create the complete feature set based ONLY on the state of the first policy.
FeaturesAtFirstPurchase AS (
    SELECT
        axa_party_id,
        -- Calculate age AT THE TIME of the first purchase.
        DATEDIFF(register_date, client_birth_date) / 365.25 AS age_at_first_purchase,
        -- The AUM feature is ONLY the AUM from that first policy.
        policy_aum AS aum_at_first_purchase,
        -- The portfolio features describe the first and ONLY product they have.
        product_category AS initial_product_purchased,
        CASE WHEN product_category = 'Life Insurance' THEN 1 ELSE 0 END AS has_life_insurance,
        CASE WHEN product_category = 'Annuities' THEN 1 ELSE 0 END AS has_annuities,
        CASE WHEN product_category = 'Investment Products' THEN 1 ELSE 0 END AS has_investment_products,
        CASE WHEN product_category = 'Equitable Network' THEN 1 ELSE 0 END AS has_equitable_network,
        CASE WHEN product_category = 'Disability Income' THEN 1 ELSE 0 END AS has_disability_income,
        CASE WHEN product_category = 'Major Medical Insurance' THEN 1 ELSE 0 END AS has_major_medical_insurance,
        CASE WHEN product_category = 'Specialty Insurance' THEN 1 ELSE 0 END AS has_specialty_insurance
    FROM
        RankedPolicies
    WHERE
        policy_rank = 1 -- This filter is the key to the entire "Time Machine".
),

-- Step 3: Identify the target variable (the second policy), same as before.
NextProductTarget AS (
    SELECT
        axa_party_id,
        product_category AS next_product_purchased
    FROM
        RankedPolicies
    WHERE
        policy_rank = 2
),

-- (Agent and Activity CTEs remain the same, as we accept current/recent data as a proxy)
AgentLatestSnapshot AS (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY agent_code ORDER BY yearmo DESC) as rn FROM wealth_management_rpt_agents
),
ClientActivityMetrics AS (
    SELECT account_axa_party_id_c AS client_id, COUNT(task_id) AS total_activities_last_12m
    FROM wealth_management_activities WHERE activity_date >= ADD_MONTHS(CURRENT_DATE, -12) GROUP BY account_axa_party_id_c
)

-- Final SELECT: Assemble the true point-in-time dataset
SELECT
    -- ===== Primary Key =====
    ret.axa_party_id,

    -- ===== POINT-IN-TIME Features from First Purchase =====
    f.age_at_first_purchase,
    f.aum_at_first_purchase,
    f.initial_product_purchased,
    f.has_life_insurance,
    f.has_annuities,
    f.has_investment_products,
    f.has_equitable_network,
    f.has_disability_income,
    f.has_major_medical_insurance,
    f.has_specialty_insurance,
    
    -- ===== Static / Proxy Features (Acceptable as current state) =====
    -- Note: 'active_policy_count' from retention table is no longer a valid feature, as it should always be 1 at the time of prediction.
    ret.city AS client_city,
    ret.state AS client_state,
    ret.client_segment, -- This is a snapshot, but can be a useful proxy
    ret.client_type,
    COALESCE(act.total_activities_last_12m, 0) AS total_activities_last_12m,
    ret.agent_code,
    agent.rpt_lgth_of_svc AS agent_length_of_service,
    agent.rank_desc AS agent_rank,
    agent.ppg_membership AS agent_is_ppg_member,

    -- ===== SINGLE TARGET VARIABLE (What we want to predict) =====
    tgt.next_product_purchased
FROM
    -- We can start with our features table as the base, as it represents the clients we want to model
    FeaturesAtFirstPurchase AS f
-- Join the target variable
LEFT JOIN NextProductTarget AS tgt ON f.axa_party_id = tgt.axa_party_id
-- Join the retention table to get some useful static client info
LEFT JOIN (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY business_month DESC) as rn
    FROM wealth_management_pcpg_retention
) AS ret ON f.axa_party_id = ret.axa_party_id AND ret.rn = 1
-- Join other summary/static info
LEFT JOIN AgentLatestSnapshot AS agent ON ret.agent_code = agent.agent_code AND agent.rn = 1
LEFT JOIN ClientActivityMetrics AS act ON f.axa_party_id = act.client_id
WHERE
    f.axa_party_id IS NOT NULL;
