-- FINAL ROBUST VERSION - DEDUPLICATION FIRST
-- This query prevents the "fan-out" effect by creating a clean list of distinct policies before any ranking or aggregation.

WITH
-- Step 1: Create a clean, deduplicated list of policies for each client. THIS IS THE KEY FIX.
DistinctPolicies AS (
    SELECT DISTINCT
        axa_party_id,
        policy_no,
        wti_lob_txt,
        register_date
    FROM
        wealth_management_client_metrics
    WHERE
        axa_party_id IS NOT NULL AND policy_no IS NOT NULL AND register_date IS NOT NULL
),

-- Step 2: Create the product portfolio from the CLEAN policy list.
ClientProductPortfolio AS (
    SELECT
        axa_party_id,
        -- Now we can use a simple COUNT, as duplicates are already removed.
        COUNT(CASE WHEN wti_lob_txt = 'Life Insurance' THEN 1 END) AS count_life_insurance,
        COUNT(CASE WHEN wti_lob_txt = 'Annuities' THEN 1 END) AS count_annuities,
        COUNT(CASE WHEN wti_lob_txt = 'Investment Products' THEN 1 END) AS count_investment_products,
        COUNT(CASE WHEN wti_lob_txt = 'Equitable Network' THEN 1 END) AS count_equitable_network,
        COUNT(CASE WHEN wti_lob_txt = 'Disability Income' THEN 1 END) AS count_disability_income,
        COUNT(CASE WHEN wti_lob_txt = 'Major Medical Insurance' THEN 1 END) AS count_major_medical_insurance,
        COUNT(CASE WHEN wti_lob_txt = 'Specialty Insurance' THEN 1 END) AS count_specialty_insurance
    FROM
        DistinctPolicies -- Using the clean source
    GROUP BY
        axa_party_id
),

-- Step 3: Identify the first product using the CLEAN policy list.
InitialProduct AS (
    WITH RankedPolicies AS (
        SELECT axa_party_id, wti_lob_txt AS product_category,
               ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC, policy_no ASC) as policy_rank
        FROM DistinctPolicies -- Using the clean source
    )
    SELECT axa_party_id, product_category AS initial_product_purchased
    FROM RankedPolicies
    WHERE policy_rank = 1
),

-- Step 4: Identify the second product (our target) using the CLEAN policy list.
NextProductTarget AS (
    WITH RankedPolicies AS (
        SELECT axa_party_id, wti_lob_txt AS product_category,
               ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC, policy_no ASC) as policy_rank
        FROM DistinctPolicies -- Using the clean source
    )
    SELECT axa_party_id, product_category AS next_product_purchased
    FROM RankedPolicies
    WHERE policy_rank = 2
),

-- (The other feature CTEs that do not depend on client_metrics remain the same)
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
ClientPolicySummary AS (
    -- This CTE can still use the raw table, as GROUP BY handles duplication for these simple aggregations.
    SELECT axa_party_id, AVG(DATEDIFF(CURRENT_DATE, isrd_brth_date) / 365.25) as avg_client_age, MAX(wc_total_assets) as estimated_total_wealth
    FROM wealth_management_client_metrics
    GROUP BY axa_party_id
)

-- Final SELECT: Assembling the clean and correct data
SELECT
    ret.axa_party_id,
    cps.avg_client_age, ret.client_age_band, ret.city AS client_city, ret.state AS client_state, ret.client_segment, ret.aum_sum AS total_aum_with_us, ret.aum_band, cps.estimated_total_wealth, ret.client_tenure, ret.active_policy_count, ret.client_type,
    ip.initial_product_purchased,
    COALESCE(prod.count_life_insurance, 0) AS count_life_insurance,
    COALESCE(prod.count_annuities, 0) AS count_annuities,
    COALESCE(prod.count_investment_products, 0) AS count_investment_products,
    COALESCE(prod.count_equitable_network, 0) AS count_equitable_network,
    COALESCE(prod.count_disability_income, 0) AS count_disability_income,
    COALESCE(prod.count_major_medical_insurance, 0) AS count_major_medical_insurance,
    COALESCE(prod.count_specialty_insurance, 0) AS count_specialty_insurance,
    COALESCE(act.total_activities_last_12m, 0) AS total_activities_last_12m,
    ret.agent_code,
    agent.rpt_lgth_of_svc AS agent_length_of_service,
    agent.rank_desc AS agent_rank,
    agent.ppg_membership AS agent_is_ppg_member,
    tgt.next_product_purchased
FROM
    wealth_management_pcpg_retention AS ret
LEFT JOIN ClientPolicySummary AS cps ON ret.axa_party_id = cps.axa_party_id
LEFT JOIN AgentLatestSnapshot AS agent ON ret.agent_code = agent.agent_code AND agent.rn = 1
LEFT JOIN ClientActivityMetrics AS act ON ret.axa_party_id = act.client_id
LEFT JOIN InitialProduct AS ip ON ret.axa_party_id = ip.axa_party_id
LEFT JOIN NextProductTarget AS tgt ON ret.axa_party_id = tgt.axa_party_id
LEFT JOIN ClientProductPortfolio AS prod ON ret.axa_party_id = prod.axa_party_id
WHERE
    ret.axa_party_id IS NOT NULL;
