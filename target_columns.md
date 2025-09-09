-- FINAL ROBUST VERSION WITH CONSISTENT FEATURES AND TARGET
-- This query creates a unified view where product features directly match the target categories.

WITH
-- NEW CTE: Create a complete product portfolio for each client based on the target categories.
ClientProductPortfolio AS (
    SELECT
        axa_party_id,
        -- We create a count for each product category that can appear in our target variable.
        -- These names now directly correspond to the values in 'wti_lob_txt'.
        COUNT(CASE WHEN wti_lob_txt = 'Life Insurance' THEN policy_no END) AS count_life_insurance,
        COUNT(CASE WHEN wti_lob_txt = 'Annuities' THEN policy_no END) AS count_annuities,
        COUNT(CASE WHEN wti_lob_txt = 'Investment Products' THEN policy_no END) AS count_investment_products,
        COUNT(CASE WHEN wti_lob_txt = 'Equitable Network' THEN policy_no END) AS count_equitable_network,
        COUNT(CASE WHEN wti_lob_txt = 'Disability Income' THEN policy_no END) AS count_disability_income,
        COUNT(CASE WHEN wti_lob_txt = 'Major Medical Insurance' THEN policy_no END) AS count_major_medical_insurance,
        COUNT(CASE WHEN wti_lob_txt = 'Specialty Insurance' THEN policy_no END) AS count_specialty_insurance
        -- Add more CASE WHEN statements here if there are other product categories in 'wti_lob_txt'.
    FROM
        wealth_management_client_metrics
    GROUP BY
        axa_party_id
),

-- CTE to identify the first product purchased.
InitialProduct AS (
    WITH RankedPolicies AS (
        SELECT axa_party_id, wti_lob_txt AS product_category, ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC) as policy_rank
        FROM wealth_management_client_metrics
        WHERE axa_party_id IS NOT NULL AND register_date IS NOT NULL
    )
    SELECT axa_party_id, product_category AS initial_product_purchased
    FROM RankedPolicies
    WHERE policy_rank = 1
),

-- CTE to identify the second product purchased (our target).
NextProductTarget AS (
    WITH RankedPolicies AS (
        SELECT axa_party_id, wti_lob_txt AS product_category, ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC) as policy_rank
        FROM wealth_management_client_metrics
        WHERE axa_party_id IS NOT NULL AND register_date IS NOT NULL
    )
    SELECT axa_party_id, product_category AS next_product_purchased
    FROM RankedPolicies
    WHERE policy_rank = 2
),

-- (The other feature CTEs remain the same)
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
    SELECT axa_party_id, AVG(DATEDIFF(CURRENT_DATE, isrd_brth_date) / 365.25) as avg_client_age, MAX(wc_total_assets) as estimated_total_wealth
    FROM wealth_management_client_metrics
    GROUP BY axa_party_id
)

-- Final SELECT: Using the new, consistent product features
SELECT
    -- ===== Primary Key =====
    ret.axa_party_id,

    -- ===== Client & Financial Features =====
    cps.avg_client_age, ret.client_age_band, ret.city AS client_city, ret.state AS client_state, ret.client_segment, ret.aum_sum AS total_aum_with_us, ret.aum_band, cps.estimated_total_wealth, ret.client_tenure, ret.active_policy_count, ret.client_type,
    
    -- ===== Initial Product Feature =====
    ip.initial_product_purchased,

    -- ===== NEW & CONSISTENT Product Portfolio Features =====
    COALESCE(prod.count_life_insurance, 0) AS count_life_insurance,
    COALESCE(prod.count_annuities, 0) AS count_annuities,
    COALESCE(prod.count_investment_products, 0) AS count_investment_products,
    COALESCE(prod.count_equitable_network, 0) AS count_equitable_network,
    COALESCE(prod.count_disability_income, 0) AS count_disability_income,
    COALESCE(prod.count_major_medical_insurance, 0) AS count_major_medical_insurance,
    COALESCE(prod.count_specialty_insurance, 0) AS count_specialty_insurance,
    
    -- ===== Engagement and Agent Features =====
    COALESCE(act.total_activities_last_12m, 0) AS total_activities_last_12m,
    ret.agent_code,
    agent.rpt_lgth_of_svc AS agent_length_of_service,
    agent.rank_desc AS agent_rank,
    agent.ppg_membership AS agent_is_ppg_member,

    -- ===== SINGLE TARGET VARIABLE (What we want to predict) =====
    tgt.next_product_purchased

FROM
    wealth_management_pcpg_retention AS ret
LEFT JOIN ClientPolicySummary AS cps ON ret.axa_party_id = cps.axa_party_id
LEFT JOIN AgentLatestSnapshot AS agent ON ret.agent_code = agent.agent_code AND agent.rn = 1
LEFT JOIN ClientActivityMetrics AS act ON ret.axa_party_id = act.client_id
LEFT JOIN InitialProduct AS ip ON ret.axa_party_id = ip.axa_party_id
LEFT JOIN NextProductTarget AS tgt ON ret.axa_party_id = tgt.axa_party_id
-- We now join our new, consistent product portfolio
LEFT JOIN ClientProductPortfolio AS prod ON ret.axa_party_id = prod.axa_party_id
WHERE
    ret.axa_party_id IS NOT NULL;
