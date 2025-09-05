-- CORRECTED AND FINAL VERSION WITH A SINGLE TARGET VARIABLE
-- This query creates a unified view to predict the immediate next product a client will buy.

WITH
-- NEW CTE: Identify the second product ever purchased by a client.
NextProductTarget AS (
    -- First, we need to rank all policies for each client chronologically.
    WITH RankedPolicies AS (
        SELECT
            axa_party_id,
            wti_lob_txt AS product_category, -- This is the product name (e.g., 'Life', 'Annuities')
            ROW_NUMBER() OVER(PARTITION BY axa_party_id ORDER BY register_date ASC) as policy_rank
        FROM
            wealth_management_client_metrics
        WHERE
            axa_party_id IS NOT NULL AND register_date IS NOT NULL
    )
    -- Now, select only the policy that has a rank of 2 for each client.
    SELECT
        axa_party_id,
        product_category AS next_product_purchased -- This is our single target column!
    FROM
        RankedPolicies
    WHERE
        policy_rank = 2 -- This is the key step: isolates the immediate next purchase.
),

-- (The rest of the CTEs for creating features remain exactly the same)
AgentLatestSnapshot AS (
    SELECT *, ROW_NUMBER() OVER(PARTITION BY agent_code ORDER BY yearmo DESC) as rn
    FROM wealth_management_rpt_agents
),
ClientActivityMetrics AS (
    SELECT account_axa_party_id_c AS client_id, COUNT(task_id) AS total_activities_all_time, COUNT(CASE WHEN activity_date >= ADD_MONTHS(CURRENT_DATE, -12) THEN task_id END) AS total_activities_last_12m, COUNT(CASE WHEN call_outcome_c IN ('Connected', 'Appointment Set', 'Referral Provided') THEN task_id END) AS positive_outcomes
    FROM wealth_management_activities
    GROUP BY account_axa_party_id_c
),
ClientOpportunityMetrics AS (
    SELECT axa_party_id_c AS client_id, COUNT(opportunity_id) AS total_opportunities, COUNT(CASE WHEN stage_name = 'Closed Won' THEN opportunity_id END) AS won_opportunities, AVG(estimated_pcs_gdc_c) AS avg_estimated_gdc
    FROM wealth_management_opportunities
    GROUP BY axa_party_id_c
),
ClientPolicySummary AS (
    SELECT axa_party_id, AVG(DATEDIFF(CURRENT_DATE, isrd_brth_date) / 365.25) as avg_client_age, MAX(wc_total_assets) as estimated_total_wealth, MAX(wc_assetmix_stocks) as estimated_stock_wealth, MAX(wc_assetmix_bonds) as estimated_bond_wealth
    FROM wealth_management_client_metrics
    GROUP BY axa_party_id
)

-- Final SELECT: Join all the features and the new, single target variable
SELECT
    -- ===== Primary Key =====
    ret.axa_party_id,

    -- ===== Client Features (The predictors) =====
    cps.avg_client_age, ret.client_age_band, ret.city AS client_city, ret.state AS client_state, ret.client_segment, ret.aum_sum AS total_aum_with_us, ret.aum_band, cps.estimated_total_wealth, cps.estimated_stock_wealth, cps.estimated_bond_wealth, ret.client_tenure, ret.active_policy_count, ret.client_type, ret.ir AS has_individual_retirement, ret.gr AS has_group_retirement, ret.bd AS has_broker_dealer, ret.life AS has_life_insurance, ret.network AS has_network_product, ret.eb AS has_employee_benefit, COALESCE(act.total_activities_last_12m, 0) AS total_activities_last_12m, COALESCE(act.positive_outcomes, 0) AS positive_outcomes, COALESCE(opp.won_opportunities, 0) AS won_opportunities, ret.agent_code, agent.rpt_lgth_of_svc AS agent_length_of_service, agent.rank_desc AS agent_rank, agent.ppg_membership AS agent_is_ppg_member,

    -- ===== SINGLE TARGET VARIABLE (What we want to predict) =====
    tgt.next_product_purchased

FROM
    wealth_management_pcpg_retention AS ret
LEFT JOIN ClientPolicySummary AS cps ON ret.axa_party_id = cps.axa_party_id
LEFT JOIN AgentLatestSnapshot AS agent ON ret.agent_code = agent.agent_code AND agent.rn = 1
LEFT JOIN ClientActivityMetrics AS act ON ret.axa_party_id = act.client_id
LEFT JOIN ClientOpportunityMetrics AS opp ON ret.axa_party_id = opp.client_id
LEFT JOIN NextProductTarget AS tgt ON ret.axa_party_id = tgt.axa_party_id -- Join our new target CTE
WHERE
    ret.axa_party_id IS NOT NULL;
