-- CORRECTED VERSION
-- This query creates a unified, 360-degree view of each client for ML modeling.
-- Each row represents a single client, identified by axa_party_id.

WITH
-- CTE 1: Get the most recent snapshot for each reporting agent
AgentLatestSnapshot AS (
    SELECT
        *,
        ROW_NUMBER() OVER(PARTITION BY agent_code ORDER BY yearmo DESC) as rn
    FROM wealth_management_rpt_agents
),

-- CTE 2: Aggregate client activities to get engagement metrics
ClientActivityMetrics AS (
    SELECT
        account_axa_party_id_o AS client_id,
        COUNT(taskid) AS total_activities_all_time,
        COUNT(CASE WHEN activity_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH) THEN taskid END) AS total_activities_last_12m,
        COUNT(CASE WHEN call_outcome_c IN ('Connected', 'Appointment Set', 'Referral Provided') THEN taskid END) AS positive_outcomes
    FROM wealth_management_activities
    GROUP BY account_axa_party_id_o
),

-- CTE 3: Aggregate client opportunities to get sales pipeline metrics
ClientOpportunityMetrics AS (
    SELECT
        axa_party_id_c AS client_id,
        COUNT(opportunity_id) AS total_opportunities,
        COUNT(CASE WHEN stage_name = 'Closed Won' THEN opportunity_id END) AS won_opportunities,
        COUNT(CASE WHEN stage_name = 'Closed Lost' THEN opportunity_id END) AS lost_opportunities,
        AVG(estimated_pcs_gdc_c) AS avg_estimated_gdc
    FROM wealth_management_opportunities
    GROUP BY axa_party_id_c
),

-- CTE 4: Aggregate distinct client policy metrics to avoid duplication
ClientPolicySummary AS (
    SELECT
        axa_party_id,
        AVG(DATEDIFF(CURRENT_DATE, isrd_brth_date) / 365.25) as avg_client_age,
        -- **FIXED HERE: Changed 'we_' to 'wc_'**
        MAX(wc_total_assets) as estimated_total_wealth,
        MAX(wc_assetmix_stocks) as estimated_stock_wealth,
        MAX(wc_assetmix_bonds) as estimated_bond_wealth
    FROM wealth_management_client_metrics
    GROUP BY axa_party_id
)


-- Final SELECT: Join all the CTEs and base tables to build the unified view
SELECT
    -- ===== Primary Key =====
    ret.axa_party_id,

    -- ===== Client Profile Features =====
    cps.avg_client_age,
    ret.client_age_band,
    ret.city AS client_city,
    ret.state AS client_state,
    ret.client_segment,

    -- ===== Financial & Relationship Features =====
    ret.aum_sum AS total_aum_with_us,
    ret.aum_band,
    -- **FIXED HERE: These columns now correctly reference the fixed CTE**
    cps.estimated_total_wealth,
    cps.estimated_stock_wealth,
    cps.estimated_bond_wealth,
    ret.client_tenure,
    ret.active_policy_count,
    ret.client_type,

    -- ===== Existing Product Portfolio (already flagged in retention table) =====
    ret.ir AS has_individual_retirement,
    ret.gr AS has_group_retirement,
    ret.bd AS has_broker_dealer,
    ret.life AS has_life_insurance,
    ret.network AS has_network_product,
    ret.eb AS has_employee_benefit,
    --ret.others AS has_other_product,

    -- ===== Behavioral & Engagement Features (from CTEs) =====
    COALESCE(act.total_activities_all_time, 0) AS total_activities_all_time,
    COALESCE(act.total_activities_last_12m, 0) AS total_activities_last_12m,
    COALESCE(act.positive_outcomes, 0) AS positive_outcomes,
    COALESCE(opp.total_opportunities, 0) AS total_opportunities,
    COALESCE(opp.won_opportunities, 0) AS won_opportunities,

    -- ===== Servicing Agent Features (from CTE) =====
    ret.agent_code,
    agent.full_name AS agent_name,
    agent.rpt_lgth_of_svc AS agent_length_of_service,
    agent.rank_desc AS agent_rank,
    agent.class_desc AS agent_class,
    agent.ppg_membership AS agent_is_ppg_member,
    agent.sterling_membership AS agent_is_sterling_member,
    agent.division_name AS agent_division

FROM
    wealth_management_pcpg_retention AS ret

LEFT JOIN ClientPolicySummary AS cps
    ON ret.axa_party_id = cps.axa_party_id

LEFT JOIN AgentLatestSnapshot AS agent
    ON ret.agent_code = agent.agent_code AND agent.rn = 1 -- Ensure we only get the latest record

LEFT JOIN ClientActivityMetrics AS act
    ON ret.axa_party_id = act.client_id

LEFT JOIN ClientOpportunityMetrics AS opp
    ON ret.axa_party_id = opp.client_id

WHERE
    ret.axa_party_id IS NOT NULL; -- Ensure we only model for clients with a valid ID
