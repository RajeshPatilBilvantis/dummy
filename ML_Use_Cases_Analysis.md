# ML Use Cases Analysis for Wealth Management Tables

## Executive Summary
Based on the analysis of 8 wealth management tables, this document identifies key machine learning use cases that can leverage the rich data available across policies, agents, clients, transactions, and retention metrics.

---

## Table Overview

1. **wealth_management_policy** - Policy-level data with financial metrics, product details, and hierarchy
2. **wealth_management_agents** - Agent demographics, performance, hierarchy, and contract details
3. **wealth_management_clients** - Client demographics, assets, interests, contract roles, and addresses
4. **wealth_management_transactions** - Transaction-level financial data
5. **wealth_management_rpt_agents** - Agent reporting metrics with performance indicators
6. **remote_advise_sales_reporting** - Sales reporting for remote advice agents
7. **wealth_management_client_metrics** - Aggregated client metrics combining policy, agent, and client data
8. **wealth_management_pcpg_retention** - Client retention and program participation data

---

## ML Use Cases

### 1. **Client Churn Prediction & Retention**
**Objective:** Predict which clients are likely to terminate policies or leave programs

**Key Features:**
- From `wealth_management_pcpg_retention`: `term_pol_count`, `term_rsn_nm`, `term_dt`, `sweep_out_program`, `client_tenure`, `program_tenure`
- From `wealth_management_policy`: `trmn_eff_date`, `trmn_typ_code`, `stat_code`, `cash_val_amt`, `loan_bal_amt`
- From `wealth_management_clients`: `termination_dt`, `term_rsn_cd`, `contract_role_end_dt`, `years_held`
- From `wealth_management_client_metrics`: `policy_status`, `client_seg`, `aum_band`, `client_type`

**Target Variable:** `term_pol_count > 0` or `sweep_out_program = 'Yes'`

**Business Value:** Proactive intervention to retain high-value clients, reduce policy terminations

---

### 2. **Agent Performance Prediction**
**Objective:** Predict future agent performance and identify top performers

**Key Features:**
- From `wealth_management_agents`: `rank_code`, `status_code`, `class_code`, `appointment_date`, `honor_club_code`, `tot_nic_count`, `pcr`, `life_pc_total_amt`
- From `wealth_management_rpt_agents`: `ytd_pcs_rolling12`, `career_pcs`, `life_persistency_index`, `rpt_rank_code`, `rpt_class_code`, `rpt_lgth_of_svc`
- From `remote_advise_sales_reporting`: `total_pc`, `agent_pc`, `prem_amt`, `aum`
- From `wealth_management_client_metrics`: `agent_segment`, `designation`, `final_designation`

**Target Variable:** `ytd_pcs_rolling12` (next period) or `career_pcs` growth

**Business Value:** Identify high-potential agents, optimize training and resource allocation

---

### 3. **Client Lifetime Value (CLV) Prediction**
**Objective:** Estimate the total value a client will bring over their lifetime

**Key Features:**
- From `wealth_management_client_metrics`: `wc_total_assets`, `aum_band`, `monthly_preminum_amount`, `cash_val_amt`, `acct_val_amt`
- From `wealth_management_policy`: `ann_prem_amt`, `face_amt`, `cash_val_amt`, `nxt_prem_amt`
- From `wealth_management_transactions`: `dr_cr_amount`, `pc_amt` (aggregated)
- From `wealth_management_pcpg_retention`: `aum_sum`, `active_policy_count`, `sow` (share of wallet)
- From `wealth_management_clients`: `years_held`, `wc_assetmix_*` (asset mix breakdown)

**Target Variable:** Predicted total revenue over next 3-5 years

**Business Value:** Prioritize high-value clients, optimize marketing spend, improve ROI

---

### 4. **Product Recommendation System**
**Objective:** Recommend the best product for each client based on their profile

**Key Features:**
- From `wealth_management_clients`: `wc_total_assets`, `wc_assetmix_*`, `demo_hh_income_band`, `demo_hh_marital_st`, `psn_age`, `demo_hh_occupation`
- From `wealth_management_policy`: `prod_code`, `plan_code`, `mkt_prod_cat_desc`, `mkt_prod_typ_desc`, `wti_lob_txt`
- From `wealth_management_client_metrics`: `prod_lob`, `sub_product_level_1`, `sub_product_level_2`, `product_combinations`
- From `wealth_management_pcpg_retention`: `ir`, `gr`, `bd`, `life`, `network`, `eb`, `other` (product flags)

**Target Variable:** Product purchase probability for each product type

**Business Value:** Increase cross-selling, improve product-market fit, enhance client satisfaction

---

### 5. **Policy Lapse Risk Prediction**
**Objective:** Predict which policies are at risk of lapsing

**Key Features:**
- From `wealth_management_policy`: `nxt_prem_date`, `nxt_prem_amt`, `nxt_prem_mode_code`, `stat_code`, `cash_val_amt`, `loan_bal_amt`
- From `wealth_management_clients`: `payment_id`, `cl_bank_account_id`, `years_held`
- From `wealth_management_transactions`: `acctg_date`, `dr_cr_amount` (premium payment patterns)
- From `wealth_management_client_metrics`: `monthly_preminum_amount`, `policy_status`

**Target Variable:** Policy lapse within next 90 days

**Business Value:** Proactive outreach to prevent lapses, reduce revenue loss

---

### 6. **Agent-Client Matching Optimization**
**Objective:** Match clients with the most suitable agents

**Key Features:**
- From `wealth_management_agents`: `class_code`, `rank_code`, `honor_club_code`, `business_city`, `business_state_cod`, `pam_classification`
- From `wealth_management_clients`: `city_name`, `prov_state_tp_cd`, `demo_hh_income_band`, `wc_total_assets`, `demo_ethnic_code`, `demo_language_code`
- From `wealth_management_client_metrics`: `agent_segment`, `client_seg`, `designation`
- Historical performance: Agent success rate by client segment

**Target Variable:** Agent-client match success (policy sold, retention, satisfaction)

**Business Value:** Improve sales conversion, enhance client-agent relationships, optimize territory management

---

### 7. **Fraud Detection & Anomaly Detection**
**Objective:** Identify suspicious transactions, policies, or agent behaviors

**Key Features:**
- From `wealth_management_transactions`: `dr_cr_amount`, `acct_code`, `jrnl_id_no`, `rollup_id`, `rollup_source`
- From `wealth_management_policy`: `face_amt`, `cash_val_amt`, `ann_prem_amt` (unusual patterns)
- From `wealth_management_agents`: `agt_pct`, `pol_pct` (commission anomalies)
- From `remote_advise_sales_reporting`: `agent_pct`, `prem_amt`, `total_pc` (unusual splits)

**Target Variable:** Fraud flag (binary classification) or anomaly score

**Business Value:** Prevent financial losses, ensure regulatory compliance, maintain trust

---

### 8. **Sales Forecasting**
**Objective:** Predict future sales volumes and revenue

**Key Features:**
- From `remote_advise_sales_reporting`: `prem_amt`, `total_pc`, `agent_pc`, `aum` (historical trends)
- From `wealth_management_rpt_agents`: `ytd_pcs_rolling12`, `career_pcs`, `new_hires_headcount`
- From `wealth_management_policy`: `register_date`, `ann_prem_amt` (seasonal patterns)
- From `wealth_management_client_metrics`: `business_year`, `quarter`, `business_month`

**Target Variable:** Next period sales (premium, PC, AUM)

**Business Value:** Improve planning, resource allocation, goal setting

---

### 9. **Client Segmentation & Clustering**
**Objective:** Identify distinct client segments for targeted marketing

**Key Features:**
- From `wealth_management_clients`: `wc_total_assets`, `wc_assetmix_*`, `demo_hh_income_band`, `demo_hh_marital_st`, `demo_hh_education`, `demo_hh_occupation`, `demo_ethnic_code`, `psn_age`
- From `wealth_management_client_metrics`: `client_seg`, `client_seg_1`, `aum_band`, `client_type`
- From `wealth_management_pcpg_retention`: `product_combinations`, `client_segment`, `client_investible_asset`
- From `wealth_management_clients`: Interest fields (`demo_int_*`)

**Target Variable:** Unsupervised clustering (K-means, hierarchical)

**Business Value:** Personalized marketing, product development, service customization

---

### 10. **Agent Attrition Prediction**
**Objective:** Predict which agents are likely to leave

**Key Features:**
- From `wealth_management_agents`: `status_code`, `term_dt`, `appointment_date`, `class_date`, `rpt_status_code`
- From `wealth_management_rpt_agents`: `rpt_status_code`, `term_dt`, `term_dec`, `ytd_pcs_rolling12`, `career_pcs`
- From `wealth_management_agents`: `rank_code`, `honor_club_code`, `tot_nic_count`, `pcr`
- Performance trends: Declining production credits, status changes

**Target Variable:** Agent termination within next 6-12 months

**Business Value:** Reduce agent turnover, improve retention strategies, reduce recruitment costs

---

### 11. **Premium Payment Default Prediction**
**Objective:** Predict clients who may default on premium payments

**Key Features:**
- From `wealth_management_policy`: `nxt_prem_date`, `nxt_prem_amt`, `nxt_prem_mode_code`, `cash_val_amt`, `loan_bal_amt`
- From `wealth_management_clients`: `payment_id`, `cl_bank_account_id`, `years_held`
- From `wealth_management_transactions`: Historical payment patterns, `acctg_date`, `dr_cr_amount`
- From `wealth_management_client_metrics`: `monthly_preminum_amount`, `policy_status`

**Target Variable:** Premium payment default within next 30-60 days

**Business Value:** Proactive collection efforts, reduce bad debt, improve cash flow

---

### 12. **Cross-Sell & Upsell Opportunity Identification**
**Objective:** Identify clients likely to purchase additional products

**Key Features:**
- From `wealth_management_pcpg_retention`: `active_policy_count`, `product_combinations`, `ir`, `gr`, `bd`, `life`, `network`, `eb`, `other`
- From `wealth_management_client_metrics`: `wc_total_assets`, `aum_band`, `client_seg`, `prod_lob`
- From `wealth_management_clients`: `wc_assetmix_*`, `demo_hh_income_band`, `years_held`
- From `wealth_management_policy`: Current product portfolio

**Target Variable:** Probability of purchasing additional product type

**Business Value:** Increase wallet share, improve revenue per client, enhance client relationships

---

### 13. **Program Participation Prediction**
**Objective:** Predict which clients will join retention programs (PCPG)

**Key Features:**
- From `wealth_management_pcpg_retention`: `program`, `program_tenure`, `sweep_in_program`, `active_policy_count`
- From `wealth_management_client_metrics`: `client_seg`, `aum_band`, `client_type`, `policy_status`
- From `wealth_management_clients`: `wc_total_assets`, `years_held`, `demo_hh_income_band`
- From `wealth_management_policy`: `cash_val_amt`, `acct_val_amt`, `stat_code`

**Target Variable:** Program participation probability

**Business Value:** Improve program targeting, increase participation rates, enhance retention

---

### 14. **Agent Commission Optimization**
**Objective:** Optimize commission structures and predict commission amounts

**Key Features:**
- From `wealth_management_agents`: `agt_pct`, `pol_pct`, `agt_int_pct`, `agt_role_code`
- From `remote_advise_sales_reporting`: `agent_pct`, `agent_participation_pct`, `agent_pc`, `total_pc`
- From `wealth_management_transactions`: `pc_amt` (production credits)
- From `wealth_management_rpt_agents`: `ytd_pcs_rolling12`, `career_pcs`

**Target Variable:** Commission amount or optimal commission percentage

**Business Value:** Fair compensation, incentive alignment, cost optimization

---

### 15. **Geographic Sales Opportunity Analysis**
**Objective:** Identify high-potential geographic markets

**Key Features:**
- From `wealth_management_clients`: `prov_state_tp_cd`, `city_name`, `postal_code`, `geo_cbsa_code`
- From `wealth_management_agents`: `business_state_cod`, `business_city`, `business_zip_code`, `division_name`
- From `wealth_management_client_metrics`: `branch_name`, `division`, `business_state_cod`
- From `wealth_management_pcpg_retention`: `state`, `city`, `branch_name`, `division`
- Aggregated metrics: Policy counts, AUM, agent density by geography

**Target Variable:** Sales potential score by geography

**Business Value:** Territory expansion, agent placement, market penetration strategy

---

## Implementation Priority Recommendations

### High Priority (Quick Wins)
1. **Client Churn Prediction** - Direct revenue impact
2. **Policy Lapse Risk Prediction** - Immediate actionable insights
3. **Agent Performance Prediction** - Resource optimization

### Medium Priority (Strategic Value)
4. **Client Lifetime Value** - Long-term planning
5. **Product Recommendation** - Revenue growth
6. **Cross-Sell Opportunity** - Wallet share expansion

### Lower Priority (Advanced Analytics)
7. **Fraud Detection** - Risk mitigation
8. **Agent-Client Matching** - Operational efficiency
9. **Geographic Analysis** - Strategic planning

---

## Data Requirements & Considerations

### Data Quality
- Ensure consistent date formats across tables
- Handle missing values in demographic and financial fields
- Address data inconsistencies in agent/client identifiers

### Feature Engineering Opportunities
- Time-based features: Days since policy registration, agent tenure
- Ratio features: Loan-to-cash-value, premium-to-face-amount
- Aggregated features: Total policies per client, average policy value
- Trend features: Rolling averages, growth rates

### Model Selection Recommendations
- **Classification:** Random Forest, XGBoost, LightGBM (for churn, fraud, default)
- **Regression:** Gradient Boosting, Neural Networks (for CLV, sales forecasting)
- **Clustering:** K-means, DBSCAN (for segmentation)
- **Recommendation:** Collaborative Filtering, Content-Based (for product recommendations)

---

## Key Join Relationships

- `pol_no` / `policy_no` - Links policies across tables
- `agent_code` / `agt_no` - Links agents across tables
- `axa_party_id` / `client_id` / `cl_party_id` - Links clients across tables
- `business_month` - Common partition key for time-based analysis

---

## Next Steps

1. **Data Exploration:** Conduct EDA on each table to understand distributions and relationships
2. **Feature Engineering:** Create derived features based on domain knowledge
3. **Pilot Project:** Start with Client Churn Prediction (highest business value)
4. **Model Development:** Build and validate models using cross-validation
5. **Deployment:** Integrate models into business workflows
6. **Monitoring:** Track model performance and retrain periodically

