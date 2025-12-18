# Diagnostics Pipeline Explanation - Step by Step Guide

This document provides a detailed explanation of each step in the `diagnostics_copy4-transformer.ipynb` notebook, including dummy table representations showing the data transformation at each stage.

---

## **STEP 1: Environment Setup and Data Loading**

### **What Happens:**
- Installs required libraries (commented out in notebook)
- Loads the raw data from Spark table `dl_tenants_daas.us_wealth_management.wealth_management_client_metrics`
- This is the initial raw dataset containing all client metrics

### **Dummy Table Representation:**

| cont_id | prod_lob | sub_product_level_1 | sub_product_level_2 | register_date | acct_val_amt | psn_age | client_seg | policy_status | branchoffice_code |
|---------|----------|---------------------|---------------------|---------------|--------------|---------|------------|---------------|-------------------|
| C001    | LIFE     | VLI                 | VARIABLE LIFE       | 2020-01-15    | 50000        | 45      | Premium    | Active        | 83                |
| C002    | BROKER DEALER | INVESTMENT PRODUCT - DIRECT | Investment Account | 2019-06-20    | 120000       | 52      | Standard   | Active        | 83                |
| C003    | GROUP RETIREMENT | RETIREMENT 401K | 401K Plan          | 2021-03-10    | 250000       | 38      | Premium    | Active        | 83                |

**Output:** Raw Spark DataFrame `df_raw` with ~48.9M rows (after 20% sampling)

---

## **STEP 2: Product Category Mapping**

### **What Happens:**
- Creates a new column `product_category` by mapping various product fields (`prod_lob`, `sub_product_level_1`, `sub_product_level_2`) to standardized categories
- Categories include: `LIFE_INSURANCE`, `RETIREMENT`, `INVESTMENT`, `NETWORK_PRODUCTS`, `DISABILITY`, `HEALTH`, `OTHER`
- Uses cascading `when()` conditions to check multiple fields

### **Dummy Table Representation:**

**Before:**
| cont_id | prod_lob | sub_product_level_1 | sub_product_level_2 | register_date |
|---------|----------|---------------------|---------------------|---------------|
| C001    | LIFE     | VLI                 | VARIABLE LIFE       | 2020-01-15    |
| C002    | BROKER DEALER | INVESTMENT PRODUCT - DIRECT | Investment Account | 2019-06-20    |
| C003    | GROUP RETIREMENT | RETIREMENT 401K | 401K Plan          | 2021-03-10    |

**After:**
| cont_id | prod_lob | sub_product_level_1 | sub_product_level_2 | product_category | register_date |
|---------|----------|---------------------|---------------------|------------------|---------------|
| C001    | LIFE     | VLI                 | VARIABLE LIFE       | **LIFE_INSURANCE** | 2020-01-15    |
| C002    | BROKER DEALER | INVESTMENT PRODUCT - DIRECT | Investment Account | **INVESTMENT** | 2019-06-20    |
| C003    | GROUP RETIREMENT | RETIREMENT 401K | 401K Plan          | **RETIREMENT** | 2021-03-10    |

**Logic:** 
- If `prod_lob == "LIFE"` → `LIFE_INSURANCE`
- If `sub_product_level_1` contains "RETIREMENT 401K" → `RETIREMENT`
- If `prod_lob == "BROKER DEALER"` → `INVESTMENT`
- And so on...

---

## **STEP 3: Data Filtering and Event Selection**

### **What Happens:**
- Filters to keep only rows with:
  - Non-null `cont_id` (client ID)
  - Non-null `register_date` (event date)
  - Non-null `product_category` (valid product)
- Applies 20% random sampling (`SAMPLE_FRACTION = 0.2`) to reduce data size
- Filters to only `Active` policies
- Selects relevant columns for modeling

### **Dummy Table Representation:**

**After Filtering:**
| cont_id | product_category | register_date | acct_val_amt | psn_age | client_seg | policy_status |
|---------|------------------|---------------|--------------|---------|------------|---------------|
| C001    | LIFE_INSURANCE   | 2020-01-15    | 50000        | 45      | Premium    | Active        |
| C001    | INVESTMENT       | 2020-06-20    | 75000        | 45      | Premium    | Active        |
| C002    | INVESTMENT       | 2019-06-20    | 120000       | 52      | Standard   | Active        |
| C002    | RETIREMENT       | 2020-03-10    | 150000       | 52      | Standard   | Active        |
| C003    | RETIREMENT       | 2021-03-10    | 250000       | 38      | Premium    | Active        |

**Output:** `df_events` with ~48.9M rows (after sampling)

---

## **STEP 4: Event Ordering and Indexing**

### **What Happens:**
- Converts `register_date` to timestamp
- Creates `event_idx` column using window function to number events chronologically per client
- Orders events by `register_ts` within each `cont_id`

### **Dummy Table Representation:**

**Before:**
| cont_id | product_category | register_date |
|---------|------------------|---------------|
| C001    | LIFE_INSURANCE   | 2020-01-15    |
| C001    | INVESTMENT       | 2020-06-20    |
| C002    | RETIREMENT       | 2020-03-10    |
| C002    | INVESTMENT       | 2019-06-20    |

**After:**
| cont_id | product_category | register_ts | event_idx |
|---------|------------------|-------------|-----------|
| C001    | LIFE_INSURANCE   | 2020-01-15  | 1         |
| C001    | INVESTMENT       | 2020-06-20  | 2         |
| C002    | INVESTMENT       | 2019-06-20  | 1         |
| C002    | RETIREMENT       | 2020-03-10  | 2         |

**Note:** Events are ordered chronologically per client, with `event_idx` starting at 1

---

## **STEP 5: Product Vocabulary Building**

### **What Happens:**
- Extracts all unique `product_category` values from the dataset
- Creates mapping dictionaries:
  - `prod2id`: Maps product name → numeric ID (1, 2, 3, ...)
  - `id2prod`: Maps numeric ID → product name
- Sets `NUM_CLASSES` to the number of unique products (7 in this case)

### **Dummy Table Representation:**

**Product Vocabulary:**
| Product ID | Product Name |
|------------|--------------|
| 1          | DISABILITY   |
| 2          | HEALTH       |
| 3          | INVESTMENT   |
| 4          | LIFE_INSURANCE |
| 5          | NETWORK_PRODUCTS |
| 6          | OTHER        |
| 7          | RETIREMENT   |

**Mapping Example:**
- `prod2id["RETIREMENT"]` → `7`
- `id2prod[4]` → `"LIFE_INSURANCE"`

**Output:** `NUM_CLASSES = 7` (7 product categories)

---

## **STEP 6: Sequence Building and Deduplication**

### **What Happens:**
- Groups events by `cont_id` to create purchase sequences
- Removes consecutive duplicate products (keeps only transitions)
- Filters out clients with fewer than `MIN_EVENTS` (2) events

### **Dummy Table Representation:**

**Before Deduplication:**
| cont_id | Sequence |
|---------|----------|
| C001    | [LIFE_INSURANCE, LIFE_INSURANCE, INVESTMENT, INVESTMENT, RETIREMENT] |
| C002    | [INVESTMENT, RETIREMENT, RETIREMENT, INVESTMENT] |

**After Deduplication:**
| cont_id | Sequence |
|---------|----------|
| C001    | [LIFE_INSURANCE, INVESTMENT, RETIREMENT] |
| C002    | [INVESTMENT, RETIREMENT, INVESTMENT] |

**After Filtering (MIN_EVENTS >= 2):**
| cont_id | Sequence | Sequence Length |
|---------|----------|-----------------|
| C001    | [LIFE_INSURANCE, INVESTMENT, RETIREMENT] | 3 |
| C002    | [INVESTMENT, RETIREMENT, INVESTMENT] | 3 |

**Output:** ~388K users with sequences

---

## **STEP 7: Training Example Generation (Sliding Window)**

### **What Happens:**
- Creates training examples using sliding window approach
- For each sequence, generates examples where:
  - **History**: Previous products (up to `MAX_SEQ_LEN = 10`)
  - **Label**: Next product to predict
- Converts product names to numeric IDs using `prod2id`

### **Dummy Table Representation:**

**Input Sequence (cont_id = C001):**
```
[LIFE_INSURANCE, INVESTMENT, RETIREMENT]
```

**Converted to IDs:**
```
[4, 3, 7]  (LIFE_INSURANCE=4, INVESTMENT=3, RETIREMENT=7)
```

**Generated Examples:**

| cont_id | history (as IDs) | label (ID) | Explanation |
|---------|------------------|------------|-------------|
| C001    | [4]              | 3          | Given LIFE_INSURANCE → predict INVESTMENT |
| C001    | [4, 3]           | 7          | Given [LIFE_INSURANCE, INVESTMENT] → predict RETIREMENT |

**If sequence is longer than MAX_SEQ_LEN (10):**
- History is truncated to last 10 products
- Example: If history has 15 products, only last 10 are used

**Output:** ~580K training examples

---

## **STEP 8: Feature Engineering - History-Derived Features**

### **What Happens:**
- Computes statistical features from purchase history:
  - `seq_len`: Length of history sequence
  - `last_1`: Most recent product ID
  - `last_2`: Second most recent product ID
  - `unique_prior`: Number of unique products in history
  - `num_switches`: Number of product transitions
  - `freq_1` to `freq_N`: Frequency of each product in history

### **Dummy Table Representation:**

**Input Example:**
| cont_id | history (IDs) | label |
|---------|---------------|-------|
| C001    | [4, 3, 4, 7]  | 3     |

**Feature Extraction:**

| Feature | Value | Calculation |
|---------|-------|-------------|
| seq_len | 4 | len([4, 3, 4, 7]) |
| last_1 | 7 | history[-1] |
| last_2 | 4 | history[-2] |
| unique_prior | 3 | len(set([4, 3, 4, 7])) = {4, 3, 7} |
| num_switches | 3 | Transitions: 4→3, 3→4, 4→7 |
| freq_1 (DISABILITY) | 0 | Count of 1 in history |
| freq_2 (HEALTH) | 0 | Count of 2 in history |
| freq_3 (INVESTMENT) | 1 | Count of 3 in history |
| freq_4 (LIFE_INSURANCE) | 2 | Count of 4 in history |
| freq_5 (NETWORK_PRODUCTS) | 0 | Count of 5 in history |
| freq_6 (OTHER) | 0 | Count of 6 in history |
| freq_7 (RETIREMENT) | 1 | Count of 7 in history |

**Output Table:**
| cont_id | seq_len | last_1 | last_2 | unique_prior | num_switches | freq_1 | freq_2 | freq_3 | freq_4 | freq_5 | freq_6 | freq_7 | label |
|---------|---------|--------|--------|--------------|--------------|--------|--------|--------|--------|--------|--------|--------|-------|
| C001    | 4       | 7      | 4      | 3            | 3            | 0      | 0      | 1      | 2      | 0      | 0      | 1      | 3     |

**Output:** ~580K examples with features

---

## **STEP 9: Static Feature Joining**

### **What Happens:**
- Gets the most recent snapshot of client static features (demographics, assets, etc.)
- Uses window function to select the latest record per client
- Left joins static features to training examples

### **Dummy Table Representation:**

**Client Snapshots (Most Recent per Client):**
| cont_id | acct_val_amt | psn_age | client_seg | aum_band | channel | branchoffice_code |
|---------|--------------|---------|------------|----------|---------|-------------------|
| C001    | 75000        | 45      | Premium    | High     | Direct  | 83                |
| C002    | 150000       | 52      | Standard   | Medium   | Broker  | 83                |

**After Join:**
| cont_id | seq_len | last_1 | ... | acct_val_amt | psn_age | client_seg | aum_band | channel | label |
|---------|---------|--------|-----|--------------|---------|------------|----------|---------|-------|
| C001    | 4       | 7      | ... | 75000        | 45      | Premium    | High     | Direct  | 3     |
| C002    | 3       | 3      | ... | 150000       | 52      | Standard   | Medium   | Broker  | 7     |

**Output:** ~580K examples with all features

---

## **STEP 10: Missing Value Imputation**

### **What Happens:**
- Numeric columns: Fill nulls with 0
- Categorical columns: Fill nulls with mode (most frequent value) or "UNKNOWN"

### **Dummy Table Representation:**

**Before Imputation:**
| cont_id | acct_val_amt | psn_age | client_seg | aum_band |
|---------|--------------|---------|------------|----------|
| C001    | 75000        | 45      | Premium    | High     |
| C002    | NULL         | NULL    | NULL       | Medium   |
| C003    | 50000        | 38      | Premium    | NULL     |

**After Imputation:**
| cont_id | acct_val_amt | psn_age | client_seg | aum_band |
|---------|--------------|---------|------------|----------|
| C001    | 75000        | 45      | Premium    | High     |
| C002    | 0            | 0       | Premium    | Medium   |
| C003    | 50000        | 38      | Premium    | High     |

**Note:** Mode for `client_seg` = "Premium", mode for `aum_band` = "High"

---

## **STEP 11: History Padding and Expansion**

### **What Happens:**
- Pads history sequences to fixed length `MAX_SEQ_LEN = 10`
- Expands padded history into separate columns: `hist_0`, `hist_1`, ..., `hist_9`
- Uses 0 for padding (0 is reserved for padding, product IDs start at 1)

### **Dummy Table Representation:**

**Before Padding:**
| cont_id | history (IDs) |
|---------|---------------|
| C001    | [4, 3, 7]     |
| C002    | [3, 7, 3, 4, 7, 3, 4, 7, 3, 4, 7] |

**After Padding (MAX_SEQ_LEN = 10):**
| cont_id | hist_0 | hist_1 | hist_2 | hist_3 | hist_4 | hist_5 | hist_6 | hist_7 | hist_8 | hist_9 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| C001    | 0      | 0      | 0      | 0      | 0      | 0      | 0      | **4**  | **3**  | **7**  |
| C002    | **3**  | **7**  | **3**  | **4**  | **7**  | **3**  | **4**  | **7**  | **3**  | **4**  |

**Logic:**
- Short sequences: Pad with 0s on the left
- Long sequences: Keep only last 10 products
- C001: [4, 3, 7] → [0, 0, 0, 0, 0, 0, 0, 4, 3, 7]
- C002: Last 10 products are kept

---

## **STEP 12: Categorical Encoding**

### **What Happens:**
- Converts categorical string columns to numeric indices
- Creates new columns: `client_seg_idx`, `aum_band_idx`, `channel_idx`, etc.
- Maps each unique category value to an integer

### **Dummy Table Representation:**

**Before Encoding:**
| cont_id | client_seg | aum_band | channel |
|---------|------------|----------|---------|
| C001    | Premium    | High     | Direct  |
| C002    | Standard   | Medium   | Broker  |
| C003    | Premium    | Low      | Direct  |

**Mapping Created:**
- `client_seg`: {"Premium": 0, "Standard": 1}
- `aum_band`: {"High": 0, "Low": 1, "Medium": 2}
- `channel`: {"Broker": 0, "Direct": 1}

**After Encoding:**
| cont_id | client_seg | client_seg_idx | aum_band | aum_band_idx | channel | channel_idx |
|---------|------------|----------------|----------|--------------|---------|-------------|
| C001    | Premium    | 0              | High     | 0            | Direct  | 1           |
| C002    | Standard   | 1              | Medium   | 2            | Broker  | 0           |
| C003    | Premium    | 0              | Low      | 1            | Direct  | 1           |

---

## **STEP 13: Train/Validation/Test Split**

### **What Happens:**
- Randomly splits data into:
  - **Train**: 80% (`TRAIN_FRAC = 0.8`)
  - **Validation**: 10% (`VAL_FRAC = 0.1`)
  - **Test**: 10% (`TEST_FRAC = 0.1`)
- Converts Spark DataFrames to Pandas DataFrames for LightGBM

### **Dummy Table Representation:**

**Split Distribution:**
| Dataset | Count | Percentage |
|---------|-------|------------|
| Train   | 464,415 | 80% |
| Validation | 57,870 | 10% |
| Test    | 58,333 | 10% |
| **Total** | **580,618** | **100%** |

**Final Feature Columns (42 total):**
- History: `hist_0` to `hist_9` (10 columns)
- Sequence features: `seq_len`, `last_1`, `last_2`, `unique_prior`, `num_switches` (5 columns)
- Frequency features: `freq_1` to `freq_7` (7 columns)
- Static numeric: `acct_val_amt`, `face_amt`, `cash_val_amt`, `wc_total_assets`, `wc_assetmix_*`, `psn_age` (12 columns)
- Categorical indices: `client_seg_idx`, `client_seg_1_idx`, `aum_band_idx`, `channel_idx`, `agent_segment_idx`, `branchoffice_code_idx` (6 columns)
- Label: `label0` (0-based, 0-6 for 7 classes)

---

## **STEP 14: Label Remapping**

### **What Happens:**
- Converts labels from product IDs (1-7) to 0-based indices (0-6) for LightGBM
- Creates `label_map` to track the mapping

### **Dummy Table Representation:**

**Before Remapping:**
| cont_id | label (product ID) |
|---------|-------------------|
| C001    | 3                 |
| C002    | 7                 |
| C003    | 4                 |

**Label Mapping:**
| Original Product ID | 0-Based Label | Product Name |
|---------------------|---------------|--------------|
| 1                   | 0             | DISABILITY   |
| 2                   | 1             | HEALTH       |
| 3                   | 2             | INVESTMENT   |
| 4                   | 3             | LIFE_INSURANCE |
| 5                   | 4             | NETWORK_PRODUCTS |
| 6                   | 5             | OTHER        |
| 7                   | 6             | RETIREMENT   |

**After Remapping:**
| cont_id | label (product ID) | label0 (0-based) |
|---------|-------------------|------------------|
| C001    | 3                 | 2                |
| C002    | 7                 | 6                |
| C003    | 4                 | 3                |

---

## **STEP 15: LightGBM Model Training**

### **What Happens:**
- Trains a LightGBM gradient boosting model for multiclass classification
- Uses parameters:
  - `objective`: "multiclass"
  - `num_class`: 7
  - `learning_rate`: 0.05
  - `num_leaves`: 64
  - Early stopping after 50 rounds without improvement
- Trains for up to 2000 boosting rounds

### **Training Process:**
```
Round 1: train's multi_logloss: 1.234567, val's multi_logloss: 1.345678
Round 2: train's multi_logloss: 1.123456, val's multi_logloss: 1.234567
...
Round 481: train's multi_logloss: 0.598748, val's multi_logloss: 0.686169
Early stopping, best iteration is: [481]
```

**Output:** Trained LightGBM model

---

## **STEP 16: Model Evaluation**

### **What Happens:**
- Makes predictions on test set
- Calculates metrics:
  - Accuracy: 74.8%
  - F1-weighted: 74.5%
  - F1-macro: 57.2%

### **Dummy Table Representation:**

**Test Predictions:**
| cont_id | True Label | Predicted Label | Confidence |
|---------|------------|-----------------|------------|
| C001    | 2 (INVESTMENT) | 2 (INVESTMENT) | 0.85 |
| C002    | 6 (RETIREMENT) | 6 (RETIREMENT) | 0.78 |
| C003    | 3 (LIFE_INSURANCE) | 2 (INVESTMENT) | 0.65 |

**Confusion Matrix (7x7):**
| Actual \ Predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-------------------|---|---|---|---|---|---|---|
| 0 (DISABILITY)     | 189 | 0 | 50 | 30 | 20 | 30 | 31 |
| 1 (HEALTH)         | 0 | 0 | 5 | 3 | 2 | 5 | 9 |
| 2 (INVESTMENT)     | 500 | 10 | 11075 | 1500 | 800 | 200 | 90 |
| 3 (LIFE_INSURANCE) | 300 | 5 | 1200 | 10823 | 800 | 400 | 328 |
| 4 (NETWORK_PRODUCTS) | 200 | 3 | 800 | 600 | 7666 | 500 | 341 |
| 5 (OTHER)          | 150 | 2 | 400 | 200 | 300 | 370 | 98 |
| 6 (RETIREMENT)     | 200 | 5 | 800 | 500 | 400 | 200 | 13461 |

**Classification Report:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.75      | 0.54   | 0.63     | 350     |
| 1     | 0.00      | 0.00   | 0.00     | 24      |
| 2     | 0.73      | 0.73   | 0.73     | 15,175  |
| 3     | 0.78      | 0.77   | 0.77     | 14,056  |
| 4     | 0.74      | 0.69   | 0.71     | 11,110  |
| 5     | 0.56      | 0.28   | 0.37     | 1,320   |
| 6     | 0.76      | 0.83   | 0.79     | 16,166  |

---

## **STEP 17: Prediction Pipeline for Branch Office 83**

### **What Happens:**
- Loads data filtered for `branchoffice_code = "83"` and most recent `business_month`
- Repeats similar preprocessing steps as training:
  - Product category mapping
  - Event ordering
  - Sequence building
  - Feature engineering
- Creates one prediction example per client using their full history

### **Dummy Table Representation:**

**Input: Clients in Branch Office 83 (Most Recent Month)**
| cont_id | product_category | register_date | business_month |
|---------|------------------|---------------|----------------|
| C100    | RETIREMENT       | 2025-10-15    | 202511         |
| C101    | LIFE_INSURANCE   | 2025-10-20    | 202511         |

**Full History for Prediction:**
| cont_id | Historical Sequence (all time) |
|---------|--------------------------------|
| C100    | [INVESTMENT, RETIREMENT, RETIREMENT, INVESTMENT, RETIREMENT] |
| C101    | [LIFE_INSURANCE, INVESTMENT, LIFE_INSURANCE] |

**Prediction Examples (One per Client):**
| cont_id | history (last 10) | seq_len | last_1 | ... | psn_age | client_seg_idx |
|---------|-------------------|---------|--------|-----|---------|----------------|
| C100    | [3, 7, 7, 3, 7]   | 5       | 7      | ... | 45      | 0              |
| C101    | [4, 3, 4]         | 3       | 4      | ... | 52      | 1              |

**Output:** 245,729 prediction examples

---

## **STEP 18: Making Predictions**

### **What Happens:**
- Uses trained model to predict next product for each client
- Outputs:
  - Predicted class ID
  - Predicted product name
  - Prediction probability (confidence)
  - Probabilities for all classes

### **Dummy Table Representation:**

**Predictions:**
| cont_id | pred_class_id | pred_product | pred_prob | prob_0 | prob_1 | prob_2 | prob_3 | prob_4 | prob_5 | prob_6 |
|---------|---------------|--------------|-----------|--------|--------|--------|--------|--------|--------|--------|
| C100    | 6             | RETIREMENT   | 0.82      | 0.01   | 0.00   | 0.05   | 0.02   | 0.03   | 0.07   | **0.82** |
| C101    | 3             | LIFE_INSURANCE | 0.65    | 0.02   | 0.00   | 0.15   | **0.65** | 0.10   | 0.05   | 0.03   |

**Prediction Summary:**
| Product | Count | Percentage |
|---------|-------|------------|
| RETIREMENT | 135,438 | 55.1% |
| LIFE_INSURANCE | 32,786 | 13.3% |
| INVESTMENT | 31,383 | 12.8% |
| OTHER | 26,809 | 10.9% |
| NETWORK_PRODUCTS | 17,365 | 7.1% |
| DISABILITY | 1,929 | 0.8% |
| HEALTH | 19 | 0.0% |

---

## **STEP 19: Adding Client Demographics**

### **What Happens:**
- Joins additional client information:
  - `axa_party_id`
  - `division_name`, `branch_name`
  - `business_city`, `business_state_cod`
  - `client_tenure` (calculated from register_date)

### **Dummy Table Representation:**

**Final Predictions with Demographics:**
| cont_id | axa_party_id | pred_product | pred_prob | psn_age | client_seg | aum_band | channel | division_name | branch_name | business_city | client_tenure |
|---------|--------------|--------------|-----------|---------|------------|----------|---------|----------------|------------|---------------|---------------|
| C100    | AXA001       | RETIREMENT   | 0.82      | 45      | Premium    | High     | Direct  | North Division | Branch A   | New York      | 5.2           |
| C101    | AXA002       | LIFE_INSURANCE | 0.65    | 52      | Standard   | Medium   | Broker  | South Division | Branch B   | Los Angeles   | 3.8           |

**Output:** 245,729 rows with 23 columns

---

## **STEP 20: SHAP Analysis for Explainability**

### **What Happens:**
- Computes SHAP (SHapley Additive exPlanations) values for predictions
- SHAP values quantify how much each feature contributes to the prediction
- Generates:
  - Overall feature importance rankings
  - Per-product feature importance
  - Individual client explanations

### **Dummy Table Representation:**

**Overall Feature Importance (Top 10):**
| Rank | Feature | Mean \|SHAP Value\| |
|------|---------|---------------------|
| 1    | last_1  | 0.245               |
| 2    | hist_9  | 0.198               |
| 3    | hist_8  | 0.165               |
| 4    | freq_7  | 0.142               |
| 5    | psn_age | 0.128               |
| 6    | freq_3  | 0.115               |
| 7    | seq_len | 0.098               |
| 8    | acct_val_amt | 0.087          |
| 9    | hist_7  | 0.075               |
| 10   | client_seg_idx | 0.062        |

**Individual Client SHAP Explanation (C100):**
| Feature | Feature Value | SHAP Value | Contribution |
|---------|---------------|------------|--------------|
| last_1  | 7 (RETIREMENT) | +0.15      | Strong positive |
| hist_9  | 7 (RETIREMENT) | +0.12      | Positive |
| freq_7  | 3             | +0.10      | Positive |
| psn_age | 45            | +0.08      | Positive |
| hist_8  | 3 (INVESTMENT) | -0.05     | Negative |
| ...     | ...           | ...        | ... |

**Prediction:** RETIREMENT (82% confidence)
- **Base value:** 0.20
- **Sum of SHAP values:** +0.62
- **Final prediction score:** 0.82

---

## **STEP 21: Agent Talking Points Generation**

### **What Happens:**
- Creates personalized talking points for each client based on:
  - SHAP feature contributions
  - Client demographics
  - Purchase history
- Formats as natural language recommendations

### **Dummy Table Representation:**

**Talking Points for Client C100:**
```
Based on your client profile and purchase history, we recommend RETIREMENT with 82.0% confidence.

Key reasons for this recommendation:
  1. Your most recent product (RETIREMENT) shows a natural progression to RETIREMENT.
  2. Your frequent engagement with RETIREMENT indicates readiness for RETIREMENT.
  3. At age 45, RETIREMENT aligns well with your life stage and financial planning needs.
  4. Your asset profile indicates RETIREMENT would be an excellent fit for your portfolio.
  5. With 5.2 years as our client, RETIREMENT represents a natural next step in your relationship with us.

Considerations:
  - Even with limited prior engagement in this category, RETIREMENT presents a strong opportunity.

Next Steps: Let's schedule a consultation to discuss how RETIREMENT can help you achieve your financial goals.
```

**Talking Points Summary:**
| cont_id | predicted_product | confidence | top_positive_features | top_negative_features |
|---------|-------------------|------------|----------------------|----------------------|
| C100    | RETIREMENT        | 82.0%      | last_1, freq_7, psn_age | hist_8 |
| C101    | LIFE_INSURANCE    | 65.0%      | last_1, hist_9, psn_age | freq_3 |

---

## **Summary of Key Parameters**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLE_FRACTION` | 0.2 | 20% random sampling of data |
| `MIN_EVENTS` | 2 | Minimum events per client |
| `MAX_SEQ_LEN` | 10 | Maximum history length |
| `TRAIN_FRAC` | 0.8 | Training set percentage |
| `VAL_FRAC` | 0.1 | Validation set percentage |
| `TEST_FRAC` | 0.1 | Test set percentage |
| `NUM_CLASSES` | 7 | Number of product categories |
| `NUM_BOOST_ROUND` | 2000 | Maximum LightGBM iterations |
| `EARLY_STOP` | 50 | Early stopping rounds |

---

## **Data Flow Summary**

```
Raw Data (48.9M rows)
    ↓
Product Category Mapping
    ↓
Filtering & Sampling (20%)
    ↓
Event Ordering & Indexing
    ↓
Sequence Building & Deduplication (388K users)
    ↓
Sliding Window Examples (580K examples)
    ↓
Feature Engineering (42 features)
    ↓
Train/Val/Test Split (80/10/10)
    ↓
LightGBM Training (481 rounds)
    ↓
Model Evaluation (74.8% accuracy)
    ↓
Predictions for Branch 83 (245K clients)
    ↓
SHAP Analysis & Talking Points
    ↓
Final Output (23 columns, 245K rows)
```

---

## **Key Insights**

1. **Sequence-Based Approach**: The model uses purchase history sequences to predict next product
2. **Feature Engineering**: Combines sequence features (history, frequency) with static features (demographics, assets)
3. **Multiclass Classification**: Predicts one of 7 product categories
4. **Explainability**: SHAP values provide interpretable explanations for each prediction
5. **Production Ready**: Includes talking points for agents to use in client conversations

---

*End of Documentation*

