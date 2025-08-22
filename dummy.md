## **Solution 1: Cross-Sell Propensity Model (Yes/No Classification)**

### **Objective**

Predict whether an existing customer (who has only one current product/plan) is **likely to purchase another product**.

### **Type of Problem**

* **Binary Classification** → Output is **Yes/No**.

### **Steps Involved**

1. **Data Extraction (from Hadoop / Hue)**

   * Data resides in **Hive tables** inside Hadoop.

   * Use **PySpark / Hive queries** in Python to fetch relevant data.

   * Example: Customer demographics, policy details, transactions, claims history.

   * *Why?* → Data must be centralized before preprocessing and model training.

2. **Data Preprocessing**

   * Handle **missing values** (e.g., impute with mean/median or “Unknown”).

   * Encode **categorical variables** (OneHotEncoding/LabelEncoding).

   * Normalize or scale numerical features if needed.

   * Remove outliers or inconsistent values.

   * *Why?* → Clean, consistent data improves model performance and reduces bias.

3. **Feature Engineering**

   * Customer age groups (e.g., \<30, 30-50, \>50).

   * Policy tenure buckets.

   * Claim frequency.

   * Interaction frequency (e.g., website logins, agent meetings).

   * *Why?* → Derived features capture patterns that raw data might miss.

4. **Train-Test Split**

   * Divide dataset into **train (70–80%) and test (20–30%)**.

   * *Why?* → To evaluate generalization and avoid overfitting.

5. **Model Selection & Training**

   * Start with simple models: **Logistic Regression, Decision Tree**.

   * Move to advanced models: **Random Forest, XGBoost, LightGBM**.

   * *Why?* → Logistic gives interpretability, while ensemble models improve accuracy.

6. **Model Evaluation**

   * Metrics: **Accuracy, Precision, Recall, F1-score, ROC-AUC**.

   * *Why?* → Cross-sell use cases need recall (don’t miss a potential buyer) and precision (don’t annoy customers with wrong offers).

7. **Deployment & Monitoring**

   * Expose as **REST API** (FastAPI/Flask) for real-time scoring.

   * Monitor drift and retrain periodically.

   * *Why?* → Business environment and customer behavior changes over time.

---

## **Solution 2: Next Best Product Recommendation (Multi-Class / Ranking Problem)**

### **Objective**

For customers with only one current product, predict **which product they are most likely to buy next**.

### **Type of Problem**

* **Multi-Class Classification** → Predict a product class (e.g., Health, Life, Vehicle).

* OR **Ranking / Recommendation** → Rank top-N products for each customer.

### **Steps Involved**

1. **Data Extraction (Hadoop / Hue)**

   * Pull **transaction \+ policy \+ product history** from Hive tables.

   * *Why?* → Recommendation requires **past purchase patterns**.

2. **Data Preprocessing**

   * Similar steps as Solution 1 (missing values, encoding, scaling).

   * Map customers to **historical purchases** and **sequence of products**.

   * *Why?* → Sequence/order matters in recommendations (e.g., Life → Health → Vehicle).

3. **Feature Engineering**

   * **Customer Profile Features**: Age, income, family size.

   * **Behavioral Features**: Previous claims, frequency of renewals.

   * **Product Affinity Features**: Which products are often purchased together.

   * *Why?* → Captures both customer behavior and product relationships.

4. **Problem Framing**

   * **Option A (Classification)**: Each product type is a class → predict which one.

   * **Option B (Recommendation System)**: Use **Collaborative Filtering / Matrix Factorization / Neural Recommenders**.

   * *Why?* → Classification is simple, Recommendation is more powerful but complex.

5. **Model Training**

   * Start with **multi-class classification models** (Random Forest, XGBoost).

   * Try **Recommendation models**:

     * Collaborative Filtering (ALS in Spark MLlib).

     * Neural Recommenders (e.g., DeepFM, LightFM).

   * *Why?* → Product purchase is not purely independent → CF learns latent patterns.

6. **Model Evaluation**

   * **Classification Metrics**: Accuracy, Top-3 Recall.

   * **Recommendation Metrics**: Precision@K, Recall@K, NDCG.

   * *Why?* → Business cares if the correct product appears in Top-3 recommendations.

7. **Deployment & Monitoring**

   * Provide **Top-N product recommendations via API**.

   * Track **conversion rate uplift** (business KPI).

   * *Why?* → Continuous improvement is required for personalization.  
