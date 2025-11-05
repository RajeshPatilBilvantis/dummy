# Feature 1: product1_to_product2_propensity
# This feature captures the general tendency for one product to follow another. We will calculate it as a frequency count. A higher count means a more common cross-sell path.

# --- Calculate Product Propensity ---

# Group by the first product and count how many times it has been cross-sold.
# This serves as our denominator to calculate ratios or can be a feature itself.
prod_total_cross_sells = cross_sell_history_df.groupBy("first_prod_code") \
    .count() \
    .withColumnRenamed("count", "p1_total_cross_sells")

# Now, let's create a more granular feature: the count of a specific cross-sell path (e.g., TermLife -> Annuity)
# This is a very powerful feature. We will join it back to the main df based on the first product.
# For a multi-class model, we can't join on the label directly, so we use the count of the first product.
# Let's call the feature `p1_cross_sell_popularity`.

propensity_prod_df = prod_total_cross_sells.withColumnRenamed("first_prod_code", "prod_code") \
                                           .withColumnRenamed("p1_total_cross_sells", "p1_cross_sell_popularity")

# --- Let's also create the MODE feature as requested (Most Common Second Product) ---

# 1. Count each specific path
path_counts = cross_sell_history_df.groupBy("first_prod_code", "second_prod_code").count()

# 2. Use a window function to find the most common path
window_spec = Window.partitionBy("first_prod_code").orderBy(F.col("count").desc())
most_common_path_df = path_counts.withColumn("rank", F.row_number().over(window_spec)) \
    .filter(F.col("rank") == 1) \
    .select(
        F.col("first_prod_code").alias("prod_code"),
        F.col("second_prod_code").alias("p1_most_common_next_prod")
    )

print("Product Propensity Lookup Table:")
propensity_prod_df.show(5)
# +---------+--------------------------+
# |prod_code|p1_cross_sell_popularity|
# +---------+--------------------------+
# |   PROD_A|                      1520|
# |   PROD_B|                       850|
# ...

print("Most Common Next Product Lookup Table:")
most_common_path_df.show(5)
# +---------+----------------------------+
# |prod_code|p1_most_common_next_prod|
# +---------+----------------------------+
# |   PROD_A|                      PROD_C|
# |   PROD_B|                      PROD_A|


######################################################################

# Feature 2: agent_cross_sell_propensity
# This feature captures an individual agent's habits. How many times has this specific agent sold any second product after selling this initial product?

# --- Calculate Agent Propensity ---

# Group by agent and the first product they sold, then count occurrences.
propensity_agent_df = cross_sell_history_df.groupBy("agt_no", "first_prod_code") \
    .count() \
    .withColumnRenamed("count", "agent_p1_cross_sell_count") \
    .withColumnRenamed("first_prod_code", "prod_code")

print("Agent Propensity Lookup Table:")
propensity_agent_df.show(5)
# +------+---------+---------------------------+
# |agt_no|prod_code|agent_p1_cross_sell_count|
# +------+---------+---------------------------+
# | 12345|   PROD_A|                         45|
# | 12345|   PROD_B|                         12|
# | 67890|   PROD_A|                         88|
# ...

#######################################################################

# Feature 3: branch_cross_sell_propensity
# This is identical to the agent propensity but aggregated at the branch level, capturing regional or office-level sales patterns.

# --- Calculate Branch Propensity ---

# Group by branch and the first product sold, then count occurrences.
propensity_branch_df = cross_sell_history_df.groupBy("branchoffice_code", "first_prod_code") \
    .count() \
    .withColumnRenamed("count", "branch_p1_cross_sell_count") \
    .withColumnRenamed("first_prod_code", "prod_code")

print("Branch Propensity Lookup Table:")
propensity_branch_df.show(5)
# +-----------------+---------+----------------------------+
# |branchoffice_code|prod_code|branch_p1_cross_sell_count|
# +-----------------+---------+----------------------------+
# |           BRANCH1|   PROD_A|                         350|
# |           BRANCH1|   PROD_B|                         120|
# |           BRANCH2|   PROD_A|                         410|
# ...



#######################################################################################
#######################################################################################

# Step 2: Applying the Features to Your Datasets
# Now, you must join these lookup tables back to both your training and testing sets. Using a left_join is crucial. It ensures that you don't lose any rows from your original datasets. Any product, agent, or branch that appeared in the test set but not the training set will get a null value, which we must handle.

def add_propensity_features(df, lookup_prod, lookup_mode, lookup_agent, lookup_branch):
    """
    Joins the pre-computed propensity lookup tables to a dataframe.
    """
    # Join the product-level features
    df_with_features = df.join(lookup_prod, on="prod_code", how="left")
    df_with_features = df_with_features.join(lookup_mode, on="prod_code", how="left")
    
    # Join the agent-level features
    df_with_features = df_with_features.join(lookup_agent, on=["agt_no", "prod_code"], how="left")

    # Join the branch-level features
    df_with_features = df_with_features.join(lookup_branch, on=["branchoffice_code", "prod_code"], how="left")

    # Handle nulls for "cold start" scenarios (e.g., a new agent in the test set)
    # Filling with 0 is a reasonable strategy, as it implies "no observed history".
    df_with_features = df_with_features.na.fill(0, [
        "p1_cross_sell_popularity",
        "agent_p1_cross_sell_count",
        "branch_p1_cross_sell_count"
    ])
    
    # For the categorical 'mode' feature, you can fill with a specific string like 'UNKNOWN'
    df_with_features = df_with_features.na.fill("UNKNOWN", ["p1_most_common_next_prod"])
    
    return df_with_features

# Apply the function to both train and test DataFrames
train_df_final = add_propensity_features(train_df, propensity_prod_df, most_common_path_df, propensity_agent_df, propensity_branch_df)
test_df_final = add_propensity_features(test_df, propensity_prod_df, most_common_path_df, propensity_agent_df, propensity_branch_df)

print("Final training data schema with new features:")
train_df_final.printSchema()

train_df_final.select("prod_code", "agt_no", "p1_cross_sell_popularity", "agent_p1_cross_sell_count", "p1_most_common_next_prod").show(10)
