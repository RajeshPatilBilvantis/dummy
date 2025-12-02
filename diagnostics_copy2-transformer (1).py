# Databricks notebook source
# %pip install synapseml
# dbutils.library.restartPython()

# COMMAND ----------

# Install CatBoost using pip
# Best practice: Use %pip to install packages in Databricks notebooks to ensure the environment is updated.
# %pip install catboost

# COMMAND ----------

# Use Spark DataFrame for large data exploration
# This will load the complete table as a Spark DataFrame (distributed, scalable)
df_raw = spark.table("dl_tenants_daas.us_wealth_management.wealth_management_client_metrics")

# Show schema and first few rows
# df_raw.printSchema()
# display(df_raw)

# COMMAND ----------



from pyspark.sql import functions as F

df_raw = df_raw.withColumn(
    "product_category",
    F.when(F.col("prod_lob") == "LIFE", "LIFE_INSURANCE")
    .when(F.col("sub_product_level_1").isin("VLI", "WL", "UL/IUL", "TERM", "PROTECTIVE PRODUCT"), "LIFE_INSURANCE")
    .when(F.col("sub_product_level_2").like("%LIFE%"), "LIFE_INSURANCE")
    .when(F.col("sub_product_level_2").isin(
        "VARIABLE UNIVERSAL LIFE", "WHOLE LIFE", "UNIVERSAL LIFE",
        "INDEX UNIVERSAL LIFE", "TERM PRODUCT", "VARIABLE LIFE",
        "SURVIVORSHIP WHOLE LIFE", "MONY PROTECTIVE PRODUCT"
    ), "LIFE_INSURANCE")
    .when(F.col("prod_lob").isin("GROUP RETIREMENT", "INDIVIDUAL RETIREMENT"), "RETIREMENT")
    .when(F.col("sub_product_level_1").isin(
        "EQUIVEST", "RETIREMENT 401K", "ACCUMULATOR",
        "RETIREMENT CORNERSTONE", "SCS", "INVESTMENT EDGE"
    ), "RETIREMENT")
    .when(
        (F.col("sub_product_level_2").like("%403B%")) |
        (F.col("sub_product_level_2").like("%401%")) |
        (F.col("sub_product_level_2").like("%IRA%")) |
        (F.col("sub_product_level_2").like("%SEP%")),
        "RETIREMENT"
    )
    .when(F.col("prod_lob") == "BROKER DEALER", "INVESTMENT")
    .when(F.col("sub_product_level_1").isin(
        "INVESTMENT PRODUCT - DIRECT", "INVESTMENT PRODUCT - BROKERAGE",
        "INVESTMENT PRODUCT - ADVISORY", "DIRECT", "BROKERAGE",
        "ADVISORY", "CASH SOLICITOR"
    ), "INVESTMENT")
    .when(
        (F.col("sub_product_level_2").like("%Investment%")) |
        (F.col("sub_product_level_2").like("%Brokerage%")) |
        (F.col("sub_product_level_2").like("%Advisory%")),
        "INVESTMENT"
    )
    .when(F.col("prod_lob") == "NETWORK", "NETWORK_PRODUCTS")
    .when(
        (F.col("sub_product_level_1") == "NETWORK PRODUCTS") |
        (F.col("sub_product_level_2") == "NETWORK PRODUCTS"),
        "NETWORK_PRODUCTS"
    )
    .when(
        (F.col("prod_lob") == "OTHERS") & (F.col("sub_product_level_1") == "HAS"),
        "DISABILITY"
    )
    .when(F.col("sub_product_level_2") == "HAS - DISABILITY", "DISABILITY"
    )
    .when(F.col("prod_lob") == "OTHERS", "HEALTH")
    .when(F.col("sub_product_level_2") == "GROUP HEALTH PRODUCTS", "HEALTH")
    .otherwise("OTHER")
)




# =====================
# 3. Exclude OTHER (not part of the 7-class problem)
# =====================
df_raw = df_raw.filter(F.col("product_category") != "OTHER")


# =====================
# 4. Basic cleaning - remove policies without dates or customer id
# =====================
df_raw = df_raw.filter(
    F.col("strt_date").isNotNull() & F.col("end_date").isNotNull() & F.col("axa_party_id").isNotNull()
)


# =====================
# 5. Cast dates correctly (if they are strings)
# =====================
df_raw = (
    df_raw
    .withColumn("strt_date", F.to_date("strt_date"))
    .withColumn("end_date", F.to_date("end_date"))
)


# =====================
# 6. Cache for performance
# =====================
df_raw = df_raw.cache()
df_raw.count()   # trigger caching



# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

# STEP 2: keep only active policies (to avoid terminated ones)
df = df_raw.filter(F.col("policy_status") == "Active")

# STEP 3: find purchase order per client (sorted by register date)
w = Window.partitionBy("cont_id").orderBy("register_date")

df = df.withColumn("rn", F.row_number().over(w))

# STEP 4: first product and second product
df_pivot = df.withColumn("product_category_next",
                         F.lead("product_category").over(w))

# Keep only customers with at least 2 products
df_two = df_pivot.filter(F.col("product_category_next").isNotNull())

display(df_two.select("cont_id", "product_category", "product_category_next").limit(10))
print("Final training population:", df_two.count())




# COMMAND ----------


feature_cols = [
    "face_amt", "cash_val_amt", "acct_val_amt",
    "monthly_preminum_amount", "wc_total_assets",
    "wc_assetmix_stocks","wc_assetmix_bonds",
    "wc_assetmix_mutual_funds", "wc_assetmix_annuity",
    "wc_assetmix_deposits", "wc_assetmix_other_assets",
    "psn_age", "client_seg", "client_seg_1", "aum_band",
    "policy_type", "designation", "channel"
]

df_train = df_two.select(
    "cont_id",
   "product_category",
    "product_category_next",   # ★ LABEL WE WILL PREDICT
    *feature_cols
)

display(df_train)
print("Training DF Count:", df_train.count())

# COMMAND ----------

# CELL 1: pick and sanitize columns (run on Spark)
from pyspark.sql import functions as F

# Parameters
SAMPLE_FRACTION = 0.2   # e.g. 0.1 for 10% sample; set to None to use full data
MIN_EVENTS = 3           # keep customers with >= MIN_EVENTS
MAX_SEQ_LEN = 32
OUTPARQUET = "/dbfs/tmp/seqs_parquet/"

# Choose columns required for sequence model (lightweight)
keep_cols = [
    "cont_id", "axa_party_id", "policy_no", "register_date",
    "product_category", "prod_lob", "sub_product_level_1", "sub_product_level_2",
    "acct_val_amt","face_amt","cash_val_amt","wc_total_assets",
    "wc_assetmix_stocks","wc_assetmix_bonds","wc_assetmix_mutual_funds",
    "wc_assetmix_annuity","wc_assetmix_deposits","wc_assetmix_other_assets",
    "psn_age","client_seg","client_seg_1","aum_band",
    "channel","agent_segment","branchoffice_code"
]

# Ensure columns exist
cols_present = [c for c in keep_cols if c in df_raw.columns]
print("Using columns:", len(cols_present), "->", cols_present)

# Filter out rows with null register_date or null cont_id or product_category
df_events = df_raw.select(*cols_present).filter(
    (F.col("cont_id").isNotNull()) & (F.col("register_date").isNotNull()) & (F.col("product_category").isNotNull())
)

# optional sampling to reduce size (set SAMPLE_FRACTION to float if needed)
if SAMPLE_FRACTION is not None:
    print("Sampling fraction:", SAMPLE_FRACTION)
    df_events = df_events.sample(withReplacement=False, fraction=float(SAMPLE_FRACTION), seed=42)

print("Event rows after filter (approx):", df_events.count())

# COMMAND ----------

# CELL 2: deduplicate and prepare lightweight event tuples
from pyspark.sql import Window

# dedupe exact duplicates
df_events = df_events.dropDuplicates(["cont_id", "register_date", "policy_no", "product_category"])

# make sure register_date is a timestamp or date
df_events = df_events.withColumn("register_date_ts", F.to_timestamp("register_date"))

# sort per user and assign event_index
w = Window.partitionBy("cont_id").orderBy(F.col("register_date_ts").asc(), F.col("policy_no").asc())
df_events = df_events.withColumn("event_idx", F.row_number().over(w))

# compute per-user count
df_counts = df_events.groupBy("cont_id").agg(F.count("*").alias("n_events"))
print("Unique users with events:", df_counts.count())

# join counts back
df_events = df_events.join(df_counts, on="cont_id", how="left")

# drop users with fewer than 1 if needed (we do filtering later)
print("Sample events:")
display(df_events.limit(10))

# COMMAND ----------

# CELL 3: build ordered sequences using RDD. Each item => (cont_id, dict{...})
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

prod_col = "product_category"
num_cols_seq = ["acct_val_amt", "face_amt", "cash_val_amt", "wc_total_assets", "psn_age"]

keep_for_rdd = ["cont_id", "event_idx", "register_date_ts", prod_col] + [c for c in num_cols_seq if c in df_events.columns]

# Avoid using non-serializable objects in lambdas; extract column list outside
num_cols_present = [c for c in num_cols_seq if c in df_events.columns]

def row_to_tuple(r):
    return (
        r["cont_id"],
        {
            "event_idx": int(r["event_idx"]),
            "ts": r["register_date_ts"].isoformat() if r["register_date_ts"] is not None else None,
            "prod": r[prod_col],
            "nums": [r[c] if r[c] is not None else None for c in num_cols_present]
        }
    )

df_rdd = df_events.select(*keep_for_rdd).rdd.map(row_to_tuple)

grouped = df_rdd.groupByKey().mapValues(list)

def build_sequence(events_list):
    events_sorted = sorted(events_list, key=lambda x: x["event_idx"])
    prods = [e["prod"] for e in events_sorted]
    timestamps = [e["ts"] for e in events_sorted]
    nums = [e["nums"] for e in events_sorted]
    return (prods, timestamps, nums)

seqs_rdd = grouped.mapValues(build_sequence).filter(lambda kv: len(kv[1][0]) >= MIN_EVENTS)

def to_row(kv):
    cont_id = kv[0]
    prods, ts, nums = kv[1]
    return (cont_id, prods, ts, nums)

final_rdd = seqs_rdd.map(to_row)

schema = StructType([
    StructField("cont_id", StringType(), False),
    StructField("prod_seq", ArrayType(StringType()), False),
    StructField("ts_seq", ArrayType(StringType()), False),
    StructField("num_seq", ArrayType(ArrayType(StringType())), False)
])

final_rdd_for_df = final_rdd.map(lambda x: (x[0], x[1], x[2], [[str(v) if v is not None else None for v in row] for row in x[3]]))
df_seqs = spark.createDataFrame(final_rdd_for_df, schema)

print("Users with >= {} events: {}".format(MIN_EVENTS, df_seqs.count()))
display(df_seqs.limit(5))

# COMMAND ----------


# CELL 4: build product vocabulary and map sequences to ids
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql import functions as F

# build product vocabulary from DF
prod_counts = df_seqs.selectExpr("explode(prod_seq) as prod").groupBy("prod").count().orderBy(F.desc("count"))
prod_list = [row["prod"] for row in prod_counts.collect()]
print("Unique products in sequences:", len(prod_list))
prod2id = {p:i for i,p in enumerate(prod_list)}
print("Top products:", prod_list[:10])

# Broadcast mapping
sc = spark.sparkContext
bmap = sc.broadcast(prod2id)

# UDF to map product list to id list, truncate to last MAX_SEQ_LEN (most recent)
def map_and_truncate(prod_seq):
    ids = [ bmap.value.get(p, -1) for p in prod_seq ]
    # keep last MAX_SEQ_LEN events (most recent)
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[-MAX_SEQ_LEN:]
    return ids

map_udf = F.udf(map_and_truncate, ArrayType(IntegerType()))
df_seqs = df_seqs.withColumn("seq_prod_ids", map_udf(F.col("prod_seq")))

# also compute seq_len
df_seqs = df_seqs.withColumn("seq_len", F.size("seq_prod_ids"))

# filter again to ensure seq_len >= MIN_EVENTS (after truncation)
df_seqs = df_seqs.filter(F.col("seq_len") >= MIN_EVENTS)

print("After truncation users:", df_seqs.count())
display(df_seqs.select("cont_id","seq_len","seq_prod_ids").limit(5))
# CELL 5: extract label (next product) and align sequences
# We'll use the original prod_seq and ts_seq to get the label index.
from pyspark.sql.types import IntegerType, StringType

# UDF to get label (product id) following the truncated history
def get_label_and_trim(prod_seq, seq_prod_ids):
    # prod_seq: full prod list
    # seq_prod_ids: truncated last N product ids we kept
    # Find where the truncated tail appears in prod_seq (match by product ids)
    # Naive approach: label is the element immediately after the last kept event in the original sequence.
    # We'll compute label_index = len(prod_seq) - len(seq_prod_ids)
    try:
        full_len = len(prod_seq)
        kept_len = len(seq_prod_ids)
        label_index = full_len - kept_len  # this is where the next event is in full sequence
        # next product exists if label_index < full_len - 0 (i.e., there is at least one more element after the kept tail)
        # But we actually want the event immediately AFTER the kept tail, which is at index full_len - 1 (last) + 1 -> invalid.
        # Simpler: we will use the original full prod_seq: the label is the element at index (full_len - kept_len)
        # However safe compute:
        if full_len > kept_len:
            # label is the element at position full_len - kept_len (0-based) ??? revise: example full=[a,b,c,d], kept last 2 -> [c,d], full_len=4 kept_len=2 label is element after d -> none.
            # We actually want to predict the next product after the last item in truncated history, which exists only if there is an element after the truncated last element.
            # To avoid confusion, simpler approach: we choose to make training examples where label = element immediately after the *truncated* history in original sequence.
            # We find index_of_last_kept = len(prod_seq) - 1  (if we kept last K), then label is prod_seq[-1] ??? that's wrong.
            # Correct approach: If we kept last K items from the beginning to some point, the label is the item at position full_len - (K) + ??? This is messy in UDF.
            # Simpler robust approach: create shifted windows earlier in RDD building. To avoid errors, we'll fallback to building training by sliding windows in RDD instead.
            return None
        else:
            return None
    except Exception as e:
        return None

# The above UDF logic is intentionally left incomplete, because we will create training examples more robustly using RDD sliding windows in the next cell.
print('Skipping label UDF; will create training examples using sliding-window RDD approach in next step.')
# CELL 6: sliding-window training examples (RDD)
# Start from the original grouped RDD (before truncation) or reuse grouped from earlier (seqs_rdd)
# We'll re-create grouped for safety from df_events.rdd as (cont_id, sorted_events)

# Build (cont_id, [prod_ids in order])
prod_map = bmap.value  # name from earlier cell

# Build an RDD of (cont_id, [prod_id ints in order])
prod_seq_rdd = df_events.select("cont_id", "event_idx", "product_category").rdd.map(lambda r: (r["cont_id"], (int(r["event_idx"]), r["product_category"]))) \
    .groupByKey().mapValues(lambda evs: sorted(list(evs), key=lambda x: x[0])) \
    .mapValues(lambda evs: [ prod_map.get(p[1], -1) for p in evs ])

# filter by MIN_EVENTS
prod_seq_rdd = prod_seq_rdd.filter(lambda kv: len(kv[1]) >= MIN_EVENTS)

# sliding window generator: emits (cont_id, history_ids, label_id)
def sliding_examples(kv):
    cont_id, seq = kv
    examples = []
    n = len(seq)
    # generate windows where history length from 1..MAX_SEQ_LEN (or choose min)
    # we will produce only windows where label exists (i.e., history_end_index < n-1)
    for end in range(0, n-1):
        # history is seq[max(0, end - MAX_SEQ_LEN + 1) : end+1]
        start = max(0, end - (MAX_SEQ_LEN - 1))
        history = seq[start:end+1]
        label = seq[end+1]
        # optionally skip very short history if you want
        examples.append((cont_id, history, label))
    return examples

examples_rdd = prod_seq_rdd.flatMap(sliding_examples)
# Optionally sample examples if too many
# examples_rdd = examples_rdd.sample(False, 0.2, seed=42)

# Convert to DataFrame with schema (cont_id, history (array<int>), label int)
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, StringType
examples_df = examples_rdd.map(lambda x: (str(x[0]), x[1], int(x[2]))).toDF(["cont_id", "history_ids", "label_id"])

print("Total training examples (sliding windows):", examples_df.count())
display(examples_df.limit(10))
# CELL 7: split and save
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

# For time-based split it's better to split by latest event timestamp; for speed we random-split examples (ok for initial dev)
train_df, val_df, test_df = examples_df.randomSplit([train_frac, val_frac, test_frac], seed=42)

# Persist and save to DBFS parquet
train_df.write.mode("overwrite").parquet(OUTPARQUET + "train")
val_df.write.mode("overwrite").parquet(OUTPARQUET + "val")
test_df.write.mode("overwrite").parquet(OUTPARQUET + "test")

print("Saved train/val/test parquet to:", OUTPARQUET)
print("Train count:", train_df.count(), "Val:", val_df.count(), "Test:", test_df.count())


# COMMAND ----------

from pyspark.sql import functions as F

# static and financial features to merge
feature_cols = [
    "psn_age", "client_seg", "client_seg_1", "aum_band", "channel", "agent_segment",
    "branchoffice_code",
    "wc_total_assets", "wc_assetmix_stocks", "wc_assetmix_bonds", 
    "wc_assetmix_mutual_funds", "wc_assetmix_annuity", "wc_assetmix_deposits",
    "wc_assetmix_other_assets",
    "acct_val_amt", "face_amt", "cash_val_amt"
]

# take last known feature snapshot per customer
from pyspark.sql.window import Window
w = Window.partitionBy("cont_id").orderBy(F.col("register_date_ts").desc())

client_features = (
    df_events
    .withColumn("rn", F.row_number().over(w))
    .filter("rn = 1")
    .select(["cont_id"] + feature_cols)
)

# attach these features to seq training parquet
train = spark.read.parquet("/dbfs/tmp/seqs_parquet/train")
val   = spark.read.parquet("/dbfs/tmp/seqs_parquet/val")
test  = spark.read.parquet("/dbfs/tmp/seqs_parquet/test")

train = train.join(client_features, "cont_id", "left")
val   = val.join(client_features, "cont_id", "left")
test  = test.join(client_features, "cont_id", "left")

train.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/train_aug")
val.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/val_aug")
test.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/test_aug")

print("Merged static + financial features")
print(train.count(), val.count(), test.count())

# COMMAND ----------


# 1A - counts we expect (from earlier)
train_before = spark.read.parquet("/dbfs/tmp/seqs_parquet/train")
val_before   = spark.read.parquet("/dbfs/tmp/seqs_parquet/val")
test_before  = spark.read.parquet("/dbfs/tmp/seqs_parquet/test")

print("Examples BEFORE merge -> train/val/test counts:", train_before.count(), val_before.count(), test_before.count())
print("Unique cont_ids BEFORE ->",
      train_before.select("cont_id").distinct().count(),
      val_before.select("cont_id").distinct().count(),
      test_before.select("cont_id").distinct().count())

# 1B - client feature counts and sample
client_features = (
    df_events
    .withColumn("rn", F.row_number().over(Window.partitionBy("cont_id").orderBy(F.col("register_date_ts").desc())))
    .filter(F.col("rn") == 1)
    .select(["cont_id"] + feature_cols)
)

print("Client features rows:", client_features.count())
print("Unique cont_ids in client_features:", client_features.select("cont_id").distinct().count())
display(client_features.limit(10))

# 2A - inner join count (how many examples have a client_features match)
train_inner = train_before.join(client_features, "cont_id", "inner")
print("Inner join matches (train):", train_inner.count())

# 2B - left-join + count of rows with NULL feature -> these didn't have client_features
train_left = train_before.join(client_features, "cont_id", "left")
null_counts = train_left.filter(F.col(feature_cols[0]).isNull()).count()  # using first feature as proxy
print("Left-join total rows:", train_left.count(), "Rows with NULL static_features:", null_counts)


## 3) Check for duplicate `cont_id` in `client_features` (the dangerous bug)

dup_cf = client_features.groupBy("cont_id").count().filter("count > 1").limit(10)
print("Any duplicated cont_id in client_features? ->", client_features.groupBy("cont_id").count().filter("count > 1").count())
display(dup_cf)

client_features = client_features.dropDuplicates(["cont_id"])
print("After dropDuplicates, client_features rows:", client_features.count())


print("train_before.cont_id type:", [f.dataType for f in train_before.schema.fields if f.name=='cont_id'])
print("client_features.cont_id type:", [f.dataType for f in client_features.schema.fields if f.name=='cont_id'])

train_before = train_before.withColumn("cont_id", F.col("cont_id").cast("string"))
client_features = client_features.withColumn("cont_id", F.col("cont_id").cast("string"))


## 5) Re-run the merge with safe dedupe + left join and verify counts

# enforce string key and unique client_features
train_safe = train_before.withColumn("cont_id", F.col("cont_id").cast("string"))
val_safe   = val_before.withColumn("cont_id", F.col("cont_id").cast("string"))
test_safe  = test_before.withColumn("cont_id", F.col("cont_id").cast("string"))

client_features = client_features.withColumn("cont_id", F.col("cont_id").cast("string")).dropDuplicates(["cont_id"])

# left join (must preserve row counts)
train_merged = train_safe.join(client_features, "cont_id", "left")
val_merged   = val_safe.join(client_features, "cont_id", "left")
test_merged  = test_safe.join(client_features, "cont_id", "left")

print("After merge counts:", train_merged.count(), val_merged.count(), test_merged.count())
print("Expected counts (before merge):", train_safe.count(), val_safe.count(), test_safe.count())


## 6) If you *still* see fewer rows — find the missing cont_ids

train_ids = train_before.select("cont_id").distinct()
merged_ids = train_merged.select("cont_id").distinct()
missing = train_ids.join(merged_ids, "cont_id", "left_anti").limit(20)
print("Sample missing cont_ids (should be zero rows):")
display(missing)
print("Missing count:", train_ids.count() - merged_ids.count())


## 7) Finally: write fixed merged data (only AFTER checks pass)


train_merged.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/train_aug")
val_merged.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/val_aug")
test_merged.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/test_aug")
print("Saved merged files; counts:", train_merged.count(), val_merged.count(), test_merged.count())


# COMMAND ----------


from pyspark.sql.functions import col, when, lit

# For numeric static features → fill zero
numeric_features = [f for f in feature_cols if df_events.schema[f].dataType.simpleString() in ["int", "bigint", "double", "float"]]

# For categorical static features → fill mode (most frequent value)
categorical_features = list(set(feature_cols) - set(numeric_features))
mode_dict = {c: client_features.groupBy(c).count().orderBy("count", ascending=False).first()[0] for c in categorical_features}

train_filled = train_merged
for f in numeric_features:
    train_filled = train_filled.withColumn(f, when(col(f).isNull(), lit(0)).otherwise(col(f)))
for f in categorical_features:
    train_filled = train_filled.withColumn(f, when(col(f).isNull(), lit(mode_dict[f])).otherwise(col(f)))

# quick fill: numeric → 0, string → "UNKNOWN"
train_filled = train_merged.fillna(0).fillna("UNKNOWN")
val_filled = val_merged.fillna(0).fillna("UNKNOWN")
test_filled = test_merged.fillna(0).fillna("UNKNOWN")

# COMMAND ----------

from collections import Counter

def extract_history_features(history, num_classes=7):
    # clean
    hist = [int(x) for x in history if isinstance(x, (int, float))]
    if len(hist) == 0:
        return {
            "num_prior": 0,
            "unique_prior": 0,
            "num_switches": 0,
            "last_1": 0,
            "last_2": 0,
            **{f"freq_{i}":0 for i in range(num_classes)}
        }

    # core stats
    num_prior = len(hist)
    unique_prior = len(set(hist))
    num_switches = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i-1])

    last_1 = hist[-1]
    last_2 = hist[-2] if len(hist) >= 2 else 0

    freq = Counter(hist)
    freq_features = {f"freq_{i}": freq.get(i, 0) for i in range(num_classes)}

    return {
        "num_prior": num_prior,
        "unique_prior": unique_prior,
        "num_switches": num_switches,
        "last_1": last_1,
        "last_2": last_2,
        **freq_features
    }

for df in [train_pd, val_pd, test_pd]:
    extracted = df["history_ids"].apply(extract_history_features)
    expanded = pd.DataFrame(list(extracted))
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, expanded], axis=1)


for df in [train_pd, val_pd, test_pd]:
    df.drop(columns=["history_ids"], inplace=True)


train_pd[['num_prior','unique_prior','num_switches','last_1','last_2']].head()


# COMMAND ----------

train_pd = train_filled.toPandas()
val_pd = val_filled.toPandas()
test_pd = test_filled.toPandas()
import numpy as np

def expand_history(arr, max_len=10):
    arr = arr[-max_len:]  # keep most recent events
    padded = [0] * (max_len - len(arr)) + arr
    return padded

max_len = 10
for df in [train_pd, val_pd, test_pd]:
    expanded = np.vstack(df['history_ids'].apply(lambda x: expand_history(x, max_len)))
    for i in range(max_len):
        df[f'hist_{i}'] = expanded[:, i]

train_pd = train_pd.drop(columns=['history_ids'])
val_pd   = val_pd.drop(columns=['history_ids'])
test_pd  = test_pd.drop(columns=['history_ids'])
label_col = "label_id"

seq_cols = [f"hist_{i}" for i in range(max_len)]
static_cols = [c for c in train_pd.columns if c not in [label_col] + seq_cols + ["cont_id"]]

feature_cols = seq_cols + static_cols
print(len(feature_cols), "features")
import lightgbm as lgb

train_ds = lgb.Dataset(train_pd[feature_cols], label=train_pd[label_col])
val_ds   = lgb.Dataset(val_pd[feature_cols], label=val_pd[label_col])

params = {
    "objective": "multiclass",
    "num_class": 7,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "lambda_l2": 2.0,
    "verbosity": -1
}

model = lgb.train(
    params,
    train_ds,
    valid_sets=[train_ds, val_ds],
    valid_names=["train", "val"],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(50)]
)
from sklearn.metrics import f1_score, accuracy_score

test_pred_prob = model.predict(test_pd[feature_cols])
test_pred = test_pred_prob.argmax(axis=1)

acc = accuracy_score(test_pd[label_col], test_pred)
f1 = f1_score(test_pd[label_col], test_pred, average="weighted")

print("Test Accuracy:", acc)
print("Test F1 Score:", f1)

# COMMAND ----------

import ast

def normalize_history(x):
    # convert None to empty list
    if x is None:
        return []
    # convert scalar (float/int) to list
    if isinstance(x, (int, float)):
        return [int(x)]
    # convert tuple → list
    if isinstance(x, tuple):
        return list(x)
    # convert numpy array → list
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except:
            pass
    # convert nested lists by flattening
    if isinstance(x, list):
        flat = []
        for e in x:
            if isinstance(e, list):
                flat.extend(e)
            else:
                flat.append(e)
        return flat
    # convert string representation of list → real list
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return parsed if isinstance(parsed, list) else [parsed]
        except:
            return []
    # fallback
    return []

for df in [train_pd, val_pd, test_pd]:
    df["history_ids"] = df["history_ids"].apply(normalize_history)


import numpy as np

max_len = 10

def expand_history(arr, max_len=10):
    # remove None and unexpected elements
    arr = [int(x) for x in arr if isinstance(x, (int, float))]
    if len(arr) >= max_len:
        return arr[-max_len:]
    return [0] * (max_len - len(arr)) + arr

for df in [train_pd, val_pd, test_pd]:
    expanded = np.vstack(df['history_ids'].apply(lambda x: expand_history(x, max_len)))
    for i in range(max_len):
        df[f'hist_{i}'] = expanded[:, i]

# remove original column to avoid leakage
for df in [train_pd, val_pd, test_pd]:
    df.drop(columns=["history_ids"], inplace=True)

print(train_pd.filter(regex="hist_").head())
print(val_pd.filter(regex="hist_").head())
print(test_pd.filter(regex="hist_").head())
print("Unique values distribution:", len(set(train_pd['label_id'])))


# COMMAND ----------

# MAGIC %md
# MAGIC