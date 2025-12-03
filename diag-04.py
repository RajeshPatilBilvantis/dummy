Training until validation scores don't improve for 50 rounds
Early stopping, best iteration is:
[478]	train's multi_logloss: 0.59743	val's multi_logloss: 0.678443

Test Accuracy: 0.7457049429920933
Test F1 weighted: 0.7424291660510403
Test F1 macro: 0.5690433376747243
Classification report:
              precision    recall  f1-score   support

           0       0.72      0.56      0.63       371
           1       0.00      0.00      0.00        29
           2       0.73      0.74      0.74     15251
           3       0.76      0.77      0.77     13855
           4       0.74      0.68      0.71     10966
           5       0.58      0.26      0.36      1227
           6       0.75      0.82      0.79     16100

    accuracy                           0.75     57799
   macro avg       0.61      0.55      0.57     57799
weighted avg       0.74      0.75      0.74     57799

Confusion matrix shape: (7, 7)
Done.


# ------------- PARAMETERS -------------
SAMPLE_FRACTION = 0.2   # set None to use full data (be careful with memory)
MIN_EVENTS = 2          # minimum number of prior events to produce an example (2 => at least 1 history item + label)
MAX_SEQ_LEN = 10        # history length used for padded features
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
RANDOM_SEED = 42

# LightGBM training params
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 6,               # we'll set dynamically later
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "lambda_l2": 2.0,
    "verbosity": -1
}
NUM_BOOST_ROUND = 2000
EARLY_STOP = 50

# ------------- IMPORTS -------------
from pyspark.sql import functions as F, Window
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType
import numpy as np
import pandas as pd
from collections import Counter
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# ------------- 1) Load and optional sample -------------
# Assumes df_raw exists (you had populated df_raw earlier). If not, re-load:
# df_raw = spark.table("dl_tenants_daas.us_wealth_management.wealth_management_client_metrics")

# Keep only rows with product_category already populated
df_events = df_raw.select("cont_id", "product_category", "register_date",
                          "acct_val_amt","face_amt","cash_val_amt","wc_total_assets",
                          "wc_assetmix_stocks","wc_assetmix_bonds","wc_assetmix_mutual_funds",
                          "wc_assetmix_annuity","wc_assetmix_deposits","wc_assetmix_other_assets",
                          "psn_age","client_seg","client_seg_1","aum_band","channel","agent_segment",
                          "branchoffice_code","policy_status"
                         ).filter(
    (F.col("cont_id").isNotNull()) &
    (F.col("register_date").isNotNull()) &
    (F.col("product_category").isNotNull())
)

if SAMPLE_FRACTION is not None:
    print("Sampling fraction:", SAMPLE_FRACTION)
    df_events = df_events.sample(withReplacement=False, fraction=float(SAMPLE_FRACTION), seed=RANDOM_SEED)

print("Event rows (approx):", df_events.count())

# ------------- 2) Keep only Active policies (optional but recommended) -------------
df_events = df_events.filter(F.col("policy_status") == "Active")

# ------------- 3) convert and order events per user -------------
df_events = df_events.withColumn("register_ts", F.to_timestamp("register_date"))
w = Window.partitionBy("cont_id").orderBy("register_ts")
df_events = df_events.withColumn("event_idx", F.row_number().over(w))

# ------------- 4) Build full product vocabulary from the ENTIRE (sampled) df_events -------------
prod_list = df_events.select("product_category").distinct().rdd.map(lambda r: r[0]).collect()
prod_list = sorted([p for p in prod_list if p is not None])
prod2id = {p: i+1 for i, p in enumerate(prod_list)}   # start ids at 1; reserve 0 for padding
id2prod = {v:k for k,v in prod2id.items()}
NUM_CLASSES = len(prod2id)
print("Vocabulary size (classes):", NUM_CLASSES)
LGB_PARAMS["num_class"] = NUM_CLASSES + 1   # +1 if you want to reserve 0? We'll keep labels in 1..N

# ------------- 5) Build grouped sequences RDD and dedupe consecutive -------------
# Create RDD of (cont_id, (event_idx, product_category))
rdd = df_events.select("cont_id","event_idx","product_category").rdd.map(lambda r: (r["cont_id"], (int(r["event_idx"]), r["product_category"])))

# group by cont_id
grouped = rdd.groupByKey().mapValues(lambda evs: [p for _, p in sorted(list(evs), key=lambda x: x[0])])

# remove consecutive duplicates (keeps only transitions)
def dedupe_consecutive(seq):
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out

grouped = grouped.mapValues(dedupe_consecutive).filter(lambda kv: len(kv[1]) >= MIN_EVENTS)

print("Users with >= MIN_EVENTS (after dedupe):", grouped.count())

# ------------- 6) Sliding-window training example generation -------------
def make_examples(kv):
    cont_id, seq = kv
    # map to ids, unknown -> 0 (shouldn't happen because vocab built from df_events)
    seq_ids = [prod2id.get(x, 0) for x in seq]
    n = len(seq_ids)
    samples = []
    for i in range(1, n):   # i is index of label in seq_ids
        history = seq_ids[max(0, i - MAX_SEQ_LEN): i]   # history (most recent up to MAX_SEQ_LEN)
        label = seq_ids[i]
        if len(history) >= 1:   # history length >= 1
            samples.append((cont_id, history, label))
    return samples

examples_rdd = grouped.flatMap(make_examples)

# OPTIONAL: reduce volume by sampling examples (if still huge); comment out if you want full
# examples_rdd = examples_rdd.sample(False, 1.0, seed=RANDOM_SEED)

# Convert to DataFrame
schema = StructType([
    StructField("cont_id", StringType(), True),
    StructField("hist_seq", ArrayType(IntegerType()), True),
    StructField("label", IntegerType(), True),
])
examples_df = spark.createDataFrame(examples_rdd, schema).cache()

print("Total training examples:", examples_df.count())
display(examples_df.limit(10))

# ------------- 7) Create tabular history-derived features (in Spark) -------------
# We'll compute: seq_len, last_1, last_2, num_prior, unique_prior, num_switches, freq for each product id (sparse)
def history_features_udf(hist):
    # build summary stats as dict: we will compute in Python side for quicker dev, but create columns in Spark later
    return None

# easier: convert RDD -> DataFrame by mapping to tuples of features (done in Python for flexibility)
def hist_to_features_row(x):
    cont_id, hist, label = x
    seq_len = len(hist)
    last_1 = hist[-1] if seq_len >= 1 else 0
    last_2 = hist[-2] if seq_len >= 2 else 0
    unique_prior = len(set(hist))
    num_switches = sum(1 for i in range(1, seq_len) if hist[i] != hist[i-1])
    freq = Counter(hist)
    freq_features = [freq.get(i, 0) for i in range(1, NUM_CLASSES+1)]  # product ids are 1..NUM_CLASSES
    return (str(cont_id), hist, label, seq_len, last_1, last_2, unique_prior, num_switches, freq_features)

# Because we will convert to Pandas, do a mapPartitions to Python to build features and then to Spark DF
sampled = examples_rdd  # rename
# WARNING: converting huge RDD to driver is expensive. We'll convert partitions to rows and then to Spark DF
rows_rdd = sampled.map(hist_to_features_row)

# define schema for features DF
from pyspark.sql.types import ArrayType, LongType
feat_schema = StructType([
    StructField("cont_id", StringType(), True),
    StructField("hist_seq", ArrayType(IntegerType()), True),
    StructField("label", IntegerType(), True),
    StructField("seq_len", IntegerType(), True),
    StructField("last_1", IntegerType(), True),
    StructField("last_2", IntegerType(), True),
    StructField("unique_prior", IntegerType(), True),
    StructField("num_switches", IntegerType(), True),
    StructField("freq_list", ArrayType(IntegerType()), True),
])

examples_feats_df = spark.createDataFrame(rows_rdd, feat_schema).cache()
print("Examples with features:", examples_feats_df.count())
display(examples_feats_df.limit(10))

# ------------- 8) Expand freq_list into separate columns (Spark) -------------
# create columns freq_1 ... freq_N
for i in range(1, NUM_CLASSES+1):
    examples_feats_df = examples_feats_df.withColumn(f"freq_{i}", F.col("freq_list")[i-1])
# drop freq_list
examples_feats_df = examples_feats_df.drop("freq_list")

# ------------- 9) Build last snapshot static features per user and join -------------
# last known static snapshot from df_events (we already had these cols in df_events)
w2 = Window.partitionBy("cont_id").orderBy(F.col("register_ts").desc())
client_snapshot = (df_events
                   .withColumn("rn", F.row_number().over(w2))
                   .filter(F.col("rn") == 1)
                   .select("cont_id",
                           "acct_val_amt","face_amt","cash_val_amt","wc_total_assets",
                           "wc_assetmix_stocks","wc_assetmix_bonds","wc_assetmix_mutual_funds",
                           "wc_assetmix_annuity","wc_assetmix_deposits","wc_assetmix_other_assets",
                           "psn_age","client_seg","client_seg_1","aum_band","channel","agent_segment","branchoffice_code"))

# Left join - preserve all examples
examples_full = examples_feats_df.join(client_snapshot, on="cont_id", how="left")

print("Examples after join:", examples_full.count())

# ------------- 10) Fill missing values sensibly -------------
# numeric cols to fill 0
numeric_cols = [c for c, t in examples_full.dtypes if t in ("int", "double", "bigint", "float") and c not in ("label","seq_len","last_1","last_2","unique_prior","num_switches")]
# we will fill numeric nulls with 0
fill_dict = {c: 0 for c in numeric_cols}

# categorical modes
categorical_cols = ["client_seg","client_seg_1","aum_band","channel","agent_segment","branchoffice_code"]
modes = {}
for c in categorical_cols:
    try:
        m = examples_full.groupBy(c).count().orderBy(F.desc("count")).first()[0]
        modes[c] = m if m is not None else "UNKNOWN"
    except:
        modes[c] = "UNKNOWN"

examples_full = examples_full.fillna(fill_dict)
for c in categorical_cols:
    examples_full = examples_full.withColumn(c, F.when(F.col(c).isNull(), F.lit(modes[c])).otherwise(F.col(c)))

# ------------- 11) Convert hist_seq to fixed-length padded columns (for LightGBM tabular) -------------
def pad_history(hist):
    arr = hist[-MAX_SEQ_LEN:] if hist is not None else []
    pad_len = MAX_SEQ_LEN - len(arr)
    return [0]*pad_len + arr

pad_udf = F.udf(lambda x: pad_history(x), ArrayType(IntegerType()))
examples_full = examples_full.withColumn("hist_padded", pad_udf(F.col("hist_seq")))

# expand hist_padded into hist_0 ... hist_{MAX_SEQ_LEN-1}
for i in range(MAX_SEQ_LEN):
    examples_full = examples_full.withColumn(f"hist_{i}", F.col("hist_padded")[i])

examples_full = examples_full.drop("hist_seq", "hist_padded")

# ------------- 12) Final column list for modeling -------------
# label = 'label' (1..N)
model_feature_cols = [f"hist_{i}" for i in range(MAX_SEQ_LEN)] + \
                     ["seq_len","last_1","last_2","unique_prior","num_switches"] + \
                     [f"freq_{i}" for i in range(1, NUM_CLASSES+1)] + \
                     ["acct_val_amt","face_amt","cash_val_amt","wc_total_assets",
                      "wc_assetmix_stocks","wc_assetmix_bonds","wc_assetmix_mutual_funds",
                      "wc_assetmix_annuity","wc_assetmix_deposits","wc_assetmix_other_assets",
                      "psn_age"]
# convert categorical strings to index using simple string->index map (lightgbm accepts categorical as int)
# create mapping for categorical_cols
for c in categorical_cols:
    vals = [r[0] for r in examples_full.select(c).distinct().collect()]
    m = {v:i for i,v in enumerate(sorted([str(x) for x in vals]))}
    b = spark.sparkContext.broadcast(m)
    examples_full = examples_full.withColumn(c + "_idx", F.coalesce(F.col(c).cast("string"), F.lit("UNKNOWN")))
    # NOTE: converting to integer index using udf
    examples_full = examples_full.withColumn(c + "_idx", F.udf(lambda s: int(b.value.get(str(s), 0)), IntegerType())(F.col(c + "_idx")))
    model_feature_cols.append(c + "_idx")

# Ensure label values are in 1..NUM_CLASSES
# If needed remap label ranges (they already are product ids 1..NUM_CLASSES)
examples_full = examples_full.withColumn("label", F.col("label").cast(IntegerType()))

# ------------- 13) Split train/val/test (random) and convert to Pandas -------------
train_spark, val_spark, test_spark = examples_full.randomSplit([TRAIN_FRAC, VAL_FRAC, TEST_FRAC], seed=RANDOM_SEED)

print("Train / Val / Test counts:", train_spark.count(), val_spark.count(), test_spark.count())

# persist to speed up conversion
train_spark = train_spark.cache()
val_spark = val_spark.cache()
test_spark = test_spark.cache()

train_pd = train_spark.select(["cont_id","label"] + model_feature_cols).toPandas()
val_pd   = val_spark.select(["cont_id","label"] + model_feature_cols).toPandas()
test_pd  = test_spark.select(["cont_id","label"] + model_feature_cols).toPandas()

# ------------- 14) Final data sanity & fillna in pandas -------------
train_pd.fillna(0, inplace=True)
val_pd.fillna(0, inplace=True)
test_pd.fillna(0, inplace=True)

# Ensure label is zero-based for LightGBM (optional) -- we'll make labels 0..K-1
label_map = {lab: i for i, lab in enumerate(sorted(train_pd["label"].unique()))}
train_pd["label0"] = train_pd["label"].map(label_map)
val_pd["label0"] = val_pd["label"].map(lambda x: label_map.get(x, 0))
test_pd["label0"] = test_pd["label"].map(lambda x: label_map.get(x, 0))

# Update params num_class to exact count
num_classes = len(label_map)
LGB_PARAMS["num_class"] = num_classes

print("Num classes:", num_classes)
print("Train shape:", train_pd.shape, "Val shape:", val_pd.shape, "Test shape:", test_pd.shape)

# ------------- 15) LightGBM training -------------
feature_cols_final = model_feature_cols  # order as defined
train_ds = lgb.Dataset(train_pd[feature_cols_final], label=train_pd["label0"])
val_ds = lgb.Dataset(val_pd[feature_cols_final], label=val_pd["label0"], reference=train_ds)

model = lgb.train(
    LGB_PARAMS,
    train_ds,
    valid_sets=[train_ds, val_ds],
    valid_names=["train","val"],
    num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOP,
    verbose_eval=50
)

# ------------- 16) Evaluation -------------
test_pred_prob = model.predict(test_pd[feature_cols_final])
test_pred = np.argmax(test_pred_prob, axis=1)

acc = accuracy_score(test_pd["label0"], test_pred)
f1_weighted = f1_score(test_pd["label0"], test_pred, average="weighted")
f1_macro = f1_score(test_pd["label0"], test_pred, average="macro")
print("Test Accuracy:", acc)
print("Test F1 weighted:", f1_weighted)
print("Test F1 macro:", f1_macro)
print("Classification report:")
print(classification_report(test_pd["label0"], test_pred))

cm = confusion_matrix(test_pd["label0"], test_pred)
print("Confusion matrix shape:", cm.shape)

# ------------- 17) Save model (optional) -------------
# model.save_model("/dbfs/tmp/next_product_lgb.txt")
print("Done.")
