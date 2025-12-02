Examples BEFORE merge -> train/val/test counts: 656383 81650 82371
Unique cont_ids BEFORE -> 265899 69684 70032
Client features rows: 3926686
Unique cont_ids in client_features: 3926686

Inner join matches (train): 656383
Left-join total rows: 656383 Rows with NULL static_features: 5772
Any duplicated cont_id in client_features? -> 0
After dropDuplicates, client_features rows: 3926686
train_before.cont_id type: [StringType()]
client_features.cont_id type: [StringType()]
After merge counts: 656383 81650 82371
Expected counts (before merge): 656383 81650 82371
Sample missing cont_ids (should be zero rows):

Missing count: 0
Saved merged files; counts: 656383 81650 82371


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
