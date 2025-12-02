Good catch — let’s debug this cleanly and fix it. I’ll be blunt: the code *shouldn’t* have dropped examples after a **left** join — so something subtle is wrong (type mismatch, duplicate keys in `client_features`, or you accidentally used a different DataFrame variable earlier). Below I give you a short, precise checklist and **exact Spark cells** to run now to find and fix the problem.

Run these cells *in order* and paste outputs if you want me to read them — but the fixes after the checks will almost certainly resolve the loss.

---

## 1) Quick verification: counts & distinct keys

This tells us whether the train/val/test splits lost examples before the join, and how many unique `cont_id`s are in each set vs the client features.

```python
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
```

---

## 2) Find how many training examples have matching client_features

This shows how many examples actually find a match. A left join should keep all examples; this count is for investigation of missing/NULL feature rows.

```python
# 2A - inner join count (how many examples have a client_features match)
train_inner = train_before.join(client_features, "cont_id", "inner")
print("Inner join matches (train):", train_inner.count())

# 2B - left-join + count of rows with NULL feature -> these didn't have client_features
train_left = train_before.join(client_features, "cont_id", "left")
null_counts = train_left.filter(F.col(feature_cols[0]).isNull()).count()  # using first feature as proxy
print("Left-join total rows:", train_left.count(), "Rows with NULL static_features:", null_counts)
```

If `train_left.count()` < `train_before.count()` — that’s impossible for a left join (it must be equal). If that happens, you previously did a different join or overwritten `train_before`. But likely `train_left.count() == train_before.count()`. The drop you observed likely arose earlier (maybe you measured different variables). Still run this to confirm.

---

## 3) Check for duplicate `cont_id` in `client_features` (the dangerous bug)

If `client_features` has **multiple rows per cont_id**, a join can *explode* rows (increase) — not reduce. But sometimes duplicates + subsequent writes/overwrites produce differences. Let's check:

```python
dup_cf = client_features.groupBy("cont_id").count().filter("count > 1").limit(10)
print("Any duplicated cont_id in client_features? ->", client_features.groupBy("cont_id").count().filter("count > 1").count())
display(dup_cf)
```

If duplicates exist, deduplicate:

```python
client_features = client_features.dropDuplicates(["cont_id"])
print("After dropDuplicates, client_features rows:", client_features.count())
```

---

## 4) Check cont_id types (string vs numeric) — must match exactly

Type mismatch can cause subtle behavior depending on how you created cont_id earlier.

```python
print("train_before.cont_id type:", [f.dataType for f in train_before.schema.fields if f.name=='cont_id'])
print("client_features.cont_id type:", [f.dataType for f in client_features.schema.fields if f.name=='cont_id'])
```

If types differ (e.g., `StringType` vs `LongType`), coerce both to string **explicitly** before join:

```python
train_before = train_before.withColumn("cont_id", F.col("cont_id").cast("string"))
client_features = client_features.withColumn("cont_id", F.col("cont_id").cast("string"))
```

---

## 5) Re-run the merge with safe dedupe + left join and verify counts

Now do a safe merge and ensure counts are preserved:

```python
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
```

If `train_merged.count()` equals `train_safe.count()`, you’re good — **no rows were dropped**. If not equal, stop and paste both outputs.

---

## 6) If you *still* see fewer rows — find the missing cont_ids

List a few cont_ids that were in `train_before` but not in `train_merged` (shouldn’t happen for left join — but run to be sure):

```python
train_ids = train_before.select("cont_id").distinct()
merged_ids = train_merged.select("cont_id").distinct()
missing = train_ids.join(merged_ids, "cont_id", "left_anti").limit(20)
print("Sample missing cont_ids (should be zero rows):")
display(missing)
print("Missing count:", train_ids.count() - merged_ids.count())
```

---

## 7) Finally: write fixed merged data (only AFTER checks pass)

When everything is verified, write the merged datasets:

```python
train_merged.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/train_aug")
val_merged.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/val_aug")
test_merged.write.mode("overwrite").parquet("/dbfs/tmp/seqs_parquet/test_aug")
print("Saved merged files; counts:", train_merged.count(), val_merged.count(), test_merged.count())
```

---

## Likely root-cause and quick fix (what I expect)

Based on your earlier outputs and common mistakes, probable causes — in order:

1. `client_features` had `cont_id` typed as `LongType` while `examples_df` cont_id values are strings (you created cont_id as `str(x[0])` earlier). Coercing both to string and re-joining fixes the mismatch.
2. `client_features` may have been accidentally created from a sampled `df_events` while train/val/test came from full examples — causing missing keys. Ensure `client_features` derived from same sampled `df_events`.
3. Some cont_ids in the earlier examples contain trailing/leading spaces or different formatting (use `F.trim()` before join).
4. Rare duplicate entries in `client_features` due to identical timestamps — drop duplicates using `.dropDuplicates(["cont_id"])`.

Do the checks above; **most likely** casting `cont_id` to string on both sides and `.dropDuplicates(["cont_id"])` resolves the issue.

---

Run the diagnostic cells now and paste the outputs (or tell me which step failed). I’ll analyze the exact output and give the precise one-line fix.
