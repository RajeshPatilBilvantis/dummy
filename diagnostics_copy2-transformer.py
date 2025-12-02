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
    early_stopping_rounds=50
)
from sklearn.metrics import f1_score, accuracy_score

test_pred_prob = model.predict(test_pd[feature_cols])
test_pred = test_pred_prob.argmax(axis=1)

acc = accuracy_score(test_pd[label_col], test_pred)
f1 = f1_score(test_pd[label_col], test_pred, average="weighted")

print("Test Accuracy:", acc)
print("Test F1 Score:", f1)
