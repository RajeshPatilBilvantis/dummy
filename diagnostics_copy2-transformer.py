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

[5 rows x 10 columns]
Unique values distribution: 6
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

ValueError: operands could not be broadcast together with shapes (8,) (2,) 
File <command-6604191847308488>, line 13
     11 max_len = 10
     12 for df in [train_pd, val_pd, test_pd]:
---> 13     expanded = np.vstack(df['history_ids'].apply(lambda x: expand_history(x, max_len)))
     14     for i in range(max_len):
     15         df[f'hist_{i}'] = expanded[:, i]
File <command-6604191847308488>, line 8, in expand_history(arr, max_len)
      6 def expand_history(arr, max_len=10):
      7     arr = arr[-max_len:]  # keep most recent events
----> 8     padded = [0] * (max_len - len(arr)) + arr
      9     return padded
