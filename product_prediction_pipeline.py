# product_prediction_pipeline.py
import os
import gc
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from collections import defaultdict

# ---------- SETTINGS ----------
DATA_PATH = "your_data.csv"   # <- change if needed
TARGET_COL = "second_product_category"
CUR_COL = "product_category"
USER_COL = "axa_party_id"
TS_COL = "register_date"
N_FOLDS = 5
RANDOM_STATE = 42
# output
OUT_DIR = "model_output"
os.makedirs(OUT_DIR, exist_ok=True)
# --------------------------------

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=[TS_COL])
print("Raw rows:", len(df))

# Filter to users known to buy second product (your binary model decides who goes here)
df = df[df[TARGET_COL].notna()].reset_index(drop=True)
print("Rows with target (modeling set):", len(df))

# DROP leakage / future columns
drop_prefixes = ['second_', 'years_to_second']
cols_to_drop = [c for c in df.columns if any(c.lower().startswith(p) for p in drop_prefixes)]
# also drop policy ids
for c in ['policy_no', 'second_policy_no']:
    if c in df.columns:
        cols_to_drop.append(c)
cols_to_drop = list(set(cols_to_drop))
print("Dropping leakage cols (sample):", cols_to_drop[:20])
df = df.drop(columns=cols_to_drop, errors='ignore')

# Fill minimal obvious NaNs
df[CUR_COL] = df[CUR_COL].astype(str).fillna("MISSING")
df[TARGET_COL] = df[TARGET_COL].astype(str).fillna("MISSING")

# ---------- Feature engineering ----------
print("Feature engineering...")

# 1) Time features from register_date
df['reg_year'] = df[TS_COL].dt.year
df['reg_month'] = df[TS_COL].dt.month
df['reg_day'] = df[TS_COL].dt.day
df['reg_dayofweek'] = df[TS_COL].dt.dayofweek
df['reg_yymm'] = df['reg_year'].astype(str) + "_" + df['reg_month'].astype(str)

# 2) Age related
if 'psn_age' in df.columns:
    df['age'] = pd.to_numeric(df['psn_age'], errors='coerce').fillna(-1)
else:
    df['age'] = -1
# bucketing
df['age_bucket'] = pd.cut(df['age'].replace(-1, np.nan),
                          bins=[0,25,35,45,55,65,120], labels=['<25','25-34','35-44','45-54','55-64','65+']).astype(str).fillna('MISSING')

# 3) Allocation ratio features and combos - numeric
alloc_cols = [c for c in df.columns if 'allocation_ratio' in c or c in ['acct_val_amt','face_amt','cash_val_amt','wc_total_assets']]
numeric_allocs = []
for c in alloc_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-999)
        numeric_allocs.append(c)
print("Numeric alloc/features:", numeric_allocs)

# PCA on allocation numeric group if size > 1
if len(numeric_allocs) >= 2:
    pca = PCA(n_components=min(3, len(numeric_allocs)), random_state=RANDOM_STATE)
    try:
        pca_feats = pca.fit_transform(df[numeric_allocs].values)
        for i in range(pca_feats.shape[1]):
            df[f'alloc_pca_{i}'] = pca_feats[:, i]
    except Exception as e:
        print("PCA failed:", e)

# 4) Ratios / interactions
if 'face_amt' in df.columns and 'acct_val_amt' in df.columns:
    df['face_to_acct'] = df['face_amt'].replace(-999,0) / (df['acct_val_amt'].replace(-999,0) + 1)
else:
    df['face_to_acct'] = 0.0

# 5) Agent / branch / product aggregates (out-of-fold target encoding below)
# we'll compute folds and do oof target encoding to avoid leakage

# 6) product-level transition probabilities P(next|current) built globally (useful)
print("Building P(next|current) transitions (global)...")
cur_next_counts = df.groupby([CUR_COL, TARGET_COL]).size().unstack(fill_value=0)
cur_next_probs = cur_next_counts.div(cur_next_counts.sum(axis=1).replace(0,1), axis=0)
# store top-3 next products per current
top_nexts = {}
for cur in cur_next_probs.index:
    sorted_nexts = cur_next_probs.loc[cur].sort_values(ascending=False)
    top_nexts[cur] = list(sorted_nexts.index[:3])

# Add feature: probability of most-likely next given current
df['most_likely_next'] = df[CUR_COL].map(lambda x: top_nexts.get(x, [None])[0])
# add binary whether current==most-likely-next
df['cur_eq_mln'] = (df['most_likely_next'] == df[TARGET_COL]).astype(int)

# 7) Basic categorical list for encoding
cat_cols = [
    CUR_COL, 'prod_lob', 'sub_product_level_1', 'sub_product_level_2',
    'client_seg', 'client_seg_1', 'aum_band', 'branchoffice_code', 'agt_no',
    'division_name', 'mkt_prod_hier', 'policy_status', 'channel', 'agent_segment',
    'season_of_first_policy', 'age_bucket'
]
cat_cols = [c for c in cat_cols if c in df.columns]

# 8) Convert remaining NaNs for cats and numerics
for c in cat_cols:
    df[c] = df[c].astype(str).fillna('MISSING')

num_cols = [c for c in df.columns if c not in cat_cols + [TARGET_COL, USER_COL, TS_COL, 'reg_yymm', 'reg_year', 'reg_month','reg_day','reg_dayofweek','most_likely_next'] and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
# keep explicit numeric list optionally
num_cols = [c for c in num_cols if c not in [USER_COL, TS_COL, TARGET_COL]]
# fill numna
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-999)

print("Categorical features used:", cat_cols)
print("Numeric features used:", num_cols)

# ---------- Label encoding target ----------
le_target = LabelEncoder()
df['y'] = le_target.fit_transform(df[TARGET_COL])
labels = list(le_target.classes_)
print("Labels:", labels)

# ---------- Out-of-fold target encoding for agent/branch/current product ----------
# We'll do KFold target encoding to avoid leakage.
def oof_target_encoding(df, col, target, n_splits=N_FOLDS, seed=RANDOM_STATE, min_samples_leaf=100, smoothing=10):
    """Return column name with oof target target-encoded (prob vector per class) and mapping"""
    print("OOF target encoding:", col)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=float)
    global_mean = df[target].mean()
    new_col = f"{col}_te"
    df[new_col] = np.nan
    # For simplicity we'll encode to probability of each class being the target - but because target multi-class,
    # we compute per-class probability and store as separate features; here we do encode to most frequent target probability
    # Simpler: encode to distribution of next-product among column groups -> store top-1 prob and entropy
    group = df.groupby(col)[target].agg(['count'])
    # do oof per fold
    for train_idx, val_idx in skf.split(df, df[target]):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        stats = train.groupby(col)[target].value_counts(normalize=True).rename("p").reset_index()
        # compute top-prob and entropy per group
        stats_top = stats.groupby(col).agg({'p': ['max', lambda x: - (x*np.log(x+1e-9)).sum()]})
        stats_top.columns = ['p_max','entropy']
        stats_top = stats_top.reset_index()
        mapping = stats_top.set_index(col).to_dict(orient='index')
        # map to val
        mapped = val[col].map(lambda x: mapping.get(x, {'p_max':global_mean, 'entropy':0.0}))
        df.loc[val.index, new_col] = [m['p_max'] for m in mapped]
    # fill na
    df[new_col] = df[new_col].fillna(global_mean)
    return new_col

# Apply OOF TE on agent & branch & current product
te_cols = []
for c in ['agt_no', 'branchoffice_code', CUR_COL]:
    if c in df.columns:
        te_col = oof_target_encoding(df, c, 'y', n_splits=N_FOLDS)
        te_cols.append(te_col)

# ---------- Additional group statistics (global, not oof — acceptable for features like popularity) ----------
# global product popularity by month (may leak if month is in future; but we use entire dataset for these static features)
prod_pop = df.groupby(CUR_COL).size().to_dict()
df['cur_prod_pop'] = df[CUR_COL].map(lambda x: prod_pop.get(x,0))
df['cur_prod_pop_norm'] = df['cur_prod_pop'] / df['cur_prod_pop'].max()

# agent-level distribution features (global)
if 'agt_no' in df.columns:
    agent_counts = df.groupby('agt_no').size().to_dict()
    df['agt_event_count'] = df['agt_no'].map(lambda x: agent_counts.get(x,0))

# ---------- Prepare features list ----------
feature_cols = []
# numeric:
feature_cols += num_cols
feature_cols += [c for c in df.columns if c.startswith('alloc_pca_')]
feature_cols += ['face_to_acct', 'cur_prod_pop_norm', 'agt_event_count', 'cur_eq_mln']
# target encoded cols:
feature_cols += te_cols
# categorical (as label-encoded ints)
cat_to_labelencode = []
for c in cat_cols:
    if c not in [CUR_COL]:
        cat_to_labelencode.append(c)
# We'll label-encode these
for c in cat_to_labelencode:
    df[c] = df[c].astype(str).fillna('MISSING')
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    feature_cols.append(c)

# label-encode CUR_COL as well (we also have te for it)
df[CUR_COL+'_le'] = LabelEncoder().fit_transform(df[CUR_COL].astype(str))
feature_cols.append(CUR_COL+'_le')

# add numeric TE columns to features
feature_cols += [c for c in te_cols]

# ensure unique
feature_cols = list(dict.fromkeys(feature_cols))
print("Final features count:", len(feature_cols))

# ---------- Cross-validated training: LightGBM + CatBoost ensemble ----------
print("Starting Stratified KFold training")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(df), len(labels)))
test_preds_avg = None

fold = 0
for train_idx, val_idx in skf.split(df, df['y']):
    fold += 1
    print(f"\nFOLD {fold}")
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]['y']
    X_val = df.iloc[val_idx][feature_cols]
    y_val = df.iloc[val_idx]['y']

    # LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    params = {
        "objective": "multiclass",
        "num_class": len(labels),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "verbosity": -1,
        "seed": RANDOM_STATE + fold
    }
    bst = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_val], early_stopping_rounds=80, verbose_eval=200)
    lgb_val_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    # store
    oof_preds[val_idx] += lgb_val_pred * 0.6  # weight lgb 0.6

    # CatBoost (on same features) -- faster with categorical handling, but we've already label-encoded cats
    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE + fold,
        loss_function='MultiClass',
        verbose=False
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    cb_val_pred = cb_model.predict_proba(X_val)
    oof_preds[val_idx] += cb_val_pred * 0.4  # weight catboost 0.4

    # metrics for this fold
    val_pred_labels = np.argmax(oof_preds[val_idx], axis=1)
    fold_macro = f1_score(y_val, val_pred_labels, average='macro')
    print(f"Fold {fold} macro F1 (ensemble on val): {fold_macro:.4f}")
    # save models
    bst.save_model(os.path.join(OUT_DIR, f"lgb_fold{fold}.txt"))
    cb_model.save_model(os.path.join(OUT_DIR, f"catb_fold{fold}.cbm"))

# OOF metrics
oof_preds_labels = np.argmax(oof_preds, axis=1)
oof_macro = f1_score(df['y'], oof_preds_labels, average='macro')
print("\nOOF Macro F1:", oof_macro)
print("\nPer-class report:")
print(classification_report(df['y'], oof_preds_labels, target_names=labels))

# confusion matrix
cm = confusion_matrix(df['y'], oof_preds_labels)
print("Confusion matrix (rows=true, cols=pred):\n", cm)
pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

# Feature importance - use last LGB model's importance (and approximate CatBoost via shap if needed)
try:
    imp = bst.feature_importance(importance_type='gain')
    fi_df = pd.DataFrame({"feature": feature_cols, "gain": imp})
    fi_df = fi_df.sort_values("gain", ascending=False)
    fi_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    print("Saved feature importance.")
except Exception as e:
    print("Feature importance save failed:", e)

# Save OOF predictions and mapping
oof_df = pd.DataFrame(oof_preds, columns=[f"prob_{c}" for c in labels])
oof_df['y_true'] = df['y'].values
oof_df['y_true_label'] = df[TARGET_COL].values
oof_df['y_pred'] = oof_preds_labels
oof_df['y_pred_label'] = [labels[i] for i in oof_preds_labels]
oof_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

# Save summary
summary = {
    "oof_macro_f1": float(oof_macro),
    "labels": labels,
    "rows": int(len(df)),
    "features_count": len(feature_cols)
}
with open(os.path.join(OUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Output saved to", OUT_DIR)
