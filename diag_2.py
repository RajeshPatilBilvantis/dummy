# candidate_ranking_model.py
import os, gc, json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb

# ---------- SETTINGS ----------
DATA_PATH = "your_data.csv"   # <- change if needed
OUT_DIR = "ranker_output"
os.makedirs(OUT_DIR, exist_ok=True)
TARGET_COL = "second_product_category"
CUR_COL = "product_category"
USER_COL = "axa_party_id"
TS_COL = "register_date"
SEED = 42
N_FOLDS = 5
CAND_K = 5   # number of transition candidates; popularity candidates added too
NEG_SAMPLING = None  # set None to use all negative candidates for each row (small here since classes=6)
# -----------------------------

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=[TS_COL])
print("Total rows:", len(df))

# filter to rows with a real second product (we model conditional problem)
df = df[df[TARGET_COL].notna()].reset_index(drop=True)
print("Rows with target (modeling set):", len(df))

# drop obvious leakage/future columns
drop_prefixes = ['second_', 'years_to_second']
cols_to_drop = [c for c in df.columns if any(c.lower().startswith(p) for p in drop_prefixes)]
for c in ['policy_no', 'second_policy_no']:
    if c in df.columns:
        cols_to_drop.append(c)
cols_to_drop = list(set(cols_to_drop))
print("Dropping leakage columns sample:", cols_to_drop[:20])
df = df.drop(columns=cols_to_drop, errors='ignore')

# basic cleanup / types
df[CUR_COL] = df[CUR_COL].astype(str).fillna("MISSING")
df[TARGET_COL] = df[TARGET_COL].astype(str).fillna("MISSING")
if TS_COL in df.columns:
    df = df.sort_values(TS_COL).reset_index(drop=True)

# label encoders for product classes
le_prod = LabelEncoder()
all_products = list(pd.unique(df[CUR_COL].tolist() + df[TARGET_COL].tolist()))
le_prod.fit(all_products)
df['cur_le'] = le_prod.transform(df[CUR_COL])
df['target_le'] = le_prod.transform(df[TARGET_COL])
product_list = list(le_prod.classes_)
n_products = len(product_list)
print("Unique products:", n_products)

# ----- build transition probabilities P(next | current) globally -----
print("Building transition counts P(next|current)...")
trans_counts = defaultdict(Counter)
for cur, nxt in zip(df[CUR_COL], df[TARGET_COL]):
    trans_counts[cur][nxt] += 1
trans_probs = {}
for cur, ctr in trans_counts.items():
    total = float(sum(ctr.values()))
    probs = {k: v/total for k, v in ctr.items()}
    # sort by prob desc
    trans_probs[cur] = sorted(probs.items(), key=lambda x: -x[1])

# global popularity ranking for fallback candidates
global_counts = Counter(df[TARGET_COL].tolist())
global_rank = [p for p,_ in global_counts.most_common()]

# ----- candidate generation per row -----
def generate_candidates_for_row(cur_prod, k=CAND_K):
    cands = []
    # top-k transition candidates
    if cur_prod in trans_probs:
        cands += [p for p,_ in trans_probs[cur_prod][:k]]
    # add global popularity until we have at least k candidates
    i = 0
    while len(cands) < k:
        if i >= len(global_rank): break
        gp = global_rank[i]
        if gp not in cands:
            cands.append(gp)
        i += 1
    # ensure unique and keep order
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# quick check
example = df.iloc[0]
print("Example current:", example[CUR_COL], "candidates:", generate_candidates_for_row(example[CUR_COL], k=CAND_K))

# ----- build candidate-level training dataframe -----
rows = []
for idx, r in df.iterrows():
    cur = r[CUR_COL]
    true = r[TARGET_COL]
    user = r[USER_COL] if USER_COL in r else None

    cands = generate_candidates_for_row(cur, k=CAND_K)
    # keep unique and allow true label if not in list (ensure positive present)
    if true not in cands:
        # replace last candidate with true to ensure positive candidate present
        if len(cands) > 0:
            cands[-1] = true
        else:
            cands = [true]

    for rank_pos, cand in enumerate(cands):
        row = {
            "idx": idx,
            "user": user,
            "cur_prod": cur,
            "cand_prod": cand,
            "cand_le": le_prod.transform([cand])[0],
            "is_true": 1 if cand == true else 0,
            "rank_pos": rank_pos
        }
        # transition prob feature
        prob = 0.0
        if cur in trans_probs:
            for c,p in trans_probs[cur]:
                if c == cand:
                    prob = p
                    break
        row['trans_prob'] = prob
        # candidate popularity (global)
        row['cand_pop'] = global_counts.get(cand, 0)
        # basic numeric user/context features (add more if available)
        # include allocation ratios and amounts if present
        for c in ['acct_val_amt','face_amt','cash_val_amt','stock_allocation_ratio','bond_allocation_ratio','annuity_allocation_ratio','mutual_fund_allocation_ratio','aum_to_asset_ratio','policy_value_to_assets_ratio','psn_age']:
            if c in df.columns:
                row[c] = r.get(c, np.nan)
        # categorical context: branch, agent, channel encoded via hashing or label encoding
        for c in ['branchoffice_code','agt_no','channel','mkt_prod_hier','prod_lob']:
            if c in df.columns:
                val = r.get(c, "MISSING")
                row[c] = str(val)
        rows.append(row)
    if idx % 200000 == 0 and idx>0:
        print("Built rows:", idx)
rows_df = pd.DataFrame(rows)
print("Candidate-level rows:", len(rows_df))

# ----- encode categorical text fields with LabelEncoder (on candidate-level df) -----
cat_cols = ['branchoffice_code','agt_no','channel','mkt_prod_hier','prod_lob']
for c in cat_cols:
    if c in rows_df.columns:
        rows_df[c] = rows_df[c].astype(str).fillna("MISSING")
        le = LabelEncoder()
        rows_df[c] = le.fit_transform(rows_df[c])

# numeric fill
num_cols = ['trans_prob','cand_pop','acct_val_amt','face_amt','cash_val_amt','stock_allocation_ratio','bond_allocation_ratio','annuity_allocation_ratio','mutual_fund_allocation_ratio','aum_to_asset_ratio','policy_value_to_assets_ratio','psn_age']
num_cols = [c for c in num_cols if c in rows_df.columns]
rows_df[num_cols] = rows_df[num_cols].fillna(-999)

# ----- sample weighting to balance positives vs negatives per class -----
# compute class frequencies at product level (positive examples)
pos_counts = rows_df[rows_df['is_true']==1]['cand_prod'].value_counts().to_dict()
# weight inversely to pos frequency to boost rare classes
rows_df['pos_weight'] = rows_df['cand_prod'].map(lambda x: 1.0/pos_counts.get(x,1))

# For negative examples, set weight smaller (so positives more important)
rows_df['sample_weight'] = rows_df['pos_weight'] * (1.0 + 0.0 * rows_df['is_true'])
# scale negatives down
rows_df.loc[rows_df['is_true']==0, 'sample_weight'] *= 0.25

# ----- split for CV by original row index stratified by true label of the original row -----
# We need labels per original example; map idx->true label
orig_true = df[TARGET_COL].values
idx_to_true = dict(enumerate(orig_true))
rows_df['orig_true'] = rows_df['idx'].map(idx_to_true)
# encode orig_true to int for stratification
le_orig = LabelEncoder()
rows_df['orig_true_le'] = le_orig.fit_transform(rows_df['orig_true'])

# We'll create folds at original example level to avoid leakage: first map unique idx -> fold
unique_idx = rows_df['idx'].unique()
# build a mapping idx->fold by StratifiedKFold on orig_true_le at unique idx level
tmp = df[[TARGET_COL]].reset_index().rename(columns={'index':'idx'})
tmp['orig_true_le'] = le_orig.transform(tmp[TARGET_COL])
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
idx2fold = {}
for fold, (train_idx, val_idx) in enumerate(skf.split(tmp['idx'], tmp['orig_true_le'])):
    for i in val_idx:
        idx2fold[tmp.loc[i,'idx']] = fold
rows_df['fold'] = rows_df['idx'].map(idx2fold)

# ensure no missing fold
assert rows_df['fold'].isnull().sum() == 0

# ----- features & label -----
feature_cols = ['cand_le','trans_prob','cand_pop'] + num_cols + cat_cols + ['rank_pos']
feature_cols = [c for c in feature_cols if c in rows_df.columns]
label_col = 'is_true'

print("Training features:", feature_cols)

# ----- Cross-validated LightGBM ranker (binary) -----
oof_scores = np.zeros(len(rows_df))
oof_preds_best = np.zeros(len(rows_df))
rows_df['pred_score'] = 0.0

for fold in range(N_FOLDS):
    print("Fold", fold)
    tr_mask = rows_df['fold'] != fold
    val_mask = rows_df['fold'] == fold
    tr = rows_df[tr_mask]
    val = rows_df[val_mask]
    dtrain = lgb.Dataset(tr[feature_cols], label=tr[label_col], weight=tr['sample_weight'])
    dval = lgb.Dataset(val[feature_cols], label=val[label_col], weight=val['sample_weight'], reference=dtrain)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED + fold
    }
    bst = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], early_stopping_rounds=80, verbose_eval=100)
    val_preds = bst.predict(val[feature_cols], num_iteration=bst.best_iteration)
    rows_df.loc[val_mask, 'pred_score'] = val_preds
    # save model
    bst.save_model(os.path.join(OUT_DIR, f"lgb_ranker_fold{fold}.txt"))

# ----- evaluate by selecting top-1 candidate per original row via predicted scores -----
print("Evaluating ranking performance (top-1)...")
best_pred_df = rows_df.loc[rows_df.groupby('idx')['pred_score'].idxmax()].sort_values('idx')
# best_pred_df contains one row per original example with predicted best candidate
# compute predicted label and true label
preds = best_pred_df['cand_prod'].values
truths = df.loc[best_pred_df['idx'], TARGET_COL].values

# compute macro F1 using label encoder of products
le = LabelEncoder()
le.fit(list(pd.unique(np.concatenate([preds, truths]))))
y_true = le.transform(truths)
y_pred = le.transform(preds)
macro = f1_score(y_true, y_pred, average='macro')
print("Ranking model macro F1 (top-1):", macro)
print("\nClassification report (ranking top-1):")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

# save outputs
best_pred_df[['idx','cur_prod','cand_prod','pred_score','is_true']].to_csv(os.path.join(OUT_DIR, "pred_per_example_top1.csv"), index=False)
rows_df.to_csv(os.path.join(OUT_DIR, "candidate_level_rows.csv"), index=False)
with open(os.path.join(OUT_DIR, "ranking_eval.json"), "w") as f:
    json.dump({"macro_f1": float(macro)}, f)

print("Saved outputs to", OUT_DIR)
