"""
Run this script to produce the diagnostics I requested.

Outputs:
 - Printed summary: columns, dtypes, sample rows, row/user/product counts, label distribution summary.
 - Files saved to working dir:
    - diagnostic_sample_rows.csv       (first 10 rows + 10 random samples)
    - diagnostic_label_counts.csv      (full label counts)
    - diagnostic_transitions_topk.csv  (top transition probabilities for top current products)
    - diagnostic_report.json           (JSON with many numeric summaries)
"""

import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict
from sklearn.metrics import recall_score

# ---------- USER SETTINGS ----------
DATA_PATH = "your_data.csv"   # <<-- change to your file path
# If your dataset uses different column names, set them here. If left None, script tries to infer.
USER_COL = None          # e.g. "user_id"
CURRENT_COL = None       # e.g. "current_product" or "product"
NEXT_COL = None          # e.g. "next_product" or "label"
TS_COL = None            # e.g. "timestamp" or "ts" (optional but recommended)
# -----------------------------------

# helper to guess columns from a list of candidates
def guess_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# load
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loaded rows:", len(df))

# infer columns if not provided
cols = df.columns.tolist()
USER_COL = USER_COL or guess_col(cols, ["user_id", "userid", "customer_id", "cust_id", "uid", "user"])
CURRENT_COL = CURRENT_COL or guess_col(cols, ["current_product", "product", "product_id", "curr_product", "last_product", "product_current"])
NEXT_COL = NEXT_COL or guess_col(cols, ["next_product", "next_item", "label", "target", "product_next", "future_product"])
TS_COL = TS_COL or guess_col(cols, ["timestamp", "ts", "time", "event_time", "created_at", "date"])

print("\nInferred columns (change variables at top if incorrect):")
print(" user id column:", USER_COL)
print(" current product column:", CURRENT_COL)
print(" next product (label) column:", NEXT_COL)
print(" timestamp column:", TS_COL)

# Basic schema
print("\nColumns and dtypes:")
print(df.dtypes)

# Show sample rows: head and random
n_head = 10
n_rand = 10
samples = pd.concat([df.head(n_head), df.sample(n_rand, random_state=42)], ignore_index=False)
samples.to_csv("diagnostic_sample_rows.csv", index=False)
print(f"\nSaved sample rows to diagnostic_sample_rows.csv (first {n_head} + {n_rand} random).")

# If key columns exist, show them; otherwise show first 10 cols of samples
print("\nSample rows (displaying key columns if available):")
if CURRENT_COL and NEXT_COL and USER_COL:
    display_cols = [c for c in [USER_COL, TS_COL, CURRENT_COL, NEXT_COL] if c in df.columns]
else:
    display_cols = df.columns[:10].tolist()
print(samples[display_cols].head(20).to_string(index=False))

# Basic counts
n_rows = len(df)
n_users = df[USER_COL].nunique() if USER_COL in df.columns else None
n_products = df[NEXT_COL].nunique() if NEXT_COL in df.columns else None
print("\nCounts:")
print(" total rows:", n_rows)
print(" unique users:", n_users)
print(" unique target products (next):", n_products)

# Label distribution analysis (if NEXT_COL exists)
report = {}
if NEXT_COL in df.columns:
    label_counts = df[NEXT_COL].value_counts().sort_values(ascending=False)
    label_counts.to_csv("diagnostic_label_counts.csv", header=["count"])
    print("\nSaved full label counts to diagnostic_label_counts.csv")
    print("\nTop 20 target products by count:")
    print(label_counts.head(20).to_string())

    report['n_labels'] = int(label_counts.shape[0])
    report['label_counts_top10'] = label_counts.head(10).to_dict()
    report['label_count_median'] = float(label_counts.median())
    report['label_count_mean'] = float(label_counts.mean())
    for N in [1,5,10,20,50]:
        report[f'labels_less_than_{N}'] = int((label_counts < N).sum())
        report[f'labels_less_than_{N}_fraction'] = float((label_counts < N).sum() / label_counts.shape[0])
    # percentiles
    for p in [50,75,90,95,99]:
        report[f'label_count_p{p}'] = int(np.percentile(label_counts.values, p))

else:
    print("\nNo NEXT_COL found; skipping label distribution.")
    report['n_labels'] = None

# If USER_COL & TS_COL present, compute basic user behavior stats
if USER_COL in df.columns:
    grp = df.groupby(USER_COL).size()
    report['median_events_per_user'] = int(grp.median())
    report['mean_events_per_user'] = float(grp.mean())
    report['users_with_1_event'] = int((grp==1).sum())
    print("\nUser event counts summary:")
    print(grp.describe().to_string())
else:
    print("\nNo USER_COL found; skipping per-user stats.")

# Try to compute transition matrix (simple Markov P(next | current)) if current & next exist
trans = {}
if CURRENT_COL in df.columns and NEXT_COL in df.columns and USER_COL in df.columns:
    print("\nBuilding product->product transition counts (requires user_id + ordering by timestamp if available).")
    # sort by user and timestamp if timestamp available
    if TS_COL in df.columns:
        df_sorted = df.sort_values([USER_COL, TS_COL])
    else:
        df_sorted = df.sort_values([USER_COL])
    for _, g in df_sorted.groupby(USER_COL):
        seq_curr = g[CURRENT_COL].astype(str).tolist()
        seq_next = g[NEXT_COL].astype(str).tolist()
        # if sequences align (current -> next on same row) also consider successive pairs within user history
        # We'll also create pairs from successive product events if next is missing or to augment transitions.
        for a,b in zip(seq_curr, seq_next):
            trans.setdefault(a, {})
            trans[a][b] = trans[a].get(b,0) + 1
        # also from successive events consider product_i -> product_{i+1}
        prod_seq = []
        # attempt to use NEXT_COL if it's the next product field; fallback to CURRENT_COL sequence
        if len(seq_next) >= 2:
            prod_seq = seq_next
        else:
            prod_seq = seq_curr
        for a,b in zip(prod_seq, prod_seq[1:]):
            trans.setdefault(a, {})
            trans[a][b] = trans[a].get(b,0) + 1

    # normalize to probabilities and save top transitions
    trans_probs = {}
    rows = []
    for a, d in trans.items():
        s = sum(d.values())
        for b, cnt in sorted(d.items(), key=lambda x: -x[1]):
            rows.append((a, b, cnt, cnt/s))
        trans_probs[a] = {b: cnt/s for b,cnt in d.items()}
    trans_df = pd.DataFrame(rows, columns=['from_product','to_product','count','prob'])
    trans_df.to_csv("diagnostic_transitions_topk.csv", index=False)
    print("Saved transitions to diagnostic_transitions_topk.csv (first rows):")
    print(trans_df.head(20).to_string(index=False))
    report['n_from_products_with_transitions'] = int(len(trans_probs))
else:
    print("\nSkipping transition matrix (need CURRENT_COL, NEXT_COL and USER_COL).")

# Quick baseline predictions & top-k evaluation (requires NEXT_COL)
def topk_recall(y_true, y_pred_candidates, k):
    # y_pred_candidates: list of candidate lists (in order) per row
    hits = 0
    for yt, cand in zip(y_true, y_pred_candidates):
        if yt in cand[:k]:
            hits += 1
    return hits / len(y_true)

if NEXT_COL in df.columns:
    y_true = df[NEXT_COL].astype(str).tolist()
    # Most frequent baseline
    most_common = label_counts.idxmax() if NEXT_COL in df.columns else None
    mf_preds = [[most_common] for _ in range(len(df))]
    report['baseline_mf_top1'] = topk_recall(y_true, mf_preds, 1)
    report['baseline_mf_top3'] = topk_recall(y_true, mf_preds, 3)
    report['baseline_mf_top5'] = topk_recall(y_true, mf_preds, 5)
    print("\nBaseline (most frequent) recall@1/3/5:", report['baseline_mf_top1'], report['baseline_mf_top3'], report['baseline_mf_top5'])

    # Last-product-per-user baseline: predict the previous product seen for same user
    last_pred_list = []
    if USER_COL in df.columns and CURRENT_COL in df.columns:
        # build last seen product per user by iterating in order
        last_seen = {}
        if TS_COL in df.columns:
            df_order = df.sort_values([USER_COL, TS_COL])
        else:
            df_order = df.sort_values([USER_COL])
        for idx, row in df_order.iterrows():
            u = row[USER_COL]
            last = last_seen.get(u, None)
            last_pred_list.append(last if last is not None else most_common)
            # update last_seen with current product if exists; else with next
            if CURRENT_COL in row and not pd.isna(row[CURRENT_COL]):
                last_seen[u] = str(row[CURRENT_COL])
        # reorder to original index order
        last_pred_array = pd.Series(last_pred_list, index=df_order.index).reindex(df.index).fillna(most_common).astype(str).tolist()
        last_preds = [[p] for p in last_pred_array]
        report['baseline_last_top1'] = topk_recall(y_true, last_preds, 1)
        report['baseline_last_top3'] = topk_recall(y_true, last_preds, 3)
        report['baseline_last_top5'] = topk_recall(y_true, last_preds, 5)
        print("Baseline (last-per-user) recall@1/3/5:", report['baseline_last_top1'], report['baseline_last_top3'], report['baseline_last_top5'])
    else:
        print("Skipping last-product baseline (need USER_COL and CURRENT_COL).")

    # Markov baseline: predict top-k most probable next products given current product using transition probs
    if CURRENT_COL in df.columns and trans:
        # build candidate list per row
        cand_list = []
        for curr in df[CURRENT_COL].astype(str).tolist():
            if curr in trans_probs:
                # sort candidates by prob desc
                cands = sorted(trans_probs[curr].items(), key=lambda x: -x[1])
                cand_list.append([c for c,_ in cands])
            else:
                cand_list.append([most_common])
        report['baseline_markov_top1'] = topk_recall(y_true, cand_list, 1)
        report['baseline_markov_top3'] = topk_recall(y_true, cand_list, 3)
        report['baseline_markov_top5'] = topk_recall(y_true, cand_list, 5)
        print("Baseline (Markov P(next|current)) recall@1/3/5:", report['baseline_markov_top1'], report['baseline_markov_top3'], report['baseline_markov_top5'])
    else:
        print("Skipping Markov baseline (need transitions).")
else:
    print("\nNEXT_COL not found, skipping baseline evaluation.")

# Save JSON report
with open("diagnostic_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved numeric report to diagnostic_report.json")

print("\nFiles produced:")
print(" - diagnostic_sample_rows.csv")
print(" - diagnostic_label_counts.csv  (if label column found)")
print(" - diagnostic_transitions_topk.csv (if transitions built)")
print(" - diagnostic_report.json")

print("\nWhat to paste here:")
print(" 1) The inferred column names printed at top (confirm if correct).")
print(" 2) diagnostic_report.json contents (or paste key fields below).")
print(" 3) First 20 rows of diagnostic_sample_rows.csv (or attach the file).")
print(" 4) diagnostic_label_counts.csv top 50 rows (if available).")
print("\nOnce you paste those, I'll run the targeted diagnosis and give a prioritized experiment plan.")
