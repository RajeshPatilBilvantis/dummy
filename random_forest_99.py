"""
Cross-Sell Product Prediction Model
Predicts the second policy (product) a client will take given their first policy.
Target: wti_lob_txt (product category)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    RandomizedSearchCV, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, make_scorer
)
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

# For feature importance visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("CROSS-SELL PRODUCT PREDICTION MODEL")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("\n[STEP 1] Loading data...")
# Note: Update this path to your actual dataset
# df = pd.read_csv('your_dataset.csv')
# For now, we'll create a function that expects the data to be loaded

def load_data(file_path=None):
    """
    Load the dataset. If file_path is None, expects df to be loaded externally.
    Handles CSV parsing errors with flexible parameters.
    """
    if file_path:
        print(f"  - Loading data from: {file_path}")

        # Check pandas version to use appropriate parameter
        pandas_version = pd.__version__
        version_parts = [int(x) for x in pandas_version.split('.')[:2]]
        use_on_bad_lines = version_parts[0] > 1 or (version_parts[0] == 1 and version_parts[1] >= 3)

        # Try multiple strategies
        strategies = [
            # Strategy 1: Standard read
            {
                'params': {'low_memory': False},
                'name': 'standard'
            },
            # Strategy 2: Skip bad lines with Python engine (no low_memory with python engine)
            {
                'params': {
                    'engine': 'python',
                    'on_bad_lines': 'skip' if use_on_bad_lines else None,
                    'error_bad_lines': False if not use_on_bad_lines else None,
                    'warn_bad_lines': False if not use_on_bad_lines else None
                },
                'name': 'skip_bad_lines'
            },
            # Strategy 3: With quoting (no low_memory with python engine)
            {
                'params': {
                    'engine': 'python',
                    'quoting': 1,  # QUOTE_ALL
                    'on_bad_lines': 'skip' if use_on_bad_lines else None,
                    'error_bad_lines': False if not use_on_bad_lines else None,
                    'warn_bad_lines': False if not use_on_bad_lines else None
                },
                'name': 'with_quoting'
            },
            # Strategy 4: Most permissive (no low_memory with python engine)
            {
                'params': {
                    'engine': 'python',
                    'sep': ',',
                    'on_bad_lines': 'skip' if use_on_bad_lines else None,
                    'error_bad_lines': False if not use_on_bad_lines else None,
                    'warn_bad_lines': False if not use_on_bad_lines else None,
                    'quoting': 3  # QUOTE_NONE
                },
                'name': 'permissive'
            }
        ]

        # Remove None values from params
        for strategy in strategies:
            strategy['params'] = {k: v for k, v in strategy['params'].items() if v is not None}

        df = None
        last_error = None

        for strategy in strategies:
            try:
                print(f"  - Trying strategy: {strategy['name']}...")
                df = pd.read_csv(file_path, **strategy['params'])
                print(f"  - Success! Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                break
            except (pd.errors.ParserError, pd.errors.EmptyDataError, Exception) as e:
                last_error = e
                print(f"  - Strategy '{strategy['name']}' failed: {str(e)[:100]}")
                continue

        if df is None:
            raise ValueError(
                f"Could not load data from {file_path}. All strategies failed.\n"
                f"Last error: {last_error}\n"
                f"Please check if the file is a valid CSV or try loading it manually with appropriate parameters."
            )

        print(f"  - Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        # If running in Colab, user should upload the file
        # For now, we'll assume df is already loaded
        raise ValueError("Please provide file_path or load the dataframe as 'df'")

    return df

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
def clean_data(df):
    """
    Perform comprehensive data cleaning.
    """
    print("\n[STEP 2] Cleaning data...")
    df_clean = df.copy()

    # Convert date columns
    date_columns = ['register_date', 'trmn_eff_date', 'strt_date', 'end_date',
                    'isrd_brth_date', 'birth_dt', 'hlp_date', 'clb_date']

    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # Handle 'NULL' strings and convert to actual NaN
    df_clean = df_clean.replace(['NULL', 'null', 'Null', ''], np.nan)

    # Convert numeric columns
    numeric_columns = ['face_amt', 'cash_val_amt', 'acct_val_amt', 'monthly_preminum_amount',
                       'wc_total_assets', 'wc_assetmix_stocks', 'wc_assetmix_bonds',
                       'wc_assetmix_mutual_funds', 'wc_assetmix_annuity',
                       'wc_assetmix_deposits', 'wc_assetmix_other_assets', 'psn_age',
                       'wm_headcount', 'headcount']

    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Remove rows where axa_party_id is missing (critical for grouping)
    df_clean = df_clean.dropna(subset=['axa_party_id'])

    # Remove rows where target is missing
    if 'wti_lob_txt' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['wti_lob_txt'])

    print(f"  - Data shape after cleaning: {df_clean.shape}")
    print(f"  - Unique clients: {df_clean['axa_party_id'].nunique()}")
    print(f"  - Total policies: {len(df_clean)}")

    return df_clean

# ============================================================================
# STEP 3: CREATE CROSS-SELL DATASET
# ============================================================================
def create_cross_sell_dataset(df_clean):
    """
    Create dataset for cross-sell prediction:
    - Identify clients with at least 2 policies
    - Use first policy features to predict second policy category
    """
    print("\n[STEP 3] Creating cross-sell dataset...")

    # Sort by client and register_date to identify first and second policies
    df_sorted = df_clean.sort_values(['axa_party_id', 'register_date']).copy()

    # Add policy sequence number per client
    df_sorted['policy_sequence'] = df_sorted.groupby('axa_party_id').cumcount() + 1

    # Filter clients with at least 2 policies
    client_policy_counts = df_sorted.groupby('axa_party_id').size()
    clients_with_multiple = client_policy_counts[client_policy_counts >= 2].index

    print(f"  - Clients with multiple policies: {len(clients_with_multiple)}")

    # Create cross-sell pairs: (first policy features, second policy target)
    cross_sell_data = []

    for client_id in clients_with_multiple:
        client_policies = df_sorted[df_sorted['axa_party_id'] == client_id].copy()

        # Get first policy (features)
        first_policy = client_policies[client_policies['policy_sequence'] == 1].iloc[0]

        # Get second policy (target)
        second_policy = client_policies[client_policies['policy_sequence'] == 2].iloc[0]

        # Create feature row
        # feature_row = first_policy.copy()
        # feature_row['target_product'] = second_policy['wti_lob_txt']

        # Create feature row
        feature_row = first_policy.copy()
        feature_row['target_product'] = second_policy['wti_lob_txt']
        # Rename first policy's wti_lob_txt to first_policy_product
        if 'wti_lob_txt' in feature_row:
            feature_row['first_policy_product'] = feature_row['wti_lob_txt']
            feature_row = feature_row.drop('wti_lob_txt')  # Remove the original column

        feature_row['days_to_second_policy'] = (
            second_policy['register_date'] - first_policy['register_date']
        ).days if pd.notna(second_policy['register_date']) and pd.notna(first_policy['register_date']) else np.nan

        cross_sell_data.append(feature_row)

    cross_sell_df = pd.DataFrame(cross_sell_data)

    print(f"  - Cross-sell dataset shape: {cross_sell_df.shape}")
    print(f"  - Target distribution:\n{cross_sell_df['target_product'].value_counts()}")

    return cross_sell_df

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """
    Create comprehensive features for the model.
    """
    print("\n[STEP 4] Engineering features...")

    df_fe = df.copy()

    # 1. Temporal Features
    if 'register_date' in df_fe.columns:
        df_fe['register_year'] = df_fe['register_date'].dt.year
        df_fe['register_month'] = df_fe['register_date'].dt.month
        df_fe['register_quarter'] = df_fe['register_date'].dt.quarter
        df_fe['register_day_of_week'] = df_fe['register_date'].dt.dayofweek

    if 'birth_dt' in df_fe.columns and 'register_date' in df_fe.columns:
        df_fe['age_at_policy'] = (
            df_fe['register_date'] - df_fe['birth_dt']
        ).dt.days / 365.25

    # 2. Financial Features
    financial_cols = ['face_amt', 'cash_val_amt', 'acct_val_amt', 'monthly_preminum_amount']
    for col in financial_cols:
        if col in df_fe.columns:
            df_fe[f'{col}_log'] = np.log1p(df_fe[col].fillna(0))
            df_fe[f'{col}_is_zero'] = (df_fe[col].fillna(0) == 0).astype(int)

    # Asset mix ratios
    if 'wc_total_assets' in df_fe.columns:
        asset_cols = ['wc_assetmix_stocks', 'wc_assetmix_bonds', 'wc_assetmix_mutual_funds',
                     'wc_assetmix_annuity', 'wc_assetmix_deposits', 'wc_assetmix_other_assets']
        for col in asset_cols:
            if col in df_fe.columns:
                df_fe[f'{col}_ratio'] = (
                    df_fe[col].fillna(0) / (df_fe['wc_total_assets'].fillna(0) + 1)
                )

    # 3. Policy Characteristics
    if 'policy_status' in df_fe.columns:
        df_fe['is_active'] = (df_fe['policy_status'] == 'Active').astype(int)

    if 'policy_type' in df_fe.columns:
        df_fe['is_old_policy'] = (df_fe['policy_type'] == 'Old Policy').astype(int)

    # 4. Client Segment Features
    segment_cols = ['client_seg', 'client_seg_1', 'aum_band', 'client_type']
    for col in segment_cols:
        if col in df_fe.columns:
            df_fe[f'{col}_encoded'] = df_fe[col].astype(str)

    # 5. Agent Features
    if 'agt_class' in df_fe.columns:
        df_fe['agent_class_encoded'] = df_fe['agt_class'].astype(str)

    if 'agent_segment' in df_fe.columns:
        df_fe['agent_segment_encoded'] = df_fe['agent_segment'].astype(str)

    # 6. Geographic Features
    if 'business_state_cod' in df_fe.columns:
        df_fe['state_encoded'] = df_fe['business_state_cod'].astype(str)

    # 7. Product Hierarchy Features
    hierarchy_cols = ['prod_lob', 'sub_product_level_1', 'sub_product_level_2',
                     'mkt_prod_hier']
    for col in hierarchy_cols:
        if col in df_fe.columns:
            df_fe[f'{col}_encoded'] = df_fe[col].astype(str)

    # 8. Channel Features
    if 'channel' in df_fe.columns:
        df_fe['channel_encoded'] = df_fe['channel'].astype(str)

    # 9. Division Features
    if 'division_name' in df_fe.columns:
        df_fe['division_encoded'] = df_fe['division_name'].astype(str)

    #### First Policy Product Feature (important!)
    if 'first_policy_product' in df_fe.columns:
        df_fe['first_policy_product_encoded'] = df_fe['first_policy_product'].astype(str)

    # 10. Time to second policy (if available)
    if 'days_to_second_policy' in df_fe.columns:
        df_fe['days_to_second_policy_log'] = np.log1p(df_fe['days_to_second_policy'].fillna(0))
        df_fe['days_to_second_policy_is_na'] = df_fe['days_to_second_policy'].isna().astype(int)

    # 11. Interaction Features (combinations that might be predictive)
    if 'face_amt' in df_fe.columns and 'psn_age' in df_fe.columns:
        df_fe['face_amt_per_age'] = (
            df_fe['face_amt'].fillna(0) / (df_fe['psn_age'].fillna(1) + 1)
        )

    if 'acct_val_amt' in df_fe.columns and 'face_amt' in df_fe.columns:
        df_fe['acct_to_face_ratio'] = (
            df_fe['acct_val_amt'].fillna(0) / (df_fe['face_amt'].fillna(0) + 1)
        )

    if 'monthly_preminum_amount' in df_fe.columns and 'face_amt' in df_fe.columns:
        df_fe['premium_to_face_ratio'] = (
            df_fe['monthly_preminum_amount'].fillna(0) / (df_fe['face_amt'].fillna(0) + 1)
        )

    # 12. Aggregated Features (if we have multiple policies per client in original data)
    # These would be calculated from the original dataset before creating cross-sell pairs
    # For now, we'll add flags for common patterns

    # 13. Binary combinations
    if 'is_active' in df_fe.columns and 'is_old_policy' in df_fe.columns:
        df_fe['active_old_policy'] = (
            df_fe['is_active'].fillna(0) * df_fe['is_old_policy'].fillna(0)
        ).astype(int)

    # 14. Financial health indicators
    if 'cash_val_amt' in df_fe.columns and 'acct_val_amt' in df_fe.columns:
        df_fe['cash_to_account_ratio'] = (
            df_fe['cash_val_amt'].fillna(0) / (df_fe['acct_val_amt'].fillna(0) + 1)
        )

    # 15. Policy value bands (categorical from continuous)
    if 'face_amt' in df_fe.columns:
        df_fe['face_amt_band'] = pd.cut(
            df_fe['face_amt'].fillna(0),
            bins=[0, 10000, 50000, 100000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        ).astype(str)

    if 'acct_val_amt' in df_fe.columns:
        df_fe['acct_val_band'] = pd.cut(
            df_fe['acct_val_amt'].fillna(0),
            bins=[0, 10000, 50000, 100000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        ).astype(str)

    print(f"  - Total features after engineering: {df_fe.shape[1]}")

    return df_fe

# ============================================================================
# STEP 5: FEATURE SELECTION AND PREPROCESSING
# ============================================================================
def prepare_features(df_fe, target_col='target_product'):
    """
    Select and preprocess features for modeling.
    """
    print("\n[STEP 5] Preparing features for modeling...")

    # Define feature columns (exclude identifiers and target)
    # exclude_cols = [
    #     'policy_no', 'axa_party_id', 'cont_id', 'ref_num', 'hhkey',
    #     'agt_no', 'target_product', 'wti_lob_txt',  # target
    #     'register_date', 'trmn_eff_date', 'strt_date', 'end_date',
    #     'isrd_brth_date', 'birth_dt', 'hlp_date', 'clb_date',  # dates (we use derived features)
    #     'policy_sequence',  # sequence number
    #     'agt_first_name', 'agt_last_name',  # names
    #     'business_city', 'business_zip_code', 'prod_code',  # high cardinality
    # ]

    exclude_cols = [
    'policy_no', 'axa_party_id', 'cont_id', 'ref_num', 'hhkey',
    'agt_no', 'target_product',  # target
    'register_date', 'trmn_eff_date', 'strt_date', 'end_date',
    'isrd_brth_date', 'birth_dt', 'hlp_date', 'clb_date',  # dates (we use derived features)
    'policy_sequence',  # sequence number
    'agt_first_name', 'agt_last_name',  # names
    'business_city', 'business_zip_code', 'prod_code', 'source_sys_id', 'prod_code', 'plan_code',
    'plan_subcd_code', 'business_state_cod', 'active_head_count_flag', 'wm_headcount', 
    'elas_y_n', 'headcount', 'designation', 'final_designation', 'esf_yr', 'esf_hire',
    'transaction_policy', 'hlp', 'columbia', 'business_month', 'days_to_second_policy', 
    'register_year', 'register_month', 'register_day_of_week',  'sub_product_level_1',
    'sub_product_level_2', 'client_seg', 'client_seg_1', 'aum_band', 'class_code',
    'client_type', 'channel', 'agent_segment', 'prod_lob', 'mkt_prod_hier', 'division',
    'days_to_second_policy_log', 'days_to_second_policy_is_na', 'prod_lob_encoded',
    'first_policy_product'
    ]

    # Get feature columns
    feature_cols = [col for col in df_fe.columns if col not in exclude_cols]
    print(f"  - Total features after excluding columns: {len(feature_cols)}")
    print(len(feature_cols))

    # Separate features and target
    X = df_fe[feature_cols].copy()
    y = df_fe[target_col].copy()

    # Handle missing values
    print("  - Handling missing values...")

    # Fill numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median() if X[col].notna().sum() > 0 else 0)

    # Fill categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna('Missing')

    # Encode categorical variables
    print("  - Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    print(f"  - Final feature matrix shape: {X.shape}")
    print(f"  - Number of classes: {len(np.unique(y_encoded))}")
    print(f"  - Class distribution:\n{pd.Series(y_encoded).value_counts().sort_index()}")

    return X, y_encoded, feature_cols, label_encoders, target_encoder

# ============================================================================
# STEP 6: MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================
def train_models(X, y, test_size=0.2, random_state=42, tune_hyperparameters=True, use_smote=False):
    """
    Train multiple models with hyperparameter tuning and select the best one based on F1 score.
    """
    print("\n[STEP 6] Training models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  - Training set size: {X_train.shape[0]}")
    print(f"  - Test set size: {X_test.shape[0]}")

    # Apply SMOTE if requested (only on training data, never on test!)
    if use_smote:
        print("  - Applying SMOTE for class imbalance handling...")
        try:
            # Determine k_neighbors based on smallest class size
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            min_class_count = class_counts.min()
            k_neighbors = min(5, min_class_count - 1)
            k_neighbors = max(1, k_neighbors)  # Ensure at least 1

            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  - Training set size after SMOTE: {X_train.shape[0]}")
        except Exception as e:
            print(f"  - SMOTE failed: {e}. Continuing without SMOTE...")
            use_smote = False

    # Scale features (for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # F1 scorer for cross-validation
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Define base models
    base_models = {
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        # 'Gradient Boosting': GradientBoostingClassifier(
        #     random_state=random_state
        # ),
        # 'XGBoost': xgb.XGBClassifier(
        #     random_state=random_state,
        #     eval_metric='mlogloss',
        #     use_label_encoder=False,
        #     tree_method='hist'
        # ),
        # 'LightGBM': lgb.LGBMClassifier(
        #     class_weight='balanced',
        #     random_state=random_state,
        #     verbose=-1
        # )
    }

    # Hyperparameter grids for tuning
    param_grids = {
        'Random Forest': {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        # 'Gradient Boosting': {
        #     'n_estimators': [200, 300],
        #     'max_depth': [8, 10, 12],
        #     'learning_rate': [0.05, 0.1, 0.15],
        #     'min_samples_split': [2, 5],
        #     'min_samples_leaf': [1, 2]
        # },
        # 'XGBoost': {
        #     'n_estimators': [200, 300, 400],
        #     'max_depth': [8, 10, 12],
        #     'learning_rate': [0.05, 0.1, 0.15],
        #     'subsample': [0.8, 0.9, 1.0],
        #     'colsample_bytree': [0.8, 0.9, 1.0],
        #     'min_child_weight': [1, 3, 5]
        # },
        # 'LightGBM': {
        #     'n_estimators': [200, 300, 400],
        #     'max_depth': [8, 10, 12, -1],
        #     'learning_rate': [0.05, 0.1, 0.15],
        #     'num_leaves': [31, 50, 70],
        #     'subsample': [0.8, 0.9, 1.0],
        #     'colsample_bytree': [0.8, 0.9, 1.0],
        #     'min_child_samples': [10, 20, 30]
        # }
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, base_model in base_models.items():
        print(f"\n  Training {name}...")

        try:
            if tune_hyperparameters and name in param_grids:
                print(f"    Tuning hyperparameters...")
                # Use RandomizedSearchCV for faster tuning
                search = RandomizedSearchCV(
                    base_model,
                    param_grids[name],
                    n_iter=20,  # Number of parameter settings sampled
                    cv=cv,
                    scoring=f1_scorer,
                    n_jobs=-1,
                    random_state=random_state,
                    verbose=0
                )

                # Train with appropriate data format
                if name in ['Random Forest']:
                # if name in ['Random Forest', 'Gradient Boosting']:
                    search.fit(X_train_scaled, y_train)
                    best_model = search.best_estimator_
                    y_pred = best_model.predict(X_test_scaled)
                else:
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    y_pred = best_model.predict(X_test)

                print(f"    Best params: {search.best_params_}")
            else:
                # Train without tuning
                if name in ['Random Forest']:
                # if name in ['Random Forest', 'Gradient Boosting']:
                    base_model.fit(X_train_scaled, y_train)
                    best_model = base_model
                    y_pred = best_model.predict(X_test_scaled)
                else:
                    base_model.fit(X_train, y_train)
                    best_model = base_model
                    y_pred = best_model.predict(X_test)

            # Cross-validation score
            if name in ['Random Forest']:
            # if name in ['Random Forest', 'Gradient Boosting']:
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train,
                                           cv=cv, scoring=f1_scorer, n_jobs=-1)
            else:
                cv_scores = cross_val_score(best_model, X_train, y_train,
                                           cv=cv, scoring=f1_scorer, n_jobs=-1)

            # Calculate metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            results[name] = {
                'model': best_model,
                'f1_score': f1,
                'f1_macro': f1_macro,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'y_pred': y_pred,
                'y_test': y_test,
                'X_train': X_train,  # Add this
                'y_train': y_train,  # Add this
                'X_test': X_test,    # Add this (already have y_test)
            }

            print(f"    CV F1 Score (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"    Test F1 Score: {f1:.4f}")
            print(f"    Test Accuracy: {accuracy:.4f}")
            print(f"    Test Precision: {precision:.4f}")
            print(f"    Test Recall: {recall:.4f}")

        except Exception as e:
            print(f"    Error training {name}: {e}")
            continue

    if not results:
        raise ValueError("No models were successfully trained!")

    # Select best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]

    print(f"\n  Best Model: {best_model_name}")
    print(f"  Best F1 Score: {best_model['f1_score']:.4f}")
    print(f"  Best CV F1 Score: {best_model['cv_f1_mean']:.4f} ± {best_model['cv_f1_std']:.4f}")

    return results, best_model_name, X_train, X_test, y_train, y_test, scaler

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
def plot_feature_importance(model, feature_cols, model_name, top_n=20):
    """
    Plot feature importance for tree-based models.
    """
    print(f"\n[STEP 7] Extracting feature importance for {model_name}...")

    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            print("  Model does not support feature importance extraction.")
            return

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\n  Top {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))

        # Plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        return importance_df
    except Exception as e:
        print(f"  Error plotting feature importance: {e}")
        return None

# ============================================================================
# STEP 8: EVALUATION METRICS
# ============================================================================
def evaluate_model(y_test, y_pred, target_encoder, model_name):
    """
    Comprehensive model evaluation.
    """
    print(f"\n[STEP 8] Detailed Evaluation for {model_name}")
    print("=" * 80)

    # Overall metrics
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"  Weighted F1 Score: {f1_weighted:.4f}")
    print(f"  Macro F1 Score: {f1_macro:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Weighted Precision: {precision_weighted:.4f}")
    print(f"  Weighted Recall: {recall_weighted:.4f}")

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(classification_report(y_test, y_pred,
                                target_names=target_encoder.classes_,
                                zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main_pipeline(file_path=None, df=None):
    """
    Complete ML pipeline for cross-sell prediction.
    """
    # Load data
    if df is None:
        df = load_data(file_path)

    # Step 1: Clean data
    df_clean = clean_data(df)

    # Step 2: Create cross-sell dataset
    cross_sell_df = create_cross_sell_dataset(df_clean)

    # Step 3: Engineer features
    df_fe = engineer_features(cross_sell_df)

    # Step 4: Prepare features
    X, y, feature_cols, label_encoders, target_encoder = prepare_features(df_fe)

    # Step 5: Train models (with hyperparameter tuning)
    # Try with SMOTE first, if F1 < 0.80, try without
    results, best_model_name, X_train, X_test, y_train, y_test, scaler = train_models(
        X, y, tune_hyperparameters=True, use_smote=True
    )

    # If F1 score is still below 0.80, try without SMOTE
    best_f1 = results[best_model_name]['f1_score']
    if best_f1 < 0.80:
        print("\n  F1 score below 0.80, trying without SMOTE...")
        results_no_smote, best_model_name_no_smote, _, _, _, _, _ = train_models(
            X, y, tune_hyperparameters=True, use_smote=False
        )
        if results_no_smote[best_model_name_no_smote]['f1_score'] > best_f1:
            results = results_no_smote
            best_model_name = best_model_name_no_smote
            print(f"  Better result without SMOTE: {results[best_model_name]['f1_score']:.4f}")

    # Step 6: Feature importance
    best_model = results[best_model_name]['model']
    importance_df = plot_feature_importance(best_model, feature_cols, best_model_name)

    # Step 7: Detailed evaluation
    best_results = results[best_model_name]
    X_train = best_results.get('X_train')
    y_train = best_results.get('y_train')
    metrics = evaluate_model(
        best_results['y_test'],
        best_results['y_pred'],
        target_encoder,
        best_model_name
    )

    # Check if F1 score meets requirement
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Model: {best_model_name}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")

    if metrics['f1_weighted'] >= 0.80:
        print("\n✓ SUCCESS: F1 score meets the requirement (>= 0.80)")
    else:
        print("\n⚠ WARNING: F1 score is below 0.80. Consider:")
        print("  - Further hyperparameter tuning (increase n_iter in RandomizedSearchCV)")
        print("  - Additional feature engineering (interaction features, aggregations)")
        print("  - Ensemble methods (voting, stacking)")
        print("  - SMOTE or other oversampling techniques for class imbalance")
        print("  - Collecting more training data")
        print("  - Feature selection to remove noise")

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'results': results,
        'metrics': metrics,
        'feature_cols': feature_cols,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'importance_df': importance_df,
        'scaler': scaler,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,  # Add this
        'y_train': y_train,  # Add this
    }


if __name__ == "__main__":
    # Example usage (uncomment and modify as needed)
    results = main_pipeline(file_path="/content/eq_client_metrics.csv")
    
    # ========================================================================
    # PRINT TRAINING DATAFRAME
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING DATAFRAME")
    print("="*80)
    
    # Get training data from results
    best_model_name = results['best_model_name']
    X_train = results['results'][best_model_name].get('X_train')
    y_train = results['results'][best_model_name].get('y_train')
    
    # If not in results, get from train_models return (you'll need to modify main_pipeline)
    # For now, let's get what we can:
    feature_cols = results['feature_cols']
    target_encoder = results['target_encoder']
    
    # Note: X_train might be scaled, so we'll show feature names
    print(f"\nTraining Features ({len(feature_cols)} features):")
    print(feature_cols)
    print(f"\nNote: Training data is scaled. Showing feature names above.")
    
    # ========================================================================
    # PRINT PREDICTION RESULTS TABLE
    # ========================================================================
    best_model_name = results['best_model_name']
    y_pred = results['results'][best_model_name]['y_pred']
    y_test = results['results'][best_model_name]['y_test']
    target_encoder = results['target_encoder']

    # Convert encoded predictions back to class names
    y_pred_classes = target_encoder.inverse_transform(y_pred)
    y_test_classes = target_encoder.inverse_transform(y_test)

    # Create comprehensive prediction table
    prediction_df = pd.DataFrame({
        'Actual': y_test_classes,
        'Predicted': y_pred_classes,
        'Correct': y_test_classes == y_pred_classes,
        'Actual_Encoded': y_test,
        'Predicted_Encoded': y_pred
    })

    # Print the full table
    print("\n" + "="*80)
    print("PREDICTION RESULTS TABLE (Full)")
    print("="*80)
    print(prediction_df)
    
    # Save to CSV if needed
    # prediction_df.to_csv('predictions.csv', index=False)
    # print("\nPredictions saved to 'predictions.csv'")

    # Print summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total predictions: {len(prediction_df)}")
    print(f"Correct predictions: {prediction_df['Correct'].sum()}")
    print(f"Accuracy: {prediction_df['Correct'].mean():.2%}")
    
    # Print accuracy by class
    print("\n" + "="*80)
    print("ACCURACY BY CLASS")
    print("="*80)
    for class_name in target_encoder.classes_:
        class_mask = y_test_classes == class_name
        if class_mask.sum() > 0:
            correct = (y_pred_classes[class_mask] == y_test_classes[class_mask]).sum()
            total = class_mask.sum()
            accuracy = correct / total
            print(f"{class_name}: {accuracy:.2%} ({correct}/{total} correct)")
    
    # Print confusion matrix as dataframe
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {name}" for name in target_encoder.classes_],
        columns=[f"Pred: {name}" for name in target_encoder.classes_]
    )
    print("\n" + "="*80)
    print("CONFUSION MATRIX (as DataFrame)")
    print("="*80)
    print(cm_df)
