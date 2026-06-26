import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger

from src.feature_engineering import Cell2CellFeatureEngineer


# ── Feature type definitions ──────────────────────────────────────────────────
# These are checked for existence at runtime — missing columns are skipped

# Raw numerical features (present before FE)
RAW_NUM_FEATURES = [
    'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
    'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
    'PercChangeMinutes', 'PercChangeRevenues', 'DroppedCalls',
    'UnansweredCalls', 'CustomerCareCalls',    # float, max=327.3
    'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls',
    'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls',
    'CallWaitingCalls', 'MonthsInService', 'UniqueSubs', 'ActiveSubs',
    'Handsets', 'HandsetModels', 'CurrentEquipmentDays',
    'AgeHH1', 'AgeHH2', 'RetentionCalls', 'RetentionOffersAccepted',
    'IncomeGroup', 'AdjustmentsToCreditRating', 'HandsetPrice',
]

# Engineered numerical features (present after FE)
ENG_NUM_FEATURES = [
    'total_complaint_calls', 'log_care_calls', 'retention_success_rate',
    'equipment_age_months', 'log_equipment_days', 'log_overage',
    'revenue_per_minute', 'device_diversity', 'total_call_volume',
    'peak_ratio', 'dropped_rate', 'inactive_subs_ratio',
    'credit_risk_score', 'service_area_encoded', 'occupation_encoded',
    'prizm_encoded',
]

# All numerical (raw + engineered combined)
ALL_NUM_FEATURES = RAW_NUM_FEATURES + ENG_NUM_FEATURES

# Categorical → OneHotEncoder
# NOTE: ServiceArea (747 unique) is target-encoded → goes to NUM, not here
# NOTE: Occupation and PrizmCode also target-encoded → go to NUM
# Only low-cardinality categoricals with natural groupings go to OHE
CAT_OHE_FEATURES = [
    'Homeownership',          # Known/Unknown — 2 values
    'BuysViaMailOrder',       # Yes/No
    'RespondsToMailOffers',   # Yes/No
    'OptOutMailings',         # Yes/No
    'NonUSTravel',            # Yes/No
    'OwnsComputer',           # Yes/No
    'HasCreditCard',          # Yes/No
    'NewCellphoneUser',       # Yes/No
    'OwnsMotorcycle',         # Yes/No
    'MadeCallToRetentionTeam',# Yes/No — critical feature
    'ChildrenInHH',           # Yes/No
    'HandsetRefurbished',     # Yes/No
    'HandsetWebCapable',      # Yes/No
    'TruckOwner',             # Yes/No
    'RVOwner',                # Yes/No
    'MaritalStatus',          # Yes/No/Unknown — 3 values (Unknown ≠ No)
    'tenure_group',           # very_new/new/growing/established/loyal
    'handset_tier',           # budget/mid/premium/flagship/ultra
]

# Categorical → OrdinalEncoder (natural order)
CAT_ORD_FEATURES = ['CreditRating']
CAT_ORD_ORDER = [[
    '1-Highest','2-High','3-Good','4-Medium',
    '5-Low','6-VeryLow','7-Lowest'
]]

# Binary engineered features (already 0/1) → impute only
BINARY_FEATURES = [
    'escalated_customer', 'retention_failed', 'has_overage',
    'old_equipment', 'very_old_equipment', 'revenue_declining',
    'revenue_growing', 'usage_declining', 'usage_declining_severe',
    'premium_dissatisfied', 'is_family_plan', 'premium_handset',
    'is_low_credit', 'early_tenure', 'loyal_customer',
]


# ── Sub-pipelines ─────────────────────────────────────────────────────────────

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

cat_ohe_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='if_binary',
    )),
])

cat_ord_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=CAT_ORD_ORDER,
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    )),
])

binary_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
])


def build_preprocessor(X_transformed: pd.DataFrame) -> ColumnTransformer:
    """
    Build ColumnTransformer using only columns that actually exist
    in X after feature engineering. Safely skips missing columns.
    """
    
    def existing(lst, group_name):
        existing_cols = [
            c for c in lst
            if c in X_transformed.columns
        ]

        missing_cols = sorted(
            set(lst) - set(existing_cols)
        )

        if missing_cols:
            logger.warning(
                f"{group_name}: skipped missing columns -> {missing_cols}"
            )

        return existing_cols

    e_num = existing(
        ALL_NUM_FEATURES,
        "Numerical"
    )

    e_ohe = existing(
        CAT_OHE_FEATURES,
        "OneHot"
    )

    e_ord = existing(
        CAT_ORD_FEATURES,
        "Ordinal"
    )

    e_binary = existing(
        BINARY_FEATURES,
        "Binary"
    )    

    logger.debug(f"Preprocessor columns — "
                 f"num:{len(e_num)} ohe:{len(e_ohe)} "
                 f"ord:{len(e_ord)} binary:{len(e_binary)}")

    return ColumnTransformer(
        transformers=[
            ('num',    num_pipe,     e_num),
            ('ohe',    cat_ohe_pipe, e_ohe),
            ('ord',    cat_ord_pipe, e_ord),
            ('binary', binary_pipe,  e_binary),
        ],
        remainder='drop',
        verbose_feature_names_out=True,
    )


def build_full_pipeline(classifier,
                         X_train_sample: pd.DataFrame,
                         y_train_sample: pd.Series,
                         use_smote: bool = False) -> Pipeline:
    """
    Build complete pipeline: FE → Preprocessor → (SMOTE) → Classifier.

    Args:
        classifier:      sklearn-compatible estimator
        X_train_sample:  small sample of training data (used to detect
                         which engineered columns exist — NOT fitted on)
        y_train_sample:  corresponding labels for sample
        use_smote:       if True, uses ImbPipeline with SMOTETomek
    """
    #fe = Cell2CellFeatureEngineer()
    #X_sample_t = fe.fit_transform(X_train_sample, y_train_sample)
    #preprocessor = build_preprocessor(X_sample_t)
    # ------------------------------------------------------------------
    # UPDATE:
    #
    # Previously we created a temporary Feature Engineer and used it
    # to discover which engineered columns existed.
    # That worked, but it created a separate fitted Feature Engineer
    # outside the pipeline.
    # Now we explicitly fit this temporary FE only for schema discovery.
    # The actual pipeline still contains its own Feature Engineer that
    # is fitted during pipeline.fit().
    # This keeps preprocessing column detection deterministic while
    # avoiding any accidental dependence on an unfitted transformer.
    # ------------------------------------------------------------------

    schema_fe = Cell2CellFeatureEngineer()

    schema_fe.fit(X_train_sample, y_train_sample)

    X_sample_t = schema_fe.transform(X_train_sample)

    preprocessor = build_preprocessor(X_sample_t)

    if use_smote:
        from imblearn.combine import SMOTETomek
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([
            ('fe',    Cell2CellFeatureEngineer()),
            ('pre',   preprocessor),
            ('smote', SMOTETomek(random_state=42, sampling_strategy=0.6)),
            ('clf',   classifier),
        ])
    else:
        return Pipeline([
            ('fe',  Cell2CellFeatureEngineer()),
            ('pre', preprocessor),
            ('clf', classifier),
        ])


#def get_feature_names_after_preprocessing(pipeline: Pipeline,
#                                            X_sample: pd.DataFrame,
#                                            y_sample: pd.Series) -> list[str]:
#    """
#    Extract human-readable feature names after ColumnTransformer.
#    Used by SHAP explainer to label features.
#    """
#    fe = Cell2CellFeatureEngineer()
#    X_t = fe.fit_transform(X_sample, y_sample)
#    pre = pipeline.named_steps['pre']
#    pre.fit(X_t, y_sample)
#    return list(pre.get_feature_names_out())
def get_feature_names_after_preprocessing(
    pipeline: Pipeline
) -> list[str]:
    """
    Return feature names from the already-fitted preprocessing step.

    ------------------------------------------------------------------
    UPDATE
    ------------------------------------------------------------------
    OLD BEHAVIOUR
    -------------
    The previous implementation created a NEW Feature Engineer,
    transformed the sample data again and then called:

        pre.fit(...)

    on the already-trained ColumnTransformer.

    This unintentionally re-fitted the preprocessing pipeline,
    meaning the encoder categories could change depending on the
    supplied sample.

    Example:

        Pipeline trained on 40,837 rows
                ↓
        Notebook calls this function
                ↓
        Preprocessor re-fitted on only 200 rows
                ↓
        Different number of encoded columns
                ↓
        Prediction errors such as:

            X has 90 features
            LogisticRegression expects 92 features

    NEW BEHAVIOUR
    -------------
    The preprocessing pipeline is already fitted during:

        pipeline.fit(X_train, y_train)

    Therefore we simply return the feature names from the fitted
    ColumnTransformer without fitting anything again.

    This guarantees that the feature names always match the
    transformed data used by the classifier and is also the
    recommended scikit-learn approach.
    ------------------------------------------------------------------
    """

    return list(
        pipeline.named_steps["pre"].get_feature_names_out()
    )