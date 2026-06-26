import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from loguru import logger
import joblib


def load_config() -> dict:
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CELL2CELL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_cell2cell_train(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load Cell2Cell training data.

    Returns:
        X : features
        y : encoded target
    """

    logger.info("Loading Cell2Cell training data...")

    df = pd.read_csv(
        config['data']['cell2cell_train']
    )

    logger.info(
        f"Loaded: {df.shape}"
    )

    # Target encoding
    y = (
        df['Churn'] == 'Yes'
    ).astype(int)

    logger.info(
        f"Churn rate: "
        f"{y.mean():.3f} "
        f"({y.sum()} / {len(y)} churners)"
    )

    # Drop unwanted columns
    drop_cols = (
        config['cell2cell_drop_cols']
        + ['Churn']
    )

    existing_drops = [
        c for c in drop_cols
        if c in df.columns
    ]

    df = df.drop(
        columns=existing_drops
    )

    logger.info(
        f"After drops: {df.shape}"
    )

    # HandsetPrice
    df['HandsetPrice'] = pd.to_numeric(
        df['HandsetPrice'],
        errors='coerce'
    )

    # Revenue cannot be negative
    df['MonthlyRevenue'] = (
        df['MonthlyRevenue']
        .clip(lower=0)
    )

    # Equipment days cannot be negative
    df['CurrentEquipmentDays'] = (
        df['CurrentEquipmentDays']
        .clip(lower=0)
    )

    return df, y


# ─────────────────────────────────────────────────────────────────────────────
# CELL2CELL HOLDOUT
# ─────────────────────────────────────────────────────────────────────────────

def load_cell2cell_holdout(config: dict):

    """
    Holdout may not contain labels.

    Returns:
        X only if Churn missing.
        X, y if labels exist.
    """

    logger.warning(
        "Loading holdout — only use for final evaluation."
    )

    df = pd.read_csv(
        config['data']['cell2cell_holdout']
    )

    if 'Churn' in df.columns:

        if df['Churn'].isna().all():

            logger.info(
                "Holdout contains no labels."
            )

            df = df.drop(
                columns=['Churn']
            )

        else:

            y = (
                df['Churn']
                == 'Yes'
            ).astype(int)

    drop_cols = [

        c

        for c in config[
            'cell2cell_drop_cols'
        ]

        if c in df.columns
    ]

    df = df.drop(
        columns=drop_cols
    )

    df['HandsetPrice'] = pd.to_numeric(
        df['HandsetPrice'],
        errors='coerce'
    )

    df['MonthlyRevenue'] = (
        df['MonthlyRevenue']
        .clip(lower=0)
    )

    df['CurrentEquipmentDays'] = (
        df['CurrentEquipmentDays']
        .clip(lower=0)
    )

    if 'y' in locals():
        return df, y

    return df


# ─────────────────────────────────────────────────────────────────────────────
# TELCO LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_telco(config: dict):

    logger.info(
        "Loading Telco XLSX..."
    )

    df = pd.read_excel(
        config['data']['telco_xlsx']
    )

    logger.info(
        f"Loaded: {df.shape}"
    )

    y = df['Churn Value'].astype(
        int
    )

    # Fix Total Charges
    df['Total Charges'] = pd.to_numeric(
        df['Total Charges'],
        errors='coerce'
    )

    df['Total Charges'] = (
        df['Total Charges']
        .fillna(0.0)
    )

    drop_cols = (
        config['telco_drop_cols']
        + ['Churn Value']
    )

    existing_drops = [

        c

        for c in drop_cols

        if c in df.columns
    ]

    df = df.drop(
        columns=existing_drops
    )

    df.columns = (
        df.columns
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )

    if df['Senior_Citizen'].dtype == object:

        df['Senior_Citizen'] = (
            df['Senior_Citizen']
            == 'Yes'
        ).astype(int)

    logger.info(
        f"Total_Charges NaN: "
        f"{df['Total_Charges'].isna().sum()}"
    )

    logger.info(
        f"Senior_Citizen values: "
        f"{df['Senior_Citizen'].unique()}"
    )

    return df, y


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config['data']['test_split'],
        random_state=config['data']['random_state'],
        stratify=y
    )

    logger.info(
        f"Train: {X_train.shape} | "
        f"Churn rate: {y_train.mean():.3f}"
    )

    logger.info(
        f"Test: {X_test.shape} | "
        f"Churn rate: {y_test.mean():.3f}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHURN REASONS
# ─────────────────────────────────────────────────────────────────────────────

def get_churn_reasons(config: dict):

    df = pd.read_excel(
        config['data']['telco_xlsx']
    )

    reasons = (
        df['Churn Reason']
        .dropna()
        .value_counts()
    )

    logger.info(
        "Top churn reasons:"
    )

    logger.info(
        f"\n{reasons.head(10)}"
    )

    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    config = load_config()

    # Load Cell2Cell
    X, y = load_cell2cell_train(
        config
    )

    X_train, X_test, y_train, y_test = (
        split_data(
            X,
            y,
            config
        )
    )

    # ------------------------------------------------------------------
    # CHANGE
    #
    # WHY:
    # Future phases (feature engineering, modeling, validation)
    # can directly load these splits.
    # ------------------------------------------------------------------

    joblib.dump(

        (
            X_train,
            X_test,
            y_train,
            y_test
        ),

        "data/processed/cell2cell_split.pkl"
    )

    logger.info(
        "Saved: data/processed/cell2cell_split.pkl"
    )

    logger.info(
        f"HandsetPrice dtype: "
        f"{X['HandsetPrice'].dtype}"
    )

    logger.info(
        f"Unknown strings remaining: "
        f"{X['HandsetPrice'].astype(str).str.contains('Unknown').sum()}"
    )

    logger.info(
        f"Minimum MonthlyRevenue: "
        f"{X['MonthlyRevenue'].min()}"
    )

    logger.info(
        f"Minimum CurrentEquipmentDays: "
        f"{X['CurrentEquipmentDays'].min()}"
    )

    logger.info(
        "Top missing values:\n"
        f"{X.isna().sum().sort_values(ascending=False).head(10)}"
    )

    # ------------------------------------------------------------------
    # TELCO VERIFICATION
    # ------------------------------------------------------------------

    X_telco, y_telco = load_telco(
        config
    )

    # ------------------------------------------------------------------
    # CHANGE
    #
    # WHY:
    # data_validation.py loads:
    #
    # data/processed/telco_clean.pkl
    #
    # Without this file:
    #
    # FileNotFoundError:
    # data/processed/telco_clean.pkl
    #
    # ------------------------------------------------------------------

    joblib.dump(

        (
            X_telco,
            y_telco
        ),

        "data/processed/telco_clean.pkl"
    )

    logger.info(
        "Saved: data/processed/telco_clean.pkl"
    )

    logger.info(
        f"Telco shape: "
        f"{X_telco.shape}"
    )

    logger.info(
        f"Telco churn rate: "
        f"{y_telco.mean():.3f}"
    )

    logger.info(
        f"Total_Charges NaN: "
        f"{X_telco['Total_Charges'].isna().sum()}"
    )

    logger.info(
        f"Senior_Citizen values: "
        f"{X_telco['Senior_Citizen'].unique()}"
    )

    reasons = get_churn_reasons(
        config
    )

    logger.success(
        "Data loading complete."
    )