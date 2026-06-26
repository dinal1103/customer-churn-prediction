# ============================================================
# DATA VALIDATION PIPELINE
# ============================================================
#
# PURPOSE:
# Validate cleaned datasets BEFORE feature engineering.
#
# Pipeline:
#
# Raw CSV
#      ↓
# data_loader.py
#      ↓
# data_validation.py  ← YOU ARE HERE
#      ↓
# feature_engineering.py
#      ↓
# model_training.py
#
# If validation fails:
#
#     STOP THE PIPELINE
#
# Because training a model on bad data is worse than
# not training a model at all.
#
# ============================================================

import pandas as pd
import numpy as np

# Great Expectations:
# Industry library for data testing.
# Think:
#
# pytest ----> tests code
# Great Expectations ----> tests data
#
import great_expectations as ge

# Better logging than print()
from loguru import logger

# Save reports
import json
import os


# ============================================================
# CELL2CELL VALIDATION
# ============================================================
def validate_cell2cell(
    df: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Validate Cell2Cell training data.

    INPUT:
        df = features only
        y  = target column

    OUTPUT:
        dictionary containing validation results

    WHY:
        This function acts like automated QA.

    Example:

        CustomerCareCalls suddenly becomes 5000.

    Validation catches it immediately.

    Instead of discovering the problem after training.
    """

    # Dictionary to store all results.
    # This eventually becomes JSON.
    results = {}

    # ========================================================
    # 1. SHAPE VALIDATION
    # ========================================================
    #
    # Question:
    #
    # Did we load the correct amount of data?
    #
    # If expected:
    #
    #     51047 rows
    #
    # but loaded:
    #
    #     2000 rows
    #
    # something is seriously wrong.
    #
    # Assertions stop execution immediately.
    #
    assert df.shape[0] >= 50000, (
        f"Expected >=50000 rows, got {df.shape[0]}"
    )

    assert df.shape[1] >= 50, (
        f"Expected >=50 columns, got {df.shape[1]}"
    )

    results["shape_ok"] = True
    results["shape"] = list(df.shape)

    logger.info(f"Shape OK: {df.shape}")

    # ========================================================
    # 2. TARGET DISTRIBUTION
    # ========================================================
    #
    # The current churn rate is roughly:
    #
    # 28.8%
    #
    # Tomorrow if we suddenly get:
    #
    # 4%
    #
    # then:
    #
    # - wrong file
    # - wrong encoding
    # - loader bug
    #
    # Therefore:
    #
    # expected range:
    #
    # 25%–35%
    #
    churn_rate = float(y.mean())

    assert 0.25 <= churn_rate <= 0.35, (
        f"Unexpected churn rate {churn_rate:.4f}"
    )

    results["churn_rate"] = round(churn_rate, 4)

    logger.info(
        f"Churn rate {churn_rate:.4f} OK"
    )

    # ========================================================
    # 3. TARGET LEAKAGE CHECK
    # ========================================================
    #
    # Churn must NEVER exist inside features.
    #
    # If it exists:
    #
    # model learns answer directly.
    #
    # ROC-AUC = 1.00
    #
    # Completely fake.
    #
    assert "Churn" not in df.columns, (
        "Target leakage detected."
    )

    results["no_target_leak"] = True

    # ========================================================
    # 4. CONFIRM DROPPED COLUMNS
    # ========================================================
    #
    # Phase 1 decided these columns are useless.
    #
    # If they appear again:
    #
    # data_loader.py is broken.
    #
    must_be_absent = [
        "CustomerID",
        "NotNewCellphoneUser",
        "CallForwardingCalls",
        "ReferralsMadeBySubscriber",
        "BlockedCalls"
    ]

    for col in must_be_absent:

        assert col not in df.columns, (
            f"{col} should have been dropped."
        )

    results["dropped_cols_absent"] = True

    # ========================================================
    # 5. KNOWN BUG FIXES
    # ========================================================
    #
    # These bugs existed in raw data.
    #
    # Loader fixed them.
    #
    # Validation confirms the fixes happened.
    #
    # MonthlyRevenue:
    #
    # raw minimum:
    #
    # -6.17
    #
    # after clipping:
    #
    # 0
    #
    assert df["MonthlyRevenue"].min() >= 0, (
        "MonthlyRevenue clipping failed."
    )

    # CurrentEquipmentDays had negatives.

    assert df["CurrentEquipmentDays"].min() >= 0, (
        "CurrentEquipmentDays clipping failed."
    )

    # HandsetPrice originally contained:
    #
    # "Unknown"
    #
    # After:
    #
    # pd.to_numeric(..., errors='coerce')
    #
    # it becomes float.
    #
    assert df["HandsetPrice"].dtype in [
        np.float64,
        np.float32
    ], (
        f"HandsetPrice should be float, "
        f"got {df['HandsetPrice'].dtype}"
    )

    results["bug_fixes_applied"] = True

    # ========================================================
    # GREAT EXPECTATIONS
    # ========================================================
    #
    # Convert pandas dataframe:
    #
    # df
    #
    # into:
    #
    # gdf
    #
    # which supports:
    #
    # expect_...
    #
    gdf = ge.from_pandas(df)

    # Store every expectation result.
    checks = []

    # ========================================================
    # NUMERICAL RANGES
    # ========================================================
    #
    # Why ranges?
    #
    # Example:
    #
    # CustomerCareCalls = 5000
    #
    # Obviously impossible.
    #
    checks.append(
        gdf.expect_column_values_to_be_between(
            "MonthsInService",
            0,
            100
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "MonthlyRevenue",
            0,
            2000
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "CurrentEquipmentDays",
            0,
            3000
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "CustomerCareCalls",
            0,
            500
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "RetentionCalls",
            0,
            10
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "RetentionOffersAccepted",
            0,
            10
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "IncomeGroup",
            0,
            9
        )
    )

    # ========================================================
    # CATEGORICAL VALIDATION
    # ========================================================
    #
    # These columns have limited values.
    #
    # Example:
    #
    # CreditRating cannot suddenly become:
    #
    # "Excellent"
    #
    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "CreditRating",
            [
                "1-Highest",
                "2-High",
                "3-Good",
                "4-Medium",
                "5-Low",
                "6-VeryLow",
                "7-Lowest"
            ]
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "PrizmCode",
            [
                "Suburban",
                "Town",
                "Rural",
                "Other"
            ]
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "MaritalStatus",
            [
                "Yes",
                "No",
                "Unknown"
            ]
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "Homeownership",
            [
                "Known",
                "Unknown"
            ]
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "Occupation",
            [
                "Professional",
                "Crafts",
                "Other",
                "Self",
                "Retired",
                "Homemaker",
                "Clerical",
                "Student"
            ]
        )
    )

    # ========================================================
    # BINARY COLUMNS
    # ========================================================
    #
    # Expected:
    #
    # Yes
    # No
    #
    # Not:
    #
    # Y
    # N
    # TRUE
    #
    binary_cols = [
        "ChildrenInHH",
        "HandsetRefurbished",
        "HandsetWebCapable",
        "TruckOwner",
        "RVOwner",
        "BuysViaMailOrder",
        "RespondsToMailOffers",
        "OptOutMailings",
        "NonUSTravel",
        "OwnsComputer",
        "HasCreditCard",
        "NewCellphoneUser",
        "OwnsMotorcycle",
        "MadeCallToRetentionTeam"
    ]

    for col in binary_cols:

        if col in df.columns:

            checks.append(
                gdf.expect_column_values_to_be_in_set(
                    col,
                    ["Yes", "No"]
                )
            )

    # ========================================================
    # CRITICAL NULL CHECKS
    # ========================================================
    #
    # Some columns are allowed to have missing values.
    #
    # Some are not.
    #
    # Example:
    #
    # MonthsInService missing:
    #
    # impossible.
    #

    critical_no_null = [
    "MonthsInService",
    "CustomerCareCalls",
    "RetentionCalls"
    ]

    for col in critical_no_null:

        if col in df.columns:

            checks.append(
                gdf.expect_column_values_to_not_be_null(
                    col
                )
            )

    # ========================================================
    # SUMMARY
    # ========================================================
    #
    # Great Expectations returns dictionaries.
    #
    # Each contains:
    #
    # success = True/False
    #
    passed = sum(
        1 for c in checks
        if c["success"]
    )

    failed = len(checks) - passed

    results["checks_passed"] = passed
    results["checks_failed"] = failed
    results["all_passed"] = failed == 0

    # ========================================================
    # SHOW FAILURES
    # ========================================================
    #
    # Example output:
    #
    # FAIL:
    # expect_column_values_to_be_between
    # on CustomerCareCalls
    #
    for c in checks:

        if not c["success"]:

            col = c["expectation_config"][
                "kwargs"
            ].get("column", "")

            exp = c["expectation_config"][
                "expectation_type"
            ]

            logger.error(
                f"FAIL: {exp} on {col}"
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================
    if failed == 0:

        logger.success(
            f"All {passed} checks passed ✓"
        )

    else:

        logger.error(
            f"{failed} validation checks failed."
        )

    return results


# ============================================================
# TELCO VALIDATION
# ============================================================
def validate_telco(
    df: pd.DataFrame,
    y: pd.Series
):

    results = {}

    # Exact row count.
    #
    # Telco always has 7043 rows.
    #
    assert df.shape[0] == 7043

    results["shape_ok"] = True

    # Churn rate check.

    churn_rate = float(y.mean())

    assert 0.20 <= churn_rate <= 0.35

    results["churn_rate"] = round(
        churn_rate,
        4
    )

    # Senior Citizen conversion check.
    #
    # Originally:
    #
    # Yes/No
    #
    # After conversion:
    #
    # 0/1
    #
    assert df["Senior_Citizen"].dtype in [
        np.int64,
        np.int32,
        int
    ]

    # Total Charges bug check.
    #
    # Some rows had empty strings.
    #
    assert (
        df["Total_Charges"]
        .isna()
        .sum()
        == 0
    )

    gdf = ge.from_pandas(df)

    checks = []

    checks.append(
        gdf.expect_column_values_to_be_between(
            "Tenure_Months",
            0,
            72
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "Monthly_Charges",
            0,
            200
        )
    )

    checks.append(
        gdf.expect_column_values_to_not_be_null(
            "Total_Charges"
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_in_set(
            "Internet_Service",
            [
                "Fiber optic",
                "DSL",
                "No"
            ]
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "Churn_Score",
            0,
            100
        )
    )

    checks.append(
        gdf.expect_column_values_to_be_between(
            "CLTV",
            2000,
            7000
        )
    )

    passed = sum(
        1 for c in checks
        if c["success"]
    )

    failed = len(checks) - passed

    results["checks_passed"] = passed
    results["checks_failed"] = failed
    results["all_passed"] = failed == 0

    return results


# ============================================================
# MAIN EXECUTION
# ============================================================
#
# This runs only if:
#
# python src/data_validation.py
#
# is executed directly.
#
# It will:
#
# 1. Load saved datasets.
# 2. Run validation.
# 3. Save JSON report.
#
if __name__ == "__main__":

    from src.data_loader import (
        load_config,
        load_cell2cell_train,
        load_telco
    )

    config = load_config()

    # Full Cell2Cell dataset
    X_c2c, y_c2c = load_cell2cell_train(config)

    c2c_results = validate_cell2cell(
        X_c2c,
        y_c2c
    )

    # Telco dataset
    X_telco, y_telco = load_telco(config)

    tel_results = validate_telco(
        X_telco,
        y_telco
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    with open(
        "reports/validation_report.json",
        "w"
    ) as f:

        json.dump(
            {
                "cell2cell": c2c_results,
                "telco": tel_results
            },
            f,
            indent=2
        )

    logger.success(
        "Validation report saved."
    )
