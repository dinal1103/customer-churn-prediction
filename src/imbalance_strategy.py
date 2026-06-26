import numpy as np
import pandas as pd
import os
import joblib
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from src.data_loader import load_config
from src.pipeline_builder import build_full_pipeline
from src.paths import REPORTS_DIR,PROCESSED_DATA_DIR




def compute_class_weights(y_train: pd.Series) -> dict:
    """
    Compute class weights for XGBoost scale_pos_weight parameter.
    Cell2Cell: neg=36,336 / pos=14,711 ≈ 2.47
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    if pos == 0:
        raise ValueError("No positive samples found in y_train.")

    scale_pos_weight = neg / pos

    logger.info(f"Class distribution:")
    logger.info(f"  Negative (No Churn): {neg:,} ({neg/len(y_train):.3f})")
    logger.info(f"  Positive (Churn):    {pos:,} ({pos/len(y_train):.3f})")
    logger.info(f"  scale_pos_weight:    {scale_pos_weight:.4f}")

    return {
        'neg': int(neg),
        'pos': int(pos),
        'scale_pos_weight': round(float(scale_pos_weight), 4),
        'ratio': f"{neg/pos:.2f}:1"
    }


def find_optimal_threshold(model,
                            X_val: pd.DataFrame,
                            y_val: pd.Series,
                            fp_cost: float = 200.0,
                            fn_cost: float = 1000.0) -> dict:
    """
    Find threshold that maximises expected business profit.

    Args:
        model:    fitted pipeline with predict_proba
        X_val:    validation features (test split, NOT holdout)
        y_val:    validation labels
        fp_cost:  cost of calling a non-churner
        fn_cost:  cost of missing a churner

    Returns:
        dict with cost-optimal, F1-optimal, and business-optimal thresholds
    """
    probs = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, probs)

    # Method 1: F1-optimal threshold
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    f1_threshold = float(thresholds[f1_scores[:-1].argmax()])

    # Method 2: Cost-optimal (analytical)
    cost_threshold = fp_cost / (fp_cost + fn_cost)  # 200/1200 = 0.1667

    # Method 3: Business-optimal (maximise profit on validation set)
    best_profit = -np.inf
    best_t = 0.35
    threshold_results = []
    for t in np.arange(0.01, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        tn = int(((preds == 0) & (y_val == 0)).sum())

        # Profit = churners saved - retention campaign cost
        profit = tp * fn_cost - (tp + fp) * fp_cost - fn * fn_cost
        threshold_results.append({
            'threshold': round(t, 2), 'tp': tp, 'fp': fp,
            'fn': fn, 'tn': tn, 'profit': int(profit)
        })

        if profit > best_profit:
            best_profit = profit
            best_t = t

    logger.info(f"Threshold analysis:")
    logger.info(f"  Cost-optimal:     {cost_threshold:.3f}")
    logger.info(f"  F1-optimal:       {f1_threshold:.3f}")
    logger.info(f"  Business-optimal: {best_t:.3f}  "
                f"(profit=₹{best_profit:,.0f})")

    # Save threshold analysis
    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(threshold_results).to_csv(
        REPORTS_DIR/'threshold_analysis.csv', index=False)

    return {
        'cost_optimal':     round(cost_threshold, 4),
        'f1_optimal':       round(f1_threshold, 4),
        'business_optimal': round(float(best_t), 4),
        'best_profit':      int(best_profit),
        'recommended':      round(float(best_t), 4),
    }


if __name__ == "__main__":


    config = load_config()
    X_train, X_test, y_train, y_test = joblib.load(
        PROCESSED_DATA_DIR/'cell2cell_split.pkl')

    weights = compute_class_weights(y_train)
    print(f"\nscale_pos_weight for XGBoost: {weights['scale_pos_weight']}")

    # Quick LR model for threshold analysis example
    lr = LogisticRegression(C=0.1, max_iter=1000, class_weight='balanced')
    pipeline = build_full_pipeline(lr, X_train.iloc[:500], y_train.iloc[:500])
    pipeline.fit(X_train, y_train)

    thresholds = find_optimal_threshold(pipeline, X_test, y_test)
    print(f"\nRecommended threshold: {thresholds['recommended']}")
    print(f"{REPORTS_DIR / 'threshold_analysis.csv'} saved.") 