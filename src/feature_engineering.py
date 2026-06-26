import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class Cell2CellFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    All feature engineering for Cell2Cell dataset.
    Fit/transform pattern: statistics computed on training data ONLY.
    ServiceArea has 747 unique values — target-encoded, not OHE.
    CustomerCareCalls is float (max=327.3) — kept as float throughout.
    """

    def __init__(self,
                 care_calls_threshold: float = 3.0,
                 overage_log: bool = True):
        self.care_calls_threshold = care_calls_threshold
        self.overage_log = overage_log
        # Target-encoding dictionaries (fit on train only)
        self.service_area_churn_rate_: dict = {}
        self.occupation_churn_rate_:   dict = {}
        self.prizm_churn_rate_:        dict = {}
        self.credit_churn_rate_:       dict = {}
        self.global_mean_: float = 0.288  # Cell2Cell training churn rate
        self.revenue_p75_: float | None = None
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y=None):
        """Learn group-level churn rates from training data"""
        if y is not None:
            df = X.copy()
            y_vals = y.values if hasattr(y, 'values') else np.array(y)
            df['__target__'] = y_vals
            self.global_mean_ = float(y_vals.mean())
            # Learn revenue threshold from TRAINING data only
            if "MonthlyRevenue" in df.columns:
                self.revenue_p75_ = float(
                    df["MonthlyRevenue"].quantile(0.75)
                )

            encoding_map = {
                'ServiceArea': 'service_area_churn_rate_',
                'Occupation':  'occupation_churn_rate_',
                'PrizmCode':   'prizm_churn_rate_',
                'CreditRating':'credit_churn_rate_',
            }
            for col, attr in encoding_map.items():
                if col in df.columns:
                    rate_dict = (df.groupby(col)['__target__']
                                 .mean().to_dict())
                    setattr(self, attr, rate_dict)
                    logger.debug(f"Target-encoded {col}: {len(rate_dict)} levels")

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ──────────────────────────────────────────────────────────────────────
        # TIER 1: Call behaviour and dissatisfaction signals
        # ──────────────────────────────────────────────────────────────────────

        # Escalated customer: called care AND retention team
        
        if all(c in df for c in ['CustomerCareCalls','RetentionCalls']):
            df['escalated_customer'] = (
                (df['CustomerCareCalls'] >= self.care_calls_threshold) &
                (df['RetentionCalls'] >= 1)
            ).astype(int)
            df['total_complaint_calls'] = (
                df['CustomerCareCalls'] + df['RetentionCalls']
            )
            # CustomerCareCalls is float — log-transform reduces outlier effect
            df['log_care_calls'] = np.log1p(df['CustomerCareCalls'])

        # Retention called but no offer accepted — very high risk
        if all(c in df for c in ['RetentionCalls','RetentionOffersAccepted']):
            df['retention_failed'] = (
                (df['RetentionCalls'] > 0) &
                (df['RetentionOffersAccepted'] == 0)
            ).astype(int)
            df['retention_success_rate'] = (
                df['RetentionOffersAccepted'] /
                (df['RetentionCalls'] + 1)
            )

        # ──────────────────────────────────────────────────────────────────────
        # TIER 2: Equipment and tenure signals
        # ──────────────────────────────────────────────────────────────────────

        if 'CurrentEquipmentDays' in df:
            df['old_equipment'] = (
                df['CurrentEquipmentDays'] > 365).astype(int)
            df['very_old_equipment'] = (
                df['CurrentEquipmentDays'] > 730).astype(int)
            df['equipment_age_months'] = df['CurrentEquipmentDays'] / 30.4
            df['log_equipment_days'] = np.log1p(df['CurrentEquipmentDays'])

        if 'MonthsInService' in df:
            # MonthsInService range: 6–61 in training data
            df['tenure_group'] = pd.cut(
                df['MonthsInService'],
                bins=[0, 12, 24, 36, 48, 100],
                labels=['very_new','new','growing','established','loyal']
            )
            df['early_tenure'] = (df['MonthsInService'] <= 12).astype(int)
            df['loyal_customer'] = (df['MonthsInService'] >= 48).astype(int)

        # ──────────────────────────────────────────────────────────────────────
        # TIER 3: Usage and revenue signals
        # ──────────────────────────────────────────────────────────────────────

        if 'OverageMinutes' in df:
            df['has_overage'] = (df['OverageMinutes'] > 0).astype(int)
            if self.overage_log:
                df['log_overage'] = np.log1p(df['OverageMinutes'])

        if all(c in df for c in ['MonthlyRevenue','MonthlyMinutes']):
            df['revenue_per_minute'] = (
                df['MonthlyRevenue'] / (df['MonthlyMinutes'] + 1)
            )

        if 'PercChangeRevenues' in df:
            df['revenue_declining'] = (
                df['PercChangeRevenues'] < -10).astype(int)
            df['revenue_growing']   = (
                df['PercChangeRevenues'] > 10).astype(int)

        if 'PercChangeMinutes' in df:
            df['usage_declining']        = (
                df['PercChangeMinutes'] < -20).astype(int)
            df['usage_declining_severe'] = (
                df['PercChangeMinutes'] < -50).astype(int)

        # Premium dissatisfied: high revenue + high care calls
        if all(c in df for c in ['MonthlyRevenue', 'CustomerCareCalls']):
        
            revenue_threshold = (
                self.revenue_p75_
                if self.revenue_p75_ is not None
                else float(df["MonthlyRevenue"].quantile(0.75))
            )

            df["premium_dissatisfied"] = (
                (df["MonthlyRevenue"] > revenue_threshold) &
                (df["CustomerCareCalls"] >= 2)
            ).astype(int)
        #if all(c in df for c in ['MonthlyRevenue','CustomerCareCalls']):
        #    rev_p75 = df['MonthlyRevenue'].quantile(0.75)
        #    df['premium_dissatisfied'] = (
        #        (df['MonthlyRevenue'] > rev_p75) &
        #        (df['CustomerCareCalls'] >= 2)
        #    ).astype(int)

        # ──────────────────────────────────────────────────────────────────────
        # TIER 4: Subscription and device signals
        # ──────────────────────────────────────────────────────────────────────

        if all(c in df for c in ['UniqueSubs','ActiveSubs']):
            df['is_family_plan'] = (df['UniqueSubs'] > 1).astype(int)
            df['inactive_subs_ratio'] = (
                (df['UniqueSubs'] - df['ActiveSubs']) /
                (df['UniqueSubs'] + 1)
            )

        if all(c in df for c in ['Handsets','HandsetModels']):
            df['device_diversity'] = df['Handsets'] * df['HandsetModels']

        if 'HandsetPrice' in df:
            df['handset_tier'] = pd.cut(
                df['HandsetPrice'],
                bins=[0, 50, 100, 200, 400, 600],
                labels=['budget','mid','premium','flagship','ultra']
            )
            df['premium_handset'] = (df['HandsetPrice'] >= 150).astype(int)

        # ──────────────────────────────────────────────────────────────────────
        # TIER 5: Credit risk ordinal encoding
        # ──────────────────────────────────────────────────────────────────────

        if 'CreditRating' in df:
            credit_map = {
                '1-Highest': 1, '2-High': 2, '3-Good': 3,
                '4-Medium':  4, '5-Low': 5, '6-VeryLow': 6, '7-Lowest': 7
            }
            df['credit_risk_score'] = (
                df['CreditRating'].map(credit_map).fillna(4)
            )  # unknown → medium
            df['is_low_credit'] = (
                df['credit_risk_score'] >= 5).astype(int)

        # ──────────────────────────────────────────────────────────────────────
        # TIER 6: Call pattern ratios
        # ──────────────────────────────────────────────────────────────────────

        call_cols = ['PeakCallsInOut','OffPeakCallsInOut',
                     'ReceivedCalls','OutboundCalls','InboundCalls']
        existing_calls = [c for c in call_cols if c in df.columns]
        if existing_calls:
            df['total_call_volume'] = df[existing_calls].sum(axis=1)

        if all(c in df for c in ['PeakCallsInOut','OffPeakCallsInOut']):
            df['peak_ratio'] = (
                df['PeakCallsInOut'] /
                (df['OffPeakCallsInOut'] + 1)
            )

        if all(c in df for c in ['DroppedCalls','MonthlyMinutes']):
            df['dropped_rate'] = (
                df['DroppedCalls'] /
                (df['MonthlyMinutes'] + 1)
            )

        # ──────────────────────────────────────────────────────────────────────
        # TIER 7: Target-encoded segment churn rates
        # Applied to BOTH train and test using rates learned from train only
        # ──────────────────────────────────────────────────────────────────────

        encoding_map = {
            'ServiceArea': ('service_area_churn_rate_',
                            'service_area_encoded'),
            'Occupation':  ('occupation_churn_rate_',
                            'occupation_encoded'),
            'PrizmCode':   ('prizm_churn_rate_',
                            'prizm_encoded'),
        }
        for col, (attr, new_col) in encoding_map.items():
            rate_dict = getattr(self, attr)
            if col in df.columns and rate_dict:
                df[new_col] = (
                    df[col].map(rate_dict)
                    .fillna(self.global_mean_)
                )

        logger.debug(f"FE complete. Shape: {df.shape}")
        return df


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for Telco XLSX dataset"""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Service add-on count
        service_cols = [
            'Online_Security','Online_Backup','Device_Protection',
            'Tech_Support','Streaming_TV','Streaming_Movies'
        ]
        df['service_count'] = sum(
            (df[c] == 'Yes').astype(int)
            for c in service_cols if c in df.columns
        )

        if 'Contract' in df.columns:
            df['is_month_to_month'] = (
                df['Contract'] == 'Month-to-month').astype(int)

        if 'Payment_Method' in df.columns:
            df['is_auto_pay'] = df['Payment_Method'].isin([
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ]).astype(int)
            df['is_echeck'] = (
                df['Payment_Method'] == 'Electronic check'
            ).astype(int)

        if 'Monthly_Charges' in df.columns:
            df['charge_per_service'] = (
                df['Monthly_Charges'] / (df['service_count'] + 1)
            )

        if 'CLTV' in df.columns:
            df['cltv_segment'] = pd.cut(
                df['CLTV'],
                bins=[0, 3000, 4500, 5500, 7000],
                labels=['low','medium','high','premium']
            )

        if 'Internet_Service' in df.columns:
            df['has_fiber'] = (
                df['Internet_Service'] == 'Fiber optic').astype(int)

        if 'Tenure_Months' in df.columns:
            df['tenure_group'] = pd.cut(
                df['Tenure_Months'],
                bins=[0, 12, 24, 36, 48, 72],
                labels=['very_new','new','growing','established','loyal']
            )

        return df