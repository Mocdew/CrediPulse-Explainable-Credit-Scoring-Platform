import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def get_model_path():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model (1).pkl"),
        os.path.abspath("Model (1).pkl"),
        r"c:\Users\user\Documents\Credit\Model (1).pkl"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

MODEL_FILE = get_model_path()

CATEGORY_MAPS = {
    'checking_status': [
        '< 0 DM',
        '0-200 DM',
        '>= 200 DM',
        'no checking account'
    ],
    'credit_history': [
        'critical account / other credits elsewhere',
        'existing credits paid duly till now',
        'delay in paying off in the past',
        'all credits at this bank paid duly',
        'no credits taken / all paid duly'
    ],
    'purpose': [
        'radio/television',
        'car (new)',
        'furniture/equipment',
        'car (used)',
        'business',
        'education',
        'repairs',
        'domestic appliances',
        'retraining',
        'others'
    ],
    'savings_status': [
        '< 100 DM',
        '100-500 DM',
        '500-1000 DM',
        '>= 1000 DM',
        'unknown/no savings account'
    ],
    'employment_since': [
        'unemployed',
        '< 1 year',
        '1-4 years',
        '4-7 years',
        '>= 7 years'
    ],
    'personal_status_sex': [
        'male: single',
        'female: divorced/separated/married',
        'male: married/widowed',
        'male: divorced/separated',
        'female: single'
    ],
    'other_debtors': [
        'none',
        'guarantor',
        'co-applicant'
    ],
    'property': [
        'real estate',
        'building society savings/life insurance',
        'car or other property',
        'unknown/no property'
    ],
    'other_installment_plans': [
        'none',
        'bank',
        'stores'
    ],
    'housing': [
        'own',
        'rent',
        'for free'
    ],
    'job': [
        'skilled employee/official',
        'unskilled resident',
        'management/self-employed/highly qualified',
        'unemployed/unskilled non-resident'
    ],
    'telephone': [
        'none',
        'yes, registered'
    ],
    'foreign_worker': [
        'yes',
        'no'
    ]
}

NUMERIC_FIELDS = {
    'duration_months': {'min': 4, 'max': 72, 'default': 24, 'label': 'Loan Duration (Months)', 'step': 1},
    'credit_amount': {'min': 250, 'max': 20000, 'default': 3000, 'label': 'Credit Amount (DM)', 'step': 50},
    'installment_rate': {'min': 1, 'max': 4, 'default': 2, 'label': 'Installment Rate (% of Income)', 'step': 1},
    'residence_since': {'min': 1, 'max': 4, 'default': 2, 'label': 'Present Residence Since (Years)', 'step': 1},
    'age': {'min': 18, 'max': 80, 'default': 35, 'label': 'Age (Years)', 'step': 1},
    'existing_credits': {'min': 1, 'max': 4, 'default': 1, 'label': 'Number of Existing Credits', 'step': 1},
    'num_dependents': {'min': 1, 'max': 2, 'default': 1, 'label': 'Number of Dependents', 'step': 1}
}

FEATURE_ORDER = [
    'checking_status', 'duration_months', 'credit_history', 'purpose',
    'credit_amount', 'savings_status', 'employment_since', 'installment_rate',
    'personal_status_sex', 'other_debtors', 'residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'existing_credits',
    'job', 'num_dependents', 'telephone', 'foreign_worker'
]

PRESET_PERSONAS = {
    "prime": {
        "name": "Prime Applicant (Low Risk)",
        "description": "Established professional with high savings, owned home, critical/clean credit record.",
        "data": {
            "checking_status": ">= 200 DM",
            "duration_months": 12,
            "credit_history": "critical account / other credits elsewhere",
            "purpose": "car (new)",
            "credit_amount": 2000,
            "savings_status": ">= 1000 DM",
            "employment_since": ">= 7 years",
            "installment_rate": 2,
            "personal_status_sex": "male: single",
            "other_debtors": "none",
            "residence_since": 4,
            "property": "real estate",
            "age": 45,
            "other_installment_plans": "none",
            "housing": "own",
            "existing_credits": 2,
            "job": "management/self-employed/highly qualified",
            "num_dependents": 1,
            "telephone": "yes, registered",
            "foreign_worker": "yes"
        }
    },
    "borderline": {
        "name": "Borderline Applicant (Moderate Risk)",
        "description": "Young worker with modest savings, renting, medium loan amount.",
        "data": {
            "checking_status": "0-200 DM",
            "duration_months": 24,
            "credit_history": "existing credits paid duly till now",
            "purpose": "furniture/equipment",
            "credit_amount": 3500,
            "savings_status": "100-500 DM",
            "employment_since": "1-4 years",
            "installment_rate": 3,
            "personal_status_sex": "female: divorced/separated/married",
            "other_debtors": "none",
            "residence_since": 2,
            "property": "building society savings/life insurance",
            "age": 28,
            "other_installment_plans": "none",
            "housing": "rent",
            "existing_credits": 1,
            "job": "skilled employee/official",
            "num_dependents": 1,
            "telephone": "none",
            "foreign_worker": "yes"
        }
    },
    "high_risk": {
        "name": "High-Risk Applicant (High Default Chance)",
        "description": "Negative balance checking account, long loan duration, high installment rate, low savings.",
        "data": {
            "checking_status": "< 0 DM",
            "duration_months": 48,
            "credit_history": "no credits taken / all paid duly",
            "purpose": "education",
            "credit_amount": 8000,
            "savings_status": "< 100 DM",
            "employment_since": "< 1 year",
            "installment_rate": 4,
            "personal_status_sex": "female: divorced/separated/married",
            "other_debtors": "none",
            "residence_since": 1,
            "property": "unknown/no property",
            "age": 22,
            "other_installment_plans": "bank",
            "housing": "for free",
            "existing_credits": 1,
            "job": "unskilled resident",
            "num_dependents": 1,
            "telephone": "none",
            "foreign_worker": "yes"
        }
    }
}


class CreditModelEngine:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or get_model_path()
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def prepare_dataframe(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        row_data = {}
        for col in FEATURE_ORDER:
            val = input_dict.get(col)
            if val is None:
                if col in NUMERIC_FIELDS:
                    val = NUMERIC_FIELDS[col]['default']
                else:
                    val = CATEGORY_MAPS[col][0]
            elif col in NUMERIC_FIELDS:
                try:
                    val = float(val) if '.' in str(val) else int(val)
                except ValueError:
                    val = NUMERIC_FIELDS[col]['default']
            row_data[col] = val

        df = pd.DataFrame([row_data])
        for col, categories in CATEGORY_MAPS.items():
            df[col] = pd.Categorical(df[col], categories=categories)
        return df

    def predict_single(self, input_dict: Dict[str, Any], threshold: float = 0.17) -> Dict[str, Any]:
        df = self.prepare_dataframe(input_dict)

        prob = self.model.predict_proba(df)[0]
        prob_good = float(prob[0])
        prob_bad = float(prob[1])

        is_bad = bool(prob_bad >= threshold)
        decision = "DECLINED (High Risk)" if is_bad else "APPROVED (Low Risk)"
        status_code = "declined" if is_bad else "approved"

        safe_p_bad = max(min(prob_bad, 0.999), 0.001)
        odds = (1.0 - safe_p_bad) / safe_p_bad
        factor = 20.0 / np.log(2)
        offset = 600.0 - factor * np.log(50.0)
        raw_score = offset + factor * np.log(odds)
        credit_score = int(np.clip(raw_score, 300, 850))

        if prob_bad < 0.10:
            risk_tier = "A (Excellent - Very Low Risk)"
            tier_color = "#10b981"
        elif prob_bad < 0.20:
            risk_tier = "B (Good - Low Risk)"
            tier_color = "#06b6d4"
        elif prob_bad < 0.35:
            risk_tier = "C (Fair - Moderate Risk)"
            tier_color = "#f59e0b"
        elif prob_bad < 0.55:
            risk_tier = "D (Poor - Elevated Risk)"
            tier_color = "#f97316"
        else:
            risk_tier = "E (Very Poor - Critical Risk)"
            tier_color = "#ef4444"

        contribs = self.model.booster_.predict(df, pred_contrib=True)[0]
        feat_impacts = contribs[:-1]
        base_bias = float(contribs[-1])

        impact_df = pd.DataFrame({
            'feature': FEATURE_ORDER,
            'value': [str(df.iloc[0][f]) for f in FEATURE_ORDER],
            'impact': feat_impacts
        }).sort_values(by='impact', ascending=False)

        adverse_reasons = []
        for _, row in impact_df.head(4).iterrows():
            if row['impact'] > 0:
                feat_label = NUMERIC_FIELDS.get(row['feature'], {}).get('label', row['feature'].replace('_', ' ').title())
                adverse_reasons.append({
                    'feature': row['feature'],
                    'label': feat_label,
                    'value': row['value'],
                    'impact': round(float(row['impact']), 4),
                    'direction': 'risk_increasing'
                })

        positive_factors = []
        for _, row in impact_df.tail(4).iloc[::-1].iterrows():
            if row['impact'] < 0:
                feat_label = NUMERIC_FIELDS.get(row['feature'], {}).get('label', row['feature'].replace('_', ' ').title())
                positive_factors.append({
                    'feature': row['feature'],
                    'label': feat_label,
                    'value': row['value'],
                    'impact': round(float(row['impact']), 4),
                    'direction': 'risk_decreasing'
                })

        chart_features = []
        for _, row in impact_df.iterrows():
            feat_label = NUMERIC_FIELDS.get(row['feature'], {}).get('label', row['feature'].replace('_', ' ').title())
            chart_features.append({
                'feature': row['feature'],
                'label': feat_label,
                'value': row['value'],
                'impact': round(float(row['impact']), 4)
            })

        return {
            'decision': decision,
            'status_code': status_code,
            'prob_bad': round(prob_bad, 4),
            'prob_good': round(prob_good, 4),
            'prob_bad_pct': round(prob_bad * 100, 1),
            'prob_good_pct': round(prob_good * 100, 1),
            'credit_score': credit_score,
            'risk_tier': risk_tier,
            'tier_color': tier_color,
            'threshold_used': threshold,
            'base_bias': round(base_bias, 4),
            'adverse_reasons': adverse_reasons,
            'positive_factors': positive_factors,
            'chart_features': chart_features,
            'applicant_summary': {f: str(df.iloc[0][f]) for f in FEATURE_ORDER}
        }

    def predict_batch(self, df_input: pd.DataFrame, threshold: float = 0.17) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_clean = df_input.copy()
        for col in FEATURE_ORDER:
            if col not in df_clean.columns:
                if col in NUMERIC_FIELDS:
                    df_clean[col] = NUMERIC_FIELDS[col]['default']
                else:
                    df_clean[col] = CATEGORY_MAPS[col][0]
            elif col in NUMERIC_FIELDS:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(NUMERIC_FIELDS[col]['default'])

        df_for_model = df_clean[FEATURE_ORDER].copy()
        for col, categories in CATEGORY_MAPS.items():
            df_for_model[col] = pd.Categorical(df_for_model[col], categories=categories)

        probs = self.model.predict_proba(df_for_model)
        prob_bads = probs[:, 1]
        prob_goods = probs[:, 0]
        decisions = np.where(prob_bads >= threshold, "DECLINED", "APPROVED")

        odds = np.clip((1.0 - prob_bads) / np.clip(prob_bads, 0.001, 0.999), 0.001, 1000.0)
        factor = 20.0 / np.log(2)
        offset = 600.0 - factor * np.log(50.0)
        scores = np.clip(offset + factor * np.log(odds), 300, 850).astype(int)

        df_result = df_clean.copy()
        df_result['Prediction'] = decisions
        df_result['Default_Risk_Prob'] = np.round(prob_bads, 4)
        df_result['Approval_Prob'] = np.round(prob_goods, 4)
        df_result['Credit_Score'] = scores

        total = len(df_result)
        approved_count = int((decisions == "APPROVED").sum())
        declined_count = int((decisions == "DECLINED").sum())
        avg_risk = float(np.mean(prob_bads))
        avg_score = float(np.mean(scores))

        summary = {
            'total_applicants': total,
            'approved_count': approved_count,
            'declined_count': declined_count,
            'approval_rate_pct': round((approved_count / total) * 100, 1) if total > 0 else 0,
            'avg_default_risk_pct': round(avg_risk * 100, 1),
            'avg_credit_score': round(avg_score, 1),
            'threshold_used': threshold
        }

        return df_result, summary

engine = CreditModelEngine()
