"""Inference engine for the CrediPulse scoring API.

Loads the artifact written by the training notebook (`credit_model_bundle.joblib`) and
exposes single and batch scoring plus adverse-action reason codes.

Two things this module deliberately guarantees:

1. **Prohibited bases never reach a decision or a reason code.** `personal_status_sex` and
   `foreign_worker` are not collected by the form, not passed to the current model, and
   filtered out of every explanation. If the legacy pickle is in use they are imputed with
   a single constant so the feature cannot discriminate between applicants.
2. **The threshold comes from the artifact, not from a literal in the code.** The previous
   version hard-coded 0.17, a number derived from a flat 5:1 count-based cost that ignored
   loan size. Thresholds now arrive as an exposure-banded policy fitted in the notebook.
"""

import os
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Importing the module is what allows joblib to resolve the classes inside the bundle.
from scorecard import ScorecardModel, BandedPolicy  # noqa: F401  (needed for unpickling)

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_CANDIDATES = ['credit_model_bundle.joblib', 'credit_model_bundle.pkl']
LEGACY_CANDIDATES = ['Model (1).pkl']

# Prohibited bases under ECOA / Reg B — never collected, never scored, never explained.
PROTECTED_FEATURES = ['personal_status_sex', 'foreign_worker']

# Only used when falling back to the legacy 20-feature pickle, which structurally requires
# these columns. A single constant makes the feature carry no information about the
# applicant, so the model cannot discriminate on it even though it still consumes it.
LEGACY_FILL = {'personal_status_sex': 'male: single', 'foreign_worker': 'yes'}


def _resolve(candidates: List[str]) -> str:
    for name in candidates:
        for path in (os.path.join(HERE, name), os.path.abspath(name)):
            if os.path.exists(path):
                return path
    return ''


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
        'vacation',
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
    'other_debtors': [
        'none',
        'co-applicant',
        'guarantor'
    ],
    'property': [
        'real estate',
        'building society savings/life insurance',
        'car or other property',
        'unknown/no property'
    ],
    'other_installment_plans': [
        'bank',
        'stores',
        'none'
    ],
    'housing': [
        'rent',
        'own',
        'for free'
    ],
    'job': [
        'unemployed/unskilled non-resident',
        'unskilled resident',
        'skilled employee/official',
        'management/self-employed/highly qualified'
    ],
    'telephone': [
        'none',
        'yes, registered'
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

# The 18 attributes the model is permitted to see, in a stable order.
FEATURE_ORDER = [
    'checking_status', 'duration_months', 'credit_history', 'purpose',
    'credit_amount', 'savings_status', 'employment_since', 'installment_rate',
    'other_debtors', 'residence_since', 'property', 'age',
    'other_installment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'telephone',
]

FEATURE_LABELS = {
    'checking_status': 'Checking account status', 'duration_months': 'Loan duration (months)',
    'credit_history': 'Credit history', 'purpose': 'Loan purpose',
    'credit_amount': 'Credit amount', 'savings_status': 'Savings account status',
    'employment_since': 'Employment tenure', 'installment_rate': 'Instalment rate (% of income)',
    'other_debtors': 'Other debtors / guarantors', 'residence_since': 'Years at residence',
    'property': 'Property owned', 'age': 'Age',
    'other_installment_plans': 'Other instalment plans', 'housing': 'Housing status',
    'existing_credits': 'Existing credits at this bank', 'job': 'Employment category',
    'num_dependents': 'Number of dependents', 'telephone': 'Registered telephone',
}

PRESET_PERSONAS = {
    "prime": {
        "name": "Prime Applicant (Low Risk)",
        "description": "Established professional, high savings, owns property, clean credit record.",
        "data": {
            "checking_status": ">= 200 DM",
            "duration_months": 12,
            "credit_history": "critical account / other credits elsewhere",
            "purpose": "car (new)",
            "credit_amount": 2000,
            "savings_status": ">= 1000 DM",
            "employment_since": ">= 7 years",
            "installment_rate": 2,
            "other_debtors": "none",
            "residence_since": 4,
            "property": "real estate",
            "age": 45,
            "other_installment_plans": "none",
            "housing": "own",
            "existing_credits": 2,
            "job": "management/self-employed/highly qualified",
            "num_dependents": 1,
            "telephone": "yes, registered"
        }
    },
    "borderline": {
        "name": "Borderline Applicant (Moderate Risk)",
        "description": "Modest savings, renting, medium loan amount and term.",
        "data": {
            "checking_status": "0-200 DM",
            "duration_months": 24,
            "credit_history": "existing credits paid duly till now",
            "purpose": "furniture/equipment",
            "credit_amount": 3500,
            "savings_status": "100-500 DM",
            "employment_since": "1-4 years",
            "installment_rate": 3,
            "other_debtors": "none",
            "residence_since": 2,
            "property": "building society savings/life insurance",
            "age": 28,
            "other_installment_plans": "none",
            "housing": "rent",
            "existing_credits": 1,
            "job": "skilled employee/official",
            "num_dependents": 1,
            "telephone": "none"
        }
    },
    "high_risk": {
        "name": "High-Risk Applicant (High Default Chance)",
        "description": "Negative checking balance, long term, high instalment burden, low savings.",
        "data": {
            "checking_status": "< 0 DM",
            "duration_months": 48,
            "credit_history": "no credits taken / all paid duly",
            "purpose": "education",
            "credit_amount": 8000,
            "savings_status": "< 100 DM",
            "employment_since": "< 1 year",
            "installment_rate": 4,
            "other_debtors": "none",
            "residence_since": 1,
            "property": "unknown/no property",
            "age": 22,
            "other_installment_plans": "bank",
            "housing": "for free",
            "existing_credits": 1,
            "job": "unskilled resident",
            "num_dependents": 1,
            "telephone": "none"
        }
    }
}

RISK_TIERS = [
    (0.10, 'A (Excellent - Very Low Risk)', '#10b981'),
    (0.20, 'B (Good - Low Risk)', '#06b6d4'),
    (0.35, 'C (Fair - Moderate Risk)', '#f59e0b'),
    (0.55, 'D (Poor - Elevated Risk)', '#f97316'),
    (1.01, 'E (Very Poor - Critical Risk)', '#ef4444'),
]

PDO_FACTOR = 20.0 / np.log(2)
PDO_OFFSET = 600.0 - PDO_FACTOR * np.log(50.0)


def _risk_tier(prob_bad: float) -> Tuple[str, str]:
    for ceiling, name, color in RISK_TIERS:
        if prob_bad < ceiling:
            return name, color
    return RISK_TIERS[-1][1], RISK_TIERS[-1][2]


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())


class CreditModelEngine:
    """Scores applicants against the artifact produced by the training notebook.

    Falls back to the legacy LightGBM pickle when no bundle is present so the app keeps
    running before the first retrain, but in a degraded mode that is reported through the
    API rather than hidden: the legacy model was fitted on prohibited bases and its
    threshold policy does not account for exposure.
    """

    def __init__(self, bundle_path: str = None, legacy_path: str = None):
        self.bundle = None
        self.scorecard = None
        self.policy = None
        self.legacy_model = None
        self.mode = None
        self.warnings: List[str] = []

        path = bundle_path or _resolve(BUNDLE_CANDIDATES)
        if path:
            self._load_bundle(path)
        else:
            self._load_legacy(legacy_path or _resolve(LEGACY_CANDIDATES))

    # -- loading ----------------------------------------------------------------------
    def _load_bundle(self, path: str) -> None:
        self.bundle = joblib.load(path)
        self.scorecard = self.bundle['scorecard']
        self.policy = self.bundle['policy']
        self.model_path = path
        self.mode = 'scorecard'

    def _load_legacy(self, path: str) -> None:
        if not path:
            raise FileNotFoundError(
                'No model artifact found. Run the training notebook to produce '
                'credit_model_bundle.joblib.')
        self.legacy_model = joblib.load(path)
        self.model_path = path
        self.mode = 'legacy'
        msg = (f'Running the legacy artifact ({os.path.basename(path)}). It was trained on '
               'prohibited bases and uses a flat count-based threshold. Prohibited '
               'attributes are constant-imputed and suppressed from reason codes, but the '
               'model should be replaced by re-running the training notebook.')
        self.warnings.append(msg)
        warnings.warn(msg, RuntimeWarning)

    # -- input handling ---------------------------------------------------------------
    def _coerce(self, value, column):
        if column in NUMERIC_FIELDS:
            try:
                return float(value) if '.' in str(value) else int(value)
            except (TypeError, ValueError):
                return NUMERIC_FIELDS[column]['default']
        return value

    def prepare_dataframe(self, input_dict: Dict[str, Any]) -> pd.DataFrame:
        """One applicant as a single-row frame over the permitted features only.

        Any prohibited attribute present in the payload is discarded here, so a stale
        client cannot reintroduce it.
        """
        row = {}
        for col in FEATURE_ORDER:
            value = input_dict.get(col)
            if value is None or value == '':
                value = (NUMERIC_FIELDS[col]['default'] if col in NUMERIC_FIELDS
                         else CATEGORY_MAPS[col][0])
            row[col] = self._coerce(value, col)
        return pd.DataFrame([row])[FEATURE_ORDER]

    def prepare_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        out = df_input.copy()
        for col in FEATURE_ORDER:
            if col not in out.columns:
                out[col] = (NUMERIC_FIELDS[col]['default'] if col in NUMERIC_FIELDS
                            else CATEGORY_MAPS[col][0])
            elif col in NUMERIC_FIELDS:
                out[col] = pd.to_numeric(out[col], errors='coerce').fillna(
                    NUMERIC_FIELDS[col]['default'])
            else:
                out[col] = out[col].fillna(CATEGORY_MAPS[col][0])
        return out[FEATURE_ORDER]

    def _legacy_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Re-add the prohibited columns as constants for the legacy model's schema."""
        legacy = frame.copy()
        for col, fill in LEGACY_FILL.items():
            legacy[col] = fill
        order = list(getattr(self.legacy_model, 'feature_name_', legacy.columns))
        return legacy[[c for c in order if c in legacy.columns]]

    # -- scoring ----------------------------------------------------------------------
    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mode == 'scorecard':
            return self.scorecard.proba(frame)
        return self.legacy_model.predict_proba(self._legacy_frame(frame))[:, 1]

    def credit_score(self, frame: pd.DataFrame, prob_bad: np.ndarray) -> np.ndarray:
        if self.mode == 'scorecard':
            return self.scorecard.points(frame).to_numpy()
        safe = np.clip(prob_bad, 0.001, 0.999)
        return np.clip(PDO_OFFSET + PDO_FACTOR * np.log((1 - safe) / safe), 300, 850)

    def threshold_for(self, frame: pd.DataFrame, override=None) -> np.ndarray:
        """Policy threshold per applicant, or a caller override applied to everyone."""
        n = len(frame)
        if override is not None:
            return np.full(n, float(override))
        if self.policy is not None:
            return self.policy.threshold_for(frame['credit_amount'].to_numpy(dtype=float))
        return np.full(n, 0.50)

    def contributions(self, frame: pd.DataFrame, row: int = 0) -> List[Dict[str, Any]]:
        """Per-feature risk contributions, largest risk-increasing first."""
        if self.mode == 'scorecard':
            return self.scorecard.contribution_chart(frame, row=row)

        legacy = self._legacy_frame(frame)
        contrib = self.legacy_model.booster_.predict(legacy, pred_contrib=True)[row][:-1]
        items = [{'feature': f, 'label': _label(f), 'value': str(legacy.iloc[row][f]),
                  'impact': round(float(v), 4)}
                 for f, v in zip(legacy.columns, contrib)
                 if f not in PROTECTED_FEATURES]
        return sorted(items, key=lambda d: d['impact'], reverse=True)

    # -- public API -------------------------------------------------------------------
    def predict_single(self, input_dict: Dict[str, Any], threshold=None) -> Dict[str, Any]:
        frame = self.prepare_dataframe(input_dict)
        prob_bad = float(self.probability(frame)[0])
        prob_good = 1.0 - prob_bad

        used_threshold = float(self.threshold_for(frame, threshold)[0])
        is_bad = prob_bad >= used_threshold
        tier, tier_color = _risk_tier(prob_bad)

        chart_features = self.contributions(frame, row=0)
        adverse = [dict(f, direction='risk_increasing')
                   for f in chart_features if f['impact'] > 0][:4]
        positive = [dict(f, direction='risk_decreasing')
                    for f in reversed(chart_features) if f['impact'] < 0][:4]

        return {
            'decision': 'DECLINED (High Risk)' if is_bad else 'APPROVED (Low Risk)',
            'status_code': 'declined' if is_bad else 'approved',
            'prob_bad': round(prob_bad, 4),
            'prob_good': round(prob_good, 4),
            'prob_bad_pct': round(prob_bad * 100, 1),
            'prob_good_pct': round(prob_good * 100, 1),
            'credit_score': int(self.credit_score(frame, np.array([prob_bad]))[0]),
            'risk_tier': tier,
            'tier_color': tier_color,
            'threshold_used': round(used_threshold, 4),
            'threshold_source': 'manual override' if threshold is not None else (
                'exposure-banded policy' if self.policy is not None else 'default 0.50'),
            'model_mode': self.mode,
            'adverse_reasons': adverse,
            'positive_factors': positive,
            'chart_features': chart_features,
            'applicant_summary': {f: str(frame.iloc[0][f]) for f in FEATURE_ORDER},
            'warnings': self.warnings,
        }

    def predict_batch(self, df_input: pd.DataFrame,
                      threshold=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        frame = self.prepare_batch(df_input)
        prob_bads = self.probability(frame)
        thresholds = self.threshold_for(frame, threshold)
        decisions = np.where(prob_bads >= thresholds, 'DECLINED', 'APPROVED')
        scores = self.credit_score(frame, prob_bads)

        result = df_input.copy()
        # Anything the model may not use is stripped from the scored output as well.
        result = result.drop(columns=[c for c in PROTECTED_FEATURES if c in result.columns])
        result['Prediction'] = decisions
        result['Default_Risk_Prob'] = np.round(prob_bads, 4)
        result['Approval_Prob'] = np.round(1 - prob_bads, 4)
        result['Credit_Score'] = scores.astype(int)
        result['Threshold_Applied'] = np.round(thresholds, 4)

        total = len(result)
        approved = int((decisions == 'APPROVED').sum())
        summary = {
            'total_applicants': total,
            'approved_count': approved,
            'declined_count': total - approved,
            'approval_rate_pct': round(approved / total * 100, 1) if total else 0,
            'avg_default_risk_pct': round(float(np.mean(prob_bads)) * 100, 1) if total else 0,
            'avg_credit_score': round(float(np.mean(scores)), 1) if total else 0,
            'threshold_used': ('manual override' if threshold is not None
                               else 'exposure-banded policy'),
            'threshold_range': [round(float(thresholds.min()), 3),
                                round(float(thresholds.max()), 3)] if total else [],
            'model_mode': self.mode,
            'warnings': self.warnings,
        }
        return result, summary

    # -- metadata ---------------------------------------------------------------------
    def model_info(self) -> Dict[str, Any]:
        """Model card served straight from the artifact, so the API cannot misreport it."""
        if self.mode != 'scorecard':
            return {
                'model_type': 'LEGACY LightGBM pickle (pending retrain)',
                'artifact': os.path.basename(self.model_path),
                'warnings': self.warnings,
                'metrics': {},
            }

        b = self.bundle
        m = b.get('metrics', {})
        ci = m.get('test_auc_ci') or []
        auc = m.get('test_auc')
        return {
            'model_type': 'WOE scorecard - sign-constrained logistic regression, Platt-calibrated',
            'challenger': 'LightGBM (fixed regularisation, monotonic constraints) - logged, not decisioning',
            'artifact': os.path.basename(self.model_path),
            'trained': b.get('created'),
            'schema_version': b.get('schema_version'),
            'dataset': 'Statlog German Credit (1,000 applicants)',
            'features_used': len(b.get('model_features', FEATURE_ORDER)),
            'excluded_features': b.get('protected_features', PROTECTED_FEATURES),
            'exclusion_reason': 'Prohibited bases under ECOA / Reg B',
            'metrics': {
                'test_auc': (f'{auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]'
                             if auc is not None and len(ci) == 2 else auc),
                'cv_auc': (f"{m.get('cv_auc_mean', float('nan')):.3f} "
                           f"+/- {m.get('cv_auc_std', float('nan')):.3f}"),
                'ks_statistic': round(m['test_ks'], 3) if 'test_ks' in m else None,
                'brier_score': round(m['test_brier'], 4) if 'test_brier' in m else None,
                'challenger_test_auc': (round(m['challenger_test_auc'], 3)
                                        if 'challenger_test_auc' in m else None),
                'challenger_vs_primary_p': (round(m['delong_p_value'], 3)
                                            if 'delong_p_value' in m else None),
                'psi_train_test': round(m['psi_train_test'], 4) if 'psi_train_test' in m else None,
                'n_train': m.get('n_train'),
                'n_test': m.get('n_test'),
            },
            'policy': {
                'mode': 'exposure-banded expected loss',
                'bands': self.policy.describe() if self.policy else [],
                'global_threshold': getattr(self.policy, 'global_threshold', None),
                'assumptions': {'lgd': getattr(self.policy, 'lgd', None),
                                'annual_margin': getattr(self.policy, 'annual_margin', None)},
                'note': ('Thresholds minimise expected loss in currency (exposure x LGD '
                         'against forgone margin), chosen on out-of-fold training '
                         'predictions. Confidence intervals are reported because a 250-row '
                         'test set cannot resolve AUC beyond about +/- 0.035.'),
            },
            'constraints': 'Monotonic WOE binning; all scorecard coefficients sign-constrained',
            'warnings': self.warnings,
        }


engine = CreditModelEngine()
