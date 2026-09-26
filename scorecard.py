"""Scorecard model components shared by the training notebook and the serving engine.

These classes live in an importable module rather than in the notebook for a specific
reason: a joblib artifact pickles a *reference* to the defining module, so a transformer
defined in a notebook cell unpickles as ``__main__.WOETransformer`` and fails to load in
the Flask process. Anything that ends up inside the model bundle must be defined here.

The scoring math is here for the same reason in reverse: the notebook and the API must not
each carry their own copy of the points formula, or they will drift apart. Both import it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

__all__ = [
    'WOETransformer', 'SignConstrainedLogit', 'ScorecardModel', 'BandedPolicy',
    'supervised_edges', 'enforce_monotonic',
]


# --------------------------------------------------------------------------------------
# Weight of evidence
# --------------------------------------------------------------------------------------

def _woe_from_counts(good: np.ndarray, bad: np.ndarray):
    """WOE and IV per bin from good/bad counts, with 0.5 Laplace smoothing.

    Good = target 0, Bad = target 1. A higher WOE always means a safer bin, which is the
    property the sign constraint in :class:`SignConstrainedLogit` relies on.
    """
    n_bins = len(good)
    good_pct = (good + 0.5) / (good.sum() + 0.5 * n_bins)
    bad_pct = (bad + 0.5) / (bad.sum() + 0.5 * n_bins)
    woe = np.log(good_pct / bad_pct)
    iv = (good_pct - bad_pct) * woe
    return woe, iv


def _bin_stats(binned: pd.Series, y):
    """Ordered per-bin (index, woe, iv) for an already-binned series."""
    grouped = pd.DataFrame({'bin': binned, 'y': np.asarray(y)}).groupby(
        'bin', observed=True)['y'].agg(['count', 'sum'])
    bad = grouped['sum'].to_numpy(dtype=float)
    good = (grouped['count'] - grouped['sum']).to_numpy(dtype=float)
    woe, iv = _woe_from_counts(good, bad)
    return grouped.index, woe, iv


def supervised_edges(x: pd.Series, y, max_bins: int, min_frac: float,
                     random_state: int) -> np.ndarray:
    """Bin edges from a shallow decision tree: supervised, with a minimum bin size.

    Equal-frequency binning ignores the target and, at this sample size, produces sparse
    bins whose WOE is mostly noise. ``min_samples_leaf`` puts a hard floor under every bin.
    """
    min_leaf = max(int(round(min_frac * len(x))), 20)
    tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=min_leaf,
                                  random_state=random_state)
    tree.fit(np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y))
    internal = tree.tree_.feature >= 0
    thresholds = np.sort(tree.tree_.threshold[internal])
    return np.unique(np.concatenate([[-np.inf], thresholds, [np.inf]]))


def enforce_monotonic(x: pd.Series, y, edges: np.ndarray) -> np.ndarray:
    """Merge adjacent bins until WOE is monotonic in the feature.

    Direction is read from the data rather than assumed, so this handles features where
    risk rises with the value (duration) and ones where it falls (age).
    """
    edges = list(edges)
    direction = None
    while len(edges) > 3:
        _, woe, _ = _bin_stats(pd.cut(x, bins=edges, include_lowest=True), y)
        if len(woe) < 3:
            break
        if direction is None:
            corr = np.corrcoef(np.arange(len(woe)), woe)[0, 1]
            direction = 1.0 if (np.isnan(corr) or corr >= 0) else -1.0
        diffs = np.diff(woe) * direction
        if (diffs >= 0).all():
            break
        edges.pop(int(np.argmin(diffs)) + 1)  # drop the boundary with the worst violation
    return np.array(edges)


class WOETransformer(BaseEstimator, TransformerMixin):
    """WOE encoder with supervised monotonic binning and information-value selection.

    Bins, WOE maps and feature selection are all derived inside ``fit``, so when this runs
    in a Pipeline under cross-validation every fold re-derives them from its own training
    rows and nothing leaks across the split.
    """

    def __init__(self, numeric_feats, categorical_feats, n_bins=5, min_iv=0.02,
                 min_bin_frac=0.05, random_state=42):
        self.numeric_feats = numeric_feats
        self.categorical_feats = categorical_feats
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.min_bin_frac = min_bin_frac
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        self.bin_edges_, self.woe_maps_, self.iv_ = {}, {}, {}

        for feat in self.numeric_feats:
            edges = supervised_edges(X[feat], y, self.n_bins, self.min_bin_frac,
                                     self.random_state)
            edges = enforce_monotonic(X[feat], y, edges)
            self.bin_edges_[feat] = edges
            index, woe, iv = _bin_stats(pd.cut(X[feat], bins=edges, include_lowest=True), y)
            self.woe_maps_[feat] = dict(zip(index, woe))
            self.iv_[feat] = float(iv.sum())

        for feat in self.categorical_feats:
            index, woe, iv = _bin_stats(X[feat], y)
            self.woe_maps_[feat] = dict(zip(index, woe))
            self.iv_[feat] = float(iv.sum())

        self.iv_summary_ = pd.Series(self.iv_).sort_values(ascending=False)
        self.keep_feats_ = self.iv_summary_[self.iv_summary_ >= self.min_iv].index.tolist()
        self.feature_names_out_ = [f + '_woe' for f in self.keep_feats_]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for feat in self.keep_feats_:
            if feat in self.numeric_feats:
                binned = pd.cut(X[feat], bins=self.bin_edges_[feat], include_lowest=True)
                mapped = binned.map(self.woe_maps_[feat])
            else:
                mapped = X[feat].map(self.woe_maps_[feat])
            # Unseen category at inference -> WOE 0: neutral, never a silent risk discount.
            out[feat + '_woe'] = pd.to_numeric(pd.Series(mapped, index=X.index),
                                               errors='coerce').fillna(0.0)
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class SignConstrainedLogit(BaseEstimator, ClassifierMixin):
    """L2-penalised logistic regression with every coefficient bounded to one sign.

    On WOE inputs a higher value always means safer, so on log-odds(bad) every coefficient
    must be <= 0. An unconstrained fit on correlated WOE features reliably flips one or two
    signs, which would make the scorecard award points for being riskier. The bound is
    imposed in the optimiser (L-BFGS-B) so it holds by construction rather than by luck.
    """

    def __init__(self, C=1.0, sign=-1, max_iter=500):
        self.C = C
        self.sign = sign
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        n_feat = X.shape[1]

        def nll(w):
            z = w[0] + X @ w[1:]
            return float(np.sum(np.logaddexp(0.0, z) - y * z)
                         + 0.5 / self.C * float(np.dot(w[1:], w[1:])))

        def grad(w):
            z = w[0] + X @ w[1:]
            resid = 1.0 / (1.0 + np.exp(-z)) - y
            return np.concatenate([[resid.sum()], X.T @ resid + w[1:] / self.C])

        bound = (None, 0.0) if self.sign < 0 else (0.0, None)
        res = minimize(nll, np.zeros(n_feat + 1), jac=grad, method='L-BFGS-B',
                       bounds=[(None, None)] + [bound] * n_feat,
                       options={'maxiter': self.max_iter})
        self.intercept_ = np.array([res.x[0]])
        self.coef_ = res.x[1:].reshape(1, -1)
        self.converged_ = bool(res.success)
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p_bad = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p_bad, p_bad])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------------------
# Deployed scorecard: calibrated linear model + PDO points, exactly additive
# --------------------------------------------------------------------------------------

class ScorecardModel:
    """Calibrated WOE scorecard: probability, points, and exact per-feature attribution.

    Platt calibration on a linear model stays linear in log-odds (``z_cal = a*z + b``), so
    calibration is folded into the coefficients at construction time. That keeps the points
    decomposition exact: per-feature points sum to the total score with no approximation,
    which is what makes it usable directly as an adverse-action reason code.
    """

    def __init__(self, woe, cal_coef, cal_intercept, contrib_mean, pdo,
                 feature_labels=None, protected_features=()):
        self.woe = woe
        self.cal_coef = pd.Series(cal_coef)
        self.cal_intercept = float(cal_intercept)
        self.contrib_mean = pd.Series(contrib_mean)
        self.pdo = dict(pdo)                       # factor, offset, floor, cap
        self.feature_labels = dict(feature_labels or {})
        self.protected_features = list(protected_features)

    # -- core -------------------------------------------------------------------------
    def logit(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Calibrated log-odds of BAD."""
        W = self.woe.transform(X_raw)
        return self.cal_intercept + W.to_numpy() @ self.cal_coef.to_numpy()

    def proba(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Calibrated P(bad)."""
        return 1.0 / (1.0 + np.exp(-self.logit(X_raw)))

    # -- points -----------------------------------------------------------------------
    @property
    def base_points(self) -> float:
        return ((self.pdo['offset'] - self.pdo['factor'] * self.cal_intercept)
                / len(self.cal_coef))

    def points_frame(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Per-feature points. Row sum equals the total score exactly."""
        W = self.woe.transform(X_raw)
        return self.base_points - self.pdo['factor'] * (W * self.cal_coef)

    def points(self, X_raw: pd.DataFrame) -> pd.Series:
        return self.points_frame(X_raw).sum(axis=1).clip(
            lower=self.pdo['floor'], upper=self.pdo['cap'])

    # -- attribution ------------------------------------------------------------------
    def risk_contributions(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution to log-odds(bad), centred on the training mean.

        Positive = pushes this applicant above the portfolio average risk.
        """
        W = self.woe.transform(X_raw)
        return (W * self.cal_coef) - self.contrib_mean

    def label_for(self, feature: str) -> str:
        return self.feature_labels.get(feature, feature.replace('_', ' ').title())

    def reason_codes(self, X_raw: pd.DataFrame, row: int = 0, top_k: int = 4) -> dict:
        """Adverse-action reason codes from the exact scorecard decomposition.

        Protected attributes are filtered even though they are not model inputs: defence in
        depth, so a future feature-set change cannot silently put a prohibited basis onto a
        compliance document.
        """
        contrib = self.risk_contributions(X_raw).iloc[row]
        contrib.index = [c[:-4] if c.endswith('_woe') else c for c in contrib.index]
        contrib = contrib.drop(labels=self.protected_features, errors='ignore')

        def render(items, direction):
            out = []
            for feat, value in items.items():
                raw = X_raw.iloc[row][feat] if feat in X_raw.columns else ''
                out.append({'feature': feat, 'label': self.label_for(feat),
                            'value': str(raw), 'impact': round(float(value), 4),
                            'direction': direction})
            return out

        adverse = contrib[contrib > 0].sort_values(ascending=False).head(top_k)
        mitigating = contrib[contrib < 0].sort_values().head(top_k)
        return {'adverse_reasons': render(adverse, 'risk_increasing'),
                'positive_factors': render(mitigating, 'risk_decreasing')}

    def contribution_chart(self, X_raw: pd.DataFrame, row: int = 0) -> list:
        """Every feature's contribution, largest risk first — for the UI chart."""
        contrib = self.risk_contributions(X_raw).iloc[row]
        contrib.index = [c[:-4] if c.endswith('_woe') else c for c in contrib.index]
        contrib = contrib.drop(labels=self.protected_features, errors='ignore')
        return [{'feature': f, 'label': self.label_for(f),
                 'value': str(X_raw.iloc[row][f]) if f in X_raw.columns else '',
                 'impact': round(float(v), 4)}
                for f, v in contrib.sort_values(ascending=False).items()]


class BandedPolicy:
    """Decision thresholds banded by exposure.

    A flat threshold implicitly assumes the cost of a wrong decision is the same on a
    500 DM loan and an 18,000 DM one. It is not: both sides of the loss scale with
    exposure and the decline side also scales with term, so the optimal cutoff varies
    across the book. Each band carries the threshold that minimised out-of-fold expected
    loss for that band.
    """

    def __init__(self, bands, global_threshold, band_column='credit_amount',
                 lgd=None, annual_margin=None):
        self.bands = list(bands)               # [{'max_amount': float|None, 'threshold': f}]
        self.global_threshold = float(global_threshold)
        self.band_column = band_column
        self.lgd = lgd
        self.annual_margin = annual_margin

    @property
    def thresholds(self) -> np.ndarray:
        return np.array([b['threshold'] for b in self.bands], dtype=float)

    @property
    def edges(self) -> np.ndarray:
        return np.array([b['max_amount'] for b in self.bands[:-1]], dtype=float)

    def threshold_for(self, amount) -> np.ndarray:
        """Threshold per applicant, from the exposure band they fall in."""
        amount = np.atleast_1d(np.asarray(amount, dtype=float))
        if not self.bands:
            return np.full(amount.shape, self.global_threshold)
        idx = np.searchsorted(self.edges, amount, side='right')
        return self.thresholds[np.clip(idx, 0, len(self.bands) - 1)]

    def describe(self) -> list:
        lo = 0.0
        rows = []
        for band in self.bands:
            hi = band['max_amount']
            rows.append({'from': lo, 'to': hi, 'threshold': band['threshold']})
            lo = hi if hi is not None else lo
        return rows


def expected_loss(y_true, proba, threshold, amount, duration, lgd, annual_margin):
    """Per-applicant money loss of an accept/reject decision. Decline if proba >= threshold.

    approve a bad applicant  -> credit_amount * LGD
    decline a good applicant -> credit_amount * annual_margin * (duration_months / 12)
    """
    y_true = np.asarray(y_true)
    approve = np.asarray(proba) < np.asarray(threshold)
    amount = np.asarray(amount, dtype=float)
    duration = np.asarray(duration, dtype=float)
    missed_default = approve & (y_true == 1)
    lost_good = (~approve) & (y_true == 0)
    return (np.where(missed_default, amount * lgd, 0.0)
            + np.where(lost_good, amount * annual_margin * duration / 12.0, 0.0))
