# CrediPulse — Credit Scoring, Decision Policy & Explainability

An auditable credit scorecard, the decision policy that sits on top of it, and a Flask
dashboard that scores applicants and produces adverse-action reason codes.

Built on the **Statlog (German Credit)** dataset: 1,000 applicants, 20 attributes, binary
good/bad outcome.

---

## What this project is actually about

Statlog German Credit is 1,000 rows collected in 1994, with no time dimension. The published
ceiling on it is roughly **AUC 0.78–0.80**, and essentially every modelling approach lands
there. Squeezing out another half-point of AUC is not where the value is, so the work here is
concentrated in four places that change decisions rather than metrics:

1. **A loss function denominated in money.** Expected loss is `P(default) × EAD × LGD`. A
   missed default on an 18,000 DM loan is not the same event as one on a 500 DM loan, so a
   flat cost ratio cannot produce the right cutoff. Thresholds are fitted per exposure band.
2. **A feature set that is legal to use.** `personal_status_sex` and `foreign_worker` are
   prohibited bases under ECOA / Reg B. They are excluded from the model and from the intake
   form, and retained only for measuring disparity.
3. **Uncertainty stated honestly.** On a 250-row test set the standard error on AUC is around
   ±0.035, so every headline metric is published with a bootstrap confidence interval and the
   two models are compared with a paired DeLong test rather than by eyeballing point
   estimates.
4. **One artifact, one model card.** The notebook serialises exactly the objects the API
   loads, so the documentation cannot drift away from what is in production.

**Deliberately not included:** hyperparameter search, SMOTE, stacked ensembles, extra model
families. At 750 training rows with ~45 bads per validation fold, a tuning run's selection
noise exceeds any real effect it could find. The notebook explains this where the Optuna
search used to be.

---

## Architecture

| File | Role |
|---|---|
| `credit score.ipynb` | End-to-end training, validation, policy fitting, model card. Writes the artifact. |
| `scorecard.py` | Model components shared by training and serving: WOE transformer, sign-constrained logit, `ScorecardModel`, `BandedPolicy`. |
| `credit_engine.py` | Inference engine: loads the artifact, scores single and batch applicants, produces reason codes. |
| `app.py` | Flask API and static host. |
| `credit_model_bundle.joblib` | The deployed artifact (**produced by running the notebook**). |
| `Model (1).pkl` | Legacy LightGBM pickle, kept only as a fallback. See below. |
| `templates/`, `static/` | Dashboard UI. |

Model classes live in `scorecard.py` rather than in notebook cells for a specific reason: a
joblib artifact pickles a reference to the defining module, so a class defined in a notebook
saves as `__main__.WOETransformer` and cannot be loaded by the Flask process.

---

## The model

**Primary — WOE scorecard.** Supervised monotonic binning (shallow decision tree per feature,
with a 5%-of-fold minimum bin size), information-value selection at 0.02, logistic regression
with **every coefficient sign-constrained negative**, Platt-calibrated on out-of-fold values.

Two of those choices fix real defects rather than adding polish:

- *Sign constraints.* WOE is built so a higher value always means a safer bin, so on
  log-odds(bad) every coefficient must be negative. An unconstrained fit on correlated WOE
  features reliably flips one or two signs — the scorecard would award points for being
  riskier. The bound is imposed in the optimiser, so it holds by construction.
- *Calibration folded into the coefficients.* A sigmoid on a linear model stays linear in
  log-odds, so calibration is a rescaling of the coefficients. That keeps the points
  decomposition **exact**: per-feature points sum to the total score with no approximation,
  which is what makes it directly usable as an adverse-action reason code.

**Challenger — LightGBM.** Fixed regularisation, monotonic constraints on duration (+),
amount (+) and age (−). Logged alongside, not decisioning: the DeLong test shows the AUC
difference between the two is not statistically detectable at this sample size, so there is
no performance case for paying the interpretability cost.

**Explainability.** Reason codes come from the scorecard's exact decomposition. SHAP is
computed on the challenger for the diagnostic it is genuinely good at — showing where a
flexible model would disagree with the additive one.

---

## The decision policy

Thresholds are **not** a constant in the code. They are fitted in the notebook against:

```
approve a bad applicant  ->  loss = credit_amount × LGD
decline a good applicant ->  loss = credit_amount × annual_margin × (duration_months / 12)
```

Both sides scale with exposure and the decline side also scales with term, so the implied
cost ratio varies per applicant and a single flat cutoff cannot be optimal. The policy is
banded by loan size, with each band's threshold chosen on **out-of-fold training
predictions** — never on the test set, which would fit the cutoff to the sample used to
report performance.

`LGD` and `annual_margin` are **assumptions, not measurements** — this dataset carries no
recovery or pricing data. The notebook publishes a sensitivity grid showing how the cutoff
moves across a plausible range of both.

The dashboard slider is a manual override of this policy, not the policy itself.

---

## Running it

```bash
pip install -r requirements.txt
```

Then run every cell of `credit score.ipynb` to produce `credit_model_bundle.joblib`, and
start the app:

```bash
python app.py
```

The dashboard is at `http://127.0.0.1:5000`. On Windows, `run_app.bat` does the same thing.

### Before the first notebook run

`credit_engine.py` falls back to the legacy `Model (1).pkl` if no bundle is present, so the
app starts either way — but it runs **degraded**, and says so through `/api/model-info` and in
every prediction payload. That pickle is a 20-feature LightGBM trained *with* the prohibited
bases. On the fallback path those attributes are constant-imputed so they cannot discriminate
between applicants, and they are filtered out of every reason code, but the correct fix is to
run the notebook and replace it.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | Form schema: categories, numeric ranges, feature order, active policy. |
| `GET /api/presets` | Three example applicant profiles. |
| `GET /api/model-info` | Model card, served from the artifact rather than hard-coded. |
| `POST /api/predict` | Score one applicant. Omit `threshold` (or send `"auto"`) to use the fitted policy. |
| `POST /api/predict-batch` | Score a CSV. Returns summary, preview rows, and a scored CSV. |

---

## Known limitations

These are in the model card in the notebook too; they belong here as well.

- **Dataset ceiling.** AUC in the high 0.70s is what this data supports. Metrics carry
  confidence intervals because the point estimates are not precise beyond two decimals.
- **Reject inference.** These are observed outcomes on an already-screened book. A model
  trained on accepted applicants is biased on the population it will face. Not fixable
  here — the remedies need the rejected applications, which this dataset does not contain.
- **No temporal validation.** No dates, so no out-of-time test and no real drift measurement.
  The PSI figures compare two random splits, which is weaker than a vintage comparison.
- **Assumed loss parameters.** LGD and margin are conventional values, not measured.
- **The fairness section is a screen, not a compliance determination.** Several subgroups in a
  250-row test set are too small for the four-fifths ratio to mean anything, and are flagged
  rather than interpreted. A real fair-lending review needs a larger sample, a statistician,
  and counsel.
