# 🛡️ CrediPulse — AI Credit Scoring & Risk Explainability Platform

A full-stack credit risk decisioning and explainability web application powered by an **Optuna-tuned LightGBM Classifier** and an interactive **HTML5, CSS3, and JavaScript** frontend.

The platform predicts whether a loan applicant is **Good (Low Risk / Approved)** or **Bad (High Risk / Default)** using the 20 attributes of the **Statlog German Credit Dataset**, providing real-time credit scorecard points, risk probabilities, **5:1 asymmetric cost-calibrated decision thresholds**, and **adverse action reason codes (Tree SHAP)**.

---

## 📸 Key Features

### 🏦 1. Single Applicant Risk Assessment
- **20 Form Attributes**: Categorized intuitively across *Banking & Accounts*, *Loan Details*, *Personal & Demographics*, and *Employment & Assets*.
- **1-Click Quick Presets**: Test applicant profiles instantly (*Prime Applicant*, *Borderline Applicant*, *High-Risk Applicant*).
- **Decision & Scorecard Output**:
  - **Approval Banner**: `APPROVED (Low Risk)` vs. `DECLINED (High Risk)`.
  - **Default Probability Gauge**: $P(\text{Default})$ vs. $P(\text{Good})$ with animated progress bar.
  - **Credit Score Rating**: Scaled from 300 to 850 points.
- **5:1 Asymmetric Cost-Optimal Thresholding**:
  - Interactive threshold slider allowing risk officers to switch between **0.17 (Cost-Optimal)** and **0.50 (Standard)** cutoffs.
- **Adverse Action Reason Codes (Compliance & Explainability)**:
  - Top factors pushing risk **UP** (adverse action notices).
  - Mitigating safety factors pulling risk **DOWN**.
  - Interactive **Chart.js Tree SHAP bar chart** showing local feature contributions.

### 📁 2. Bulk / Batch CSV Assessment
- Drag-and-drop CSV file upload.
- Batch prediction with summary metrics (total applications, approved count, approval rate %, average default risk %).
- Scored table preview with **1-click Export to Scored CSV**.

### 📘 3. Model & Risk Intelligence
- **Architecture**: LightGBM Classifier tuned via Optuna (25 trials, 5-fold stratified CV).
- **Monotonic Constraints**: Enforced on *Loan Duration (+)*, *Credit Amount (+)*, and *Age (-)*.
- **Discrimination Metrics**: $ROC\text{-}AUC = 0.8089$, $KS = 0.4895$.
- **Cost Asymmetry Explanation**: Highlights why missed defaults (5x penalty) justify a 0.17 decision cutoff over the default 0.50 cutoff.

---

## 📂 Project Structure

```
Credit/
├── Model (1).pkl                 # Pre-trained LightGBM Classifier model
├── credit_engine.py              # ML inference engine & Tree SHAP explainability
├── app.py                        # Flask API server & static asset host
├── credit_score_rewritten.ipynb  # End-to-end training & analysis notebook
├── sample_applicants.csv         # Sample batch dataset for testing
├── run_app.bat                   # 1-Click launcher script for Windows
├── templates/
│   └── index.html                # Semantic HTML5 dashboard layout
└── static/
    ├── css/
    │   └── style.css             # Modern fintech styling & responsive layout
    └── js/
        └── app.js                # Async API handler, Chart.js graphs, preset loaders
```

---

## ⚙️ Installation & Prerequisites

### 1. Python Environment
Python 3.10+ is recommended. Ensure the following packages are installed:

```bash
pip install flask lightgbm scikit-learn pandas numpy joblib
```

---

## 🚀 How to Run

### Option 1: 1-Click Launcher (Windows)
Double-click [`run_app.bat`](file:///c:/Users/user/Documents/Credit/run_app.bat).  
It will automatically launch the Flask server and open `http://127.0.0.1:5000` in your default browser.

### Option 2: Command Line
1. Open terminal / PowerShell in the project directory:
   ```bash
   cd Credit
   ```
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 📡 REST API Endpoints

The Flask backend exposes clean JSON endpoints for integration into third-party loan origination systems:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the HTML5 dashboard frontend |
| `GET` | `/api/schema` | Returns categorical choices and numeric field boundaries |
| `GET` | `/api/presets` | Returns preset applicant profiles (*Prime*, *Borderline*, *High-Risk*) |
| `POST` | `/api/predict` | Computes prediction, probability, credit score, and SHAP reason codes |
| `POST` | `/api/predict-batch` | Accepts a multipart CSV file upload and returns batch-scored records |
| `GET` | `/api/model-info` | Returns model performance metrics and cost matrix parameters |

### Example Prediction Request (`POST /api/predict`)

```json
{
  "threshold": 0.17,
  "applicant": {
    "checking_status": "< 0 DM",
    "duration_months": 48,
    "credit_history": "critical account / other credits elsewhere",
    "purpose": "radio/television",
    "credit_amount": 1169,
    "savings_status": "unknown/no savings account",
    "employment_since": ">= 7 years",
    "installment_rate": 4,
    "personal_status_sex": "male: single",
    "other_debtors": "none",
    "residence_since": 4,
    "property": "real estate",
    "age": 67,
    "other_installment_plans": "none",
    "housing": "own",
    "existing_credits": 2,
    "job": "skilled employee/official",
    "num_dependents": 1,
    "telephone": "yes, registered",
    "foreign_worker": "yes"
  }
}
```

---

## 📊 Dataset Reference: Statlog (German Credit Data)
- **Observations**: 1,000 applicants (700 Good, 300 Bad).
- **Target Variable**:
  - `0 = Good` (Safe / Creditworthy)
  - `1 = Bad` (Default / High Risk)
- **Key Predictors**:
  - `checking_status`: Balance and status of existing checking account ($IV \approx 0.659$).
  - `credit_history`: Past repayment behavior ($IV \approx 0.291$).
  - `duration_months`: Loan repayment duration ($IV \approx 0.213$).
  - `savings_status`: Total savings/bonds ($IV \approx 0.188$).

---

## ⚖️ License
Educational and commercial credit risk demonstration project.
# CrediPulse-Explainable-Credit-Scoring-Platform
