@echo off
title CrediPulse - AI Credit Scoring Server
echo ========================================================
echo Starting CrediPulse Frontend (Flask + HTML/CSS/JS)
echo Model: credit_model_bundle.joblib (WOE scorecard)
echo        falls back to Model (1).pkl if the notebook has not been run
echo URL:   http://127.0.0.1:5000
echo ========================================================
echo.

start http://127.0.0.1:5000
python app.py
pause
