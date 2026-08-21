@echo off
title CrediPulse - AI Credit Scoring Server
echo ========================================================
echo Starting CrediPulse Frontend (Flask + HTML/CSS/JS)
echo Model: Model (1).pkl (LightGBM Classifier)
echo URL:   http://127.0.0.1:5000
echo ========================================================
echo.

start http://127.0.0.1:5000
python app.py
pause
