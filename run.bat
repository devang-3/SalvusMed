@echo off
cd /d "%~dp0"
echo Starting SalvusMed...
pip install -r requirements.txt -q
python app.py
pause
