@echo off
cd /d "%~dp0.."
call ".venv\Scripts\activate.bat"
python "frontend_dashboard\run_dashboard.py"
pause
