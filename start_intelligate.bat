@echo off
cd /d "%~dp0"
echo IntelliGate CPU software prototype
echo Open http://127.0.0.1:8000 and create your local administrator account.
echo Build frontend first if needed: cd module8_dashboard\frontend ^&^& npm run build
python -m uvicorn module8_dashboard.backend.main:app --host 127.0.0.1 --port 8000
