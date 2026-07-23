@echo off
echo Starting IntelliGate Monolithic System...

echo Starting Backend Server...
start "IntelliGate Backend" cmd /k "uvicorn module8_dashboard.backend.main:app --reload --port 8000"

echo Starting Frontend Server...
start "IntelliGate Frontend" cmd /k "cd module8_dashboard\frontend && npm run dev"

echo Done! Both servers are launching in separate windows.
