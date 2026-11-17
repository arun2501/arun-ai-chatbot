@echo off
title Arunkumar K R - Personal AI Chatbot
echo ===================================================
echo   Starting Arunkumar AI Chatbot Server...
echo ===================================================

cd /d "%~dp0"

set PYTHON_EXE="C:\Users\ELCOT\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if exist %PYTHON_EXE% (
    start "" http://localhost:5000
    %PYTHON_EXE% app.py
) else (
    start "" http://localhost:5000
    python app.py
)

pause
