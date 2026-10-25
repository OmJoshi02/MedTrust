@echo off
echo ==========================================================
echo MEDTRUST V1 - CLINICAL DECISION SUPPORT SYSTEM
echo ==========================================================

start "MedTrust API" python backend/server.py
cd frontend
call npm run dev
