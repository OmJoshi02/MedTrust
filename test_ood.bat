@echo off
setlocal
echo ==========================================================
echo MEDTRUST V1 - OOD MODULE VERIFICATION ^& TEST RUNNER
echo ==========================================================
echo.
echo [1/4] Verifying OOD module artifacts and model dependencies...
python -m src.OOD verify
if %errorlevel% neq 0 goto :error

echo.
echo [2/4] Running OOD unit regression test suite...
python -m unittest src/OOD/test.py
if %errorlevel% neq 0 goto :error

echo.
echo [3/4] Running sample In-Distribution inference (NIH radiograph)...
python -m src.OOD predict "dataset/images/00006440_002.png" --method combined
if %errorlevel% neq 0 goto :error

echo.
echo [4/4] Running sample Out-of-Distribution inference (Brain MRI)...
python -m src.OOD predict "test_images/mri.jpg" --method combined
if %errorlevel% neq 0 goto :error

echo.
echo ==========================================================
echo ALL OOD MODULE CHECKS ^& TESTS COMPLETED SUCCESSFULLY!
echo ==========================================================
exit /b 0

:error
echo.
echo [FAIL] An error occurred during OOD module execution.
exit /b 1

