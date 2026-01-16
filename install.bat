@echo off
setlocal EnableDelayedExpansion
TITLE GeoFuse Installer

:: --- CONFIGURATION ---
set ENV_NAME=geofuse
set PYTHON_VER=3.12.12

echo ========================================================
echo        GeoFuse Installer (MPI-Enabled)
echo ========================================================

:: --- STEP 1: AUTO-DETECT CONDA ---
set "CONDA_ROOT="
set "PATHS[0]=%USERPROFILE%\miniforge3"
set "PATHS[1]=%USERPROFILE%\anaconda3"
set "PATHS[2]=%USERPROFILE%\miniconda3"
set "PATHS[3]=C:\ProgramData\miniforge3"
set "PATHS[4]=C:\ProgramData\anaconda3"

for /L %%i in (0,1,4) do (
    call set "TEST_PATH=%%PATHS[%%i]%%"
    if exist "!TEST_PATH!\Scripts\conda.exe" (
        set "CONDA_ROOT=!TEST_PATH!"
        goto FOUND_CONDA
    )
)
:MANUAL_INPUT
if not defined CONDA_ROOT set /P "CONDA_ROOT=Paste path to Conda folder: "
:FOUND_CONDA
call "%CONDA_ROOT%\Scripts\activate.bat" base >nul 2>&1

:: --- STEP 2: CREATE LOGS DIRECTORY ---
if not exist logs mkdir logs

:: --- STEP 3: RUN PYTHON SETUP SCRIPT (handles environment creation) ---
if exist scripts\setup_env.py (
    python scripts\setup_env.py
    if %errorlevel% neq 0 (
        echo [ERROR] Setup script failed!
        pause
        exit /b 1
    )
) else (
    echo [ERROR] scripts\setup_env.py not found!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [SUCCESS] Installation complete.
echo [READY] To launch the CLI (example with 4 MPI processes):
echo    conda activate %ENV_NAME%
echo    mpiexec -n 4 python scripts/cli.py --config config.csv
echo ========================================================
pause