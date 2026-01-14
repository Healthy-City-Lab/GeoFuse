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
    if exist "!TEST_PATH!\Scripts\activate.bat" (
        set "CONDA_ROOT=!TEST_PATH!"
        goto FOUND_CONDA
    )
)
:MANUAL_INPUT
if not defined CONDA_ROOT set /P "CONDA_ROOT=Paste path to Conda folder: "
:FOUND_CONDA
call "%CONDA_ROOT%\Scripts\activate.bat" base

:: --- STEP 2: CREATE/RESET ENVIRONMENT ---
conda env list | findstr /R /C:"^%ENV_NAME% " >nul
if %errorlevel% equ 0 (
    echo [WARN] Environment '%ENV_NAME%' already exists.
    set /P DELETE="Delete and clean install? (Y/N): "
    if /I "!DELETE!"=="Y" (
        call conda remove -n %ENV_NAME% --all -y
    ) else (
        goto ACTIVATE
    )
)

echo [INFO] Creating Base Environment (Python %PYTHON_VER%)...
call conda create -n %ENV_NAME% python=%PYTHON_VER% -y

:ACTIVATE
echo [INFO] Activating Environment...
call conda activate %ENV_NAME%

:: --- STEP 3: RUN PYTHON SETUP SCRIPT ---
if exist scripts\setup_env.py (
    call "%CONDA_ROOT%\envs\%ENV_NAME%\python.exe" scripts\setup_env.py
) else (
    echo [ERROR] scripts\setup_env.py not found!
    pause
    exit /b
)

echo.
echo ========================================================
echo [READY] To launch the CLI (example with 4 MPI processes):
echo    conda activate %ENV_NAME%
echo    mpiexec -n 4 python scripts/cli.py --config config.csv
echo ========================================================
pause