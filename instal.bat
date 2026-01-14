@echo off
setlocal EnableDelayedExpansion
TITLE GeoFuse Installer

:: --- CONFIGURATION ---
set "ENV_NAME=geofuse"
set "PYTHON_VER=3.12.12"

echo ========================================================
echo        GeoFuse Installer (MPI-Enabled)
echo ========================================================

:: --- STEP 1: AUTO-DETECT CONDA ---
set "CONDA_EXEC="
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
        goto CONDA_FOUND
    )
)

:CONDA_FOUND
if defined CONDA_ROOT (
    echo [INFO] Found Conda at: %CONDA_ROOT%
    set "CONDA_EXEC=%CONDA_ROOT%\Scripts\conda.exe"
) else (
    echo [INFO] Could not find Conda in common locations. Checking PATH.
    where conda >nul 2>nul
    if !errorlevel! equ 0 (
        echo [INFO] Found Conda in system PATH.
        set "CONDA_EXEC=conda"
    ) else (
        echo [WARN] Conda not found.
        set /P "CONDA_ROOT=Please paste the path to your Conda folder: "
        if exist "!CONDA_ROOT!\Scripts\conda.exe" (
            set "CONDA_EXEC=!CONDA_ROOT!\Scripts\conda.exe"
        ) else (
            echo [ERROR] conda.exe not found at the specified path.
            pause
            exit /b 1
        )
    )
)

:: --- STEP 2: CREATE/RESET ENVIRONMENT ---
set CREATE_ENV=true
"!CONDA_EXEC!" env list | findstr /R /C:"^%ENV_NAME% " >nul
if %errorlevel% equ 0 (
    echo [WARN] Environment '%ENV_NAME%' already exists.
    set /P "DELETE=Delete and clean install? (Y/N): "
    if /I "!DELETE!"=="Y" (
        echo [INFO] Removing existing environment...
        call "!CONDA_EXEC!" remove -n %ENV_NAME% --all -y
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to remove existing environment.
            pause
            exit /b !errorlevel!
        )
    ) else (
        echo [INFO] Updating existing environment.
        set CREATE_ENV=false
    )
)

if "!CREATE_ENV!"=="true" (
    echo [INFO] Creating Base Environment (Python %PYTHON_VER%)...
    call "!CONDA_EXEC!" create -n %ENV_NAME% python=%PYTHON_VER% -y
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create Conda environment.
        pause
        exit /b !errorlevel!
    )
)

:: --- STEP 3: RUN PYTHON SETUP SCRIPT in ENV ---
echo [INFO] Installing dependencies into '%ENV_NAME%' environment...
if exist scripts\setup_env.py (
    call "!CONDA_EXEC!" run -n %ENV_NAME% python scripts\setup_env.py
    if !errorlevel! neq 0 (
        echo [ERROR] Python setup script failed.
        pause
        exit /b !errorlevel!
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