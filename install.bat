@echo off
setlocal EnableDelayedExpansion
TITLE GeoFuse Installer
:: GeoFuse installer for Windows.
:: Usage: install.bat [--yes] [--torch auto/cuda/cpu] [--env-name NAME]
:: The options are passed to scripts\setup_env.py. Without --yes the window
:: waits for a key at the end; with it (unattended runs) it does not.

cd /d "%~dp0"

set "ENV_NAME=geofuse"
set "UNATTENDED="
set "NEXT_IS_NAME="
for %%a in (%*) do (
    if defined NEXT_IS_NAME (
        set "ENV_NAME=%%~a"
        set "NEXT_IS_NAME="
    )
    if /I "%%~a"=="--env-name" set "NEXT_IS_NAME=1"
    if /I "%%~a"=="--yes" set "UNATTENDED=1"
    if /I "%%~a"=="-y" set "UNATTENDED=1"
)

echo ========================================================
echo        GeoFuse Installer
echo ========================================================

:: --- STEP 1: FIND CONDA ---
set "CONDA_ROOT="
if defined CONDA_EXE if exist "%CONDA_EXE%" (
    for %%d in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fd"
    echo [INFO] Found Conda from CONDA_EXE: !CONDA_ROOT!
    goto FOUND_CONDA
)
for %%p in (
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\miniforge3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "C:\ProgramData\miniforge3"
    "C:\ProgramData\anaconda3"
    "C:\ProgramData\miniconda3"
) do (
    if exist "%%~p\Scripts\conda.exe" (
        set "CONDA_ROOT=%%~p"
        echo [INFO] Found Conda at: %%~p
        goto FOUND_CONDA
    )
)
for /f "delims=" %%c in ('where conda 2^>nul') do (
    for %%d in ("%%~dpc..") do set "CONDA_ROOT=%%~fd"
    echo [INFO] Found Conda on PATH: !CONDA_ROOT!
    goto FOUND_CONDA
)
if defined UNATTENDED (
    echo [ERROR] Conda not found. Install Miniforge, or put conda on PATH.
    exit /b 1
)
set /P "CONDA_ROOT=Conda not found. Paste the path to your Conda folder: "

:FOUND_CONDA
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [ERROR] No Conda installation at: %CONDA_ROOT%
    if not defined UNATTENDED pause
    exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" base
if errorlevel 1 (
    echo [ERROR] Could not activate the Conda base environment.
    if not defined UNATTENDED pause
    exit /b 1
)

:: --- STEP 2: CREATE THE ENVIRONMENT AND INSTALL ---
if not exist logs mkdir logs
python scripts\setup_env.py %*
if errorlevel 1 (
    echo [ERROR] Setup script failed. See logs\install.log.
    if not defined UNATTENDED pause
    exit /b 1
)

echo.
echo ========================================================
echo [SUCCESS] Installation complete.
echo [READY] In the environment, the command line is geofuse:
echo    conda activate %ENV_NAME%
echo    geofuse doctor
echo    geofuse --help
echo ========================================================
if not defined UNATTENDED pause
exit /b 0
