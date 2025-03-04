@echo off

REM ----------------------
REM Step 1: Check for Python
REM ----------------------

echo Checking for Python...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] No Python found on PATH. Please install Python 3.12+ and re-run.
    pause
    goto end
)

REM ----------------------------------
REM 1) Get the raw version string
REM    e.g. "3.10.5" from "Python 3.10.5"
REM ----------------------------------
for /f "tokens=2 delims= " %%a in ('python --version') do (
    set FULL_VERSION=%%a
)

REM ----------------------------------
REM 2) Split FULL_VERSION by "."
REM    -> major, minor, patch
REM ----------------------------------
for /f "tokens=1,2,3 delims=." %%i in ("%FULL_VERSION%") do (
    set MAJOR=%%i
    set MINOR=%%j
    set PATCH=%%k
)

echo Detected Python version: %FULL_VERSION%
echo Major = %MAJOR%, Minor = %MINOR%, Patch = %PATCH%

REM ----------------------
REM Step 2: Check or create the venv
REM ----------------------

set VENV_DIR=venv

if exist %VENV_DIR% (
    echo Virtual environment folder "%VENV_DIR%" detected.
) else (
    echo No virtual environment folder "%VENV_DIR%" found.
    set /p CREATE_VENV="Would you like to create one now? (y/n) "
    if /i "%CREATE_VENV%"=="y" (
        echo Creating virtual environment...
        python -m venv %VENV_DIR%
        if ERRORLEVEL 1 (
            echo [ERROR] Failed to create virtual environment. Exiting...
            pause
            goto end
        )
    ) else (
        echo Skipping virtual environment creation...
    )
)

REM ----------------------
REM Step 3: Activate venv (if present)
REM ----------------------

if exist %VENV_DIR% (
    echo Activating virtual environment...
    call "%VENV_DIR%\Scripts\activate.bat"
)

REM ----------------------
REM Step 4: Install or update dependencies
REM ----------------------

if exist requirements.txt (
    echo Installing/updating dependencies from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
    if ERRORLEVEL 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        goto end
    )
) else (
    echo No requirements.txt found; skipping dependency install.
)

REM ----------------------
REM Step 5: Run your app
REM ----------------------

echo Launching the application.
REM Example: If your app has an entry point "python -m your_package"
python launch.py

:end
echo.
