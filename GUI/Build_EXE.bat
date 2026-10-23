@echo off
title Build UniMate Studio EXE
cd /d "%~dp0"

set "CONDA_ENV=%USERPROFILE%\miniconda3\envs\unimate"
if not exist "%CONDA_ENV%\python.exe" set "CONDA_ENV=%LOCALAPPDATA%\miniconda3\envs\unimate"
if not exist "%CONDA_ENV%\python.exe" set "CONDA_ENV=%USERPROFILE%\anaconda3\envs\unimate"
if not exist "%CONDA_ENV%\python.exe" set "CONDA_ENV=%PROGRAMDATA%\miniconda3\envs\unimate"
set "CONDA_PY=%CONDA_ENV%\python.exe"
set "PATH=%CONDA_ENV%;%CONDA_ENV%\Library\bin;%CONDA_ENV%\Scripts;%PATH%"
set "BUILD_TMP=%TEMP%\unimate_studio_build"

echo Installing PySide6 and PyInstaller if needed...
"%CONDA_PY%" -m pip install pyside6 pyinstaller -q
"%CONDA_PY%" -c "import PySide6" 2>nul
if errorlevel 1 (
    echo [ERROR] PySide6 could not be installed in "%CONDA_PY%".
    pause
    exit /b 1
)

echo Building UniMate_Studio.exe...
"%CONDA_PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name "UniMate_Studio" --distpath "." --workpath "%BUILD_TMP%\work" --specpath "%BUILD_TMP%\spec" --hidden-import PySide6.QtCore --hidden-import PySide6.QtGui --hidden-import PySide6.QtWidgets --exclude-module torch --exclude-module torchvision --exclude-module bpy --exclude-module scipy --exclude-module matplotlib --exclude-module transformers --exclude-module diffusers --exclude-module trimesh "unimate_gui.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed! See messages above.
    pause
    exit /b 1
)

echo.
echo Build complete! You can now launch:
echo   %~dp0UniMate_Studio.exe
pause
