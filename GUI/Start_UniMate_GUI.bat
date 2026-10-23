@echo off
title UniMate Animation Studio
cd /d "%~dp0\.."

set "CONDA_PY=%USERPROFILE%\miniconda3\envs\unimate\pythonw.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%LOCALAPPDATA%\miniconda3\envs\unimate\pythonw.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%USERPROFILE%\anaconda3\envs\unimate\pythonw.exe"
if not exist "%CONDA_PY%" set "CONDA_PY=%PROGRAMDATA%\miniconda3\envs\unimate\pythonw.exe"

if exist "%CONDA_PY%" (
    start "" "%CONDA_PY%" "%~dp0unimate_gui.py"
) else (
    python "%~dp0unimate_gui.py"
)
