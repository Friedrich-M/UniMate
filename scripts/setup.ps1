param([string]$Python = "")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    if ($Python) {
        & $Python -m venv (Join-Path $projectRoot ".venv")
    } else {
        py -3.10 -m venv (Join-Path $projectRoot ".venv")
    }
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }
}
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $venvPython -m pip install -r (Join-Path $projectRoot "backend\requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Runtime installation failed" }
& $venvPython (Join-Path $projectRoot "scripts\download_model.py")
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }
Write-Host "Ready. Configure the Blender add-on project folder as $projectRoot"
