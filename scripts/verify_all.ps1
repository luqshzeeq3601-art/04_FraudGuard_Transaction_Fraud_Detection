# PowerShell script for full local CI/CD quality and gate verification

$ErrorActionPreference = "Stop"

Write-Host "=== Running FraudGuard Local Quality & CI Gate Verification ===" -ForegroundColor Cyan

# 1. Ruff Linter
Write-Host "`n1. Running Ruff Linter..." -ForegroundColor Cyan
& .venv\Scripts\ruff.exe check src tests
if ($LASTEXITCODE -ne 0) { Write-Error "Ruff check failed." }
Write-Host "Ruff check passed." -ForegroundColor Green

# 2. Ruff Formatter Check
Write-Host "`n2. Running Ruff Formatter Check..." -ForegroundColor Cyan
& .venv\Scripts\ruff.exe format --check src tests
if ($LASTEXITCODE -ne 0) { Write-Error "Ruff format check failed." }
Write-Host "Ruff format check passed." -ForegroundColor Green

# 3. Pytest with Coverage and Isolation Guard
Write-Host "`n3. Running Pytest Suite with Coverage..." -ForegroundColor Cyan
& .venv\Scripts\pytest.exe -v --cov=src/fraudguard --cov-report=term-missing
if ($LASTEXITCODE -ne 0) { Write-Error "Pytest suite failed." }
Write-Host "Pytest suite passed." -ForegroundColor Green

Write-Host "`n=== All Quality Gates Passed Successfully ===" -ForegroundColor Green
