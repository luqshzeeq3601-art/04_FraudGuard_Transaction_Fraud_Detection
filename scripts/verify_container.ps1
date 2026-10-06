# PowerShell script to verify live Docker container lifecycle for FraudGuard

param(
    [string]$ImageName = "fraudguard:latest",
    [int]$Port = 8000,
    [string]$ArtifactDir = "artifacts/champion"
)

$ErrorActionPreference = "Stop"

Write-Host "=== FraudGuard Container Lifecycle Verification ===" -ForegroundColor Cyan

# 1. Check Docker Daemon
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Docker daemon is not running. Please start Docker Desktop to complete live container verification."
        exit 1
    }
} catch {
    Write-Warning "Docker command not found or daemon offline. Please start Docker Desktop."
    exit 1
}

Write-Host "Docker daemon is active." -ForegroundColor Green

# 2. Build Docker Image
Write-Host "Building Docker image: $ImageName ..." -ForegroundColor Cyan
docker build -t $ImageName -f Dockerfile .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed."
    exit 1
}

# 3. Clean up any existing container
docker rm -f fraudguard_live_test 2>$null | Out-Null

# 4. Run Container with Mounted Champion Bundle
$absArtifactPath = (Resolve-Path $ArtifactDir).Path
Write-Host "Starting container with mounted bundle: $absArtifactPath ..." -ForegroundColor Cyan
docker run -d --name fraudguard_live_test -p "${Port}:8000" -v "${absArtifactPath}:/app/artifacts/champion:ro" $ImageName

# 5. Wait for readiness
Start-Sleep -Seconds 3

# 6. Test Endpoints
try {
    $healthRes = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/health" -Method Get
    Write-Host "Health Check: $($healthRes.status)" -ForegroundColor Green

    $readyRes = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/ready" -Method Get
    Write-Host "Readiness Check: $($readyRes.status) (Model: $($readyRes.model_version))" -ForegroundColor Green

    $payload = Get-Content -Raw "examples/synthetic_transaction.json"
    $scoreRes = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/v1/score" -Method Post -Body $payload -ContentType "application/json"
    Write-Host "Score API Response: Fraud Probability = $($scoreRes.fraud_score), Decision = $($scoreRes.decision)" -ForegroundColor Green

    # Save verification report
    $report = @{
        container_verification = "PASSED"
        timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
        image = $ImageName
        health_status = $healthRes.status
        ready_status = $readyRes.status
        model_version = $readyRes.model_version
        sample_score = $scoreRes.fraud_score
        decision = $scoreRes.decision
    }
    $report | ConvertTo-Json -Depth 4 | Set-Content "reports/container_verification.json"
    Write-Host "Verification report saved to reports/container_verification.json" -ForegroundColor Green
} finally {
    Write-Host "Stopping and cleaning up container..." -ForegroundColor Cyan
    docker rm -f fraudguard_live_test | Out-Null
}

Write-Host "=== Container Lifecycle Verification Complete ===" -ForegroundColor Green
