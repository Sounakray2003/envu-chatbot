param(
    [switch]$NoBuild,
    [int]$HealthRetries = 30,
    [int]$HealthDelaySeconds = 2
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    if ($NoBuild) {
        docker compose up -d rag-ingestion-admin
    } else {
        docker compose up -d --build rag-ingestion-admin
    }

    for ($i = 1; $i -le $HealthRetries; $i++) {
        try {
            $health = Invoke-RestMethod -Method Get -Uri "http://localhost:8094/health" -TimeoutSec 5
            Write-Host "rag-ingestion-admin is healthy:" -ForegroundColor Green
            $health | ConvertTo-Json -Depth 10
            exit 0
        } catch {
            Write-Host ("waiting for health check {0}/{1}" -f $i, $HealthRetries)
            Start-Sleep -Seconds $HealthDelaySeconds
        }
    }

    throw "rag-ingestion-admin did not become healthy at http://localhost:8094/health"
} finally {
    Pop-Location
}
