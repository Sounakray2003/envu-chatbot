param(
    [string]$FolderPath = "D:\Downloads\Pest_Documents\Pest_Documents",
    [string]$BaseUrl = "http://localhost:8094",
    [switch]$Recursive,
    [int]$DelaySeconds = 1
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $FolderPath -PathType Container)) {
    throw "Folder not found: $FolderPath"
}

$supportedExtensions = @(
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
    ".json", ".csv", ".tsv", ".xlsx", ".xls", ".xlsm",
    ".html", ".htm", ".xml", ".zip",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"
)

$searchOption = if ($Recursive) { "-Recurse" } else { "" }
if ($Recursive) {
    $files = Get-ChildItem -LiteralPath $FolderPath -File -Recurse
} else {
    $files = Get-ChildItem -LiteralPath $FolderPath -File
}

$files = $files | Where-Object { $supportedExtensions -contains $_.Extension.ToLowerInvariant() }

if (-not $files -or $files.Count -eq 0) {
    Write-Host "No supported files found in: $FolderPath" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($files.Count) supported file(s)." -ForegroundColor Cyan

$successCount = 0
$failedCount = 0
$results = @()

foreach ($file in $files) {
    Write-Host ""
    Write-Host "Uploading: $($file.FullName)" -ForegroundColor Cyan

    try {
        $responseText = & curl.exe -sS -X POST "$BaseUrl/ingest/file-upload" `
            -F "file=@$($file.FullName)" `
            -F "is_active=true"

        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE"
        }

        $response = $responseText | ConvertFrom-Json
        $status = [string]$response.status

        if ($status.ToLowerInvariant() -in @("success", "partial_success")) {
            $successCount += 1
            Write-Host "Uploaded: $($file.Name) [$status]" -ForegroundColor Green
        } else {
            $failedCount += 1
            Write-Host "Failed: $($file.Name) [$status]" -ForegroundColor Red
        }

        $results += [pscustomobject]@{
            file = $file.FullName
            status = $status
            response = $response
        }
    } catch {
        $failedCount += 1
        Write-Host "Failed: $($file.Name) - $($_.Exception.Message)" -ForegroundColor Red
        $results += [pscustomobject]@{
            file = $file.FullName
            status = "error"
            error = $_.Exception.Message
        }
    }

    if ($DelaySeconds -gt 0) {
        Start-Sleep -Seconds $DelaySeconds
    }
}

Write-Host ""
Write-Host "Upload complete. Success: $successCount Failed: $failedCount Total: $($files.Count)" -ForegroundColor Cyan

$results | ConvertTo-Json -Depth 20 | Out-File -FilePath ".\upload_results.json" -Encoding utf8
Write-Host "Saved detailed results to: .\upload_results.json"
