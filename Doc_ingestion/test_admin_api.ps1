param(
    [string]$BaseUrl = "http://localhost:8094",
    [string]$FilePath,
    [switch]$DeleteAfter,
    [int]$PollSeconds = 10,
    [int]$MaxPolls = 60
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Json {
    param(
        [string]$Method,
        [string]$Uri,
        [hashtable]$Form
    )

    if ($Form) {
        return Invoke-RestMethod -Method $Method -Uri $Uri -Form $Form
    }

    return Invoke-RestMethod -Method $Method -Uri $Uri
}

Write-Step "Checking admin API health"
$health = Invoke-Json -Method Get -Uri "$BaseUrl/health"
$health | ConvertTo-Json -Depth 10

Write-Step "Listing files before upload"
$before = Invoke-Json -Method Get -Uri "$BaseUrl/admin/files"
$before | ConvertTo-Json -Depth 10

if (-not $FilePath) {
    Write-Host ""
    Write-Host "Health/list checks passed. Pass -FilePath to test upload/status/delete." -ForegroundColor Yellow
    Write-Host "Example:"
    Write-Host "  .\test_admin_api.ps1 -FilePath C:\path\sample.pdf -DeleteAfter"
    exit 0
}

if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
    throw "File not found: $FilePath"
}

Write-Step "Uploading file"
$upload = Invoke-Json -Method Post -Uri "$BaseUrl/admin/files/upload" -Form @{
    file = Get-Item -LiteralPath $FilePath
}
$upload | ConvertTo-Json -Depth 10

$fileId = $upload.file_id
if (-not $fileId) {
    throw "Upload response did not include file_id"
}

Write-Step "Polling file status for file_id=$fileId"
$finalStates = @("active", "failed", "deleted", "delete_failed")
$status = $null
for ($i = 1; $i -le $MaxPolls; $i++) {
    Start-Sleep -Seconds $PollSeconds
    $status = Invoke-Json -Method Get -Uri "$BaseUrl/admin/files/$fileId"
    Write-Host ("poll {0}/{1}: {2}" -f $i, $MaxPolls, $status.status)

    if ($finalStates -contains $status.status) {
        break
    }
}

$status | ConvertTo-Json -Depth 20

if ($status.status -ne "active") {
    throw "Expected final status 'active', got '$($status.status)'"
}

Write-Step "Listing files after upload"
$after = Invoke-Json -Method Get -Uri "$BaseUrl/admin/files"
$after | ConvertTo-Json -Depth 10

if ($DeleteAfter) {
    Write-Step "Deleting uploaded file_id=$fileId"
    $delete = Invoke-Json -Method Delete -Uri "$BaseUrl/admin/files/$fileId"
    $delete | ConvertTo-Json -Depth 10

    if ($delete.status -ne "deleted") {
        throw "Expected delete status 'deleted', got '$($delete.status)'"
    }
}

Write-Step "Admin API test completed"
