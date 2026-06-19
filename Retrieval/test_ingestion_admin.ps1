param(
    [string]$BaseUrl = "http://localhost:8094",
    [string]$FilePath,
    [switch]$DeleteAfter,
    [int]$PollSeconds = 10,
    [int]$MaxPolls = 60
)

$scriptPath = Join-Path $PSScriptRoot "Doc_ingetion\test_admin_api.ps1"

if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Test script not found: $scriptPath"
}

& $scriptPath `
    -BaseUrl $BaseUrl `
    -FilePath $FilePath `
    -DeleteAfter:$DeleteAfter `
    -PollSeconds $PollSeconds `
    -MaxPolls $MaxPolls
