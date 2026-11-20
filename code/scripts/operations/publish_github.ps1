# Run from your ordinary PowerShell account, outside the restricted agent shell.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
$repositoryUrl = 'https://github.com/kevinzb56/LightCrack-Seg.git'

function Invoke-GitChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git command failed: $($GitArguments -join ' ')" }
}

if (-not (Test-Path -LiteralPath '.git')) {
    Invoke-GitChecked init -b main
}
$remoteNames = @(& git remote)
if ($LASTEXITCODE -ne 0) { throw 'Cannot read Git configuration.' }
if ($remoteNames -contains 'origin') {
    $currentUrl = & git remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $currentUrl -ne $repositoryUrl) {
        throw 'Existing origin differs from requested repository; inspect before publishing.'
    }
} else {
    Invoke-GitChecked remote add origin $repositoryUrl
}

# Verify authentication/network before creating a local commit.
$remoteHeads = @(& git ls-remote --heads origin)
if ($LASTEXITCODE -ne 0) { throw 'GitHub access failed. Sign into GitHub through Git Credential Manager and retry.' }
Invoke-GitChecked add --all
& git diff --cached --quiet
$diffStatus = $LASTEXITCODE
if ($diffStatus -eq 1) {
    Invoke-GitChecked commit -m 'Add Phase 1 project setup, audit evidence, and frozen pilot protocol'
} elseif ($diffStatus -ne 0) {
    throw 'Could not inspect staged changes.'
}

# A nonempty remote gets a new branch, preserving all existing branches/history.
$destination = if ($remoteHeads.Count -eq 0) { 'main' } else { 'phase1-upload-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
Invoke-GitChecked push --set-upstream origin "HEAD:refs/heads/$destination"
Write-Host "Published to $repositoryUrl/tree/$destination"
Invoke-GitChecked status --short
