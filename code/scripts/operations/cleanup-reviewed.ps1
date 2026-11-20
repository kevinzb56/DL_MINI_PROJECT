[CmdletBinding()]
param([switch]$Delete)

# Preview by default. Only -Delete removes files from the reviewed manifest.
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path (Split-Path $PSScriptRoot -Parent) -Parent))
$manifestPath = Join-Path $projectRoot 'output\cleanup_2026-11-16.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

function Resolve-ProjectFile([string]$relativePath) {
    if ([IO.Path]::IsPathRooted($relativePath)) { throw "Expected relative path: $relativePath" }
    $fullPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $relativePath))
    if (-not $fullPath.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside project: $relativePath"
    }
    $cursor = $fullPath
    while ($cursor -ne $projectRoot) {
        if (Test-Path -LiteralPath $cursor) {
            $entry = Get-Item -LiteralPath $cursor -Force
            if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing linked path: $cursor"
            }
        }
        $cursor = Split-Path $cursor -Parent
    }
    return $fullPath
}

# Validate retained backups before touching any candidates.
$protectedPaths = @()
foreach ($relativePath in $manifest.kept_recovery_archives) {
    $archive = Resolve-ProjectFile $relativePath
    $checksum = "$archive.sha256"
    $expected = ((Get-Content -LiteralPath $checksum -Raw).Trim() -split '\s+')[0]
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
        throw "Retained backup checksum mismatch: $relativePath"
    }
    $protectedPaths += $archive, $checksum
}
if ($protectedPaths.Count -ne 6) { throw 'Expected three retained backups and their checksums.' }

$tracked = @(git -C $projectRoot ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Unable to check Git tracked files.' }
$candidates = @()
foreach ($item in $manifest.targets) {
    $path = Resolve-ProjectFile $item.path
    if ($protectedPaths -contains $path) { throw "Protected backup in deletion list: $path" }
    if ($tracked -contains $item.path.Replace('\', '/')) { throw "Tracked file in deletion list: $path" }
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $file = Get-Item -LiteralPath $path -Force
    if ($file.PSIsContainer) { throw "Only individual files may be deleted: $path" }
    if ($file.Length -ne $item.bytes) { throw "File size changed since review: $path" }
    $candidates += [pscustomobject]@{ Path=$path; RelativePath=$item.path; Bytes=$file.Length; Reason=$item.reason }
}

$candidates | Group-Object Reason | ForEach-Object {
    [pscustomobject]@{ Reason=$_.Name; Files=$_.Count; MiB=[math]::Round(($_.Group | Measure-Object Bytes -Sum).Sum / 1MB, 1) }
} | Format-Table -AutoSize -Wrap
$totalBytes = ($candidates | Measure-Object Bytes -Sum).Sum
Write-Host ("Reviewed candidates: {0} files, {1:N2} GiB" -f $candidates.Count, ($totalBytes / 1GB))
if (-not $Delete) {
    Write-Host 'Preview only. Run this script with -Delete to remove these files.'
    return
}

$results = foreach ($candidate in $candidates) {
    try {
        $path = Resolve-ProjectFile $candidate.RelativePath
        if ((Get-Item -LiteralPath $path -Force).Length -ne $candidate.Bytes) { throw 'File changed after preflight.' }
        Remove-Item -LiteralPath $path -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $path) { throw 'File still exists after deletion.' }
        [pscustomobject]@{ Path=$candidate.RelativePath; Bytes=$candidate.Bytes; Deleted=$true; Error=$null }
    } catch {
        [pscustomobject]@{ Path=$candidate.RelativePath; Bytes=$candidate.Bytes; Deleted=$false; Error=$_.Exception.Message }
    }
}
$logPath = Join-Path $projectRoot ('output\cleanup_execution_{0}.json' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
ConvertTo-Json -InputObject @($results) -Depth 5 | Set-Content -LiteralPath $logPath -Encoding UTF8
$deleted = @($results | Where-Object Deleted)
$failed = @($results | Where-Object { -not $_.Deleted })
Write-Host ("Deleted {0} files ({1:N2} GiB). Failures: {2}." -f $deleted.Count, (($deleted | Measure-Object Bytes -Sum).Sum / 1GB), $failed.Count)
Write-Host "Log: $logPath"
if ($failed.Count) { $failed | Format-Table Path, Error -Wrap; exit 1 }
