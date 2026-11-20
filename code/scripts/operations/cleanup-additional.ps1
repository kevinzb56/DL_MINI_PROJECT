[CmdletBinding()]
param([switch]$Delete, [switch]$IncludePreparedMilano)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path (Split-Path $PSScriptRoot -Parent) -Parent))
$audit = Get-Content -LiteralPath (Join-Path $projectRoot 'output\cleanup_additional_manifest.json') -Raw | ConvertFrom-Json

function SafePath([string]$relative) {
    if ([IO.Path]::IsPathRooted($relative)) { throw "Absolute path rejected: $relative" }
    $full = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $full.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside project: $full" }
    $part = $full
    while ($part -ne $projectRoot) {
        if ([IO.File]::Exists($part) -or [IO.Directory]::Exists($part)) {
            if ([IO.File]::GetAttributes($part) -band [IO.FileAttributes]::ReparsePoint) { throw "Linked path rejected: $part" }
        }
        $part = [IO.Path]::GetDirectoryName($part)
    }
    return $full
}
function VerifyFile($record) {
    $path = SafePath $record.path
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $record.sha256) { throw "File changed or invalid: $path" }
}

$preservedDependencies = @('web/node_modules', '.venv-multisource')
$groups = @($audit.groups | Where-Object {
    $relative = $_.path.Replace('\', '/')
    $preservedDependencies -notcontains $relative -and (-not $_.optional -or $IncludePreparedMilano)
})
$tracked = @(git -C $projectRoot ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Cannot check tracked files.' }
# Narrow allowlist: the manifest cannot authorize arbitrary directories.
$allowed = @('artifacts/phase8/CrackLab_RC1.zip', 'artifacts/phase8/CrackLab_RC1.zip.sha256')
if ($IncludePreparedMilano) {
    $allowed += @('data/external/milano_prepared_v1/train', 'data/external/milano_prepared_v1/val', 'data/external/milano_prepared_v1/test')
}
foreach ($record in $audit.required_files) { VerifyFile $record }
if ($IncludePreparedMilano) {
    Write-Host 'Verifying original Milano images and masks before removing their prepared copies...'
    foreach ($record in $audit.prepared_source_files) { VerifyFile $record }
}
$ready = @()
foreach ($group in $groups) {
    $relative = $group.path.Replace('\', '/')
    if ($allowed -notcontains $relative) { throw "Unexpected cleanup target: $relative" }
    $path = SafePath $group.path
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $entry = Get-Item -LiteralPath $path -Force
    $actual = @($entry)
    if ($entry.PSIsContainer) { $actual = @(Get-ChildItem -LiteralPath $path -Recurse -Force) }
    if (@($actual | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw "Linked entry under $path" }
    $files = @($actual | Where-Object { -not $_.PSIsContainer })
    if ($files.Count -ne @($group.files).Count) { throw "Folder contents changed: $path. Reaudit before deletion." }
    foreach ($file in $group.files) {
        $filePath = SafePath $file.path
        if ($filePath -ne $path -and -not $filePath.StartsWith($path + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest file outside its group.' }
        if ($tracked -contains $file.path.Replace('\', '/')) { throw "Tracked file: $filePath" }
        VerifyFile $file
    }
    $bytes = ($group.files | Measure-Object bytes -Sum).Sum
    $ready += [pscustomobject]@{ Path=$path; Relative=$relative; Bytes=$bytes }
}
$ready | Select-Object Relative, @{n='MiB';e={[math]::Round($_.Bytes/1MB,1)}} | Format-Table -AutoSize
Write-Host ('Total: {0:N2} GiB' -f (($ready | Measure-Object Bytes -Sum).Sum / 1GB))
Write-Host 'Preserved: Python environment, web dependencies, original datasets, Multi-Source inputs, recovery backups, checksums and reports.'
if ($IncludePreparedMilano) { Write-Host 'Milano prepared train/val/test will need rebuilding. Original dataset and preparation metadata stay intact. See docs/CLEANUP_ADDITIONAL.md.' }
if (-not $Delete) { Write-Host 'Preview only. Add -Delete to perform this cleanup.'; return }

$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(python.*|node)\.exe$' -and
    ($_.CommandLine -like "*$projectRoot*" -or -not $_.CommandLine)
})
if ($active.Count) { throw 'Stop project Python/Node processes before cleanup.' }
$results = foreach ($target in $ready) {
    try {
        $path = SafePath $target.Relative
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $path) { throw 'Target still exists.' }
        [pscustomobject]@{Path=$target.Relative; Deleted=$true; Bytes=$target.Bytes; Error=$null}
    } catch {
        [pscustomobject]@{Path=$target.Relative; Deleted=$false; Bytes=0; Error=$_.Exception.Message}
    }
}
$log = Join-Path $projectRoot ('output\cleanup_additional_execution_{0}.json' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
ConvertTo-Json -InputObject @($results) -Depth 5 | Set-Content -LiteralPath $log -Encoding UTF8
$results | Format-Table -Wrap
Write-Host "Execution log: $log"
if (@($results | Where-Object { -not $_.Deleted }).Count) { throw 'Some targets could not be fully deleted. See the log; a failed folder deletion may be partial.' }
