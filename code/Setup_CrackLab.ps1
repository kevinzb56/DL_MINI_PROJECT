param([switch]$RunTests)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install it, then rerun setup.' }
}
$crackPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath 'RELEASE_MANIFEST.json') {
    & $crackPython scripts/verify_release.py
    if ($LASTEXITCODE -ne 0) { throw 'Release verification failed. Preserve the folder and inspect it.' }
}
& $crackPython -m pip install -r requirements-release.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath 'web\dist\index.html')) {
    npm.cmd --prefix web ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    npm.cmd --prefix web run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
}
if (Test-Path -LiteralPath 'release_assets\source_replay\manifest.json') {
    & $crackPython scripts/bootstrap_release.py
} else {
    & $crackPython scripts/prepare_inspection_models.py
    if ($LASTEXITCODE -eq 0) { & $crackPython scripts/prepare_source_replay.py }
}
if ($LASTEXITCODE -ne 0) { throw 'Model/data preparation failed.' }
if ($RunTests) {
    New-Item -ItemType Directory -Force -Path 'artifacts\release-checks' | Out-Null
    $crackTestDir = 'artifacts/release-checks/test-' + [guid]::NewGuid().ToString('N')
    & $crackPython -m pytest tests -q -p no:cacheprovider "--basetemp=$crackTestDir"
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
}
Write-Host 'CrackLab is ready. Run Start_CrackLab.cmd, then open http://127.0.0.1:8765'
