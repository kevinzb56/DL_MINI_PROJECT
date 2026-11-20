# Run in a fresh PowerShell terminal. The key stays in this process environment.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $env:GROQ_API_KEY) {
    $env:GROQ_API_KEY = [Environment]::GetEnvironmentVariable('GROQ_API_KEY', 'User')
}
if (-not $env:GROQ_API_KEY) {
    $secureKey = Read-Host 'Paste your Groq API key' -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $env:GROQ_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    if (-not $env:GROQ_API_KEY) { throw 'A Groq API key is required.' }
}
try {

    $listeners = Get-NetTCPConnection -State Listen -LocalPort 8765 -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $process = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -eq 'python') {
            Write-Host "Stopping the existing CrackLab server (PID $($process.Id))..."
            Stop-Process -Id $process.Id -Force
        }
    }
    Start-Sleep -Milliseconds 500

    & .\.venv\Scripts\python.exe scripts\run_inspection.py
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    Remove-Item Env:\GROQ_API_KEY -ErrorAction SilentlyContinue
}
