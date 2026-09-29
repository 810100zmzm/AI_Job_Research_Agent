param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [string]$FrontendHost = "127.0.0.1",
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$FrontendDir = Join-Path $RootDir "frontend"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
}
else {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$Npm = (Get-Command npm.cmd -ErrorAction Stop).Source
$env:VITE_PROXY_TARGET = "http://${BackendHost}:$BackendPort"

if (-not (Test-Path (Join-Path $FrontendDir "package.json"))) {
    throw "frontend/package.json not found: $FrontendDir"
}

$LogBase = Join-Path $env:TEMP "job-research-dev-$PID"
$BackendOut = "$LogBase-backend.out.log"
$BackendErr = "$LogBase-backend.err.log"

Write-Host "Starting backend:" -ForegroundColor Cyan
Write-Host "  http://${BackendHost}:$BackendPort"
Write-Host "  logs: $BackendOut"
Write-Host "Starting frontend:" -ForegroundColor Cyan
Write-Host "  http://${FrontendHost}:$FrontendPort"

try {
    $backend = Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "uvicorn", "api.main:app",
            "--reload",
            "--host", $BackendHost,
            "--port", $BackendPort
        ) `
        -WorkingDirectory $RootDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $BackendOut `
        -RedirectStandardError $BackendErr `
        -PassThru

    Start-Sleep -Milliseconds 800
    if ($backend.HasExited) {
        if (Test-Path $BackendErr) {
            Get-Content $BackendErr | Write-Error
        }
        throw "Backend failed to start; see $BackendErr"
    }

    Push-Location $FrontendDir
    try {
        & $Npm run dev -- --host $FrontendHost --port $FrontendPort
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($backend -and -not $backend.HasExited) {
        & taskkill.exe /PID $backend.Id /T /F *> $null
    }
}
