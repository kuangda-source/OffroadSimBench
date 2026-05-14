param(
    [switch]$BeamNGConnect
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Micromamba = Join-Path $RepoRoot "BeamNG\tools\Library\bin\micromamba.exe"
$EnvPrefix = Join-Path $RepoRoot ".conda\offroad-sim-bench"
$env:MAMBA_ROOT_PREFIX = Join-Path $RepoRoot ".mamba-root"

function Invoke-AcceptanceStep {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Acceptance step failed: $Name"
    }
}

Invoke-AcceptanceStep "Python test suite" {
    & $Micromamba run -p $EnvPrefix python -m pytest -q
}

Invoke-AcceptanceStep "Backend catalog status" {
    & $Micromamba run -p $EnvPrefix python examples\check_backends.py
}

Invoke-AcceptanceStep "BeamNG runtime status" {
    & $Micromamba run -p $EnvPrefix python examples\check_beamng_runtime.py
}

Invoke-AcceptanceStep "CLI recorded smoke episode" {
    & $Micromamba run -p $EnvPrefix python -m offroad_sim.cli run --agent stop --max-steps 3 --record
}

Invoke-AcceptanceStep "Dashboard API streaming smoke" {
    & $Micromamba run -p $EnvPrefix python -m pytest tests\test_dashboard_backend.py -q
}

Invoke-AcceptanceStep "Frontend production build" {
    Push-Location (Join-Path $RepoRoot "dashboard\frontend")
    try {
        npm run build
    }
    finally {
        Pop-Location
    }
}

if ($BeamNGConnect) {
    Invoke-AcceptanceStep "BeamNG real connection smoke" {
        & $Micromamba run -p $EnvPrefix python examples\check_beamng_runtime.py --connect --steps 1
    }
}

Write-Host ""
Write-Host "Phase 2 acceptance passed." -ForegroundColor Green
