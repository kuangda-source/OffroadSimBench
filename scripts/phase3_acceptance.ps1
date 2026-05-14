param(
    [string]$OrfdRoot = "",
    [switch]$BeamNGConnect
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Micromamba = Join-Path $RepoRoot "BeamNG\tools\Library\bin\micromamba.exe"
$EnvPrefix = Join-Path $RepoRoot ".conda\offroad-sim-bench"
$env:MAMBA_ROOT_PREFIX = Join-Path $RepoRoot ".mamba-root"

if (-not $OrfdRoot) {
    $OrfdRoot = Join-Path $RepoRoot "outputs\mock_orfd_phase3"
}
$ModelOutput = Join-Path $RepoRoot "outputs\models\phase3_tiny_world_model"

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

if (-not (Test-Path $OrfdRoot)) {
    Invoke-AcceptanceStep "Create tiny ORFD fixture" {
        & $Micromamba run -p $EnvPrefix python scripts\create_mock_orfd_dataset.py $OrfdRoot --frames 8
    }
}

Invoke-AcceptanceStep "Python test suite" {
    & $Micromamba run -p $EnvPrefix python -m pytest -q
}

Invoke-AcceptanceStep "Inspect ORFD dataset" {
    & $Micromamba run -p $EnvPrefix python examples\inspect_dataset.py $OrfdRoot --adapter orfd
}

Invoke-AcceptanceStep "Train tiny learned world model" {
    & $Micromamba run -p $EnvPrefix python scripts\train_world_model.py $OrfdRoot --adapter orfd --output $ModelOutput
}

Invoke-AcceptanceStep "Dataset replay with switchable world model" {
    & $Micromamba run -p $EnvPrefix python -m offroad_sim.cli run --backend dataset_replay --dataset-root $OrfdRoot --adapter orfd --agent world_model --world-model-type tiny_learned --world-model $ModelOutput --max-steps 3 --record --record-arrays --json
}

Invoke-AcceptanceStep "BeamNG runtime status" {
    & $Micromamba run -p $EnvPrefix python examples\check_beamng_runtime.py
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
    Invoke-AcceptanceStep "BeamNG world-model connection run" {
        & $Micromamba run -p $EnvPrefix python examples\run_beamng_world_model.py --scenario configs\scenarios\beamng_orfd_eval.yaml --world-model-type tiny_learned --world-model $ModelOutput --max-steps 5 --record
    }
}

Write-Host ""
Write-Host "Phase 3 acceptance passed." -ForegroundColor Green
