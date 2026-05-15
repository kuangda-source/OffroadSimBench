param(
    [string]$OrfdRoot = "",
    [switch]$BeamNGConnect
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvPrefix = Join-Path $RepoRoot ".conda\offroad-sim-bench"
$Python = Join-Path $EnvPrefix "python.exe"
$env:MAMBA_ROOT_PREFIX = Join-Path $RepoRoot ".mamba-root"
$EnvPathEntries = @(
    $EnvPrefix,
    (Join-Path $EnvPrefix "Library\mingw-w64\bin"),
    (Join-Path $EnvPrefix "Library\usr\bin"),
    (Join-Path $EnvPrefix "Library\bin"),
    (Join-Path $EnvPrefix "Scripts")
)
$env:Path = ($EnvPathEntries -join ";") + ";" + $env:Path

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
        & $Python scripts\create_mock_orfd_dataset.py $OrfdRoot --frames 8
    }
}

Invoke-AcceptanceStep "Python test suite" {
    & $Python -m pytest -q
}

Invoke-AcceptanceStep "Inspect ORFD dataset" {
    & $Python examples\inspect_dataset.py $OrfdRoot --adapter orfd
}

Invoke-AcceptanceStep "Train tiny learned world model" {
    & $Python scripts\train_world_model.py $OrfdRoot --adapter orfd --output $ModelOutput
}

Invoke-AcceptanceStep "Dataset replay with switchable world model" {
    & $Python -m offroad_sim.cli run --backend dataset_replay --dataset-root $OrfdRoot --adapter orfd --agent world_model --world-model-type tiny_learned --world-model $ModelOutput --max-steps 3 --record --record-arrays --json
}

Invoke-AcceptanceStep "Dataset replay with CEM path planner" {
    & $Python -m offroad_sim.cli run --backend dataset_replay --dataset-root $OrfdRoot --adapter orfd --agent world_model --world-model-type tiny_learned --world-model $ModelOutput --planner world_model_cem --planner-horizon 4 --planner-samples 16 --planner-iterations 2 --max-steps 3 --record --json
}

Invoke-AcceptanceStep "BeamNG runtime status" {
    & $Python examples\check_beamng_runtime.py
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
        & $Python examples\run_beamng_world_model.py --scenario configs\scenarios\beamng_orfd_eval.yaml --world-model-type tiny_learned --world-model $ModelOutput --planner world_model_cem --max-steps 5 --record
    }
}

Write-Host ""
Write-Host "Phase 3 acceptance passed." -ForegroundColor Green
