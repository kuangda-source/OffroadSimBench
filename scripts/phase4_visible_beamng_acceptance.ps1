param(
    [switch]$BeamNGVisible,
    [string]$OrfdRoot = "datasets\ORFD_Dataset_ICRA2022_ZIP",
    [string]$SequenceId = "training/c2021_0228_1819",
    [string]$WorldModelType = "le_wm",
    [string]$WorldModelPath = "outputs\models\lewm_orfd_real_c2021_0228_1819",
    [string]$Planner = "le_wm_cem",
    [int]$MaxSteps = 80,
    [int]$BeamNGTimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$python = Join-Path (Get-Location) ".conda\offroad-sim-bench\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

& $python -m pytest tests/test_beamng_visible_config.py tests/test_beamng_backend_visible.py tests/test_route_world_model_agent.py tests/test_desktop_visible_demo.py -q
if ($LASTEXITCODE -ne 0) { throw "Phase 4 focused tests failed." }

& $python -m offroad_sim.cli list --kind all
if ($LASTEXITCODE -ne 0) { throw "CLI catalog listing failed." }

if (-not $BeamNGVisible) {
    Write-Output "Phase 4 non-launching checks passed. Add -BeamNGVisible to launch the visible BeamNG demo."
    exit 0
}

$demoArgs = @(
    "scripts\run_beamng_visible_demo.py",
    "--dataset-root", $OrfdRoot,
    "--adapter", "orfd",
    "--sequence-id", $SequenceId,
    "--world-model-type", $WorldModelType,
    "--scenario", "beamng_visible_autodrive",
    "--vehicle", "configs\vehicles\ugv_medium.yaml",
    "--max-steps", [string]$MaxSteps,
    "--pre-run-hold-sec", "0",
    "--step-delay-sec", "0",
    "--hold-open-sec", "0",
    "--close-beamng"
)
if ($WorldModelPath) {
    $demoArgs += @("--world-model", $WorldModelPath)
}
if ($Planner) {
    $demoArgs += @("--planner", $Planner)
}

$stamp = Get-Date -Format "yyyyMMddTHHmmss"
$stdoutPath = Join-Path $env:TEMP "offroad_sim_phase4_beamng_$stamp.out"
$stderrPath = Join-Path $env:TEMP "offroad_sim_phase4_beamng_$stamp.err"
$process = Start-Process -FilePath $python -ArgumentList $demoArgs -WorkingDirectory (Get-Location) -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
if (-not $process.WaitForExit($BeamNGTimeoutSeconds * 1000)) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { "" }
    $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    Write-Output $stdout
    Write-Output $stderr
    throw "Visible BeamNG demo timed out after $BeamNGTimeoutSeconds seconds."
}
$process.Refresh()
$stdoutText = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { "" }
$stderrText = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
$exitCode = $process.ExitCode
if ($null -ne $exitCode -and $exitCode -ne 0) {
    Write-Output $stdoutText
    Write-Output $stderrText
    throw "Visible BeamNG demo failed with exit code $exitCode."
}
$text = $stdoutText
$marker = $text.IndexOf('"episode_id"')
if ($marker -lt 0) {
    $text = ($stdoutText, $stderrText) -join [Environment]::NewLine
    $marker = $text.IndexOf('"episode_id"')
}
$start = if ($marker -ge 0) { $text.LastIndexOf("{", $marker) } else { -1 }
$end = $text.LastIndexOf("}")
if ($start -lt 0 -or $end -le $start) {
    $text | Write-Output
    throw "Visible BeamNG demo did not emit a JSON payload."
}
$payload = $text.Substring($start, $end - $start + 1) | ConvertFrom-Json
$metrics = $payload.metrics
if (-not $metrics.connected) { throw "BeamNG did not report connected=true." }
if ([int]$payload.steps -lt 60) { throw "Expected at least 60 steps, got $($payload.steps)." }
if ([double]$metrics.horizontal_distance_traveled -lt 10.0) { throw "Expected at least 10m horizontal travel, got $($metrics.horizontal_distance_traveled)." }
if ([double]$metrics.max_abs_vertical_deviation -gt 8.0) { throw "Vehicle left the drivable surface vertically by $($metrics.max_abs_vertical_deviation)m." }
if ([string]$metrics.level -ne "gridmap_v2") { throw "Expected gridmap_v2 level, got $($metrics.level)." }
if (-not (Test-Path -LiteralPath $payload.episode_path)) { throw "Episode path does not exist: $($payload.episode_path)" }

[pscustomobject]@{
    status = "phase4_visible_beamng_passed"
    episode_id = $payload.episode_id
    steps = $payload.steps
    distance_traveled = $metrics.distance_traveled
    connected = $metrics.connected
    level = $metrics.level
    route_waypoint_count = $metrics.route_waypoint_count
    world_model = $payload.visible_demo.world_model_type
    planner = $payload.visible_demo.planner
    episode_path = $payload.episode_path
} | ConvertTo-Json -Depth 4
