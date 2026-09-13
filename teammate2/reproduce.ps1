# Run from any directory. Outputs are new, timestamped, and remain separate
# from the historical package. Frozen annotations are AI references.
$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path $PSScriptRoot -Parent
$PythonExe = Join-Path $RepositoryRoot "venv/Scripts/python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = Join-Path $RepositoryRoot "venv/bin/python"
}
$OutputDirectory = Join-Path $RepositoryRoot ("results/reasoning/package_reproduction_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
Push-Location $RepositoryRoot
try {
    & $PythonExe -m src.evaluation.packaged_reproduction --output-dir $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "Packaged reasoning reproduction failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
}
# For live checks, separately run src.evaluation.live_completion_checks with
# --external-authorized and a fresh --output-dir after configuring local stores.
