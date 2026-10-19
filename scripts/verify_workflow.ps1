# Usage: powershell -ExecutionPolicy Bypass -File scripts\verify_workflow.ps1 [-CleanLegacy] [-CleanArtifacts]
#   -CleanLegacy     remove obsolete leftovers (toy_data\, project1\data\sh2_uniprot.fasta)
#   -CleanArtifacts  ALSO delete results\* and models\*.pt  (destructive, off by default)
# to run this : powershell -ExecutionPolicy Bypass -File scripts\verify_workflow.ps1
# should return ALL CHECKS PASSED if everything is fine 
param([switch]$CleanLegacy, [switch]$CleanArtifacts)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)         # repo root, whatever the caller's folder
$env:PYTHONWARNINGS = 'ignore::DeprecationWarning'       # hides the pandas/pyarrow notice only
Remove-Item Env:AI_PROJECT_DIR -ErrorAction Ignore       # force local/toy mode

function Invoke-Native([string]$Label, [string]$Exe, [string[]]$CmdArgs) {
    Write-Host "==> $Label"
    $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'  # PS 5.1: stderr must not throw
    $lines = & $Exe @CmdArgs 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE; $ErrorActionPreference = $prev
    $lines | ForEach-Object { Write-Host "    $_" }
    if ($code -ne 0) { throw "$Label failed (exit code $code)" }
    return $lines
}
function Remove-IfExists([string]$Path) {
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force; Write-Host "removed  $Path" }
    else { Write-Host "absent   $Path (ok)" }
}

$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$null = Invoke-Native 'python deps' $py @('-c', 'import pandas, numpy, torch, pytest; print(42)')

if ($CleanLegacy) { Remove-IfExists 'toy_data'; Remove-IfExists 'project1\data\sh2_uniprot.fasta' }
if ($CleanArtifacts) {
    Get-ChildItem results -ErrorAction Ignore | Remove-Item -Recurse -Force
    Get-ChildItem models -Filter *.pt -ErrorAction Ignore | Remove-Item -Force
}

# 1. Toy raw data: regenerate from scratch, twice, and compare
Remove-IfExists 'data\toy'
$null = Invoke-Native 'make_toy_data (run 1)' $py @('scripts\make_toy_data.py')
$raw = 'data\toy\raw\toy_oas_raw.csv'
if (-not (Test-Path $raw)) { throw "missing $raw" }
$h1 = (Get-FileHash $raw -Algorithm SHA256).Hash
$null = Invoke-Native 'make_toy_data (run 2)' $py @('scripts\make_toy_data.py')
if ((Get-FileHash $raw -Algorithm SHA256).Hash -ne $h1) { throw 'toy data is not deterministic' }
$n = @([IO.File]::ReadLines((Resolve-Path $raw).Path)).Count
if ($n -ne 241) { throw "expected 241 lines (JSON + header + 239 rows), got $n" }

# 2. Slim: 14 columns, same number of rows
$null = Invoke-Native 'slim_columns' $py @('scripts\slim_columns.py', '--inputs', $raw, '--out-dir', 'data\toy\processed\slim')
$slim = 'data\toy\processed\slim\toy_oas_raw.csv'
if ((Get-Content $slim -TotalCount 2)[1].Split(',').Count -ne 14) { throw 'slim file should have 14 columns' }
if (@([IO.File]::ReadLines((Resolve-Path $slim).Path)).Count -ne $n) { throw 'slim changed the row count' }

# 3. get_paths() must be in local toy mode
$res = Invoke-Native 'get_paths() local mode' $py @('-c', 'from utils import get_paths; print(get_paths()[''raw''])')
if (($res -join ' ') -notmatch 'data\\toy\\raw') { throw 'get_paths() is not pointing to data\toy\raw' }

# 4. Tests
$null = Invoke-Native 'pytest' $py @('-m', 'pytest', '-q', '-p', 'no:cacheprovider')

# 5. Git hygiene: nothing heavy tracked, toy data ignored
$bad = @(git ls-files) | Where-Object {
    $_ -match '^(\.venv|data|toy_data|results|models)/' -or $_ -match '\.(pt|pth|ckpt|csv|csv\.gz|fasta)$' }
if ($bad) { $bad | ForEach-Object { Write-Host "TRACKED: $_" }; throw 'forbidden files are tracked by git' }
git check-ignore -q data/toy/raw/toy_oas_raw.csv
if ($LASTEXITCODE -ne 0) { throw 'data/toy/raw/toy_oas_raw.csv is not git-ignored' }

Write-Host "`nALL CHECKS PASSED"