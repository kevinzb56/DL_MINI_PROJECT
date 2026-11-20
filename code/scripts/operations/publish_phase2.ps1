# Run under your regular Windows account from the project workspace.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
function Invoke-Phase2Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArguments -join ' ')" }
}
$remoteUrl = & git remote get-url origin
if ($LASTEXITCODE -ne 0 -or $remoteUrl -ne 'https://github.com/kevinzb56/LightCrack-Seg.git') {
    throw 'Unexpected origin. Inspect the repository before publishing.'
}
$branch = & git branch --show-current
if ($LASTEXITCODE -ne 0 -or -not $branch) { throw 'Expected a checked-out branch.' }
$staged = @(& git diff --cached --name-only)
if ($LASTEXITCODE -ne 0 -or $staged.Count -gt 0) {
    throw 'Review and clear any existing staged changes before running this script.'
}
Invoke-Phase2Git ls-remote --heads origin
$phase2Paths = @(
    'README.md', 'requirements.txt',
    'configs/baseline_r2au.json', 'configs/baseline_unet.json', 'configs/data_sources.json',
    'configs/stabilized_r2au.json', 'configs/stabilized_unet.json',
    'data/manifests/baseline_registry_v1.json',
    'docs/PHASE_2.md', 'docs/PHASE_2_FINAL_RESULTS.md',
    'docs/PHASE_2_RESULTS_V1.md', 'docs/PHASE_2_RESULTS_V2.md',
    'output/kaggle/Road_Crack_Phase2_Baselines.ipynb',
    'output/kaggle/Road_Crack_Phase2_Final_Evaluation.ipynb',
    'output/kaggle/Road_Crack_Phase2_Stabilization.ipynb',
    'scripts/authoring/build_final_evaluation_notebook.py', 'scripts/authoring/build_phase2_notebook.py',
    'scripts/evaluation/evaluate_frozen_baselines.py', 'scripts/evaluation/profile_baseline.py',
    'scripts/data/restore_datasets.py', 'scripts/experiments/train_baseline.py', 'scripts/operations/publish_phase2.ps1',
    'src/dataset.py', 'src/models.py', 'src/segmentation.py',
    'tests/test_dataset_restore.py', 'tests/test_phase2.py'
)
Invoke-Phase2Git add -- @phase2Paths
Invoke-Phase2Git diff --cached --check
Invoke-Phase2Git commit -m 'Complete Phase 2: checkpointed baselines and verified pilot evaluation'
# No force push or automatic history rewriting.
Invoke-Phase2Git push origin "HEAD:refs/heads/$branch"
Write-Host "Published Phase 2 to $remoteUrl/tree/$branch"
