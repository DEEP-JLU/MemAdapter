param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("stage1-baseline", "stage1-stage2-baseline")]
    [string]$Variant,

    [Parameter(Mandatory = $true)]
    [ValidateSet("DeepSeek", "GPT", "Qwen")]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [ValidateSet("AMEM", "Mem0", "naiveRAG", "MemoryBank", "LightMem")]
    [string]$MemorySystem,

    [int]$Workers = 4,
    [Nullable[int]]$Limit,
    [string]$RetrievalFile,
    [string]$EvidenceFile,
    [string]$OutputDir,
    [switch]$RequireTop10
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$runner = Join-Path $repoRoot "methods\memadapter\run_memadapter.py"

$common = @(
    $runner,
    "--model", $Model,
    "--memory-system", $MemorySystem,
    "--ablation", $Variant,
    "--workers", $Workers,
    "--continue-on-error"
)

if ($RequireTop10) {
    $common += "--require-top-10"
} else {
    $common += "--allow-fewer-than-top-10"
}
if ($null -ne $Limit) { $common += "--limit", $Limit }
if ($RetrievalFile) { $common += "--retrieval-file", $RetrievalFile }
if ($EvidenceFile) { $common += "--evidence-file", $EvidenceFile }
if ($OutputDir) { $common += "--output-dir", $OutputDir }

& python @($common + "generate")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python @($common + "judge")
exit $LASTEXITCODE
