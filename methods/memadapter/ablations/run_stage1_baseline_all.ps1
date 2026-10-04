param(
    [ValidateSet("DeepSeek-V4-Flash", "GPT-5.6-sol", "Qwen3-8B")]
    [string]$Model = "DeepSeek-V4-Flash",

    [int]$Workers = 4,
    [Nullable[int]]$Limit,
    [string]$RetrievalRoot,
    [string]$OutputRoot,
    [switch]$RequireTop10
)

# Run the Stage-1-only MemAdapter ablation over all five frozen retrieval sets.
# The underlying runner is resumable, so rerunning this script continues an
# interrupted system without regenerating completed records.
$ErrorActionPreference = "Stop"
$systems = @("AMEM", "Mem0", "naiveRAG", "MemoryBank", "LightMem")
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptRoot "run_ablation.ps1"

if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY) -and $Model -eq "DeepSeek-V4-Flash") {
    throw "DEEPSEEK_API_KEY is not set in this PowerShell session. Set it before launching the experiment."
}

foreach ($system in $systems) {
    $invokeParams = @{
        Variant = "stage1-baseline"
        Model = $Model
        MemorySystem = $system
        Workers = $Workers
    }
    if ($null -ne $Limit) { $invokeParams.Limit = $Limit }
    if ($RequireTop10) { $invokeParams.RequireTop10 = $true }
    if ($RetrievalRoot) {
        $invokeParams.RetrievalFile = Join-Path $RetrievalRoot "$system.jsonl"
    }
    if ($OutputRoot) {
        $invokeParams.OutputDir = Join-Path $OutputRoot $system
    }

    Write-Host "[stage1-baseline] starting $system"
    & $runner @invokeParams
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
