# MemAdapter

This is the preserved, unexecuted three-stage experiment design. It is kept
separately from the formal `baseline` and `ours/M6` archive.

## Stages

1. Counterfactual memory risk detection: create a query-agnostic risk card for
   every retrieved memory.
2. Context-aware memory reflection: use the current query and available
   evidence to assign each memory its current role and authority boundary.
3. Role-guided final generation: generate the answer under those per-memory
   role instructions.

The three system prompts are in `prompts/`. `memadapter.py` provides prompt
loading, input rendering, response validation, and the core orchestration.
`model_client.py` provides the OpenAI-compatible model adapter used by the
formal runner.

## Minimal usage

```python
from MemAdapter.memadapter import run_three_stage

result = run_three_stage(
    memories=[{"memory_id": "memory_1", "memory_text": "..."}],
    current_query="...",
    call_model=lambda system, user: client_call(system, user),
)
print(result["final_answer"])
```

The returned object preserves both structured intermediate stages for logging,
inspection, and a later judge stage.

Install dependencies before a formal run:

```powershell
python -m pip install -r benchmark/requirements.txt
```

The runner reuses frozen retrieval files, so `requirements-memory-baselines.txt`
is only needed if retrieval itself must be regenerated.

For generation, configure one model family:

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:DEEPSEEK_MODEL = "deepseek-v4-flash"
$env:DEEPSEEK_BASE_URL = "<your-authorized-endpoint>"
```

Then run a small check before the full experiment:

```powershell
python MemAdapter/run_memadapter.py run-all --model DeepSeek --memory-system AMEM --limit 10 --require-top-10
```

The full run uses the same command without `--limit`. Available memory systems
are `AMEM`, `Mem0`, and `naiveRAG`; available model families are `DeepSeek`,
`GPT`, and `Qwen`. GPT uses `EVAL_API_KEY`, `EVAL_BASE_URL`, and `EVAL_MODEL`;
Qwen uses `QWEN_API_KEY`, `QWEN_BASE_URL`, and `QWEN_MODEL`.

The frozen retrieval files currently contain fewer than ten memories for some
samples because those samples do not provide ten available memories. The
formal runner requires an explicit choice: use `--require-top-10` for a strict
Top-10 run, or `--allow-fewer-than-top-10` when deliberately reusing these
existing frozen rows. The latter records the actual memory count and is the
appropriate choice for the current pilot because some existing AMEM rows have
fewer than ten memories.

Use `--per-task-limit 30` to select the first 30 rows of each of the five tasks.
The runner verifies that every selected sample ID exists in the earlier GPT
baseline and M6 outputs before making any model calls. Use `--workers` to tune
concurrency; the default is 8.

Judge credentials default to the generation credentials. They can be separated
with `JUDGE_API_KEY`, `JUDGE_BASE_URL`, and `JUDGE_MODEL`. Results are written
to `MemAdapter/results/<model>/<memory-system>/` and contain `outputs.jsonl`,
`judge.jsonl`, `summary.json`, and `validation_report.json`. Generation and
judge runs also record:

- `efficiency_calls.jsonl`: one row per real API request, including timestamps,
  latency, attempt number, failures, and API-reported token usage when present.
- `efficiency_samples.jsonl`: one row per sample with phase totals, Stage 1/2/3
  totals for MemAdapter generation, retry counts, and wall-clock time.
- `efficiency_summary.json`: batch totals, mean/median/P95 time, batch wall-clock
  time, throughput, task breakdowns, and retrieval exclusion metadata.
- `efficiency_samples/<sample_id>.json`: lossless per-sample records used to
  rebuild the flat artifacts.

Token counts come from the API response's actual `usage` object. `max_tokens`
is stored as configuration only and is never treated as consumed tokens.
Generation and judge costs are kept in separate phases, and reused retrieval
time and tokens are explicitly excluded from both.
