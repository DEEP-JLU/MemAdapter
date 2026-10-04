# MemAdapter

This directory contains the three-stage MemAdapter implementation and its
retained ablations.

## Components

- `memadapter.py` loads the stage prompts, renders model inputs, validates
  structured intermediate outputs, and runs the three-stage method.
- `model_client.py` provides API and local model clients.
- `run_memadapter.py` runs generation and judging from frozen retrieval files.
- `prompts/` contains the three prompts used by the complete method.
- `ablations/` contains the two retained variants: Stage 1 only, and Stages 1
  and 2 only.

## Inputs

MemAdapter consumes a current request and a frozen retrieval record. The same
retrieval record is used by every compared method in an experimental cell.

## Usage

Install the repository dependencies:

```powershell
python -m pip install -r requirements.txt
```

Inspect the native runner options:

```powershell
python methods/memadapter/run_memadapter.py --help
```

Inspect one retained ablation launcher:

```powershell
powershell -ExecutionPolicy Bypass -File methods/memadapter/ablations/run_ablation.ps1 -?
```
