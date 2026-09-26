#!/usr/bin/env bash
# Run the full external-benchmark pipeline on a multi-GPU Linux server.
# DEEPSEEK_API_KEY is intentionally inherited from the invoking environment.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-python}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export RETRIEVAL_GPU_IDS="${RETRIEVAL_GPU_IDS:-2,3}"

bash external_benchmarks/launcher/run_external_retrieval.sh \
  --lane both \
  --shards "${RETRIEVAL_SHARDS:-24}" \
  --jobs "${RETRIEVAL_JOBS:-8}" \
  --min-free-mib "${RETRIEVAL_MIN_FREE_MIB:-32768}" \
  --attempts "${RETRIEVAL_ATTEMPTS:-3}"

bash external_benchmarks/launcher/run_external_arms.sh \
  --lane both \
  --no-retrieval \
  --jobs "${ARM_JOBS:-12}" \
  --workers "${ARM_WORKERS:-96}"
