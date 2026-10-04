"""Measure resident memory of a shard-like process holding the local embedder.

Measures the host-RAM cost of a retrieval shard. Every shard process builds its
own SentenceTransformer on the same local BGE-M3 snapshot. safetensors can
share mapped pages through the page cache; this utility measures the observed
memory use before choosing a worker count.

Run several at once and compare the sum of RSS against the drop in available RAM:
if the sum grows linearly while available RAM barely moves, the pages are shared
and the shard count is bounded by something other than the weights.
"""

from __future__ import annotations

import os
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from external_benchmarks import run_config as RC  # noqa: E402

RC.load_env_file()
from external_benchmarks.build_retrieval import resolve_embedding_model  # noqa: E402

path = resolve_embedding_model()
me = psutil.Process()
base = me.memory_info().rss

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(path, device="cpu")
model.encode(["warm the model with one sentence"], normalize_embeddings=True)

info = me.memory_info()
print(
    f"pid={os.getpid()} base={base / 2**20:.0f}MiB "
    f"rss={info.rss / 2**20:.0f}MiB delta={(info.rss - base) / 2**20:.0f}MiB",
    flush=True,
)
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 20)
