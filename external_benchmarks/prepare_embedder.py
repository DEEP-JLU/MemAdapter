"""Materialize a loadable local bge-m3 out of the HuggingFace cache.

Why this exists. ``BAAI/bge-m3`` ships ``pytorch_model.bin`` and no safetensors,
and ``transformers`` >= 4.57 refuses to ``torch.load`` a ``.bin`` when torch is
below 2.6, citing CVE-2025-32434. This environment has torch 2.4.1+cu121 -- which
is also the version that produced the frozen ``retrieval/`` artifacts. Bumping
torch to satisfy the guard would change the environment out from under the
existing results and pull torchvision/torchaudio along with it, for no functional
gain. Converting the checkpoint once is the smaller change: safetensors is the
format upstream would have shipped, it is what the loader asks for, and torch is
left exactly as it was.

The conversion is a format change only -- same tensors, same dtype, same values,
no quantization and no re-derivation. Afterwards the snapshot directory is a
complete model directory, so it can be named directly (no hub round-trip) and the
recorded ``MEMORY_EMBEDDING_MODEL`` matches what the earlier MemSyco runs
recorded: the snapshot path for the mem0 layers, the repo id for A-MEM.

Idempotent: returns immediately once a safetensors file is present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .build_retrieval import DEFAULT_EMBEDDING_REPO

#: A snapshot counts as complete when either a single-file or a sharded
#: safetensors index is present.
_WEIGHT_MARKERS = ("model.safetensors", "model.safetensors.index.json")


def _snapshot_dir(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, local_files_only=True))


def has_weights(directory: Path) -> bool:
    return any((directory / name).is_file() for name in _WEIGHT_MARKERS)


def ensure_local_weights(repo: str = DEFAULT_EMBEDDING_REPO) -> Path:
    """Return a snapshot directory holding loadable weights, converting if needed.

    Raises rather than silently returning a directory the loader cannot use: a
    snapshot with only config and tokenizer files resolves cleanly through the
    hub and then fails deep inside ``SentenceTransformer``, which is a much worse
    error message than a failure here.
    """
    snapshot = _snapshot_dir(repo)
    if has_weights(snapshot):
        return snapshot

    source = snapshot / "pytorch_model.bin"
    if not source.is_file():
        raise RuntimeError(
            f"No loadable weights in {snapshot}. Expected {' or '.join(_WEIGHT_MARKERS)} "
            f"(or pytorch_model.bin to convert). Re-download the model, e.g. "
            f"snapshot_download({repo!r})."
        )

    import torch
    from safetensors.torch import save_file

    print(f"[embedder] converting {source.name} -> model.safetensors in {snapshot}", flush=True)
    state = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    # save_file rejects non-contiguous tensors, and a converted checkpoint can
    # carry views. cheap next to the 2.3 GB it is about to write.
    state = {name: tensor.contiguous() for name, tensor in state.items()}

    # Write beside the target and move into place. A truncated model.safetensors
    # would satisfy has_weights() on the next call, locking in a corrupt file that
    # only fails much later, at load time.
    partial = snapshot / "model.safetensors.partial"
    save_file(state, str(partial), metadata={"format": "pt", "source": source.name})
    partial.replace(snapshot / "model.safetensors")

    tensors = sum(1 for _ in state)
    print(
        f"[embedder] wrote model.safetensors ({tensors} tensors) to {snapshot}",
        flush=True,
    )
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_EMBEDDING_REPO)
    args = parser.parse_args()

    snapshot = ensure_local_weights(args.repo)
    print(f"[embedder] usable snapshot: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
