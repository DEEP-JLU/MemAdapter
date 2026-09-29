"""Self-contained retrieval adapters for MemoryBank and LightMem.

The adapters are intentionally isolated from benchmark data and generated
artifacts.  They accept dialogue text and write their ephemeral stores only to
the caller-supplied output directory.
"""

from .base import BaselineContext, BaselineEvalConfig
from .lightmem_adapter import METHOD as LIGHTMEM, build_context as build_lightmem_context
from .memorybank_adapter import METHOD as MEMORYBANK, build_context as build_memorybank_context

BUILDERS = {
    MEMORYBANK: build_memorybank_context,
    LIGHTMEM: build_lightmem_context,
}


def build_context(method: str, *args, **kwargs) -> BaselineContext:
    try:
        return BUILDERS[method](*args, **kwargs)
    except KeyError as exc:
        raise ValueError(f"Unsupported bundled memory system: {method!r}") from exc
