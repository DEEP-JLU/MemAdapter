"""Import-path setup for the two bundled, license-preserved vendor modules."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def prepare_lightmem_import_path() -> None:
    source = ROOT / "lightmem" / "vendor"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
