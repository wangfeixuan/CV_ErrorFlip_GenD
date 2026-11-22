from __future__ import annotations

from pathlib import Path

EXPECTED_MODEL_SHA256 = (
    "d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833"
)
EXPECTED_MODEL_SIZE = 1_215_928_160
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
