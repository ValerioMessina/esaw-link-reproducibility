"""ESAW adapters for faithful hierarchical text-classification baselines."""

from __future__ import annotations

import sys
from pathlib import Path


# The HGCLR authors use top-level imports such as ``from model.graph import``.
# Make the pinned upstream repository discoverable from every local entry
# point, not only from the remote launcher's shell-specific PYTHONPATH.
_hgclr_root = Path(__file__).resolve().parents[1] / "upstream" / "contrastive-htc"
if str(_hgclr_root) not in sys.path:
    sys.path.insert(0, str(_hgclr_root))
