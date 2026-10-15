# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""WP3 command-line entry points. Every module in here is a runnable step
(`python -m scripts.<name>` from wp3/); the analysis itself lives in the
b12wp3 package under wp3/src/.

The b12wp3 package is not pip-installed -- it is imported straight from the
working tree -- so wp3/src/ is put on sys.path here, when the first
`scripts.<name>` module is imported. Same mechanism as wp2/scripts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
