# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1: write the spanwise lift and pitching-moment distributions.

Runs VSPAERO in the OpenVSP env, then post-processes in this venv.
Output: wp4/out/wp41/lift_distribution.txt and moment_distribution.txt.

    python wp4/scripts/wp41_aero_loads.py [--skip-vspaero]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.wp41 import settings  # noqa: E402
from wp41 import aero_loads  # noqa: E402

logger = logging.getLogger("wp41_aero_loads")


def main(argv: list[str] | None = None) -> None:
    """Run VSPAERO (unless --skip-vspaero), then write the two distributions."""
    parser = argparse.ArgumentParser(description="WP4.1 lift and moment distributions")
    parser.add_argument(
        "--skip-vspaero", action="store_true",
        help="post-process the existing VSPAERO files only",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.skip_vspaero:
        settings.RUN_DIR.mkdir(parents=True, exist_ok=True)
        aero_loads.run_vspaero(settings.RUN_DIR, aero_loads.openvsp_python())
    lift_path, moment_path = aero_loads.post_process(settings.RUN_DIR, settings.OUT_DIR)
    logger.info("Done: %s, %s", lift_path, moment_path)


if __name__ == "__main__":
    main()
