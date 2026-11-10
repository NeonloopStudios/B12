# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Airfoil coordinate files (.dat) for the OpenVSP wing section.

Same file formats as wp2 (b12wp2.wing.geometry.read_airfoil_dat), which wp3
does not import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MIN_POINTS: Final[int] = 10  # fewer points is not an airfoil
LEDNICER_COUNT_THRESHOLD: Final[float] = 1.5  # first "x" above this: point counts


def read_airfoil_dat(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a Selig or Lednicer airfoil file.

    Selig: one loop TE -> upper -> LE -> lower -> TE. Lednicer: a line with
    the point counts, then the upper and the lower surface, each LE -> TE.

    Args:
        path: .dat file, x/c and y/c in two columns; non-numeric lines
            (name, comments) are skipped.

    Returns:
        (upper, lower) as (N, 2) arrays of (x/c, y/c), both LE to TE.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file has fewer than MIN_POINTS points.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        logger.error("cannot read airfoil file %s: %s", path, error)
        raise
    points: list[tuple[float, float]] = []
    for line in text.splitlines():
        fields = line.replace(",", " ").split()
        if len(fields) < 2:
            continue
        try:
            points.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue  # name line or comment
    if points and points[0][0] > LEDNICER_COUNT_THRESHOLD:
        points = points[1:]
    if len(points) < MIN_POINTS:
        raise ValueError(f"{path}: {len(points)} points, need >= {MIN_POINTS}")
    arr = np.ascontiguousarray(points, dtype=np.float64)
    if arr[0, 0] > 0.5:
        i_le = int(np.argmin(arr[:, 0]))
        return arr[i_le::-1].copy(), arr[i_le:].copy()
    i_split = int(np.argmax(np.diff(arr[:, 0]) < -0.5)) + 1
    return arr[:i_split].copy(), arr[i_split:].copy()
