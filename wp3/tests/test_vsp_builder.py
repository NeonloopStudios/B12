# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""Integration test of wp3/src/b12wp3/vsp/builder.py in the OpenVSP env.

The builder imports openvsp, which exists only in the OpenVSP conda env
(Python 3.11, no pytest), so the test runs it there as a subprocess.
Skipped when that interpreter is not found.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from b12wp3.config.geometry import AircraftGeometry
from b12wp3.config.paths import OPENVSP_CONDA_ENV, WING_AIRFOIL_DAT
from b12wp3.config.vsp_model import VspModelInputs
from b12wp3.vsp import layout
from b12wp3.vsp.airfoil import read_airfoil_dat

BUILDER_PY = Path(layout.__file__).resolve().parent / "builder.py"
_EXE = "python.exe" if os.name == "nt" else "bin/python"
OPENVSP_PYTHON = Path(
    os.environ.get(
        "OPENVSP_PYTHON", Path(sys.prefix) / "envs" / OPENVSP_CONDA_ENV / _EXE
    )
)


@pytest.mark.openvsp
@pytest.mark.skipif(not OPENVSP_PYTHON.is_file(), reason="OpenVSP env not found")
def test_builder_sample_aircraft_writes_vsp3_with_every_geom(
    tmp_path: Path, sample_geometry: AircraftGeometry, sample_vsp_inputs: VspModelInputs
) -> None:
    model, _ = layout.build_model(
        sample_geometry, sample_vsp_inputs, read_airfoil_dat(WING_AIRFOIL_DAT)
    )
    json_path, vsp3_path = tmp_path / "model.json", tmp_path / "model.vsp3"
    model.write_json(json_path)

    run = subprocess.run(
        [str(OPENVSP_PYTHON), str(BUILDER_PY), str(json_path), str(vsp3_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    text = vsp3_path.read_text(encoding="utf-8") if vsp3_path.is_file() else ""
    names = [
        "Fuselage",
        "Wing",
        "Vertical tail",
        "Horizontal tail",
        "Pylon",
        "Nacelle",
        "Fan face",
        "Spinner",
        "Exhaust plug",
        "Nose gear strut",
        "Main gear strut",
        "Tires",
    ]
    assert run.returncode == 0, run.stderr
    assert all(f"<Name>{name}</Name>" in text for name in names)
    assert "<Name>B12 nacelle</Name>" in text  # custom materials stored
