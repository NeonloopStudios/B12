# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-19
"""OpenVSP model of the whole aircraft, for the 3-view figure.

Run from wp3/:  python -m scripts.vsp_model  (or as a file / Run button)

Two stages, because the OpenVSP Python API runs only in its own Python 3.11
conda env and the wp3 library needs Python 3.13+:

1. Here (wp3 env): load configs/geometry.yaml (sizes) and
   configs/vsp_model.yaml (positions, visual details), resolve every
   component (b12wp3.vsp.layout) and write results/vsp/model_inputs.json.
2. The OpenVSP env runs b12wp3/vsp/builder.py on that JSON and writes
   results/vsp/b12_aircraft.vsp3.

The OpenVSP interpreter is, in this order: --openvsp-python, the
OPENVSP_PYTHON environment variable, or the conda env "openvsp" next to the
running Python (<conda base>/envs/openvsp).

Also writes results/vsp/vsp_model_report.txt: the derived positions and the
placement checks. Open the .vsp3 in OpenVSP, select all geoms and click
"Shade" (the draw type is a GUI setting, not stored in the .vsp3).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    # run as a file (e.g. the editor's Run button) instead of
    # `python -m scripts.vsp_model`: make wp3/src/ importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from b12wp3 import vsp as vsp_package  # noqa: E402
from b12wp3.config.geometry import load_geometry  # noqa: E402
from b12wp3.config.paths import (  # noqa: E402
    OPENVSP_CONDA_ENV,
    VSP_MODEL_JSON,
    VSP_MODEL_REPORT_TXT,
    VSP_MODEL_VSP3,
    VSP_MODEL_YAML,
    WING_AIRFOIL_DAT,
    WP3_DIR,
)
from b12wp3.config.vsp_model import load_vsp_model  # noqa: E402
from b12wp3.config.yaml_io import ConfigError  # noqa: E402
from b12wp3.vsp.airfoil import read_airfoil_dat  # noqa: E402
from b12wp3.vsp.layout import LayoutReport, build_model  # noqa: E402

logger = logging.getLogger("scripts.vsp_model")

OPENVSP_PYTHON_ENV_VAR = "OPENVSP_PYTHON"
BUILDER_TIMEOUT_S = 300.0
# run as a file in the OpenVSP env; never imported here (no openvsp in this env)
BUILDER_PY = Path(vsp_package.__file__).resolve().parent / "builder.py"


def _openvsp_python(cli_value: Path | None) -> Path:
    """The OpenVSP env interpreter: CLI, environment variable, conda env."""
    if cli_value is not None:
        return cli_value
    from_env = os.environ.get(OPENVSP_PYTHON_ENV_VAR)
    if from_env:
        return Path(from_env)
    env_dir = Path(sys.prefix) / "envs" / OPENVSP_CONDA_ENV
    exe = "python.exe" if os.name == "nt" else "bin/python"
    return env_dir / exe


def _report(report: LayoutReport, vsp_yaml: Path) -> str:
    """Text report: derived positions, checks, files."""
    x_in, y_in, z_in = report.nacelle_inlet
    lines = [
        "OPENVSP 3-VIEW MODEL - derived positions",
        f"  inputs                     configs/geometry.yaml, {vsp_yaml.name}",
        "  axes                       x aft from the nose tip, z up from the "
        "cylinder axis",
        f"  wing MAC c/4 x             {report.wing_mac_c4_x_m:.3f} m",
        f"  c.g. x                     {report.cg_x_m:.3f} m "
        f"({100.0 * report.cg_mac_fraction:.1f} % MAC)",
        f"  fin root LE (x, z)         ({report.vertical_tail_root_le[0]:.3f}, "
        f"{report.vertical_tail_root_le[1]:.3f}) m",
        f"  horizontal tail root LE    ({report.horizontal_tail_root_le[0]:.3f}, "
        f"{report.horizontal_tail_root_le[1]:.3f}) m",
        f"  nacelle inlet (x, y, z)    ({x_in:.3f}, {y_in:.3f}, {z_in:.3f}) m",
        f"  nacelle ground clearance   {report.nacelle_ground_clearance_m:.3f} m",
        f"  nose strut length          {report.nose_strut_length_m:.3f} m",
        f"  main strut length          {report.main_strut_length_m:.3f} m, "
        f"rake {report.main_strut_rake_deg:.1f} deg, into the {report.main_gear_mount}",
        "",
        "CHECKS",
        *(
            [f"  WARNING: {w}" for w in report.warnings]
            if report.warnings
            else ["  all placement checks passed"]
        ),
        "",
        "FILES (relative to wp3/)",
        *(
            f"  {p.relative_to(WP3_DIR).as_posix()}"
            for p in (VSP_MODEL_JSON, VSP_MODEL_VSP3, VSP_MODEL_REPORT_TXT)
        ),
    ]
    return "\n".join(lines) + "\n"


def _run_builder(python: Path) -> None:
    """Run stage 2 in the OpenVSP env.

    Raises:
        FileNotFoundError: If the interpreter does not exist.
        subprocess.CalledProcessError: If the builder fails.
        subprocess.TimeoutExpired: If the builder hangs.
    """
    if not python.is_file():
        raise FileNotFoundError(
            f"OpenVSP Python not found at {python}; pass --openvsp-python or set "
            f"{OPENVSP_PYTHON_ENV_VAR}"
        )
    command = [str(python), str(BUILDER_PY), str(VSP_MODEL_JSON), str(VSP_MODEL_VSP3)]
    logger.info("stage 2: %s", " ".join(command))
    subprocess.run(command, check=True, timeout=BUILDER_TIMEOUT_S)


def main(argv: list[str] | None = None) -> int:
    """Resolve the model, write the JSON and the report, run the builder.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Process exit code: 0 on success, 1 on an error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=VSP_MODEL_YAML, help="VSP model YAML file"
    )
    parser.add_argument(
        "--openvsp-python", type=Path, default=None, help="OpenVSP env interpreter"
    )
    parser.add_argument(
        "--json-only", action="store_true", help="stop after stage 1 (no OpenVSP)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        geometry = load_geometry()
        inputs = load_vsp_model(args.config)
        airfoil = read_airfoil_dat(WING_AIRFOIL_DAT)
        model, report = build_model(geometry, inputs, airfoil)
        VSP_MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
        model.write_json(VSP_MODEL_JSON)
        text = _report(report, args.config)
        VSP_MODEL_REPORT_TXT.write_text(text, encoding="utf-8")
    except (ConfigError, ValueError, OSError) as error:
        logger.error("%s", error)
        return 1
    logger.info("stage 1: wrote %s\n%s", VSP_MODEL_JSON, text)
    if args.json_only:
        return 0
    try:
        _run_builder(_openvsp_python(args.openvsp_python))
    except (OSError, subprocess.SubprocessError) as error:
        logger.error("stage 2 failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
