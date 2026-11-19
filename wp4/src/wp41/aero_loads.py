# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1 aerodynamic loads: pre- and post-processing of the VSPAERO run.

Pre-processing: planform, moment reference point, airfoil reader.
vspaero_runner.py imports these, so this module must stay Python 3.11 and
numpy only.
Post-processing: move the strip moment to the local quarter chord, check
the strips against the VSPAERO totals, and write the two distribution files.
Each file has one column per (Mach, alpha) case of the VSPAERO grid.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from configs.aircraft import WING, WingGeometry
from configs.wp41 import settings

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]

STRIP_COLUMNS: Final[tuple[str, ...]] = (
    "mach", "alpha", "y", "chord", "area", "cl", "cx", "cz", "cmy",
)
TOTAL_COLUMNS: Final[tuple[str, ...]] = ("mach", "alpha", "CL", "CMy")

RUNNER_PY: Final[Path] = Path(__file__).resolve().parent / "vspaero_runner.py"


# --- pre-processing ---


@dataclass(frozen=True)
class Planform:
    """Quantities that follow from a straight-tapered planform.

    Attributes:
        span_m: Span b [m].
        half_span_m: b / 2 [m].
        root_chord_m: Root chord [m].
        tip_chord_m: Tip chord [m].
        mac_m: Mean aerodynamic chord [m].
        y_mac_m: Spanwise station of the MAC [m].
        tan_sweep_le: Tangent of the leading-edge sweep [-].
    """

    span_m: float
    half_span_m: float
    root_chord_m: float
    tip_chord_m: float
    mac_m: float
    y_mac_m: float
    tan_sweep_le: float

    def chord_at(self, s: FloatArray) -> FloatArray:
        """Local chord [m] at the panel-span stations s [m] (s = y without dihedral)."""
        slope = (self.tip_chord_m - self.root_chord_m) / self.half_span_m
        return self.root_chord_m + slope * s


def planform(wing: WingGeometry) -> Planform:
    """Span, chords, MAC and leading-edge sweep of the wing.

    Args:
        wing: Wing geometry.

    Returns:
        The derived planform quantities.
    """
    span = math.sqrt(wing.area_m2 * wing.aspect_ratio)
    half_span = span / 2.0
    lam = wing.taper
    c_root = 2.0 * wing.area_m2 / (span * (1.0 + lam))
    c_tip = lam * c_root
    mac = 2.0 / 3.0 * c_root * (1.0 + lam + lam**2) / (1.0 + lam)
    y_mac = span / 6.0 * (1.0 + 2.0 * lam) / (1.0 + lam)
    tan_le = math.tan(math.radians(wing.sweep_c4_deg)) + 0.25 * (c_root - c_tip) / half_span
    return Planform(span, half_span, c_root, c_tip, mac, y_mac, tan_le)


def moment_reference(wing: WingGeometry, pf: Planform) -> tuple[float, float]:
    """Moment reference point (x, z) of the VSPAERO run: the MAC quarter chord."""
    x_ref = wing.root_le_x_m + pf.y_mac_m * pf.tan_sweep_le + 0.25 * pf.mac_m
    return x_ref, wing.root_z_m


def quarter_chord_points(
    wing: WingGeometry, pf: Planform, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """x and z of the local quarter chord at the stations y (VSPAERO frame).

    The panel span s = y / cos(dihedral). The incidence turns the wing about
    the root LE, nose up positive. The twist does not move the quarter chord.

    Args:
        wing: Wing geometry.
        pf: Planform of the wing.
        y: Spanwise stations [m].

    Returns:
        (x, z) of the quarter chord at each station [m].
    """
    dihedral = math.radians(wing.dihedral_deg)
    s = y / math.cos(dihedral)
    x_w = s * pf.tan_sweep_le + 0.25 * pf.chord_at(s)
    z_w = s * math.sin(dihedral)
    inc = math.radians(wing.incidence_deg)
    x = wing.root_le_x_m + x_w * math.cos(inc) + z_w * math.sin(inc)
    z = wing.root_z_m - x_w * math.sin(inc) + z_w * math.cos(inc)
    return x, z


def read_airfoil_dat(path: Path) -> tuple[FloatArray, FloatArray]:
    """Read a Selig or Lednicer airfoil .dat file (copied from wp2).

    Args:
        path: Airfoil file.

    Returns:
        (upper, lower) as (N, 2) arrays of (x, y). Both run from the leading
        to the trailing edge, with unit chord and the leading edge at (0, 0).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has fewer than 10 points.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Airfoil file not found: {path}")
    points = []
    for line in path.read_text().splitlines():
        try:
            x, y = (float(v) for v in line.replace(",", " ").split()[:2])
        except ValueError:
            continue
        points.append((x, y))
    if points and points[0][0] > 1.5:  # Lednicer point counts
        points = points[1:]
    arr = np.array(points, dtype=np.float64)
    if len(arr) < 10:
        raise ValueError(f"Too few points ({len(arr)}) in airfoil file: {path}")

    if arr[0, 0] > 0.5:  # Selig: TE -> upper -> LE -> lower -> TE
        i_le = int(np.argmin(arr[:, 0]))
        upper, lower = arr[i_le::-1], arr[i_le:]
    else:  # Lednicer: LE -> upper -> TE, LE -> lower -> TE
        i_split = int(np.argmax(np.diff(arr[:, 0]) < -0.5)) + 1
        upper, lower = arr[:i_split], arr[i_split:]

    le = upper[0].copy()
    chord = max(upper[:, 0].max(), lower[:, 0].max()) - le[0]
    return (upper - le) / chord, (lower - le) / chord


def openvsp_python() -> Path:
    """Interpreter of the OpenVSP env: environment variable, then the default.

    Raises:
        FileNotFoundError: If the interpreter does not exist.
    """
    from_env = os.environ.get(settings.OPENVSP_PYTHON_ENV_VAR)
    exe = Path(from_env) if from_env else settings.OPENVSP_PYTHON_DEFAULT
    if not exe.is_file():
        raise FileNotFoundError(
            f"OpenVSP Python not found: {exe}. Set {settings.OPENVSP_PYTHON_ENV_VAR} "
            "to the python.exe of the OpenVSP env."
        )
    return exe


def run_vspaero(run_dir: Path, python_exe: Path) -> None:
    """Run vspaero_runner.py in the OpenVSP env. Its output goes to the console.

    Args:
        run_dir: Folder for the VSPAERO files, strips.csv and totals.csv.
        python_exe: Interpreter of the OpenVSP env.

    Raises:
        RuntimeError: If the runner fails or takes longer than the time limit.
    """
    logger.info("Starting VSPAERO in the OpenVSP env: %s", python_exe)
    try:
        result = subprocess.run(
            [str(python_exe), str(RUNNER_PY), str(run_dir)],
            timeout=settings.VSPAERO.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        logger.error("VSPAERO took longer than %.0f s", settings.VSPAERO.timeout_s)
        raise RuntimeError("VSPAERO run timed out") from error
    except OSError as error:
        logger.error("Cannot start %s: %s", python_exe, error)
        raise RuntimeError(f"Cannot start the OpenVSP Python {python_exe}") from error
    if result.returncode != 0:
        logger.error("vspaero_runner.py ended with exit code %d", result.returncode)
        raise RuntimeError(f"vspaero_runner.py failed (exit code {result.returncode})")


# --- post-processing ---


@dataclass(frozen=True)
class SpanTable:
    """Strip data of the right half-wing: one row per strip, one column per case.

    A case is one (Mach, alpha) pair of the VSPAERO grid. The cases are sorted
    by Mach, then by alpha.

    Attributes:
        mach: Mach number of each case, shape (n_case,).
        alpha: Angle of attack of each case [deg], shape (n_case,).
        y: Strip stations [m], shape (n_y,).
        chord: Strip chords [m], shape (n_y,).
        area: Strip areas [m^2], shape (n_y,).
        cl: Lift coefficient, shape (n_y, n_case).
        cx: x-force coefficient (x aft), shape (n_y, n_case).
        cz: z-force coefficient (z up), shape (n_y, n_case).
        cmy: Pitching moment about the reference point, nose up positive,
            on the local chord, shape (n_y, n_case).
    """

    mach: FloatArray
    alpha: FloatArray
    y: FloatArray
    chord: FloatArray
    area: FloatArray
    cl: FloatArray
    cx: FloatArray
    cz: FloatArray
    cmy: FloatArray


def read_csv(path: Path, columns: tuple[str, ...]) -> FloatArray:
    """Read a numeric CSV file with one header line and the given columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header or the shape does not match the columns.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path} (run VSPAERO first)")
    header = path.read_text().splitlines()[0].lstrip("# ").split(",")
    if tuple(header) != columns:
        raise ValueError(f"{path}: header {header}, expected {list(columns)}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
    if data.shape[0] == 0 or data.shape[1] != len(columns):
        raise ValueError(f"{path}: shape {data.shape}, expected (>0, {len(columns)})")
    return data


def sort_cases(rows: FloatArray) -> FloatArray:
    """Sort rows by Mach (column 0), then by alpha (column 1)."""
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def span_table(strips: FloatArray) -> SpanTable:
    """Sort the strip rows (columns STRIP_COLUMNS) into a SpanTable.

    Keeps the right half-wing (y > 0) only.

    Raises:
        ValueError: If no strip has y > 0, or if the strips differ between
            the cases.
    """
    right = strips[strips[:, 2] > 0.0]
    if right.shape[0] == 0:
        raise ValueError("No strips with y > 0")
    cases = np.unique(right[:, :2], axis=0)  # sorted by Mach, then alpha

    blocks = []
    for mach, alpha in cases:
        rows = right[(right[:, 0] == mach) & (right[:, 1] == alpha)]
        blocks.append(rows[np.argsort(rows[:, 2])])
    first = blocks[0]
    for (mach, alpha), rows in zip(cases, blocks):
        if rows.shape != first.shape or not np.allclose(rows[:, 2], first[:, 2]):
            raise ValueError(
                f"The strips at Mach {mach}, alpha {alpha} deg differ from the first case"
            )

    stacked = np.stack(blocks, axis=-1)  # (n_y, n_columns, n_case)
    return SpanTable(
        mach=cases[:, 0].copy(),
        alpha=cases[:, 1].copy(),
        y=first[:, 2],
        chord=first[:, 3],
        area=first[:, 4],
        cl=stacked[:, 5, :],
        cx=stacked[:, 6, :],
        cz=stacked[:, 7, :],
        cmy=stacked[:, 8, :],
    )


def quarter_chord_moment(
    cmy: FloatArray, cx: FloatArray, cz: FloatArray, chord: FloatArray,
    dx: FloatArray, dz: FloatArray,
) -> FloatArray:
    """Move a strip pitching moment from the reference point to the quarter chord.

    M_ref = M_c4 + dz Fx - dx Fz (x aft, z up, nose up positive), so
    cm_c4 = cmy - (dz / c) cx + (dx / c) cz.

    Args:
        cmy: Moment coefficient about the reference point.
        cx: x-force coefficient.
        cz: z-force coefficient.
        chord: Local chord [m].
        dx: x_c4 - x_ref [m].
        dz: z_c4 - z_ref [m].

    Returns:
        Moment coefficient about the local quarter chord.
    """
    return cmy - dz / chord * cx + dx / chord * cz


def check_totals(
    table: SpanTable, totals: FloatArray, wing: WingGeometry, pf: Planform, tolerance: float
) -> None:
    """Compare the strip integral of CL and CMy with the VSPAERO totals.

    Both halves: CL = 2 sum(cl dA) / S, CMy = 2 sum(cmy dA c) / (S MAC).

    Args:
        table: Strip data.
        totals: VSPAERO totals (columns TOTAL_COLUMNS), sorted with sort_cases.
        wing: Wing geometry.
        pf: Planform of the wing.
        tolerance: Largest allowed absolute difference.

    Raises:
        ValueError: If the cases differ, or if a difference exceeds the tolerance.
    """
    if totals.shape[0] != table.mach.size or not (
        np.allclose(totals[:, 0], table.mach) and np.allclose(totals[:, 1], table.alpha)
    ):
        raise ValueError(
            f"Cases differ: totals {totals[:, :2].tolist()}, "
            f"strips {np.column_stack([table.mach, table.alpha]).tolist()}"
        )
    cl_strips = 2.0 * (table.cl * table.area[:, None]).sum(axis=0) / wing.area_m2
    weight = (table.area * table.chord)[:, None]
    cmy_strips = 2.0 * (table.cmy * weight).sum(axis=0) / (wing.area_m2 * pf.mac_m)
    for j in range(table.mach.size):
        logger.info(
            "M = %.3f, alpha = %5.2f deg: CL strips %.5f / VSPAERO %.5f, "
            "CMy strips %.5f / VSPAERO %.5f",
            table.mach[j], table.alpha[j], cl_strips[j], totals[j, 2],
            cmy_strips[j], totals[j, 3],
        )
    d_cl = np.abs(cl_strips - totals[:, 2])
    d_cmy = np.abs(cmy_strips - totals[:, 3])
    if d_cl.max() > tolerance or d_cmy.max() > tolerance:
        logger.error("Strip check failed: |dCL| = %s, |dCMy| = %s", d_cl, d_cmy)
        raise ValueError(
            f"Strip integral differs from the VSPAERO totals by more than {tolerance}: "
            f"|dCL| = {d_cl}, |dCMy| = {d_cmy}"
        )


def _header_line(key: str, values: FloatArray, fmt: str) -> str:
    """One machine-readable header line: 'key: v1 v2 ...'."""
    return f"{key}: " + " ".join(format(v, fmt) for v in values)


def write_distribution(
    path: Path, title: str, symbol: str, table: SpanTable, values: FloatArray,
    totals: FloatArray, total_key: str, wing: WingGeometry, pf: Planform,
) -> None:
    """Write one distribution: y [m], chord [m], one column per (Mach, alpha) case.

    The header holds 'key: values' lines (keys in configs.wp41.settings) that
    wp41.lift_distribution reads back.

    Args:
        path: Output file.
        title: First header line.
        symbol: Coefficient name in the column description (cl, cm_c4).
        table: Strip data (y, chord, cases).
        values: Coefficient, shape (n_y, n_case).
        totals: Whole-wing coefficient per case, shape (n_case,).
        total_key: Header key of the whole-wing coefficient.
        wing: Wing geometry.
        pf: Planform of the wing.
    """
    header = "\n".join(
        [
            title,
            "Right half-wing, VSPAERO VLM (inviscid, fixed wake), Prandtl-Glauert per Mach.",
            "alpha is the freestream angle; the wing incidence is in the model.",
            f"{settings.KEY_S_REF}: {wing.area_m2:.6f}",
            f"{settings.KEY_HALF_SPAN}: {pf.half_span_m:.6f}",
            f"{settings.KEY_MAC}: {pf.mac_m:.6f}",
            f"{settings.KEY_INCIDENCE}: {wing.incidence_deg:.6f}",
            _header_line(settings.KEY_MACH, table.mach, ".4f"),
            _header_line(settings.KEY_ALPHA, table.alpha, ".4f"),
            _header_line(total_key, totals, ".6f"),
            f"columns: y [m], chord [m], then {symbol} for each "
            f"({settings.KEY_MACH}, {settings.KEY_ALPHA}) pair above",
        ]
    )
    data = np.column_stack([table.y, table.chord, values])
    np.savetxt(path, data, fmt="%12.6f", header=header)
    logger.info("Wrote %s", path)


def post_process(run_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Read the VSPAERO strips and write the lift and moment distributions.

    Args:
        run_dir: Folder with strips.csv and totals.csv.
        out_dir: Folder for the two text files.

    Returns:
        Paths of the lift and the moment distribution files.
    """
    pf = planform(WING)
    table = span_table(read_csv(run_dir / settings.STRIPS_FILE, STRIP_COLUMNS))
    totals = sort_cases(read_csv(run_dir / settings.TOTALS_FILE, TOTAL_COLUMNS))
    check_totals(table, totals, WING, pf, settings.CHECK_TOLERANCE)

    x_ref, z_ref = moment_reference(WING, pf)
    x_c4, z_c4 = quarter_chord_points(WING, pf, table.y)
    cm_c4 = quarter_chord_moment(
        table.cmy, table.cx, table.cz, table.chord[:, None],
        (x_c4 - x_ref)[:, None], (z_c4 - z_ref)[:, None],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    lift_path = out_dir / settings.LIFT_FILE
    moment_path = out_dir / settings.MOMENT_FILE
    write_distribution(
        lift_path, "WP4.1 spanwise lift coefficient distribution cl(y)", "cl",
        table, table.cl, totals[:, 2], settings.KEY_CL, WING, pf,
    )
    write_distribution(
        moment_path,
        "WP4.1 spanwise pitching moment coefficient about the local quarter chord "
        "cm_c4(y), nose up positive",
        "cm_c4", table, cm_c4, totals[:, 3], settings.KEY_CMY, WING, pf,
    )
    return lift_path, moment_path
