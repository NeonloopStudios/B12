# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-11-17
"""WP4.1 aerodynamic loads: pre- and post-processing of the VSPAERO run.

Runs in the repo venv. Only wp41/vspaero_runner.py needs the OpenVSP env;
main() starts it as a subprocess.

Chain of main():

1. Pre-processing: the planform, the freestream (ISA) and the moment
   reference point. vspaero_runner.py imports these functions, so the
   VSPAERO model and the post-processing use the same numbers.
2. VSPAERO: vspaero_runner.py runs in the OpenVSP env. It writes
   strips.csv and totals.csv to out/wp41/vspaero_run/.
3. Post-processing: read the strips of the right half-wing, move the strip
   pitching moment from the reference point to the local quarter chord,
   check the strips against the VSPAERO totals, then write
   out/wp41/lift_distribution.txt and out/wp41/moment_distribution.txt.

Run from the repo venv (or with the editor's Run button):

    python wp4/src/wp41/aero_loads.py
    python wp4/src/wp41/aero_loads.py --skip-vspaero    (post-processing only)

Compressibility: VSPAERO applies Prandtl-Glauert at the run Mach. No other
correction is applied. Drag is not computed.

This module must stay Python 3.11-compatible and use only numpy, because
vspaero_runner.py imports it in the OpenVSP env.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

if __package__ in (None, ""):
    # Run as a file (Run button): make wp4/ (configs) and wp4/src/ (wp41) importable.
    _WP4_DIR = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

from configs.aircraft import WING, WingGeometry  # noqa: E402
from configs.wp41 import settings  # noqa: E402
from configs.wp41.settings import FlightCondition  # noqa: E402

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float64]

# ICAO standard atmosphere (same constants as wp2/src/b12wp2/config/atmosphere.py)
ISA_T0_K: Final[float] = 288.15
ISA_P0_PA: Final[float] = 101_325.0
ISA_LAPSE_K_M: Final[float] = 0.0065
ISA_R_AIR: Final[float] = 287.05287
ISA_GAMMA: Final[float] = 1.4
ISA_G0: Final[float] = 9.80665
ISA_H_TROPOPAUSE_M: Final[float] = 11_000.0
ISA_H_MAX_M: Final[float] = 20_000.0

# Columns of the files that vspaero_runner.py writes (one source for both sides)
STRIP_COLUMNS: Final[tuple[str, ...]] = (
    "alpha", "y", "chord", "area", "cl", "cx", "cz", "cmy",
)
TOTAL_COLUMNS: Final[tuple[str, ...]] = ("alpha", "CL", "CMy")

RUNNER_PY: Final[Path] = Path(__file__).resolve().parent / "vspaero_runner.py"


# ============================================================
#  PRE-PROCESSING
# ============================================================


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
    """Moment reference point (x, z) of the VSPAERO run: the MAC quarter chord.

    Any point gives the same final result, because post-processing moves the
    moment to the local quarter chord. This point makes CMy of the whole
    wing the usual pitching moment about c/4 of the MAC.
    """
    x_ref = wing.root_le_x_m + pf.y_mac_m * pf.tan_sweep_le + 0.25 * pf.mac_m
    return x_ref, wing.root_z_m


def quarter_chord_points(
    wing: WingGeometry, pf: Planform, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """x and z of the local quarter chord at the stations y (VSPAERO frame).

    OpenVSP builds the panel in its own plane, then turns it by the dihedral
    G about the x axis. A point at panel span s is thus at y = s cos(G),
    z = s sin(G). With the root LE at the origin: x_w = s tan(Lambda_LE)
    + c(s)/4 and z_w = s sin(G). OpenVSP then turns the wing by the incidence
    i about the root LE (nose up positive) and moves it to (root_le_x,
    root_z). The twist turns each section about its quarter chord, so it does
    not move this point.

    Args:
        wing: Wing geometry.
        pf: Planform of the wing.
        y: Spanwise stations [m].

    Returns:
        (x, z) of the quarter chord at each station [m].
    """
    dihedral = math.radians(wing.dihedral_deg)
    s = y / math.cos(dihedral)  # span along the panel
    x_w = s * pf.tan_sweep_le + 0.25 * pf.chord_at(s)
    z_w = s * math.sin(dihedral)
    inc = math.radians(wing.incidence_deg)
    x = wing.root_le_x_m + x_w * math.cos(inc) + z_w * math.sin(inc)
    z = wing.root_z_m - x_w * math.sin(inc) + z_w * math.cos(inc)
    return x, z


def isa(altitude_m: float) -> tuple[float, float, float]:
    """ISA temperature [K], speed of sound [m/s] and density [kg/m^3].

    Args:
        altitude_m: Geopotential altitude, 0 to 20 km [m].

    Returns:
        (temperature, speed of sound, density).

    Raises:
        ValueError: If the altitude is outside 0 to 20 km.
    """
    if not 0.0 <= altitude_m <= ISA_H_MAX_M:
        raise ValueError(f"altitude_m must be in [0, {ISA_H_MAX_M}], got {altitude_m}")
    exponent = ISA_G0 / (ISA_LAPSE_K_M * ISA_R_AIR)
    t_tropopause = ISA_T0_K - ISA_LAPSE_K_M * ISA_H_TROPOPAUSE_M
    if altitude_m <= ISA_H_TROPOPAUSE_M:
        t = ISA_T0_K - ISA_LAPSE_K_M * altitude_m
        p = ISA_P0_PA * (t / ISA_T0_K) ** exponent
    else:
        t = t_tropopause
        p_tropopause = ISA_P0_PA * (t_tropopause / ISA_T0_K) ** exponent
        dh = altitude_m - ISA_H_TROPOPAUSE_M
        p = p_tropopause * math.exp(-ISA_G0 * dh / (ISA_R_AIR * t))
    rho = p / (ISA_R_AIR * t)
    a = math.sqrt(ISA_GAMMA * ISA_R_AIR * t)
    return t, a, rho


@dataclass(frozen=True)
class Freestream:
    """Freestream state of one flight condition.

    Attributes:
        mach: Mach number [-].
        altitude_m: Altitude [m].
        density: Density [kg/m^3].
        velocity: True airspeed [m/s].
        dynamic_pressure: 0.5 rho V^2 [Pa].
    """

    mach: float
    altitude_m: float
    density: float
    velocity: float
    dynamic_pressure: float


def freestream(condition: FlightCondition) -> Freestream:
    """Density, speed and dynamic pressure of a flight condition (ISA)."""
    _, a, rho = isa(condition.altitude_m)
    v = condition.mach * a
    return Freestream(condition.mach, condition.altitude_m, rho, v, 0.5 * rho * v**2)


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
            continue  # name line, blank line or comment
        points.append((x, y))
    if points and points[0][0] > 1.5:  # Lednicer header with the point counts
        points = points[1:]
    arr = np.array(points, dtype=np.float64)
    if len(arr) < 10:
        raise ValueError(f"Too few points ({len(arr)}) in airfoil file: {path}")

    if arr[0, 0] > 0.5:
        # Selig: TE -> upper -> LE -> lower -> TE
        i_le = int(np.argmin(arr[:, 0]))
        upper, lower = arr[i_le::-1], arr[i_le:]
    else:
        # Lednicer: LE -> upper -> TE, then LE -> lower -> TE
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


# ============================================================
#  POST-PROCESSING
# ============================================================


@dataclass(frozen=True)
class SpanTable:
    """Strip data of the right half-wing: one row per strip, one column per alpha.

    Attributes:
        alpha: Angles of attack [deg], shape (n_alpha,).
        y: Strip stations [m], shape (n_y,).
        chord: Strip chords [m], shape (n_y,).
        area: Strip areas [m^2], shape (n_y,).
        cl: Lift coefficient, shape (n_y, n_alpha).
        cx: x-force coefficient (x aft), shape (n_y, n_alpha).
        cz: z-force coefficient (z up), shape (n_y, n_alpha).
        cmy: Pitching moment about the reference point, nose up positive,
            on the local chord, shape (n_y, n_alpha).
    """

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


def span_table(strips: FloatArray) -> SpanTable:
    """Sort the strip rows (columns STRIP_COLUMNS) into a SpanTable.

    Keeps the right half-wing (y > 0) only.

    Raises:
        ValueError: If no strip has y > 0, or if the strips differ between
            the angles of attack.
    """
    right = strips[strips[:, 1] > 0.0]
    if right.shape[0] == 0:
        raise ValueError("No strips with y > 0")
    alphas = np.unique(right[:, 0])

    # one block of rows per alpha, each sorted root to tip
    blocks = []
    for alpha in alphas:
        rows = right[right[:, 0] == alpha]
        blocks.append(rows[np.argsort(rows[:, 1])])
    first = blocks[0]
    for alpha, rows in zip(alphas, blocks):
        if rows.shape != first.shape or not np.allclose(rows[:, 1], first[:, 1]):
            raise ValueError(f"The strips at alpha = {alpha} deg differ from the first alpha")

    stacked = np.stack(blocks, axis=-1)  # shape (n_y, n_columns, n_alpha)
    return SpanTable(
        alpha=alphas,
        y=first[:, 1],
        chord=first[:, 2],
        area=first[:, 3],
        cl=stacked[:, 4, :],
        cx=stacked[:, 5, :],
        cz=stacked[:, 6, :],
        cmy=stacked[:, 7, :],
    )


def quarter_chord_moment(
    cmy: FloatArray, cx: FloatArray, cz: FloatArray, chord: FloatArray,
    dx: FloatArray, dz: FloatArray,
) -> FloatArray:
    """Move a strip pitching moment from the reference point to the quarter chord.

    The moment about a point is M = M_c4 + (dz Fx - dx Fz), with (dx, dz) the
    position of the quarter chord relative to that point (x aft, z up, nose
    up positive). Divided by q S_strip c:

        cm_c4 = cmy - (dz / c) cx + (dx / c) cz

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

    Raises:
        ValueError: If the alphas differ, or if a difference exceeds the tolerance.
    """
    if not np.allclose(np.sort(totals[:, 0]), table.alpha):
        raise ValueError(f"Alphas differ: totals {totals[:, 0]}, strips {table.alpha}")
    totals = totals[np.argsort(totals[:, 0])]
    cl_strips = 2.0 * (table.cl * table.area[:, None]).sum(axis=0) / wing.area_m2
    weight = (table.area * table.chord)[:, None]
    cmy_strips = 2.0 * (table.cmy * weight).sum(axis=0) / (wing.area_m2 * pf.mac_m)
    for j, alpha in enumerate(table.alpha):
        logger.info(
            "alpha = %5.2f deg: CL strips %.5f / VSPAERO %.5f, CMy strips %.5f / VSPAERO %.5f",
            alpha, cl_strips[j], totals[j, 1], cmy_strips[j], totals[j, 2],
        )
    d_cl = np.abs(cl_strips - totals[:, 1])
    d_cmy = np.abs(cmy_strips - totals[:, 2])
    if d_cl.max() > tolerance or d_cmy.max() > tolerance:
        logger.error("Strip check failed: |dCL| = %s, |dCMy| = %s", d_cl, d_cmy)
        raise ValueError(
            f"Strip integral differs from the VSPAERO totals by more than {tolerance}: "
            f"|dCL| = {d_cl}, |dCMy| = {d_cmy}"
        )


def write_distribution(
    path: Path, title: str, symbol: str, table: SpanTable, values: FloatArray,
    totals: FloatArray, total_symbol: str, fs: Freestream, wing: WingGeometry, pf: Planform,
) -> None:
    """Write one spanwise distribution as a text file with a header.

    Columns: y [m], chord [m], then one column per angle of attack.

    Args:
        path: Output file.
        title: First header line.
        symbol: Name of the coefficient in the column names (cl, cm_c4).
        table: Strip data; gives y, chord and the alphas.
        values: Coefficient, shape (n_y, n_alpha).
        totals: Whole-wing coefficient per alpha, shape (n_alpha,).
        total_symbol: Name of the whole-wing coefficient (CL, CMy).
        fs: Freestream of the run.
        wing: Wing geometry.
        pf: Planform of the wing.
    """
    alpha_text = "  ".join(f"{a:10.4f}" for a in table.alpha)
    total_text = "  ".join(f"{t:10.6f}" for t in totals)
    names = ", ".join(f"{symbol}(alpha={a:g} deg)" for a in table.alpha)
    header = "\n".join(
        [
            title,
            "Right half-wing, VSPAERO VLM (inviscid), Prandtl-Glauert at the run Mach.",
            f"Mach = {fs.mach:.4f}, altitude = {fs.altitude_m:.1f} m, "
            f"V = {fs.velocity:.3f} m/s, rho = {fs.density:.5f} kg/m^3, "
            f"q = {fs.dynamic_pressure:.2f} Pa",
            f"S_ref = {wing.area_m2:.4f} m^2, b/2 = {pf.half_span_m:.4f} m, "
            f"MAC = {pf.mac_m:.4f} m, incidence = {wing.incidence_deg:.2f} deg "
            "(in the model: alpha is the freestream angle, not the wing angle)",
            f"alpha [deg]: {alpha_text}",
            f"{total_symbol}:{' ' * (12 - len(total_symbol))}{total_text}",
            f"columns: y [m], chord [m], {names}",
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
    fs = freestream(settings.FLIGHT)
    table = span_table(read_csv(run_dir / settings.STRIPS_FILE, STRIP_COLUMNS))
    totals = read_csv(run_dir / settings.TOTALS_FILE, TOTAL_COLUMNS)
    totals = totals[np.argsort(totals[:, 0])]
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
        table, table.cl, totals[:, 1], "CL", fs, WING, pf,
    )
    write_distribution(
        moment_path,
        "WP4.1 spanwise pitching moment coefficient about the local quarter chord "
        "cm_c4(y), nose up positive",
        "cm_c4", table, cm_c4, totals[:, 2], "CMy(c/4 MAC)", fs, WING, pf,
    )
    return lift_path, moment_path


def main(argv: list[str] | None = None) -> None:
    """Run VSPAERO (unless --skip-vspaero), then write the two distributions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--skip-vspaero", action="store_true",
        help="post-process the existing VSPAERO files only",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.skip_vspaero:
        settings.RUN_DIR.mkdir(parents=True, exist_ok=True)
        run_vspaero(settings.RUN_DIR, openvsp_python())
    post_process(settings.RUN_DIR, settings.OUT_DIR)


if __name__ == "__main__":
    main()
