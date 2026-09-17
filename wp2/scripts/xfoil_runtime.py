"""XFoil runtime wrapper: Windows DLL-loading fix, airfoil-file loading, and
a single-point-at-a-time alpha sweep that records convergence per point.

Every other wp2/scripts module that needs XFoil imports it through here, not
via `import xfoil` directly, so the DLL-directory fix below always runs
first. See wp2/README.md ("Trap 3") for why this is required: on Python
>= 3.8, Windows ctypes no longer searches PATH for a loaded DLL's
dependencies, so having the MinGW runtime on PATH is not enough -- it has
to be registered with `os.add_dll_directory()` before `import xfoil`.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MINGW_BIN = Path(r"C:\msys64\mingw64\bin")
_process_setup_done = False


def _ensure_process_setup() -> None:
    """One-time, process-wide setup that must happen before the first
    `import xfoil` / `XFoil()`:

    1. Register the MinGW runtime directory with the Windows DLL loader.
       `import xfoil` triggers `cdll.LoadLibrary` on libxfoil.dll, which
       depends on libgfortran/libgcc_s_seh/libwinpthread/libquadmath from
       this directory -- see wp2/README.md ("Trap 3") for why merely having
       it on PATH is not enough on Python >= 3.8.

    2. Fix a ctypes argtypes bug in the vendored xfoil-python package:
       `XFoil.__del__` calls `ctypes.windll.kernel32.FreeLibrary(handle)`
       with no argtypes declared, so ctypes assumes a 32-bit `c_int`. The
       DLL handle is a 64-bit pointer, which overflows that assumption
       (`OverflowError: int too long to convert`), FreeLibrary never runs,
       and the following `os.remove()` of the temp DLL copy then fails with
       `PermissionError` because the library handle is still open. Verified
       directly in this environment: every `XFoil()` instance leaked its
       library handle and its temp .dll file until FreeLibrary's argtypes
       were corrected to `c_void_p`. Declaring the correct prototype here
       fixes it for every XFoil instance in the process -- Stage 3 onward
       creates one XFoil() per airfoil/condition, so this is not optional.
    """
    global _process_setup_done
    if _process_setup_done:
        return
    if not _MINGW_BIN.is_dir():
        raise RuntimeError(
            f"MinGW runtime directory not found at {_MINGW_BIN} -- required "
            "for XFoil's compiled DLL to resolve its gfortran/gcc runtime "
            "dependencies at load time. See wp2/README.md."
        )
    os.add_dll_directory(str(_MINGW_BIN))
    ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    ctypes.windll.kernel32.FreeLibrary.restype = ctypes.c_int
    _process_setup_done = True


_ensure_process_setup()

from xfoil import XFoil  # noqa: E402  (must follow the DLL-directory fix above)
from xfoil.model import Airfoil  # noqa: E402


def _looks_like_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def load_airfoil_dat(path: Path) -> Airfoil:
    """Parse a Selig-format .dat airfoil coordinate file.

    Handles both the headerless files and the ones with a name header line
    as the first row (both forms exist in wp2/airfoils/, e.g.
    lockheed_c5a_bl758.dat has a header, WORTMANN_FX_62-K-131.dat does not).
    Coordinate order from the file is preserved as-is (XFoil's panel method
    needs a continuous loop: TE -> upper surface -> LE -> lower surface ->
    TE), not re-sorted or de-duplicated.
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path}: empty airfoil file")

    start = 0
    first_tokens = lines[0].split()
    if len(first_tokens) != 2 or not _looks_like_float(first_tokens[0]):
        start = 1  # first line is a name header, skip it

    xs: list[float] = []
    ys: list[float] = []
    for line_no, line in enumerate(lines[start:], start=start + 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected 'x y', got {line!r}")
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))

    if len(xs) < 10:
        raise ValueError(
            f"{path}: only {len(xs)} coordinate points parsed -- suspiciously "
            "few for an airfoil section, check the file format"
        )

    return Airfoil(np.array(xs, dtype=float), np.array(ys, dtype=float))


@contextlib.contextmanager
def xfoil_session(
    airfoil: Airfoil,
    *,
    mach: float,
    reynolds: float,
    ncrit: float,
    max_iter: int = 200,
    repanel: bool = True,
) -> Iterator[Any]:
    """Configure a fresh XFoil instance for one polar run.

    A new XFoil() is created per session rather than reused across
    airfoils/conditions: XFoil retains internal boundary-layer state
    between analyses, and starting each airfoil/Re/M combination from a
    clean instance is more predictable than relying on `reset_bls()` to
    fully clear it. `del xf` at the end triggers `XFoil.__del__`
    (frees the loaded DLL and removes its temp-file copy) deterministically
    via CPython refcounting rather than leaving it to GC timing.

    `repanel=True` (default) applies XFoil's standard cosine-bunched
    paneling (`repanel()`, 160 nodes) instead of analyzing the raw
    coordinate file directly -- standard XFoil practice, since imported
    .dat files often don't have paneling well-suited to the viscous solver
    (the files in wp2/airfoils/ range from 73 to 152 raw points).
    """
    xf = XFoil()
    try:
        xf.print = False
        xf.airfoil = airfoil
        if repanel:
            xf.repanel()
        xf.Re = reynolds
        xf.M = mach
        xf.n_crit = ncrit
        xf.max_iter = max_iter
        yield xf
    finally:
        del xf


def run_alpha_sweep(
    xf: Any,
    alphas: Iterable[float],
    *,
    stop_after_n_nonconverged: int = 8,
) -> pd.DataFrame:
    """Run one angle of attack at a time (not XFoil's built-in `aseq`), in
    the given order, recording each point's convergence individually.

    `aseq` runs the whole range internally and returns NaN rows for
    non-converged points with no per-point control; running one at a time
    lets the sweep (a) stop early once the boundary-layer solver is clearly
    past breakdown instead of grinding through a long non-converged tail,
    and (b) reset the boundary layer after a failed point so one bad solve
    doesn't poison the next one's initial guess. `alphas` should be given in
    a single monotonic direction so each point can warm-start from the
    previous converged solution, the same way XFoil is used interactively.

    Columns: alpha, cl, cd, cm, cp_min, converged, diverged, rms_bl, note.

    `cp_min` is the minimum pressure coefficient at that alpha -- XFoil's
    `.a()` call returns it directly, and it is exactly what Stage 5's
    Mach-critical sweep needs, so it is recorded here rather than
    recomputed later.

    `diverged` and `rms_bl` come from a patch to the vendored XFoil Fortran
    source (externals/xfoil-python/src/api.f90, `alfa_`) that exposes real
    internal solver state instead of XFoil's collapsed converged/not bool:
    `diverged` is True only if the boundary-layer Newton solve aborted on a
    NaN mid-iteration; `rms_bl` is the final RMS residual of that Newton
    system, so a non-converged point that simply ran out of iterations can
    be told apart from one that actually blew up, and "how close" a
    near-miss got is visible instead of a flat False. `note` is a plain-
    English label derived from those two fields -- not XFoil's own verbatim
    diagnostic text (that goes to stdout via Fortran WRITE statements that
    are not capturable through Python's os.dup2 on this compiled DLL,
    verified empirically -- see wp2/README.md).
    """
    rows: list[dict[str, float | bool | str]] = []
    consecutive_failures = 0
    for alpha in alphas:
        cl, cd, cm, cp_min, diverged, rms_bl = xf.a(alpha)
        converged = not np.isnan(cl)
        if converged:
            note = ""
        elif diverged:
            note = "boundary-layer Newton solve diverged (NaN) mid-iteration"
        else:
            note = f"did not converge within the iteration budget (final RMS residual {rms_bl:.3g})"
        rows.append(
            {
                "alpha": float(alpha),
                "cl": cl,
                "cd": cd,
                "cm": cm,
                "cp_min": cp_min,
                "converged": converged,
                "diverged": bool(diverged),
                "rms_bl": rms_bl,
                "note": note,
            }
        )
        if converged:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            xf.reset_bls()
        if consecutive_failures >= stop_after_n_nonconverged:
            break
    return pd.DataFrame(rows)


def cp_distribution(xf: Any) -> pd.DataFrame:
    """Full Cp(x) distribution from the airfoil's last converged analysis
    (thin wrapper around XFoil.get_cp_distribution for Stage 6 plotting).
    """
    x, y, cp = xf.get_cp_distribution()
    return pd.DataFrame({"x": x, "y": y, "cp": cp})
