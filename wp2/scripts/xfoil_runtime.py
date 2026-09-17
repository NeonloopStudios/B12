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


def _split_into_coordinate_blocks(
    lines: list[str], path: Path, line_offset: int
) -> list[tuple[list[float], list[float]]]:
    """Split lines into blank-line-separated (xs, ys) coordinate blocks."""
    blocks: list[tuple[list[float], list[float]]] = []
    xs: list[float] = []
    ys: list[float] = []
    for line_no, line in enumerate(lines, start=line_offset):
        stripped = line.strip()
        if not stripped:
            if xs:
                blocks.append((xs, ys))
                xs, ys = [], []
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected 'x y', got {line!r}")
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))
    if xs:
        blocks.append((xs, ys))
    return blocks


def _merge_lednicer_blocks(
    block_a: tuple[list[float], list[float]],
    block_b: tuple[list[float], list[float]],
    path: Path,
) -> tuple[list[float], list[float]]:
    """Merge two LE-to-TE coordinate runs (Lednicer-style: one surface per
    block, blank-line separated, each starting near x=0 and ending near
    x=1) into the single continuous TE -> ... -> LE -> ... -> TE loop
    XFoil's panel method needs. Which block is geometrically "upper" vs
    "lower" doesn't matter for this -- only that each is one LE-to-TE run;
    reversing the first and appending the second (minus its duplicate
    leading-edge point) always produces a consistent continuous contour.
    """
    for label, (xs, _ys) in (("first", block_a), ("second", block_b)):
        if not (xs[0] < 0.05 and xs[-1] > 0.95):
            raise ValueError(
                f"{path}: {label} blank-line-separated block doesn't look "
                f"like a single LE-to-TE surface run (x from {xs[0]} to "
                f"{xs[-1]}) -- cannot assume Lednicer upper/lower format"
            )
    xs_a, ys_a = block_a
    xs_b, ys_b = block_b
    xs = list(reversed(xs_a)) + xs_b[1:]
    ys = list(reversed(ys_a)) + ys_b[1:]
    return xs, ys


def load_airfoil_dat(path: Path) -> Airfoil:
    """Parse a .dat airfoil coordinate file, either format:

    - Selig: one continuous loop, TE -> upper surface -> LE -> lower
      surface -> TE (e.g. NACA_25112.dat, lockheed_c5a_bl758.dat).
    - Lednicer: two blank-line-separated blocks, each one LE-to-TE surface
      run (e.g. WORTMANN_FX_62-K-131.dat) -- merged into a single
      continuous loop via _merge_lednicer_blocks, since XFoil's panel
      method needs one continuous contour, not two separate runs.
      Concatenating the two blocks as-is (an earlier version of this
      function did exactly that) produces a self-intersecting, invalid
      panel geometry that silently fails to converge at every angle of
      attack -- verified directly: WORTMANN_FX_62-K-131 didn't converge at
      a single point, including easy, safely-subsonic cases, until this
      was fixed.

    Also handles a name header line as the first row, before either format
    (e.g. lockheed_c5a_bl758.dat has one, WORTMANN_FX_62-K-131.dat doesn't).
    """
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path}: empty airfoil file")

    start = 0
    first_tokens = lines[0].split()
    if len(first_tokens) != 2 or not _looks_like_float(first_tokens[0]):
        start = 1  # first line is a name header, skip it

    blocks = _split_into_coordinate_blocks(lines[start:], path, line_offset=start + 1)

    if len(blocks) == 1:
        xs, ys = blocks[0]
    elif len(blocks) == 2:
        xs, ys = _merge_lednicer_blocks(blocks[0], blocks[1], path)
    else:
        raise ValueError(
            f"{path}: found {len(blocks)} blank-line-separated coordinate "
            "blocks, expected 1 (Selig) or 2 (Lednicer upper/lower)"
        )

    if len(xs) < 10:
        raise ValueError(
            f"{path}: only {len(xs)} coordinate points parsed -- suspiciously "
            "few for an airfoil section, check the file format"
        )
    if not (xs[0] > 0.95 and xs[-1] > 0.95 and min(xs) < 0.05):
        raise ValueError(
            f"{path}: assembled coordinates don't form a TE->...->LE->...->TE "
            f"loop (x starts at {xs[0]}, ends at {xs[-1]}, min {min(xs)}) -- "
            "panel geometry would be invalid"
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


def run_two_leg_polar(
    airfoil: Airfoil,
    *,
    mach: float,
    reynolds: float,
    ncrit: float,
    alpha_low_deg: float,
    alpha_high_deg: float,
    alpha_step_deg: float = 0.25,
    stop_after_n_nonconverged: int = 8,
) -> pd.DataFrame:
    """Build a full polar as two independent warm-started sweeps from
    alpha=0, one up to alpha_high_deg, one down to alpha_low_deg, merged.

    Cold-starting XFoil's BL solve directly at a harsh angle reliably
    diverges for several consecutive points before reset_bls() recovers
    it -- verified directly (Stage 3, cruise polars) -- and at a fine
    alpha_step_deg that can burn the whole non-convergence budget before
    ever reaching a point that would actually converge, so a naive single-
    direction sweep from one extreme can come back completely empty.
    Warm-starting from a known-good 0 deg point and marching outward in
    each direction avoids the cold start; the eventual non-convergence
    each leg hits is the real stall/breakdown, not an artifact.

    Each leg gets its own fresh XFoil session (not one session run twice)
    so a bad excursion at one extreme (e.g. deep stall on the upper leg)
    can't leave corrupted BL state that taints the other leg's results.
    The two legs' alpha=0 point is identical by construction; the lower
    leg's copy is dropped before merging.
    """
    upper_alphas = np.arange(0.0, alpha_high_deg + alpha_step_deg, alpha_step_deg)
    lower_alphas = np.arange(0.0, alpha_low_deg - alpha_step_deg, -alpha_step_deg)

    with xfoil_session(airfoil, mach=mach, reynolds=reynolds, ncrit=ncrit) as xf:
        upper = run_alpha_sweep(xf, upper_alphas, stop_after_n_nonconverged=stop_after_n_nonconverged)

    with xfoil_session(airfoil, mach=mach, reynolds=reynolds, ncrit=ncrit) as xf:
        lower = run_alpha_sweep(xf, lower_alphas, stop_after_n_nonconverged=stop_after_n_nonconverged)
    lower = lower[lower["alpha"] != 0.0]

    polar = pd.concat([lower, upper], ignore_index=True)
    return polar.sort_values("alpha").reset_index(drop=True)


def cp_distribution(xf: Any) -> pd.DataFrame:
    """Full Cp(x) distribution from the airfoil's last converged analysis
    (thin wrapper around XFoil.get_cp_distribution for Stage 6 plotting).
    """
    x, y, cp = xf.get_cp_distribution()
    return pd.DataFrame({"x": x, "y": y, "cp": cp})
