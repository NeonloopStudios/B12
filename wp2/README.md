<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 Kajetan R. Gulaj
Created: 2026-09-17
-->

# WP2 — Airfoil Selection

XFoil-based screening of candidate 2D airfoil sections, and the 3D wing and
high-lift work built on the section that wins. This file covers the layout,
how to run each step, and how to get XFoil and OpenVSP working.

## Layout

```
wp2/
  airfoils/     candidate .dat coordinate files (the set will change; any .dat
                dropped here is picked up automatically by the scripts)
  scripts/      the runnable steps, one file per step -- nothing else
  src/b12wp2/   the analysis library every step calls
    config/     everything that is an input: paths, atmosphere, mission,
                wing, hld, solvers, scoring
    common/     airfoil geometry, compressibility relations
    xfoil/      2D: the XFoil runtime, polar post-processing, M_crit,
                lift-curve slope
    wing/       3D: planform geometry, VSPAERO, viscous/wave corrections
    hld/        high-lift device sizing and comparison
    plots/      the figures, one module per stage
    scoring/    the weighted scorecard
  tests/        pytest suite over src/b12wp2/
  results/      generated CSVs and plots, one directory per stage:
    section/    2D XFoil work -- data/ and plots/
    wing/       3D VSPAERO work, one subdirectory per airfoil
    hld/        high-lift device comparison
    scorecard/  scorecard.csv, scorecard_detail.csv

externals/
  xfoil-python/ vendored, tracked-in-repo source for XFoil's Python bindings
                (see below) -- not gitignored, so our patches to it have real
                commit history
```

## Running the analysis

Every runnable step is one file in `scripts/`, started from `wp2/` as a
module -- `python -m scripts.<name>`, not `python scripts/<name>.py`, since
the package is what puts `src/` on the path. (The two OpenVSP steps are the
exception: they add it themselves, so the editor's Run button works on
them.)

| Command | Does | Writes |
|---|---|---|
| `python -m scripts.run_all` | the whole 2D pipeline, in order | everything below except the 3D results |
| `python -m scripts.run_cruise_polars` | XFoil cruise polars, every airfoil | `results/section/data/*_cruise_polar.csv` |
| `python -m scripts.run_landing_polars` | XFoil landing polars | `results/section/data/*_landing_polar.csv` |
| `python -m scripts.mcrit_sweep` | M_crit + Korn M_dd | `results/section/data/*_mcrit.csv`, `*_baseline_cp.csv`, `mcrit_summary.csv` |
| `python -m scripts.build_scorecard` | the weighted scorecard | `results/scorecard/` |
| `python -m scripts.make_plots` | every 2D figure | `results/section/plots/` |
| `python -m scripts.lift_slope <airfoil>` | lift-curve slope by OLS, `--scan` for window sensitivity | prints only |
| `python -m scripts.vspaero_analysis` | 3D wing polar (OpenVSP env) | `results/wing/<airfoil>/` |
| `python -m scripts.hld_analysis` | high-lift device comparison (OpenVSP env) | `results/hld/` |

The steps read what earlier ones wrote, so order matters; `run_all` is that
order. The two OpenVSP steps are deliberately not in it -- they need a
different interpreter (see below).

Adding a candidate airfoil is dropping its `.dat` into `airfoils/` and
rerunning: every step discovers the current set through
`config.discover_airfoils()`. The one thing it will ask for is the Korn
`kappa_A` classification in `config/scoring.py`, which is a design judgment
and so raises rather than defaulting.

## Installing XFoil (Windows, compiled from source)

There is no working prebuilt `xfoil` wheel for modern Python/Windows, so the
Python bindings are built locally from
[`externals/xfoil-python`](../externals/xfoil-python) — a vendored copy of
[DARcorporation/xfoil-python](https://github.com/DARcorporation/xfoil-python)
(a maintained fork of daniel-de-vries/xfoil-python with a native CMake+Fortran
build), commit `0a8c2fc`, with two kinds of local patches on top:

1. Build fixes (Traps 1–2 below) needed just to compile it on Windows at all.
2. A source change to `src/api.f90` that exposes real XFoil convergence
   diagnostics through the Python bindings (see "Exposing XFoil's internal
   convergence diagnostics" below) — not upstream, specific to this project.

A plain `pip install git+https://github.com/DARcorporation/xfoil-python.git`
(the unpatched upstream) **fails out of the box on Windows** for two
independent reasons, both hit and fixed while setting this project up, and
would also silently skip the diagnostics patch. Build from
`externals/xfoil-python` using the steps below instead of the one-liner.

### Requirements

- **MSYS2** (https://www.msys2.org) — provides the mingw-w64 toolchain.
- **gfortran, gcc, make** via MSYS2's `mingw-w64-x86_64-*` packages (XFoil's core
  is Fortran 90 — this is the part a plain "Build Tools for Visual Studio"
  install cannot provide, MSVC has no Fortran compiler).
- **CMake** (standalone Windows installer or `mingw-w64-x86_64-cmake`; this
  project used the standalone installer at `C:\Program Files\CMake`).
- Python venv with `numpy` (already in [`requirements.txt`](requirements.txt)).

Install the MinGW toolchain from an MSYS2 shell:

```
pacman -S mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-make
```

(`gcc-fortran` pulls in `gcc`/`gcc-libs` as dependencies.)

Versions verified working in this environment (not hard requirements, just a
known-good baseline): `gfortran` 16.2.0, `GNU Make` 4.4.1, `cmake` 4.4.3.

### Why the plain `pip install` fails, and the fix

**Trap 1 — scikit-build hijacks the generator.**
Upstream's `pyproject.toml` lists `scikit-build` as a build requirement.
scikit-build drives CMake itself and, on Windows, defaults to the Visual Studio
generator with `-DCMAKE_GENERATOR_PLATFORM=x64`. Visual Studio has no Fortran
compiler, so configure fails with:

```
CMake Error at CMakeLists.txt:2 (project):
  No CMAKE_Fortran_COMPILER could be found.
```

Fix: build with plain `setuptools` instead of `scikit-build`. Already applied
in `externals/xfoil-python/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools", "wheel", "cmake"]
build-backend = "setuptools.build_meta"
```

**Trap 2 — the MinGW generator needs its own tools on PATH.**
Once scikit-build is out of the way, `setup.py`'s own `CMakeBuild` step forces
`-G "MinGW Makefiles"` on Windows (already present in
`CMakeBuild.build_extensions` in `externals/xfoil-python/setup.py` — if
working from a different fork/version, add `'-G', 'MinGW Makefiles'` to the
Windows `cmake_args` and make sure no `-DCMAKE_GENERATOR_PLATFORM=x64`
argument survives alongside it, the two are mutually exclusive). But CMake
still needs `mingw32-make` and `gfortran` resolvable, or you get:

```
CMake Error: CMake was unable to find a build program corresponding to "MinGW Makefiles".
CMake Error: CMAKE_Fortran_COMPILER not set, after EnableLanguage
```

Fix: put MSYS2's `mingw64\bin` (not `usr\bin`) first on `PATH` for the build,
and build with `--no-build-isolation` so pip's isolated build env can't shadow
the system `cmake`/generator:

```powershell
$env:Path = "C:\msys64\mingw64\bin;C:\Program Files\CMake\bin;" + $env:Path
pip install --no-build-isolation .\externals\xfoil-python
```

This produces `xfoil-<version>-cp3xx-cp3xx-win_amd64.whl` and installs cleanly
(`Successfully installed xfoil-1.1.1`). Re-run this after pulling in any
change to `externals/xfoil-python` (e.g. the diagnostics patch below) — pip
won't know to rebuild otherwise; uninstall first (`pip uninstall -y xfoil`)
if it doesn't pick up a source change, and delete
`externals\xfoil-python\build\` to clear stale CMake cache.

**Trap 3 — the DLL builds, but won't load at runtime.**
`import xfoil; XFoil()` then fails with:

```
FileNotFoundError: Could not find module 'C:\...\Temp\tmpXXXXXXXX.dll'
(or one of its dependencies). Try using the full path with constructor syntax.
```

The compiled `libxfoil.dll` depends on the MinGW runtime
(`libgfortran-5.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`,
`libquadmath-0.dll` — confirmed via `ldd libxfoil.dll`). Having
`mingw64\bin` on `PATH` does **not** fix this: since Python 3.8, `ctypes`'s
`CDLL`/`cdll.LoadLibrary` on Windows no longer searches `PATH` for a loaded
DLL's dependencies by default (verified in this environment — adding
`mingw64\bin` to `PATH` and re-running still throws the same error). The
supported fix is `os.add_dll_directory()`, called **before** `import xfoil`:

```python
import os
os.add_dll_directory(r"C:\msys64\mingw64\bin")
from xfoil import XFoil
xf = XFoil()  # works
```

`wp2/src/b12wp2/xfoil/runtime.py` does this once, centrally, so nothing else
in the package has to repeat it.

### Verifying the install

```powershell
$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
python -c "import os; os.add_dll_directory(r'C:\msys64\mingw64\bin'); from xfoil import XFoil; print(XFoil())"
```

Should print an `<xfoil.xfoil.XFoil object at ...>` with no traceback.

### Troubleshooting reference

| Symptom | Cause | Fix |
|---|---|---|
| `No CMAKE_Fortran_COMPILER could be found` (VS generator) | scikit-build defaulting to MSVC/Visual Studio generator | Patch `pyproject.toml` to use plain `setuptools` (Trap 1) |
| `Generator "MinGW Makefiles" does not support platform specification, but platform x64 was specified` | Both `-G "MinGW Makefiles"` and `-DCMAKE_GENERATOR_PLATFORM=x64` passed together | Remove the `-DCMAKE_GENERATOR_PLATFORM=x64` arg; MinGW Makefiles doesn't take a platform flag |
| `CMake was unable to find a build program corresponding to "MinGW Makefiles"` | `mingw32-make` not on PATH | Put `C:\msys64\mingw64\bin` first on PATH before building (Trap 2) |
| `CMAKE_Fortran_COMPILER not set, after EnableLanguage` | `gfortran` not on PATH, or MSYS2 `mingw-w64-x86_64-gcc-fortran` not installed | Install the package; ensure `mingw64\bin` (not `usr\bin`) is on PATH |
| `FileNotFoundError: Could not find module '...tmpXXXX.dll' (or one of its dependencies)` at import time | Python ≥3.8 ctypes doesn't use PATH for dependent-DLL resolution | `os.add_dll_directory(r"C:\msys64\mingw64\bin")` before `import xfoil` (Trap 3) |

### One more runtime bug: `XFoil.__del__` leaks a handle and a temp file

Not a build problem, but hit and fixed while writing
`wp2/src/b12wp2/xfoil/runtime.py`:
`xfoil.XFoil.__del__` calls `ctypes.windll.kernel32.FreeLibrary(handle)`
with no `argtypes` declared, so ctypes assumes a 32-bit `c_int`. The DLL
handle is a 64-bit pointer, which overflows that assumption
(`OverflowError: int too long to convert`); `FreeLibrary` never actually
runs, and the following `os.remove()` of the instance's temp `.dll` copy
then fails with `PermissionError` because the handle is still open. Verified
directly in this environment — every `XFoil()` instance leaked its library
handle and its temp file until this was fixed. Since the analysis pipeline
creates one `XFoil()` per airfoil/condition (dozens over a full run), this
isn't cosmetic. Fix (applied once, process-wide, in
`xfoil_runtime._ensure_process_setup`):

```python
import ctypes
ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
ctypes.windll.kernel32.FreeLibrary.restype = ctypes.c_int
```

### Exposing XFoil's internal convergence diagnostics

The unpatched binding's `XFoil.a(alpha)` only returns a `converged: bool` —
useful, but it collapses two very different situations ("ran out of Newton
iterations, residual still relatively small" vs. "the boundary-layer solve
actually diverged") into the same flag, and gives no sense of *how close* a
near-miss got. The natural instinct — capture XFoil's own printed diagnostic
text (`'VISCAL:  Convergence failed'` etc., `m_xoper.f90`, gated by the
`show_output`/`xf.print` flag) via Python's `os.dup2` stdout redirection —
**does not work**: verified directly in this environment, the redirected file
descriptor captured zero bytes while the Fortran output still printed
straight to the real console. The MinGW-compiled DLL's I/O isn't routed
through the fd table entry `os.dup2` rewrites (it resolves the console handle
through its own CRT/Win32 path); genuinely redirecting it would need
process-level `SetStdHandle` plumbing done before the DLL's first write, which
is far more fragile than the alternative below.

Instead, `externals/xfoil-python/src/api.f90`'s `alfa_` subroutine (the
Fortran routine backing `.a()`) was patched to pass out two pieces of state
it already computes internally, via `i_xfoil`'s module variables:

- **`rms_bl`** — the boundary layer Newton system's final RMS residual
  (`RMSbl`), i.e. how far from XFoil's own convergence tolerance (1e-4) the
  solve actually got.
- **`diverged`** — True only when the Newton solve aborted on a NaN
  mid-iteration (`viscal()`'s own raw return value in `m_xoper.f90`, before
  it's combined with `LVConv`), as opposed to simply exhausting the
  iteration budget without meeting tolerance.

`externals/xfoil-python/xfoil/xfoil.py`'s `.a()` was updated to match,
returning `(cl, cd, cm, cp, diverged, rms_bl)` instead of a 4-tuple.
`b12wp2.xfoil.runtime.run_alpha_sweep` consumes both fields directly
(see its docstring for the resulting `note` text). Only `alfa_`/`.a()` were
touched — `cl_`/`.cl()` (fixed-Cl mode) are unused by this project's
pipeline and were deliberately left unpatched to keep the change scoped to
what's actually exercised.

Verified directly against NACA 25112 at the cruise condition: converged
points return `diverged=False` with `rms_bl` on the order of `1e-5`–`1e-4`;
a deliberately absurd angle of attack (40°, deep past stall) returns
`diverged=True` with `rms_bl≈6.3` — a large, physically sensible residual
from the aborted iteration, not a placeholder value.

### Notes

- `externals/xfoil-python/` is tracked in this repo (not gitignored) so that
  patches to it — the diagnostics change above, and any future ones — have
  real commit history, unlike a locally-cloned build directory would.
  `externals/xfoil-python/build/`, `dist/`, and `*.egg-info/` are still
  gitignored (via that folder's own `.gitignore`) since they're regenerated
  by every build.
- The compiled package ends up in the project venv at
  `venv/Lib/site-packages/xfoil/` (gitignored, like the rest of `venv/`).
- `wp2/requirements.txt` covers the analysis dependencies (`numpy`, `matplotlib`,
  `pandas`, `scipy`) — it deliberately does **not** include `xfoil`, since that
  install is the multi-step process above, not a single pip line.

## Development

`wp2/requirements-dev.txt` adds `pytest` and `mypy`. Config lives in the repo
root `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.mypy]`); tests belong
in `wp2/tests/`, type-checked source is `wp2/src/` and `wp2/scripts/`.

`b12wp2` is imported straight from the working tree, not pip-installed:
`pyproject.toml`'s `pythonpath` puts `wp2/src` on the path for pytest, and
`scripts/__init__.py` does the same for `python -m scripts.<name>`.

```
pip install -r wp2/requirements.txt -r wp2/requirements-dev.txt
pytest
mypy
```

## Installing OpenVSP (for the VSPAERO scripts)

`scripts/vspaero_analysis.py` and `scripts/hld_analysis.py` need the OpenVSP
Python API and the `vspaero.exe` solver. Nothing has to be compiled: the
OpenVSP release ships a prebuilt extension module (`openvsp/_vsp.pyd`) and
the solver executables. The catch is that the `.pyd` is built for **one
Python version** -- Python 3.11 for OpenVSP 3.51.3 (see the release's
`python/environment.yml`) -- so it gets its own conda environment instead
of this repo's venv.

Verified working in this environment: OpenVSP 3.51.3 (VSPAERO 7.2.2),
Anaconda, Python 3.11.

### Steps (Windows, PowerShell)

1. Download the Windows 64-bit zip of OpenVSP 3.51.3 from
   https://openvsp.org/download.php and extract it to a path you will keep
   (the pip install below copies the packages, but keep the folder anyway
   for the GUI `vsp.exe`). Below it is `<OpenVSP>`, e.g.
   `C:\...\OpenVSP-3.51.3-win64`.

2. Create the environment with the Python version the `.pyd` was built for:

   ```
   conda create -n openvsp python=3.11
   conda activate openvsp
   ```

3. Install the OpenVSP Python packages from the release, **in this order**
   (`openvsp` depends on the three before it):

   ```
   cd <OpenVSP>\python
   pip install .\vsp_airfoils .\utilities .\degen_geom .\openvsp_config .\openvsp
   ```

   This is `requirements.txt` from that folder minus the unrelated rotor /
   panel-code packages (CHARM, AvlPy, pyPMARC), which these scripts don't
   use. The release's own `setup.ps1` (`conda env create -f environment.yml`
   + `pip install -r requirements-dev.txt`) also works; it creates an env
   named `vsppytools` instead of `openvsp` and installs everything in
   editable mode.

4. Add the analysis dependencies from this repo (and pytest if you want to
   run the tests from this env):

   ```
   pip install -r <repo>\wp2\requirements.txt pytest
   ```

### Verifying the install

```
python -c "import os, openvsp as vsp; print(vsp.GetVSPVersion()); print(vsp.CheckForVSPAERO(os.path.dirname(vsp.__file__)))"
```

Should print `OpenVSP 3.51.3` and `True` (`vspaero.exe` is installed next
to the package, which is where the scripts look for it).

### Running

From `wp2/`, with the env active:

```
python -m scripts.vspaero_analysis
python -m scripts.hld_analysis
```

or open either file in VS Code with the `openvsp` interpreter selected
(Ctrl+Shift+P -> "Python: Select Interpreter") and press Run -- both
scripts add `wp2/` and `wp2/src/` to `sys.path` themselves when started as a
file.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'openvsp'` | Wrong interpreter (e.g. the repo venv or system Python) | `conda activate openvsp` / select that interpreter in VS Code |
| `ImportError: DLL load failed while importing _vsp` | Env Python is not 3.11, so the prebuilt `.pyd` doesn't match | Recreate the env with `python=3.11` |
| `ModuleNotFoundError: No module named 'scripts'` | Started as a file from outside `wp2/` with an older copy of the script | Run `python -m scripts.<name>` from `wp2/` |
| `ModuleNotFoundError: No module named 'xfoil'` | Something imported `b12wp2.xfoil.runtime` | The OpenVSP env has no XFoil; only the VSPAERO scripts run there. XFoil polars are generated from the repo venv (above) |
| `RuntimeError: vspaero.exe not found` | `openvsp` installed without its executables (e.g. from another source) | Reinstall from the release's `python\openvsp` folder |

The XFoil polars in `results/section/data/` are inputs to the VSPAERO scripts, so
the two environments work in sequence: XFoil pipeline in the repo venv
first, then the VSPAERO scripts in `openvsp`.

## 3D wing analysis: VSPAERO + XFoil profile drag

`scripts/vspaero_analysis.py` analyses the wing (planform in
`config/wing.py` and `wing/geometry.py`, NACA 25112 sections) at the `config.CRUISE`
condition:

    CD = CDi (VSPAERO, Trefftz plane) + CD_profile (XFoil strips) + CD_wave (Korn/Lock)

- **VSPAERO (VLM)** gives CL, CMy and the induced drag. Its own `CDo` (a
  flat-plate skin-friction estimate) is discarded.
- **Profile drag** comes from the XFoil cruise polar
  (`results/section/data/<airfoil>_cruise_polar.csv`), strip by strip: local
  cl -> cl_n = cl / cos^2(sweep) -> cd_n(cl_n) from the converged pre-stall
  polar -> streamwise cd = cd_n (friction-dominated, conservative; the
  cd_n cos^3(sweep) variant is reported as a sensitivity) -> Reynolds
  correction (c / c_ref)^-0.2 -> integrated over the span. A strip past
  the polar's cl_max gets cd = NaN (and the whole angle of attack has no
  CD); the first such wing CL is reported as `CL_first_section_stall`.
- **Wave drag**: swept Korn equation (kappa_A from `config.kappa_a`) with
  Lock's 20 (M - M_crit)^4.

Lift and moment stay inviscid. Details in `wing/viscous_correction.py`.

It needs the OpenVSP Python API, which is not in this venv. Run it from
the OpenVSP conda env (see "Installing OpenVSP" above), from `wp2/`:

```
conda activate openvsp
python -m scripts.vspaero_analysis
```

Outputs go to `results/wing/<airfoil>/`: `polar.csv`, `span_loads.csv`,
`summary.csv`, `polar.png`, `drag_breakdown.png`, `span_loads.png`,
`transonic.png`. The raw VSPAERO run (`vspaero_run/`, including
`wing.vsp3`) is gitignored.

## High-lift device comparison

`scripts/hld_analysis.py` (same OpenVSP env, `python -m scripts.hld_analysis`
from `wp2/`) compares every trailing-edge x leading-edge device combination
(`config.hld.TE_DEVICES` x `LE_DEVICES`) on the same wing:

1. Clean-wing CL_max at M ~ 0.2 (`config.LANDING`): VSPAERO spanwise lift +
   critical-section method with the XFoil landing polar.
2. ADSEE increments per device (`hld/sizing.py`):
   dCL_max = 0.9 dcl_max (S_wf/S) cos(Lambda_hinge), S'/S, and the
   alpha_0L shift; take-off = 60% of the landing increment.
3. Pareto front of CL_max,L vs. a qualitative mechanism-complexity rank.

Span limits, spar positions and the (optional) CL_max requirements are the
settings in `config/hld.py` and `config/solvers.py`. Outputs in
`results/hld/`:
`comparison.csv`, `clean_wing.csv`, `comparison_heatmap.png`,
`cl_max_vs_complexity.png`, `lift_curves.png`, `planform_hld.png`.
