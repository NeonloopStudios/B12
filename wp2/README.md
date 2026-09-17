# WP2 — Airfoil Selection

XFoil-based screening of candidate 2D airfoil sections. The analysis plan is in
[`xfoil_plan.md`](xfoil_plan.md); this file only covers getting XFoil working.

## Layout

```
wp2/
  airfoils/     candidate .dat coordinate files (the set will change; any .dat
                dropped here is picked up automatically by the scripts)
  scripts/      the analysis pipeline (see xfoil_plan.md)
  results/      generated CSVs and plots (see xfoil_plan.md §3)
  xfoil_plan.md the stage-by-stage analysis plan
```

## Installing XFoil (Windows, compiled from source)

There is no working prebuilt `xfoil` wheel for modern Python/Windows, so the
Python bindings ([DARcorporation/xfoil-python](https://github.com/DARcorporation/xfoil-python),
a maintained fork of daniel-de-vries/xfoil-python with a native CMake+Fortran
build) have to be compiled locally. A plain

```
pip install git+https://github.com/DARcorporation/xfoil-python.git
```

**fails out of the box on Windows** for two independent reasons, both hit and
fixed while setting this project up. Reproduce it with the steps below instead
of the one-liner.

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

Fix: build with plain `setuptools` instead of `scikit-build`. Edit
`pyproject.toml` in the cloned source to:

```toml
[build-system]
requires = ["setuptools", "wheel", "cmake"]
build-backend = "setuptools.build_meta"
```

**Trap 2 — the MinGW generator needs its own tools on PATH.**
Once scikit-build is out of the way, `setup.py`'s own `CMakeBuild` step forces
`-G "MinGW Makefiles"` on Windows (already present in `CMakeBuild.build_extensions`
in this fork — if working from a different fork/version, add
`'-G', 'MinGW Makefiles'` to the Windows `cmake_args` and make sure no
`-DCMAKE_GENERATOR_PLATFORM=x64` argument survives alongside it, the two are
mutually exclusive). But CMake still needs `mingw32-make` and `gfortran`
resolvable, or you get:

```
CMake Error: CMake was unable to find a build program corresponding to "MinGW Makefiles".
CMake Error: CMAKE_Fortran_COMPILER not set, after EnableLanguage
```

Fix: put MSYS2's `mingw64\bin` (not `usr\bin`) first on `PATH` for the build,
and build with `--no-build-isolation` so pip's isolated build env can't shadow
the system `cmake`/generator:

```powershell
$env:Path = "C:\msys64\mingw64\bin;C:\Program Files\CMake\bin;" + $env:Path
git clone https://github.com/DARcorporation/xfoil-python.git tools\xfoil-python
# apply the pyproject.toml fix above inside tools\xfoil-python
pip install --no-build-isolation .\tools\xfoil-python
```

This produces `xfoil-<version>-cp3xx-cp3xx-win_amd64.whl` and installs cleanly
(`Successfully installed xfoil-1.1.1`).

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

`wp2/scripts/xfoil_runtime.py` does this once, centrally, so no other script in
`wp2/scripts/` has to repeat it.

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

### Notes

- `tools/` (the cloned `xfoil-python` source + build artifacts) is git-ignored —
  it's a vendored third-party build directory, not project source. This README
  is what makes the build reproducible instead of committing it.
- The compiled package ends up in the project venv at
  `venv/Lib/site-packages/xfoil/` (also git-ignored).
- `wp2/requirements.txt` covers the analysis dependencies (`numpy`, `matplotlib`,
  `pandas`, `scipy`) — it deliberately does **not** include `xfoil`, since that
  install is the multi-step process above, not a single pip line.

## Development

`wp2/requirements-dev.txt` adds `pytest` and `mypy`. Config lives in the repo
root `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.mypy]`); tests belong
in `wp2/tests/`, type-checked source is `wp2/scripts/`.

```
pip install -r wp2/requirements.txt -r wp2/requirements-dev.txt
pytest
mypy
```

