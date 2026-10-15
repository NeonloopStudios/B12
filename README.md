<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 Kajetan R. Gulaj
Created: 2026-09-22
-->

# System Design (AE2111-I, B12)

TU Delft AE2111-I "System Design" - aircraft design project, group B12.

Three work packages, one aircraft, done in sequence:

- **WP1** - initial sizing: mission, class I weight estimate, wing and power
  loading. Hand calculations, no code here.
- **WP2** - wing aerodynamic design: planform, airfoil selection, control
  surfaces, design synthesis. See [`wp2/`](wp2/README.md).
- **WP3** - propulsion system, overall aircraft design, class II estimations,
  final concept. See [`wp3/`](wp3/README.md).

## Layout

```
wp2/              WP2 code, data, results - see wp2/README.md
wp3/              WP3 code, data, results - see wp3/README.md
externals/        vendored third-party source (e.g. xfoil-python)
requirements.txt  the only requirements file: wp2 + wp3, tests, type checks
pyproject.toml    pytest and mypy configuration for wp2 and wp3
```

## Setup

One venv at the repo root, Python 3.13, one requirements file:

```
python -m venv venv
venv\Scripts\activate              # Windows; Unix: source venv/bin/activate
python -m pip install -r requirements.txt
```

Two tools are not pip packages and are set up separately, see
[`wp2/README.md`](wp2/README.md):

- **XFoil** (WP2 2D analysis): built into this venv from
  `externals/xfoil-python`.
- **OpenVSP** (WP2 3D analysis): its own conda environment with Python 3.11,
  because the OpenVSP Python bindings are built for that version.

Tests and type checks, from the repo root:

```
python -m pytest
python -m mypy
```

## License

GPL-3.0-or-later, full text in [`LICENSE`](LICENSE). Airfoil coordinate files
under `wp2/airfoils/` and everything in `externals/` come from third parties
and keep their own terms.
