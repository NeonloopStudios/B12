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
  surfaces, design synthesis. See [`wp2/`](wp2/README.md) - the only WP with
  code in this repo so far.
- **WP3** - propulsion system, overall aircraft design, class II estimations,
  final concept. May get its own code directory later.

## Layout

```
wp2/       WP2 code, data, results - see wp2/README.md
externals/ vendored third-party source (e.g. xfoil-python)
```
