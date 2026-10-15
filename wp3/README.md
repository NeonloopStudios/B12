<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 Mateusz Suszynski
Created: 2026-10-15
-->

# WP3 — Further Aircraft Design

Propulsion, external layout, class II drag and weight, and the iteration to
the final concept. The structure is the same as `wp2/`.

## Layout

```
wp3/
  scripts/      the runnable steps, one file per step
  src/b12wp3/   the analysis library every step calls
    config/     every input: paths, atmosphere, mission, geometry, drag settings
    common/     trapezoidal planform (wing, tails, pylons)
    drag/       WP3.3a class II drag: component drag build-up (ADSEE-II)
    propulsion/ WP3.1  engine configuration and selection
    fuselage/   WP3.2a fuselage external dimensions
    landing_gear/ WP3.2b landing gear dimensions and position
    empennage/  WP3.2c/d empennage trade-off and tail sizing
    weight/     WP3.3b class II weight estimation
    iteration/  WP3.4a class I / class II iteration, wing-loading diagram
    concept/    WP3.4b/d parameter table, compliance matrix
    plots/      the figures, one module per stage
  tests/        pytest suite over src/b12wp3/
  results/      generated CSVs and plots, one directory per stage
```

wp3 does not import the wp2 package. Values taken from WP2 (the wing
planform, the cruise condition) are typed into `config/` with their source.

## Inputs

All geometric inputs are in `src/b12wp3/config/geometry.py`, one constant per
quantity. An input that is not designed yet is `None`.
`geometry.aircraft_geometry()` raises one error that names every unset input.

The ADSEE-II method tables (laminar fraction, roughness, interference
factor, flap factor, landing gear fit, excrescence fraction) are in
`src/b12wp3/config/drag_tables.py`. The selected rows are in
`src/b12wp3/config/drag.py`:

| Setting | Value | Source |
|---|---|---|
| Laminar fraction | civil jet, classic production metal: 5 % fuselage, 10 % wing and tails | ADSEE-II table |
| Surface roughness k | smooth paint, 0.634e-5 m | ADSEE-II table |
| IF wing | un-filleted low wing, 1.4 (upper end, conservative) | ADSEE-II table |
| IF horizontal and vertical tail | cruciform tail, 1.06 | ADSEE-II table |
| IF nacelle | more than about 1 diameter away, 1.0 | ADSEE-II table |
| IF fuselage, pylon | 1.0 (no table row) | assumption |
| k_a (Korn) | 0.95, supercritical | same as WP2 |
| M_cr | 0.56 normal to the half chord (WP2), converted to free stream | WP2 M_crit sweep |
| Flap factor | Fowler TE flap: slotted, 0.0074; nose flap: plain, 0.0144 | ADSEE-II |
| Landing gear | open wheel wells, 0.05328 | ADSEE-II fit |
| Excrescence and leakage | jet transport, 3.5 % of the total CD0 | ADSEE-II range 2-5 % |

Configurations are in `src/b12wp3/config/mission.py`: `CRUISE_CONFIGURATION`,
`takeoff_configuration()` and `landing_configuration()`. The take-off speed
and the flap deflections are unset (`None`).

Nacelles use the fuselage laminar fraction and the fuselage wetted-area
formula. Pylons use the wing-and-tail laminar fraction and the tail
wetted-area formula.

## Class II drag method (WP3.3a)

```
CD0 = (1 / S_ref) * sum_c( Cf_c * FF_c * IF_c * S_wet,c ) + sum CD_misc
```

| Term | Module | Equation |
|---|---|---|
| S_wet | `drag/wetted_area.py` | wing 1.07·2·S_exp; tails, pylon 1.05·2·S_exp; body: ADSEE-II fuselage formula |
| Re | `drag/skin_friction.py` | min(ρVl/μ, cutoff); cutoff 38.21(l/k)^1.053 subsonic, 44.62(l/k)^1.053·M^1.16 transonic |
| Cf | `drag/skin_friction.py` | x_lam·1.328/√Re + (1−x_lam)·0.455/((log10 Re)^2.58 (1+0.144M²)^0.65) |
| FF | `drag/form_factor.py` | lifting surface, fuselage, nacelle formulas; f = l/√(4A_max/π) |
| Wave drag | `drag/miscellaneous.py` | Mason M_DD; 0 / 0.002[1+2.5(M_DD−M)/0.05]^-1 / 0.002[1+(M−M_DD)/0.05]^2.5 |
| Upsweep, base | `drag/miscellaneous.py` | D/q = 3.83u^2.5·A_max; D/q = [0.139+0.419(M−0.161)²]·A_base |
| Flaps | `drag/miscellaneous.py` | F_flap(c_f/c)(S_flap/S_ref)(δ−10), δ > 10° only |
| Landing gear | `drag/miscellaneous.py` | coeff·e^{5.615 S_A/(d w)}·(d w)/S_ref per unit, main and nose |
| Excrescence | `drag/miscellaneous.py` | f/(1−f)·CD0_rest, so it is f of the total |
| CD0 | `drag/buildup.py` | `aircraft_drag_buildup(geometry, configuration, settings)` |

The characteristic length l is the MAC for a lifting surface and the total
length for a body. The flow regime of the cutoff is set per flight condition
in `config/mission.py`. Cruise (M = 0.77) uses the transonic cutoff.

## Drag due to lift (WP3.3a)

```
CD = CD0 + K CL^2,   K = 1 / (pi AR_eff e)
```

| Term | Module | Equation | Source |
|---|---|---|---|
| h/b | `config/induced.py` | mean of the A320neo and CRJ900 winglets | airport planning documents, `docs/reports/winglets/` |
| AR_eff | `drag/induced.py` | AR + 1.9 (h/b) AR, h = (h/b) b | ADSEE-II slides 75, 78 |
| e | `drag/induced.py` | Brandt: 2 / (2 - AR_eff + sqrt(4 + AR_eff^2 (1 + tan^2 Lambda_c/2))) | ADSEE-II slide 72 |
| Delta e, flaps | `drag/induced.py` | 0.0026 delta_f (wing-mounted engines) | Vos (2022) eq. 7.62 |
| polar | `drag/polar.py` | CL at (L/D)max = sqrt(CD0/K), (L/D)max = 1/(2 sqrt(CD0 K)) | ADSEE-II slide 49 |

Slide 75: AR_eff is used in all lift and drag equations, so e is evaluated
at AR_eff. No wave drag and no trim drag in this term. The two Raymer
formulas of slide 72 are written to `induced_drag.csv` as a comparison only:
Raymer's swept-wing formula at AR_eff gives more induced drag with the
winglet than without, so it is not used.

Flap deflections: the trailing-edge take-off (15 deg, WP1) and landing (40 deg,
WP2 Fowler maximum) inputs of `config/mission.py`.

```
python -m scripts.induced_drag
```

writes `results/drag/induced_drag.csv/.png`, the WP3.4 input
`results/iteration/oswald_configurations.csv` (e and K of the clean,
take-off and landing configurations), and `results/drag/drag_polar.csv/.png`
once every class II CD0 input is set.

## Running the tests

From the repo root (the paths are in `pyproject.toml`):

```
python -m pytest wp3/tests
python -m mypy --strict wp3/src
```

pytest runs with `--import-mode=importlib`, because wp2 and wp3 both have a
top-level `tests` package.
