<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026
-->

# WP4.1 — Design of the force diagrams

This file describes the design of WP4.1 (shear force, bending moment, torque).
It follows the WP4-5 reader, section WP4.1 and Appendices A, B, C and E.
The layout and coding rules of `wp4/README.md` apply.

---

## 1. Goal and scope

WP4.1 gives the shear force `S_z(y)`, the bending moment `M_x(y)` and the torque `τ(y)` of the wing.
The three diagrams are required for two load cases: one with positive load factor and one with negative load factor.
The load cases come from WP4.3. WP4.1 treats them as input.

In scope:

- Out-of-plane shear force, bending moment about the chordwise axis, torque about the shear centre.
- Symmetric flight only. Control surfaces are not deflected.

Out of scope (reader, WP4.1 and Appendix A.3):

- In-plane shear force and in-plane bending moment.
- Sweep coupling at the root (kink of the elastic axis).
- The axial force diagram (see assumption A-9).

---

## 2. Data flow

Four scripts exchange data only through CSV files.
No script imports another script. No `wp41` module imports `wp2`, `wp3`, `wp42` or `wp43`.

```text
        wp41_aero_loads.py (VSPAERO run on a Mach grid, once per geometry)
        out/wp41/lift_distribution.txt, moment_distribution.txt
                          |
                          v
   [1] wp41_lift.py                 [2] wp41_weight.py
   out/wp41/<case>/lift.csv         out/wp41/<case>/weight.csv
                                    out/wp41/<case>/point_loads.csv
          |            \               /           |
          |             \             /            |
          v              v           v             v
   [3] wp41_torsion.py          [4] wp41_shear_moment.py
   out/wp41/<case>/torque.csv   out/wp41/<case>/shear_moment.csv
   + torque plot                + shear and moment plots
```

| Script | Reads | Writes |
|---|---|---|
| `wp41_lift.py` | VSPAERO distribution files, load case | `lift.csv` |
| `wp41_weight.py` | mass properties, load case | `weight.csv`, `point_loads.csv` |
| `wp41_torsion.py` | `lift.csv`, `weight.csv`, `point_loads.csv` | `torque.csv`, plot |
| `wp41_shear_moment.py` | `lift.csv`, `weight.csv`, `point_loads.csv` | `shear_moment.csv`, plots |

`wp41_aoa_grid.py` is a side script: it reads the VSPAERO files and writes `out/wp41/aoa_grid.txt` (`CL` and `α` for OEW, MZFW and MTOW over a speed range). No other script reads it.

All per-case files use the same global stations, `wp41.stations` (`N_SPAN_STATIONS` points from `y = 0` to `y = b/2`). `wp41.torsion` checks that `lift.csv` and `weight.csv` share them, so the weight script must take its stations from there.

Scripts 1 and 2 are independent. Run them first, in any order.
Scripts 3 and 4 are independent of each other. Run them after 1 and 2.
`<case>` is the load case identifier, for example `LC-1`.

---

## 3. Coordinates and signs

- Body axes (whiteboard sketch of the team): `x` forward (towards the nose), `y` right, `z` down. The frame is right-handed. Unit: SI.
- `configs/aircraft.py` and the VSPAERO files use `x` aft and `z` up. The conversion is `x_body = −x_VSPAERO` and `z_body = −z_VSPAERO`. `y` is the same.
- Span station `y` runs from the centreline (`y = 0`) to the tip (`y = b/2`).
- The root of the cantilever is the centreline. The strips inside the fuselage stay in the integrals.
- Loads are per unit span, on the right half-wing.
- Moments are right-handed about the body axes: `M_x` bends the wing (right tip goes down), `M_y` is positive nose-up.
- The torque is `τ = M_y` about the shear centre. A positive torque is nose-up.
- The load files give `N'` as a positive number for an upward force. In body axes the force is `−N'` along `z`.
- Each diagram value is the resultant of all loads outboard of `y`, taken about the point `y` on the shear centre axis.
  `f_z(y)` is the load per unit span along `z` (positive down), `t(y)` is the torque per unit span, `F_j`, `M_j` and `T_j` are point loads:

```text
S_z(y) = ∫_y^{b/2} f_z(η) dη       +  Σ F_j        (y_j >= y)
M_x(y) = ∫_y^{b/2} S_z(η) dη       +  Σ M_j        (y_j >= y)
τ(y)   = ∫_y^{b/2} t(η) dη         +  Σ T_j        (y_j >= y)
```

Differential form: `dS_z/dy = −f_z`, `dM_x/dy = −S_z`, `dτ/dy = −t`.
For a positive load factor the lift is upward, so `S_z` and `M_x` are negative. The sign follows from the axes, not from a choice of the diagram.
`f_z(y) = −N'(y) + n (w'_str(y) + w'_fuel(y))`.
---

## 4. Inputs

### 4.1 Load cases (from WP4.3)

One frozen dataclass per case, in `configs/wp41/load_cases.py`.
WP4.3 fills the values. The README gives no numbers.

| Field | Unit | Meaning |
|---|---|---|
| `name` | - | Identifier, for example `LC-1` |
| `n` | - | Load factor. Positive or negative |
| `v_eas` | m/s | Equivalent airspeed |
| `altitude_m` | m | ISA altitude |
| `mass_case` | - | `oew`, `mtow`, or `no_fuel` (the three weights of the reader) |
| `thrust_n` | N | Thrust per engine |

### 4.2 Mass properties (from WP1-3 and WP4.2)

The masses live in `configs/aircraft.py` (`MASSES`, including `engine_installed_kg`) and the engine station in `ENGINE`.
The chord fractions and the engine installation live in `configs/wp41/weight.py` (`WeightInputs`).
The rows `h_thrust_m` and `f_sc` live in `configs/wp41/torsion.py` (`TorsionInputs`) until WP4.2 replaces them.
The rows below are the full list of inputs, not one dataclass.

| Field | Unit | Meaning |
|---|---|---|
| `w_oew_n`, `w_mpw_n`, `w_fuel_n` | N | Weights of the three mass cases |
| `w_engine_n` | N | Weight of one engine with pylon |
| `y_engine_m` | m | Span station of the engine |
| `d_engine_m` | m | Distance of the engine centre of gravity ahead of the shear centre |
| `h_thrust_m` | m | Height of the thrust line below the shear centre |
| `f_fuel` | - | Chord fraction of the fuel centre of gravity |
| `f_sc` | - | Chord fraction of the shear centre (the wing box centroid) |
| `a_nd` | - | Box area ratio `A_nd = (h/c)(l/c)` |
| `eta_tank_in`, `eta_tank_out` | - | Span fraction `2y/b` where the fuel tank starts and ends |

The reader allows `f_sc` equal to the centroid of the wing box.
WP4.2 computes that centroid. Until then, `f_sc` is an input.
Every value is read from the config. No script has a default.
`f_sc` is `None` in `torsion.py` until WP4.2 gives it. The torsion script then stops and names WP4.2.
`h_thrust_m` is 0.8 m, the axis height below the chord line from `wp3/configs/vsp_model.yaml`. It is not yet the height below the centroid.

---

## 5. Script 1 — lift

**Input:** the two VSPAERO files and one load case.
The files hold `cl` and `cm_c4` at α = 0° and α = 10° for every Mach number of the grid in `settings.VSPAERO` (0.15 to 0.85).
Their header lines give the Mach number, the angle and `CL` of each column.
Module: `wp41.lift_distribution`.

**Steps:**

1. Flight state (ISA): `q = ½ ρ0 V_EAS²`, `V_TAS = V_EAS √(ρ0/ρ)`, `M = V_TAS / a`.
2. Total weight of the mass case, `W = W_oem + W_payload + W_fuel` (`case_weights`, masses in `configs/aircraft.py`):
   - `oew`: OEM only.
   - `mtow`: OEM + maximum payload + the fuel that gives MTOM.
   - `no_fuel`: OEM + maximum payload.
3. Required lift coefficient: `CL_d = n W / (q S)`.
4. Mach: a cubic spline (`scipy.interpolate.CubicSpline`) over the Mach grid gives `CL0`, `CL10`, `cl0(y)`, `cl10(y)`, `cm0(y)`, `cm10(y)` at `M`.
5. Interpolation weight: `s = (CL_d − CL0) / (CL10 − CL0)`.
   `cl_d(y) = cl0(y) + s (cl10(y) − cl0(y))` and the same for `cm_c4`.
6. Angle of attack: `α_d = α0 + s (α10 − α0)` (freestream angle; the wing incidence is in the model). It is reported, not used in the loads (see A-2).
7. Stations: `cl c` and `cm_c4 c²` go from the VSPAERO strip centres to the global stations with a monotone cubic (`scipy.interpolate.PchipInterpolator`) and the value 0 at the tip.
8. Loads per unit span:
   - `N'(y) = cl_d(y) q c(y)`
   - `m'_c4(y) = cm_c4_d(y) q c(y)²`  (nose-up positive, about the local quarter chord)

**Output `lift.csv`:** header lines `# schema: wp41.lift.v1`, case name, `n`, `W`, `q`, `CL_d`, `alpha_d`, `mach_case`, `mach_run`, `extrapolated`.
Columns: `y_m, chord_m, cl_d, cm_c4_d, n_prime_npm, m_prime_c4_nmpm`, on the global stations.
The database is evaluated at the case Mach, so `mach_run` equals `mach_case`.

**Checks (the script fails with the offending values):**

- `2 ∫ N' dy` equals `n W` within `LIFT_CHECK_TOLERANCE` (a fraction of `W`) in `settings.py`.
- A case Mach outside the database grid raises an error.
- Flag `extrapolated` is true when `s < 0` or `s > 1`. A negative-`n` case is always extrapolated, because `CL_d < 0 < CL0`.

---

## 6. Script 2 — weight

**Input:** `MASSES`, `ENGINE`, `WeightInputs`, `f_sc`, one load case, the chord `c(y)` and the stations of the planform.
Modules: `wp41.weight` and, for the fuel formula, `wp41.weight_distrubution`.

All loads are at `n = 1`. The torsion and shear scripts multiply them by `n`.

**Distributions** (`η = 2y/b`):

- Structure: `w'_str(y) = 0`. The wing structure weight is neglected (assumption A-7).
- Fuel, from the fuselage side to the tip: `w'_fuel(y) = (W_fuel/2) c(y)² / ∫_{y_f}^{b/2} c² dy`, with `y_f = d_fus / 2` (`weight_distrubution.py`).
- Inside the fuselage, `w'_fuel(y) = 0`.
- The box area ratio `A_nd` is constant for one airfoil, so it cancels.

The sketch of the design gives the reason: the box area is `A = A_nd c²`.
Fuel mass per unit span follows the box area. The fuel weight is the weight of the mass case (`case_weights`).
The formula is evaluated on the global stations. The fuel table is then scaled so that `2 ∫ w'_fuel dy = W_fuel` exactly. The scale factor must be within `WEIGHT_CHECK_TOLERANCE` of 1.

**Point loads:** one row for the engines of one half-wing. The weight is `W_engine = g × engine_installed_kg × engine_count / 2`. Thrust is not a weight and is added in scripts 3 and 4.
The engine centre of gravity is ahead of the shear centre by `d = f_sc c(y_e) + x_inlet − k l_n`:
`c(y_e)` is the chord at the engine, `x_inlet` the inlet plane ahead of the leading edge, `l_n` the nacelle length and `k` the centre of gravity position as a fraction of `l_n` (assumption A-12).

**Fuselage group:** `W_fus = W_oew − 2 W_engine` plus `W_mpw`. The wing structure weight is part of `W_oew` and is not applied to the wing (A-7). It acts at the centreline. It is not in the CSV. It is the reaction in the equilibrium check of script 4.

**Output `weight.csv`:** columns `y_m, chord_m, w_str_npm, w_fuel_npm, f_str, f_fuel`. `w_str_npm` is zero and `f_str` equals `f_fuel`. The columns stay for the contract.
**Output `point_loads.csv`:** columns `name, y_m, w_n, d_ahead_of_sc_m`.
`weight.csv` carries `# schema: wp41.weight.v1`. `point_loads.csv` carries `# schema: wp41.point_loads.v1`. Both carry `# case: <name>`.

**Checks:** The fuel scale factor is within `WEIGHT_CHECK_TOLERANCE` of 1 (`settings.py`).

---

## 7. Script 3 — torsion

**Input:** `lift.csv`, `weight.csv`, `point_loads.csv`, `f_sc`, the load case.
The chord `c(y)` and `n` come from the CSV files.

**Torque per unit span about the shear centre** (nose-up positive):

```text
t(y) =  m'_c4(y)
      + N'(y)  (f_sc − 1/4) c(y)
      − n w'_str(y)  (f_sc − f_str)  c(y)
      − n w'_fuel(y) (f_sc − f_fuel) c(y)
```

Each force term is `force × arm`. The arm is the chordwise distance of the load from the shear centre.
In body axes `M_y = z F_x − x F_z`, with `x` forward of the shear centre. An upward force (`F_z < 0`) ahead of the shear centre gives a nose-up torque. A downward force ahead of it gives a nose-down torque.

**Point torque of an engine at `y_j`:**

```text
T_j = − n W_engine d_engine  +  T cos(Λ) h_thrust
```

- `Λ` is the quarter-chord sweep. The thrust is rotated by `Λ` (reader, Appendix A.3).
- Thrust below the shear centre gives a nose-up torque.

**Integration:** the diagram `τ(y)` follows from section 3.
The data are strip values at fixed stations. The script interpolates linearly between stations. Beyond the end stations the value stays constant.
The integral uses `scipy.integrate` (reader, Appendix C).

**Output `torque.csv`:** header `# schema: wp41.torque.v1`, `# case`, `# n`. Columns `y_m, t_prime_nmpm, torque_nm`.
The stations are the strip stations, `y = 0`, `y = b/2` and each engine station.
An engine station appears twice. The first row includes the point torque. The second row excludes it.
**Plot:** `torque_<case 1>_<case 2>.png` in `out/wp41/`. Top panel: `t'(y)`. Bottom panel: `τ(y)`. Each case is one curve in each panel.
The script takes `b/2` and `Λ` from `WING` and the thrust from the load case.

---

## 8. Script 4 — shear force and bending moment

**Input:** `lift.csv`, `weight.csv`, `point_loads.csv`, the load case.

**Load per unit span along `z`** (positive down):

```text
f_z(y) = −N'(y) + n (w'_str(y) + w'_fuel(y))
```

**Point loads at `y_j`:**

- Shear force: `F_j = + n W_engine` (down, positive `z`).
- Bending moment: `M_j = + T sin(Λ) h_thrust`. The spanwise (inboard) thrust component `−T sin(Λ)` along `y` acts at the height `h_thrust` below the shear centre.
  Derivation: `M_x = r_y F_z − r_z F_y` with `r_z = h_thrust` and `F_y = −T sin(Λ)`. A test covers the sign.

**Integration:** `S_z(y)` and `M_x(y)` follow from section 3. The method is the same as in script 3.

**Equilibrium check:** the root shear `S_z(0)` equals `− n W_fus / 2` (the fuselage group pulls down, the wing resultant points up).
This holds when the tail load is zero (A-1). The script fails with the two values if the relative difference is above the tolerance.

**Output `shear_moment.csv`:** columns `y_m, fz_npm, shear_z_n, moment_x_nm`.
**Plots:** `shear_<case>.png` and `moment_<case>.png`, positive-`n` and negative-`n` case in one figure.

---

## 9. Source layout

```text
wp4/
├── configs/wp41/
│   ├── settings.py        VSPAERO Mach grid, stations, tolerances, files
│   ├── load_cases.py      frozen dataclass LoadCase, LOAD_CASES (empty until WP4.3)
│   ├── torsion.py         frozen dataclass TorsionInputs (f_sc, h_thrust_m)
│   └── weight.py          frozen dataclass WeightInputs (f_fuel, engine installation)
├── scripts/
│   ├── wp41_aero_loads.py VSPAERO database
│   ├── wp41_lift.py
│   ├── wp41_aoa_grid.py   side script: CL and alpha over weights and speeds
│   ├── wp41_weight.py
│   ├── wp41_torsion.py
│   └── wp41_shear_moment.py
├── src/wp41/
│   ├── contracts.py       frozen dataclasses of the CSV files, each with to_csv / from_csv
│   ├── stations.py        global spanwise stations of every per-case file
│   ├── atmosphere.py      ISA
│   ├── aero_loads.py      VSPAERO pre- and post-processing
│   ├── lift_distribution.py  section 5
│   ├── weight.py          section 6
│   ├── weight_distrubution.py  fuel formula of section 6 (file name as committed)
│   ├── torsion.py         section 7
│   ├── internal_loads.py  section 8
│   └── integrate.py       tip-to-station integral, shared by torsion and internal_loads
└── tests/unit/wp41/       one test file per module, same names
```

`contracts.py` is the only place that reads or writes the CSV files.
A loader checks the schema line and the column names. It raises an error with the offending value.

---

## 10. Tests

Test values are general, not round. Each module has a test with a known exact answer.

| Module | Test |
|---|---|
| `integrate.py` | Constant load `f_z` on span `L`: `S_z = f_z (L − y)`, `M_x = f_z (L − y)² / 2`. Use `f_z = 37.3`, `L = 11.7`, station `y = 4.9`. |
| `internal_loads.py` | Point load `F` at `y_j`: `S_z` is `F` inboard of `y_j` and zero outboard. Thrust moment `M_j` has the sign derived in section 8. Equilibrium check fails for a wrong `W_fus`. |
| `torsion.py` | Constant `N'` with `f_sc ≠ 1/4`: `τ = N' (f_sc − 1/4) c (L − y)` for constant chord. Sign flips with `f_sc < 1/4`. |
| `lift_distribution.py` | `s = 0` returns the α = 0° table. `s = 1` returns the α = 10° table. A value in between uses the linear closed form. `n < 0` sets `extrapolated`. `2 ∫ N' dy = n W`. Round trip to `lift.csv`. |
| `weight.py` | The fuel integral equals `W_fuel/2`. Fuel is zero inside the fuselage and follows `c²`. The structure column is zero. Hand value of the engine offset. |
| `contracts.py` | Round trip `to_csv` / `from_csv` is bitwise equal. A wrong schema line or column name raises. |
| import guard | No module in `wp41` imports `wp2`, `wp3`, `wp42` or `wp43`. |

Gate: `mypy --strict wp4/src` and `pytest wp4/tests`, as in `wp4/README.md`.

---

## 11. Assumptions

Each entry has the assumption, the result, and the validity.

| ID | Assumption | Result | Validity |
|---|---|---|---|
| A-1 | The wing carries all the lift. The tail load is zero. | `2 ∫ N' dy = n W`. The root-shear equilibrium check is valid. | A trimmed aircraft has a tail load. Its size is not estimated here. |
| A-2 | `N' = L'`. The factor `cos α` is taken as 1 and the drag term `sin α D'` is dropped. | The loads need `cl` only. | `cos 10° = 0.985`, an error of 1.5 % at α = 10°. The error grows with `α_d`. The dropped drag term has the opposite sign. Its size is not estimated. |
| A-3 | Between the Mach numbers of the VSPAERO grid, the coefficients follow a cubic spline in Mach. Prandtl-Glauert (in VSPAERO) holds up to `M = 0.85`. | One VSPAERO run (15 Mach numbers) serves all cases at their own Mach. | At `M = 0.77` the spline gives `CL` within 0.01 % of a direct run. Above `M ≈ 0.8` Prandtl-Glauert is outside its valid range on this wing. |
| A-4 | `CL` is linear in `α`. | Two angles of attack give any `CL_d` (reader, Appendix B). | The reader accepts it. Cases outside α = 0° to 10° are extrapolated and flagged. |
| A-5 | The wing is equivalent to a straight unswept wing. Dihedral is ignored. | Reader, Appendix A.3. Chord fractions replace `x` coordinates. The root kink is not modelled. | Accepted by the reader. The torque at the root is under-estimated and the bending moment is over-estimated (reader, Appendix A.2). |
| A-6 | The shear centre is at the box centroid, at the constant chord fraction `f_sc`. | One value for the whole span. | The centroid varies with `y` in WP4.2. Replace `f_sc` after WP4.2. |
| A-7 | The wing structure weight is neglected. The fuel per unit span scales with `c²`. All fuel is in the wing tank. | `w'_str = 0`. Section 6 formulas. | Conservative for shear and bending: the structure weight relieves the wing loading. For the torque it has no effect, because `f_str = f_sc`. |
| A-11 | The fuel acts at the centroid of the wing box (`f_fuel = f_sc`). | The fuel gives no torque. | True for the fuel, which fills the box. |
| A-12 | The engine centre of gravity is 48 % of the nacelle length behind the inlet. It follows from the position in the WP3 report, read by eye from the drawing. | Sets the engine arm `d`. | The report does not define the frame origin. The engine term of the torque changes in proportion to `d`. |
| A-8 | The engine, its pylon and its thrust act at one span station. The thrust line is at `h_thrust` below the shear centre. | Point loads in scripts 3 and 4. | Reader, Appendix A.3 gives the thrust direction. |
| A-9 | The axial force is not computed. | No axial force diagram. | The reader lists the axial force in deliverable (a). The report states that this design omits it. |
| A-10 | `lift.csv` holds the VSPAERO strips moved to the global stations (monotone cubic, 0 at the tip). Scripts 3 and 4 interpolate linearly between stations. | `scipy.integrate` runs on a continuous function. | The 2° dihedral projection of `y` is neglected (`cos 2° = 0.9994`). The check in section 5 shows the effect (−0.14 % at MTOW cruise). |

---

## 12. Run order

```text
python wp4/scripts/wp41_aero_loads.py          (once per geometry or Mach change)
python wp4/scripts/wp41_lift.py   --case LC-1
python wp4/scripts/wp41_aoa_grid.py            (optional: CL and alpha over weights and speeds)
python wp4/scripts/wp41_weight.py --case LC-1
python wp4/scripts/wp41_torsion.py       --case LC-1 LC-2
python wp4/scripts/wp41_shear_moment.py  --case LC-1 LC-2
```

Always call Python through the project environment (`<env>\python.exe`).
The two cases of scripts 3 and 4 are the critical positive-`n` and negative-`n` cases from WP4.3.
