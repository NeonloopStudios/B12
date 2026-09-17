<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026 Kajetan R. Gulaj
Created: 2026-09-17
-->

# WP2 — Airfoil Selection: XFoil Analysis Plan

Goal: score the candidate 2D airfoil sections against the WP2 selection matrix (Mach
critical, Cl/Cd cruise, Cl max landing, stall margin, zero-angle Cl, pitching moment)
using XFoil, with full polar data, breakdown/non-convergence diagnostics, and
matplotlib figures for every airfoil, so the weighted scorecard is computed from
real runs instead of filled in by hand.

Airfoils are **not fixed** — the set in [`wp2/airfoils/`](airfoils/) will change as the
selection narrows. Every script in this plan reads `*.dat` files from that folder
by discovery (no hardcoded airfoil list), so dropping in a new `.dat` file and
re-running the pipeline is the only step needed to re-score a new candidate.

Toolchain setup (compiling XFoil's Python bindings from source on Windows) is
documented separately in [`wp2/README.md`](README.md) — this file is the analysis
plan, not the install guide.

## 0. Scoring matrix (the target deliverable)

| Criterion | Weight |
|---|---|
| Mach Critical | 0.25 |
| Cl/Cd cruise | 0.30 |
| Cl max landing | 0.15 |
| Stall Angle − Angle in Cruise | 0.125 |
| Zero Angle Cl | 0.05 |
| Pitching Moment | 0.125 |

`wp2/scripts/build_scorecard.py` (Stage 7) reproduces this exact table, per
airfoil, computed from XFoil output rather than entered manually.

## 1. Flight conditions → 2D section conditions (sweep theory)

XFoil is a 2D panel/viscous code; the wing is swept, so every condition fed to it
first goes through simple sweep theory (infinite yawed wing approximation):

- Λ = 0.419271315 rad = 24.02°, cos Λ = 0.9135
- Cruise altitude 35,000 ft (ISA): T ≈ 218.8 K, a ≈ 296.5 m/s, ρ ≈ 0.380 kg/m³
- V∞ = M·a = 0.77 × 296.5 ≈ 228.4 m/s
- Effective normal Mach seen by the section: **M_n = M∞·cos Λ ≈ 0.703**
- Effective normal dynamic pressure: q_n = q∞·cos²Λ, so the section's required Cl is
  higher than the wing's design Cl: **Cl_n = Cl_wing / cos²Λ**
- Reynolds number: use the normal chord (c_n = c_streamwise·cos Λ) and
  V_n = V∞·cos Λ for **Re_n**

M_n ≈ 0.70 is the number actually fed to XFoil for the cruise polars — not 0.77 —
and it sits right at the edge of where a panel + integral boundary-layer method
stops being trustworthy, which is the core tension the rest of this plan manages
(Stage 4 and the validity flags in Stage 5).

The landing criterion (Cl max landing) uses a **separate, low-Mach condition**
(effectively incompressible, M ≈ 0.1–0.2) at sea-level/approach altitude and the
approach Reynolds number. Those numbers depend on the aircraft's approach speed
and wing MAC from the performance/sizing work package and are not yet fixed —
placeholders live in `wp2/scripts/config.py` (`LANDING` block) with a `# TODO`
pointing at the sizing WP; nothing here invents them.

This module (sweep theory + both flight conditions) lives in
`wp2/scripts/config.py` as the single source of truth: `CRUISE` and `LANDING`
dataclasses with M∞, altitude, Λ, chord, Ncrit, plus the derived `M_n`, `Re_n`,
`Cl_n`. Every other script imports from here — conditions are set once.

## 2. Stage-by-stage pipeline

### Stage 1 — `config.py`
Aircraft/mission constants, sweep-theory derivation (§1), airfoil auto-discovery
(`glob(wp2/airfoils/*.dat)`), and the scoring weights table (§0) as data, not
prose, so Stage 7 reads it instead of re-typing it.

### Stage 2 — `xfoil_runtime.py`
Thin wrapper around the compiled `xfoil` Python package:
- Calls `os.add_dll_directory(...)` for the MinGW runtime, and fixes a ctypes
  `FreeLibrary` argtypes bug that leaked a handle + temp file per `XFoil()`
  instance — both **before** `import xfoil` (see README — required on Python
  ≥3.8/Windows, PATH alone is not enough for the first; the second is a
  vendored-package bug hit directly in this environment).
- `load_airfoil_dat(path) -> Airfoil` — handles a name header line, the
  single-loop Selig format, and the two-block Lednicer format (blank-line
  separated upper/lower surfaces, merged into one continuous TE→LE→TE loop);
  validates the assembled geometry actually closes into a loop before
  returning it.
- `run_alpha_sweep(xf, alphas, stop_after_n_nonconverged) -> DataFrame` —
  iterates alpha one point at a time (not XFoil's built-in `aseq`) so each
  point's convergence is checked and recorded individually, with an early
  stop after N consecutive failures. Columns include `diverged`/`rms_bl` from
  the `externals/xfoil-python` Fortran patch (real solver state: NaN-abort
  vs. simply ran out of iterations, and the actual residual) — not XFoil's
  own printed diagnostic text, which is not capturable via Python (verified;
  see README).
- `run_two_leg_polar(airfoil, ..., alpha_low_deg, alpha_high_deg) -> DataFrame`
  — two independent `run_alpha_sweep` calls warm-started from α=0 (one up, one
  down), each its own fresh session, merged. Added when Stage 3's naive
  single-direction cold sweep came back completely empty (see Stage 3);
  shared by every polar-generating stage from Stage 3 onward.

### Stage 3 — `run_cruise_polars.py`
For each airfoil: `xfoil_runtime.run_two_leg_polar` at `M_n`, `Re_n`, `Ncrit`
from `CRUISE` — two independent sweeps warm-started from α=0 (one up, one
down), not a single cold pass from one extreme. A cold start directly at a
harsh angle at this Mach reliably diverges for several consecutive points
before `reset_bls()` recovers it — verified directly: at this sweep's 0.25°
resolution that burns the whole non-convergence budget before ever reaching a
point that would actually converge, so a naive single-direction sweep came
back completely empty for every airfoil until this was fixed. Writes
`wp2/results/data/<airfoil>_cruise_polar.csv` with columns
`alpha, cl, cd, cm, cp_min, converged, diverged, rms_bl, note` (Stage 2's
`run_alpha_sweep` output as-is — no `Cdp` column: the compiled binding only
exposes total `cd`, not the pressure/friction split, so this plan no longer
claims one).

From this polar, directly extract: zero-angle Cl, stall angle & Cl_max (cruise
Re/M), and — by finding α where Cl(α) = Cl_n — the cruise operating point (α, Cl,
Cd, Cm, Cl/Cd) that feeds the scorecard. At cruise Mach the sweep's own
non-convergence is the real stall signal (transonic breakdown genuinely stops
the BL solver, verified: NACA 25112 converges to exactly 5.00°, not further)
— unlike Stage 4 below, `polar_analysis.find_cl_max`'s extra stall-onset
detection is not needed here.

### Stage 4 — `run_landing_polars.py`
Same mechanics as Stage 3 (`run_two_leg_polar`), at the `LANDING` condition
(§1), clean (flaps-up, since no flap geometry is defined yet), swept wider
(−8° to 25°, vs. cruise's −6° to 20°) since low-speed stall angles run
noticeably higher than the transonic cruise polar's. Writes
`wp2/results/data/<airfoil>_landing_polar.csv`. Cl_max from this run is the
"Cl max landing" scorecard entry — it is deliberately a separate run from cruise
because Cl_max depends on Re/M, not just the airfoil.

**Cl_max here is not `converged['cl'].max()`.** Verified directly: at this
project's landing condition (M_n≈0.18, Re_n≈10.3M — nearly incompressible, no
transonic breakdown mechanism), XFoil's BL solver keeps numerically converging
all the way to α=40°, with Cl dropping from a real peak (1.80 at 17.25° for
NACA 25112) down to a non-physical plateau (~0.75) and *staying* converged
there — a known XFoil limitation, not something the `converged` flag catches.
`wp2/scripts/polar_analysis.find_cl_max` (pulled forward from Stage 7 out of
necessity — Stage 4's own sanity check needed it to report a trustworthy
number, not a hypothetical future one) detects the first *sustained* drop in
Cl(α) while sweeping upward from α=0 and reports the peak up to that point,
instead of trusting convergence alone. Stage 7 must reuse this same function
for its own Cl_max_landing extraction, not re-derive it.

### Stage 5 — `mcrit_sweep.py`
The actual "Mach Critical" deliverable, and the standard, legitimate use of a
panel method at transonic conditions (it flags an oncoming shock, it does not
resolve one):
1. Run XFoil at a low, safely subsonic Mach (M = 0.2) at the cruise-relevant Cl
   (Cl_n from Stage 3) to get the incompressible Cp distribution.
2. Apply a Karman–Tsien compressibility correction to Cp and scale the result up
   through a Mach grid.
3. Cp_min(M) is compared against Cp_crit(M) (sonic condition, function of M only)
   at each step; **M_crit** is the Mach where they cross.
4. Cross-check with the Korn equation for drag-divergence Mach:
   `M_dd + t/c + Cl/10 = κ_A` (κ_A ≈ 0.87 conventional, ≈0.95 supercritical),
   giving `M_dd` and margin `M_dd − M_n` per airfoil — a defensible *comparative*
   wave-drag ranking without CFD, consistent with the M_crit numbers from step 3.

Writes `wp2/results/data/<airfoil>_mcrit.csv` (Mach, Cp_min_corrected, Cp_crit)
and appends to `wp2/results/data/mcrit_summary.csv`
(M_crit, M_dd, margin, t/c, Cl_cruise) — one row per airfoil.

### Stage 6 — validity flagging (not a separate script — applied in Stage 7)
Carried into the scorecard, not silently dropped:
- If `M_n (0.703) > M_crit` for an airfoil, its cruise Cd is flagged
  **not physically trustworthy** (quoted only as a lower bound); the Korn-derived
  margin is used for ranking instead of raw Cd.
- Cl_max/stall numbers are least reliable near the point the boundary-layer model
  breaks down (separation) — non-convergence clustering near M_crit indicates
  shock-induced separation XFoil cannot model; Stage 3/4's `converged`/`note`
  columns surface exactly where that starts for each airfoil.
- Sweep theory itself ignores finite-span/3D effects — this whole pipeline is a
  2D screening/ranking tool, not final wing verification. Stated once here, not
  re-derived per airfoil.

### Stage 7 — `build_scorecard.py`
Reads every `results/data/*.csv`, extracts the six criteria (§0) per airfoil,
applies the validity flags (Stage 6), normalizes each criterion 0–1 across the
candidate set, applies the weights, and writes
`wp2/results/scorecard.csv` + prints the ranked table (same shape as the
screenshot table, with numbers instead of blanks, plus a `validity_notes` column).

### Stage 8 — `make_plots.py`
Matplotlib figures, per airfoil into `wp2/results/plots/<airfoil>/`, plus
cross-airfoil comparisons into `wp2/results/plots/comparison/`:

Per airfoil (cruise + landing data combined):
1. Cl vs α — full polar, stall point marked, cruise α marked, non-converged tail
   shown as a distinct marker/color (the "where it breaks" plot).
2. Cd vs α and the Cl–Cd drag polar, cruise operating point marked.
3. Cl/Cd vs α, cruise value annotated.
4. Cm vs α (pitching moment), cruise value annotated.
5. Cp distribution at the cruise α (from the M = 0.2 run), to show shape context
   for the M_crit determination.
6. Cp_min(M) vs Cp_crit(M) — the Karman–Tsien sweep from Stage 5, M_crit read
   directly off the crossing point ("why it breaks" for the transonic criterion).
7. Landing Cl vs α, Cl_max landing marked.

Cross-airfoil:
8. Grouped bar chart of the six normalized/weighted criteria per airfoil (mirrors
   the screenshot table visually).
9. Total weighted score bar chart, ranked.

Plot styling follows the project's chart-design conventions (consistent palette,
readable in the report) — applied when this script is implemented, not before.

### Stage 9 — `run_all.py`
Orchestrates Stages 3–8 in order for every airfoil currently in
`wp2/airfoils/`, so re-scoring a swapped-in airfoil is one command.

## 3. Output tree

```
wp2/results/
  data/
    <airfoil>_cruise_polar.csv
    <airfoil>_landing_polar.csv
    <airfoil>_mcrit.csv
    mcrit_summary.csv
  scorecard.csv
  plots/
    <airfoil>/  (figures 1-7)
    comparison/ (figures 8-9)
```

## 4. Open inputs

Resolved:
- Chord (`config.CHORD_M = 2.649904207 m`) — same physical wing station for
  cruise and landing.
- **Cruise required Cl (`CRUISE.cl_wing = 0.489433403`)** — from WP1's
  level-flight trim calc (`Cl = 2W/(ρV²S)`, eq. 8.13). `Cl_n` (the sweep-corrected
  section Cl XFoil's polar is matched against) is `cl_wing / cos²Λ ≈ 0.587`.
  Landing needs no equivalent — "Cl max landing" is the polar's Cl_max at the
  landing Re/M, not a trim point.
- **Landing condition** — confirmed sea level, confirmed 65 m/s approach speed
  (`config.LANDING_SPEED_MS`); `LANDING.mach_freestream` is derived from it
  (`65 / a(sea level)`), not a separate guess.
- **`CRUISE.ncrit = 8`, `LANDING.ncrit = 8`** — both confirmed.
- **Korn equation κ_A per airfoil** (`config.kappa_a`) — confirmed:
  NASA_SC(2)-0712 (supercritical, by design) → 0.95; the other three
  (conventional) → 0.87. Raises if a future airfoil swap adds an unmapped
  candidate, rather than silently assuming "conventional".

No open placeholders remain in `config.py` as of this writing — every
`FlightCondition` field on `CRUISE` and `LANDING` is a confirmed value or a
value derived from one.

## 5. Status

Stages 1–7 implemented and run for all 4 candidates (`config.py`,
`xfoil_runtime.py`, `run_cruise_polars.py`, `run_landing_polars.py`,
`mcrit_sweep.py`, `build_scorecard.py`, plus supporting modules
`polar_analysis.py`, `geometry.py`, `compressibility.py`). Stages 8–9
(`make_plots.py`, `run_all.py`) not yet built.

Stage 5 results (M_n=0.7033 cruise): **all four candidates currently have
M_crit below cruise M_n** (0.525–0.555 vs. 0.703) — every cruise Cd in the
Stage 3 polars is in the "not physically trustworthy" territory Stage 6
already anticipates, and the Korn-derived margin (M_dd − M_n) is the number
to actually rank on. Only NASA_SC(2)-0712 (the supercritical section) has a
positive margin (+0.068); the three conventional sections are all
essentially at or just past their drag-divergence Mach at this cruise
condition (margins −0.002 to −0.012) — consistent with what a supercritical
section is specifically designed to do, not a coincidence in the numbers.

Stage 7 scorecard (weighted total, ranked):

| Airfoil | Total score |
|---|---|
| NACA_25112 | 0.7085 |
| NASA_SC(2)-0712 | 0.4892 |
| lockheed_c5a_bl758 | 0.4594 |
| NACA_64212 | 0.3866 |

Scoring direction for the two ambiguous criteria confirmed directly (not
assumed): pitching moment scores on smaller `|Cm|` (not simply higher or
lower), zero-angle Cl scores higher-is-better. `pitching_moment` and
`cl_cd_cruise` (0.125 + 0.30 = 0.425 combined weight) are the biggest swing
factors in NACA_25112's lead — it has the best (smallest-magnitude) Cm and
a strong Cl/Cd, while NASA_SC(2)-0712 trades that off for the best Mach
Critical and Cl max landing scores. A real bug was caught building this
stage: `polar_analysis.interpolate_at_cl` initially restricted its search
to `alpha >= 0` (copied from `find_cl_max`'s stall-detection restriction,
which doesn't apply here) and raised a false "not bracketed" error for
NASA_SC(2)-0712, whose cruise operating point genuinely falls at a negative
alpha (≈−0.99°) due to its heavy camber.
