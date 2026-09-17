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
- Calls `os.add_dll_directory(...)` for the MinGW runtime **before** `import xfoil`
  (see README — required on Python ≥3.8/Windows, PATH alone is not enough).
- `load_airfoil(path) -> Airfoil` (handles both headerless Selig-format files and
  files with a name header line, since the current set mixes both).
- `run_alpha_sweep(airfoil, M, Re, Ncrit, alphas) -> DataFrame` — iterates alpha
  one point at a time (not XFoil's built-in ASeq) so each point's convergence can
  be checked and recorded individually instead of silently skipped.
- Every result row carries a `converged: bool` and, on failure, the reason XFoil
  reported (BL calculation failed to converge, max iterations hit, etc.) — this is
  what makes "where it breaks and why" a data column instead of a footnote.

### Stage 3 — `run_cruise_polars.py`
For each airfoil: alpha sweep at `M_n`, `Re_n`, `Ncrit` from `CRUISE`, from a few
degrees below zero-lift through well past stall (sweep continues even after
convergence starts failing, to record *how* it breaks, then stops after N
consecutive non-converged points). Writes
`wp2/results/data/<airfoil>_cruise_polar.csv` with columns
`alpha, Cl, Cd, Cdp, Cm, converged, note`.

From this polar, directly extract: zero-angle Cl, stall angle & Cl_max (cruise
Re/M), and — by finding α where Cl(α) = Cl_n — the cruise operating point (α, Cl,
Cd, Cm, Cl/Cd) that feeds the scorecard.

### Stage 4 — `run_landing_polars.py`
Same mechanics as Stage 3, at the `LANDING` condition (§1), clean (flaps-up,
since no flap geometry is defined yet). Writes
`wp2/results/data/<airfoil>_landing_polar.csv`. Cl_max from this run is the
"Cl max landing" scorecard entry — it is deliberately a separate run from cruise
because Cl_max depends on Re/M, not just the airfoil.

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

## 4. Open inputs needed before Stage 4/7 can run for real

- Landing condition: approach speed, altitude, wing MAC (→ Re_landing) — from the
  performance/sizing WP, not assumed here.
- Ncrit (turbulence/surface quality assumption) for cruise vs. landing — a
  placeholder value is set in `config.py`, to be confirmed against the assumed
  manufacturing/surface finish quality.
- κ_A in the Korn equation (conventional vs. supercritical) per airfoil, if the
  candidate set includes a supercritical section.

## 5. Status

Environment verified working (see `wp2/README.md`): XFoil's Python bindings are
compiled and load correctly, including the Windows DLL-loading fix. Stages 1–9
above are the implementation plan; `wp2/scripts/` will be built out stage by
stage against this document.
