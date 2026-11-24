# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Javier Cortina
# Created: 2026-11-24
"""WP4.3: (V_EAS-n) manoeuvre diagrams.

One diagram per mass (MTOM, MZFM, OEM) and altitude (configs/wp43/settings.py).
Masses and wing area from configs/aircraft.py. Writes one PDF per diagram to
wp4/out/wp43/.

    python wp4/scripts/wp43_vn.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_WP4_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_WP4_DIR), str(_WP4_DIR / "src")]

from configs.aircraft import MASSES, WING  # noqa: E402
from configs.wp43 import settings  # noqa: E402
from wp43.atmosphere import isa  # noqa: E402
from wp43.vn_diagram import VnEnvelope, manoeuvre_envelope, vn_speeds  # noqa: E402

logger = logging.getLogger("wp43_vn")


def plot_vn(env: VnEnvelope, title: str, path: Path) -> None:
    """Plot one V-n diagram and save it to path.

    Clean envelope (solid, lightly filled), flaps-down curve (dashed), the
    design speeds as dotted vertical lines and the corner points labelled.
    """
    vs = env.speeds
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for label, v in (
        ("$V_{S0}$", vs.v_s0), ("$V_{S1}$", vs.v_s1), ("$V_A$", vs.v_a),
        ("$V_F$", vs.v_f), ("$V_C$", vs.v_c), ("$V_D$", vs.v_d),
    ):
        ax.axvline(v, color="0.75", linestyle=":", linewidth=0.8, zorder=1)
        ax.annotate(
            label, (v, settings.N_MIN - 0.45), xytext=(2, 0),
            textcoords="offset points", fontsize=8, color="0.35",
        )
    ax.axhline(0.0, color="0.0", linewidth=0.6, zorder=1)

    ax.fill(env.v_eas, env.n, color="C0", alpha=0.08, zorder=2)
    ax.plot(env.v_eas, env.n, color="C0", linewidth=1.6, label="Flaps up", zorder=3)
    ax.plot(
        env.v_flap, env.n_flap, color="C1", linewidth=1.4, linestyle="--",
        label="Flaps down", zorder=3,
    )
    for corner in env.corners:
        ax.plot(corner.v_eas_ms, corner.n, "o", color="C0", markersize=4, zorder=4)
        ax.annotate(
            corner.name, (corner.v_eas_ms, corner.n), xytext=(4, 4),
            textcoords="offset points", fontsize=9, fontweight="bold",
        )

    ax.set_xlim(0.0, 1.1 * vs.v_d)
    ax.set_ylim(settings.N_MIN - 0.5, vs.n_max + 0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("$V_{EAS}$ [m/s]")
    ax.set_ylabel("$n$ [-]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    """Plot the V-n diagram of every mass and altitude."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    settings.OUT_DIR.mkdir(parents=True, exist_ok=True)
    masses = {"mtom": MASSES.mtom_kg, "mzfm": MASSES.mzfm_kg, "oem": MASSES.oem_kg}
    for name, mass_kg in masses.items():
        for altitude_m in settings.ALTITUDES_M:
            vs = vn_speeds(
                mass_kg, MASSES.mtom_kg, WING.area_m2, settings.CL, settings.SPEEDS,
                isa(altitude_m),
            )
            title = f"{name.upper()} ({mass_kg:.0f} kg), h = {altitude_m:.0f} m"
            path = settings.OUT_DIR / f"vn_{name}_{altitude_m:.0f}m.pdf"
            env = manoeuvre_envelope(vs)
            plot_vn(env, title, path)
            corners = ", ".join(
                f"{c.name} ({c.v_eas_ms:.1f} m/s, {c.n:+.2f})" for c in env.corners
            )
            logger.info("wrote %s: %s", path, corners)


if __name__ == "__main__":
    main()
