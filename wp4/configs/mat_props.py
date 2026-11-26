# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Rodrigo and Yajvin
# Created: 2026-11-26
"""Material properties configuration for Aluminum 2024-T81 (WP4 & WP5)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IsotropicMaterial:
    """
    Isotropic linear-elastic material properties in SI base units (kg, m, s, Pa).

    Primary mechanical design allowables are B-basis values for bare AL2024-T81
    sheet/plate (thickness 1.60–6.32 mm / 0.063–0.249 in, longitudinal grain
    direction) from MMPDS-01 / MIL-HDBK-5J, Table 3.2.3.0(b), and physical
    constants from ASM Handbook Vol. 2.
    """

    name: str
    density_kgm3: float          # rho [kg/m^3]
    youngs_modulus_pa: float     # E [Pa] (tensile elastic modulus)
    shear_modulus_pa: float      # G [Pa]
    poissons_ratio: float        # nu [-]
    tensile_yield_pa: float      # F_ty [Pa] (WP5.2 tension / yielding check)
    tensile_ultimate_pa: float   # F_tu [Pa] (WP5.2 ultimate tension check)
    compressive_yield_pa: float  # F_cy [Pa] (WP5.1 compression / column buckling)
    shear_ultimate_pa: float     # F_su [Pa] (WP5.1 web shear strength)


AL2024_T81 = IsotropicMaterial(
    name="AL2024-T81",
    # Physical properties (ASM Handbook Vol. 2; MMPDS-01 Sec. 3.2.3.0)
    density_kgm3=2780.0,           # 0.100 lb/in^3
    youngs_modulus_pa=72.4e9,      # 10.5e6 psi (tensile); compressive E_c = 73.8 GPa
    shear_modulus_pa=27.6e9,       # 4.0e6 psi (often rounded to 28.0 GPa in literature)
    poissons_ratio=0.33,           # Standard wrought 2024 aluminum value
    # B-basis strength allowables, L-direction (MMPDS-01 Table 3.2.3.0(b))
    tensile_yield_pa=407.0e6,      # 59 ksi (typical ASM handbook value: 450 MPa)
    tensile_ultimate_pa=462.0e6,   # 67 ksi (typical ASM handbook value: 485 MPa)
    compressive_yield_pa=400.0e6,  # 58 ksi
    shear_ultimate_pa=276.0e6,     # 40 ksi (typical ASM handbook value: 295 MPa)
)

# Active project material alias (mirrors `from configs.aircraft import WING`)
MATERIAL = AL2024_T81