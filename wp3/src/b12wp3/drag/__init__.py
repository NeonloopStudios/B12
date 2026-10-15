# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Class II drag estimation: the ADSEE-II component drag build-up.

CD0 = (1 / S_ref) * sum_c( Cf_c * FF_c * IF_c * S_wet,c ) + sum CD_misc

wetted_area    S_wet of the wing, the tails and the bodies
skin_friction  Reynolds number with roughness cutoff, Cf laminar /
               turbulent / mixed
form_factor    FF of lifting surfaces, fuselage and nacelles
buildup        one ComponentDrag per component, summed to CD0
"""
