# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Component drag build-up (ADSEE-II, "The component drag build-up method").

    CD0 = (1 / S_ref) * sum_c( Cf_c * FF_c * IF_c * S_wet,c ) + sum CD_misc

Characteristic length l of each component (Reynolds number):

* wing, tails, pylon: mean aerodynamic chord
* fuselage, nacelle: total body length L1 + L2 + L3

The reference area S_ref is the wing planform area.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from b12wp3.config.drag import InterferenceFactors, LaminarFraction
from b12wp3.config.geometry import AircraftGeometry, BodyInputs, LiftingSurfaceInputs
from b12wp3.config.mission import FlightCondition
from b12wp3.drag.form_factor import (
    fineness_ratio,
    fuselage_form_factor,
    lifting_surface_form_factor,
    nacelle_form_factor,
)
from b12wp3.drag.skin_friction import (
    cf_mixed,
    effective_reynolds,
    reynolds_actual,
    reynolds_cutoff,
)
from b12wp3.drag.wetted_area import body_wetted_area, tail_wetted_area, wing_wetted_area

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentDrag:
    """Drag terms of one component type.

    Attributes:
        name: Component label.
        count: Number of identical items (for example nacelles).
        wetted_area_m2: Wetted area of one item [m^2].
        reference_length_m: Characteristic length l [m].
        reynolds: Reynolds number used in Cf [-].
        laminar_fraction: Laminar fraction x_lam [-].
        skin_friction: Cf [-].
        form_factor: FF [-].
        interference_factor: IF [-].
    """

    name: str
    count: int
    wetted_area_m2: float
    reference_length_m: float
    reynolds: float
    laminar_fraction: float
    skin_friction: float
    form_factor: float
    interference_factor: float

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"{self.name}: count must be >= 1, got {self.count}")
        if self.interference_factor <= 0.0:
            raise ValueError(
                f"{self.name}: interference_factor must be > 0, "
                f"got {self.interference_factor}"
            )

    @property
    def drag_area_m2(self) -> float:
        """count * Cf * FF * IF * S_wet [m^2]."""
        return (
            self.count
            * self.skin_friction
            * self.form_factor
            * self.interference_factor
            * self.wetted_area_m2
        )


@dataclass(frozen=True)
class DragBuildup:
    """Zero-lift drag coefficient from its components.

    Attributes:
        components: Drag terms of each component type.
        reference_area_m2: Reference area S_ref [m^2].
        cd_misc: Sum of the miscellaneous drag coefficients [-].
    """

    components: tuple[ComponentDrag, ...]
    reference_area_m2: float
    cd_misc: float

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("DragBuildup needs at least one component")
        if self.reference_area_m2 <= 0.0:
            raise ValueError(
                f"reference_area_m2 must be > 0, got {self.reference_area_m2}"
            )
        if self.cd_misc < 0.0:
            raise ValueError(f"cd_misc must be >= 0, got {self.cd_misc}")

    def contribution(self, component: ComponentDrag) -> float:
        """CD0 contribution of one component type [-]."""
        return component.drag_area_m2 / self.reference_area_m2

    @property
    def cd0(self) -> float:
        """Total zero-lift drag coefficient [-]."""
        return sum(self.contribution(c) for c in self.components) + self.cd_misc

    def to_dataframe(self) -> pd.DataFrame:
        """One row per component, then a miscellaneous row and a total row."""
        nan = math.nan
        rows: list[tuple[str | int | float, ...]] = [
            (
                c.name,
                c.count,
                c.wetted_area_m2,
                c.reference_length_m,
                c.reynolds,
                c.laminar_fraction,
                c.skin_friction,
                c.form_factor,
                c.interference_factor,
                c.drag_area_m2,
                self.contribution(c),
            )
            for c in self.components
        ]
        no_terms = (nan,) * 8
        rows.append(("miscellaneous", 1, *no_terms, self.cd_misc))
        rows.append(("total", 1, *no_terms, self.cd0))
        frame = pd.DataFrame(
            rows,
            columns=[
                "component",
                "count",
                "wetted_area_m2",
                "reference_length_m",
                "reynolds",
                "laminar_fraction",
                "cf",
                "form_factor",
                "interference_factor",
                "drag_area_m2",
                "cd0",
            ],
        )
        frame["share_pct"] = 100.0 * frame["cd0"] / self.cd0
        return frame


def component_drag(
    name: str,
    *,
    count: int,
    wetted_area_m2: float,
    reference_length_m: float,
    form_factor: float,
    interference_factor: float,
    laminar_fraction: float,
    condition: FlightCondition,
    roughness_m: float,
) -> ComponentDrag:
    """Skin friction of one component, combined with its other drag terms.

    Args:
        name: Component label.
        count: Number of identical items.
        wetted_area_m2: Wetted area of one item [m^2].
        reference_length_m: Characteristic length l [m].
        form_factor: FF [-].
        interference_factor: IF [-].
        laminar_fraction: Laminar fraction x_lam [-].
        condition: Flight condition.
        roughness_m: Equivalent sand roughness k [m].

    Returns:
        The component drag terms.
    """
    actual = reynolds_actual(
        condition.density_kg_m3,
        condition.velocity_m_s,
        reference_length_m,
        condition.viscosity_pa_s,
    )
    cutoff = reynolds_cutoff(
        reference_length_m, roughness_m, condition.mach, condition.regime
    )
    reynolds = effective_reynolds(actual, cutoff)
    cf = cf_mixed(reynolds, condition.mach, laminar_fraction)
    logger.debug(
        "%s: Re_actual=%.4g Re_cutoff=%.4g Cf=%.5g FF=%.4f",
        name,
        actual,
        cutoff,
        cf,
        form_factor,
    )
    return ComponentDrag(
        name=name,
        count=count,
        wetted_area_m2=wetted_area_m2,
        reference_length_m=reference_length_m,
        reynolds=reynolds,
        laminar_fraction=laminar_fraction,
        skin_friction=cf,
        form_factor=form_factor,
        interference_factor=interference_factor,
    )


def lifting_surface_drag(
    name: str,
    surface: LiftingSurfaceInputs,
    *,
    wetted_area_fn: Callable[[float], float],
    count: int,
    interference_factor: float,
    laminar_fraction: float,
    condition: FlightCondition,
    roughness_m: float,
) -> ComponentDrag:
    """Drag terms of a wing, tail or pylon.

    Args:
        name: Component label.
        surface: Geometric inputs of the surface.
        wetted_area_fn: S_exp -> S_wet (wing or tail formula).
        count: Number of identical surfaces.
        interference_factor: IF [-].
        laminar_fraction: Laminar fraction x_lam [-].
        condition: Flight condition.
        roughness_m: Equivalent sand roughness k [m].

    Returns:
        The component drag terms. l is the mean aerodynamic chord.
    """
    planform = surface.planform()
    wetted = wetted_area_fn(planform.exposed_area(surface.covered_span_m))
    form_factor = lifting_surface_form_factor(
        surface.thickness_to_chord,
        surface.x_c_max_thickness,
        condition.mach,
        planform.sweep_at(surface.x_c_max_thickness),
    )
    return component_drag(
        name,
        count=count,
        wetted_area_m2=wetted,
        reference_length_m=planform.mac_m,
        form_factor=form_factor,
        interference_factor=interference_factor,
        laminar_fraction=laminar_fraction,
        condition=condition,
        roughness_m=roughness_m,
    )


def body_drag(
    name: str,
    body: BodyInputs,
    *,
    form_factor_fn: Callable[[float], float],
    count: int,
    interference_factor: float,
    laminar_fraction: float,
    condition: FlightCondition,
    roughness_m: float,
) -> ComponentDrag:
    """Drag terms of a fuselage or nacelle.

    Args:
        name: Component label.
        body: Geometric inputs of the body.
        form_factor_fn: Fineness ratio -> FF (fuselage or nacelle formula).
        count: Number of identical bodies.
        interference_factor: IF [-].
        laminar_fraction: Laminar fraction x_lam [-].
        condition: Flight condition.
        roughness_m: Equivalent sand roughness k [m].

    Returns:
        The component drag terms. l is the total body length.
    """
    wetted = body_wetted_area(
        body.diameter_m,
        body.nose_length_m,
        body.cylinder_length_m,
        body.tail_length_m,
    )
    max_frontal_area = math.pi * body.diameter_m**2 / 4.0
    fineness = fineness_ratio(body.length_m, max_frontal_area)
    return component_drag(
        name,
        count=count,
        wetted_area_m2=wetted,
        reference_length_m=body.length_m,
        form_factor=form_factor_fn(fineness),
        interference_factor=interference_factor,
        laminar_fraction=laminar_fraction,
        condition=condition,
        roughness_m=roughness_m,
    )


def aircraft_drag_buildup(
    geometry: AircraftGeometry,
    condition: FlightCondition,
    *,
    interference: InterferenceFactors,
    cd_misc: float,
    laminar: LaminarFraction,
    roughness_m: float,
) -> DragBuildup:
    """CD0 build-up of the full aircraft.

    Components: wing, horizontal tail, vertical tail, fuselage, and one
    nacelle and one pylon per engine. Nacelles use the fuselage laminar
    fraction; the pylons use the wing-and-tail laminar fraction.

    Args:
        geometry: All geometric inputs.
        condition: Flight condition.
        interference: Interference factor of each component.
        cd_misc: Sum of the miscellaneous drag coefficients [-].
        laminar: Laminar fractions of the selected aircraft category.
        roughness_m: Equivalent sand roughness k [m].

    Returns:
        The drag build-up. S_ref is the wing planform area.
    """
    lifting = laminar.wing_and_tail
    engines = geometry.engine_count
    components = (
        lifting_surface_drag(
            "wing",
            geometry.wing,
            wetted_area_fn=wing_wetted_area,
            count=1,
            interference_factor=interference.wing,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness_m,
        ),
        lifting_surface_drag(
            "horizontal_tail",
            geometry.horizontal_tail,
            wetted_area_fn=tail_wetted_area,
            count=1,
            interference_factor=interference.horizontal_tail,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness_m,
        ),
        lifting_surface_drag(
            "vertical_tail",
            geometry.vertical_tail,
            wetted_area_fn=tail_wetted_area,
            count=1,
            interference_factor=interference.vertical_tail,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness_m,
        ),
        body_drag(
            "fuselage",
            geometry.fuselage,
            form_factor_fn=fuselage_form_factor,
            count=1,
            interference_factor=interference.fuselage,
            laminar_fraction=laminar.fuselage,
            condition=condition,
            roughness_m=roughness_m,
        ),
        body_drag(
            "nacelle",
            geometry.nacelle,
            form_factor_fn=nacelle_form_factor,
            count=engines,
            interference_factor=interference.nacelle,
            laminar_fraction=laminar.fuselage,
            condition=condition,
            roughness_m=roughness_m,
        ),
        lifting_surface_drag(
            "pylon",
            geometry.pylon,
            wetted_area_fn=tail_wetted_area,
            count=engines,
            interference_factor=interference.pylon,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness_m,
        ),
    )
    buildup = DragBuildup(
        components=components,
        reference_area_m2=geometry.wing.area_m2,
        cd_misc=cd_misc,
    )
    logger.info("Class II drag at %s: CD0 = %.5f", condition.name, buildup.cd0)
    return buildup
