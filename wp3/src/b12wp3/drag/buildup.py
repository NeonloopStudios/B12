# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Component drag build-up (ADSEE-II, "The component drag build-up method").

    CD0 = (1 / S_ref) * sum_c( Cf_c * FF_c * IF_c * S_wet,c ) + sum CD_misc

Miscellaneous terms (b12wp3.drag.miscellaneous), per configuration:

* every configuration: wing wave drag, fuselage upsweep and base drag
* flaps: trailing-edge and leading-edge flap drag (zero at 10 deg or less)
* gear down: main and nose landing gear drag
* last: excrescence and leakage, a fraction of the final total CD0

Characteristic length l of each component (Reynolds number):

* wing, tails: mean aerodynamic chord
* fuselage: total body length L1 + L2 + L3; nacelle: nacelle length l_n

The reference area S_ref is the wing planform area.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from b12wp3.config.drag import DragSettings
from b12wp3.config.geometry import (
    AircraftGeometry,
    BodyInputs,
    FlapInputs,
    GearInputs,
    LiftingSurfaceInputs,
    NacelleInputs,
)
from b12wp3.config.mission import DragConfiguration, FlightCondition
from b12wp3.drag.form_factor import (
    fineness_ratio,
    fuselage_form_factor,
    lifting_surface_form_factor,
    nacelle_form_factor,
)
from b12wp3.drag.miscellaneous import (
    base_drag_area,
    drag_divergence_mach,
    excrescence_drag,
    flap_drag,
    landing_gear_drag,
    streamwise_critical_mach,
    upsweep_drag_area,
    wave_drag,
)
from b12wp3.drag.skin_friction import (
    cf_mixed,
    effective_reynolds,
    reynolds_actual,
    reynolds_cutoff,
)
from b12wp3.drag.wetted_area import (
    body_wetted_area,
    nacelle_wetted_area,
    tail_wetted_area,
    wing_wetted_area,
)

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
    def total_wetted_area_m2(self) -> float:
        """Wetted area of all items, count * S_wet [m^2]."""
        return self.count * self.wetted_area_m2

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
class MiscDrag:
    """One miscellaneous drag term.

    Attributes:
        name: Term label.
        delta_cd: Drag coefficient on S_ref [-], >= 0.
    """

    name: str
    delta_cd: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta_cd) or self.delta_cd < 0.0:
            raise ValueError(f"{self.name}: delta_cd must be >= 0, got {self.delta_cd}")


@dataclass(frozen=True)
class DragBuildup:
    """Zero-lift drag coefficient from its components and miscellaneous terms.

    Attributes:
        components: Drag terms of each component type.
        reference_area_m2: Reference area S_ref [m^2].
        misc: Miscellaneous drag terms.
    """

    components: tuple[ComponentDrag, ...]
    reference_area_m2: float
    misc: tuple[MiscDrag, ...] = ()

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("DragBuildup needs at least one component")
        if self.reference_area_m2 <= 0.0:
            raise ValueError(
                f"reference_area_m2 must be > 0, got {self.reference_area_m2}"
            )

    def contribution(self, component: ComponentDrag) -> float:
        """CD0 contribution of one component type [-]."""
        return component.drag_area_m2 / self.reference_area_m2

    @property
    def total_wetted_area_m2(self) -> float:
        """Wetted area of the whole aircraft, sum of count * S_wet [m^2]."""
        return sum(c.total_wetted_area_m2 for c in self.components)

    @property
    def cd_components(self) -> float:
        """Sum of the component contributions [-]."""
        return sum(self.contribution(c) for c in self.components)

    @property
    def cd_misc(self) -> float:
        """Sum of the miscellaneous terms [-]."""
        return sum(m.delta_cd for m in self.misc)

    @property
    def cd0(self) -> float:
        """Total zero-lift drag coefficient [-]."""
        return self.cd_components + self.cd_misc

    def to_dataframe(self) -> pd.DataFrame:
        """One row per component, one per miscellaneous term, then a total row.

        wetted_area_m2 is one item; total_wetted_area_m2 is all items (count
        times). The total row holds the aircraft wetted area and CD0.
        """
        nan = math.nan
        rows: list[tuple[str | int | float, ...]] = [
            (
                c.name,
                c.count,
                c.wetted_area_m2,
                c.total_wetted_area_m2,
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
        no_terms = (nan,) * 9
        rows.extend((m.name, 1, *no_terms, m.delta_cd) for m in self.misc)
        rows.append(("total", 1, nan, self.total_wetted_area_m2, *(nan,) * 7, self.cd0))
        frame = pd.DataFrame(
            rows,
            columns=[
                "component",
                "count",
                "wetted_area_m2",
                "total_wetted_area_m2",
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
    """Drag terms of a wing or tail.

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
    """Drag terms of a body of revolution (the fuselage).

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


def nacelle_drag(
    nacelle: NacelleInputs,
    *,
    count: int,
    interference_factor: float,
    laminar_fraction: float,
    condition: FlightCondition,
    roughness_m: float,
) -> ComponentDrag:
    """Drag terms of the nacelles (Type B outline).

    Args:
        nacelle: Outer geometry of one nacelle.
        count: Number of nacelles.
        interference_factor: IF [-].
        laminar_fraction: Laminar fraction x_lam [-].
        condition: Flight condition.
        roughness_m: Equivalent sand roughness k [m].

    Returns:
        The component drag terms. l is the nacelle length; FF uses the
        nacelle formula with f = l_n / D_n.
    """
    wetted = nacelle_wetted_area(
        nacelle.length_m,
        nacelle.highlight_diameter_m,
        nacelle.max_diameter_m,
        nacelle.max_diameter_position,
        nacelle.exit_diameter_m,
    )
    fineness = fineness_ratio(nacelle.length_m, nacelle.max_frontal_area_m2)
    return component_drag(
        "nacelle",
        count=count,
        wetted_area_m2=wetted,
        reference_length_m=nacelle.length_m,
        form_factor=nacelle_form_factor(fineness),
        interference_factor=interference_factor,
        laminar_fraction=laminar_fraction,
        condition=condition,
        roughness_m=roughness_m,
    )


def _components(
    geometry: AircraftGeometry,
    condition: FlightCondition,
    settings: DragSettings,
) -> tuple[ComponentDrag, ...]:
    """Skin-friction drag terms of every component.

    Nacelles use the fuselage laminar fraction. The pylons are neglected at
    this design stage.
    """
    lifting = settings.laminar.wing_and_tail
    body = settings.laminar.fuselage
    interference = settings.interference
    roughness = settings.roughness_m
    engines = geometry.engine_count
    return (
        lifting_surface_drag(
            "wing",
            geometry.wing,
            wetted_area_fn=wing_wetted_area,
            count=1,
            interference_factor=interference.wing,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness,
        ),
        lifting_surface_drag(
            "horizontal_tail",
            geometry.horizontal_tail,
            wetted_area_fn=tail_wetted_area,
            count=1,
            interference_factor=interference.horizontal_tail,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness,
        ),
        lifting_surface_drag(
            "vertical_tail",
            geometry.vertical_tail,
            wetted_area_fn=tail_wetted_area,
            count=1,
            interference_factor=interference.vertical_tail,
            laminar_fraction=lifting,
            condition=condition,
            roughness_m=roughness,
        ),
        body_drag(
            "fuselage",
            geometry.fuselage,
            form_factor_fn=fuselage_form_factor,
            count=1,
            interference_factor=interference.fuselage,
            laminar_fraction=body,
            condition=condition,
            roughness_m=roughness,
        ),
        nacelle_drag(
            geometry.nacelle,
            count=engines,
            interference_factor=interference.nacelle,
            laminar_fraction=body,
            condition=condition,
            roughness_m=roughness,
        ),
    )


def wing_wave_drag(
    geometry: AircraftGeometry,
    configuration: DragConfiguration,
    settings: DragSettings,
) -> float:
    """Wave drag of the wing, ADSEE-II three-branch relation.

    M_cr is the streamwise value of the sweep-normal section M_cr. M_DD is
    only computed if M >= M_cr, so a low-speed configuration needs no C_L.

    Args:
        geometry: All geometric inputs.
        configuration: Aircraft configuration.
        settings: Drag settings.

    Returns:
        Delta C_D of the wave drag [-].

    Raises:
        ValueError: If M >= M_cr and the configuration has no C_L.
    """
    wing = geometry.wing
    sweep = wing.planform().sweep_at(settings.wave_sweep_chord_fraction)
    mach = configuration.condition.mach
    mach_critical = streamwise_critical_mach(settings.wing_mach_critical_normal, sweep)
    if mach < mach_critical:
        return 0.0
    if configuration.lift_coefficient is None:
        raise ValueError(
            f"DragConfiguration({configuration.name!r}): M = {mach} >= M_cr = "
            f"{mach_critical}, so the wave drag needs a lift_coefficient"
        )
    mach_dd = drag_divergence_mach(
        settings.kappa_a,
        wing.thickness_to_chord,
        configuration.lift_coefficient,
        sweep,
    )
    logger.debug("wave drag: M=%.4f M_cr=%.4f M_DD=%.4f", mach, mach_critical, mach_dd)
    return wave_drag(mach, mach_critical, mach_dd)


def _flap_term(
    flap: FlapInputs,
    flap_factor: float,
    deflection_deg: float,
    geometry: AircraftGeometry,
) -> float:
    """Flap drag of one flap; S_flap is the wing strip between its ends."""
    flap_area = geometry.wing.planform().area_between(flap.eta_in, flap.eta_out)
    return flap_drag(
        flap_factor,
        flap.chord_ratio,
        flap_area,
        geometry.wing.area_m2,
        deflection_deg,
    )


def _gear_term(
    gear: GearInputs, settings: DragSettings, reference_area: float
) -> float:
    """Drag of every unit of one landing gear type, on S_ref."""
    unit = landing_gear_drag(
        settings.wheel_well_coeff,
        gear.frontal_area_m2,
        gear.tire_width_m,
        gear.tire_diameter_m,
        reference_area,
    )
    return gear.count * unit


def _miscellaneous(
    geometry: AircraftGeometry,
    configuration: DragConfiguration,
    settings: DragSettings,
) -> tuple[MiscDrag, ...]:
    """Every miscellaneous term except excrescence and leakage."""
    reference_area = geometry.wing.area_m2
    fuselage_aft = geometry.fuselage_aft
    upsweep = upsweep_drag_area(
        math.radians(fuselage_aft.upsweep_deg), geometry.fuselage.max_frontal_area_m2
    )
    base = base_drag_area(configuration.condition.mach, fuselage_aft.base_area_m2)
    te_flap = _flap_term(
        geometry.te_flap,
        settings.te_flap_factor,
        configuration.te_flap_deflection_deg,
        geometry,
    )
    le_flap = _flap_term(
        geometry.le_flap,
        settings.le_flap_factor,
        configuration.le_flap_deflection_deg,
        geometry,
    )
    terms = [
        MiscDrag("wave_drag", wing_wave_drag(geometry, configuration, settings)),
        MiscDrag("fuselage_upsweep", upsweep / reference_area),
        MiscDrag("fuselage_base", base / reference_area),
        MiscDrag("te_flap", te_flap),
        MiscDrag("le_flap", le_flap),
    ]
    if configuration.gear_down:
        main = _gear_term(geometry.main_gear, settings, reference_area)
        nose = _gear_term(geometry.nose_gear, settings, reference_area)
        terms.extend((MiscDrag("main_gear", main), MiscDrag("nose_gear", nose)))
    return tuple(terms)


def aircraft_drag_buildup(
    geometry: AircraftGeometry,
    configuration: DragConfiguration,
    settings: DragSettings,
) -> DragBuildup:
    """CD0 build-up of the full aircraft in one configuration.

    Args:
        geometry: All geometric inputs.
        configuration: Flight condition, flap deflections and gear state.
        settings: Laminar fraction, roughness, interference factors and the
            miscellaneous-drag method inputs.

    Returns:
        The drag build-up. S_ref is the wing planform area. The last
        miscellaneous term is excrescence and leakage, a fraction
        settings.excrescence_fraction of the returned CD0.
    """
    components = _components(geometry, configuration.condition, settings)
    misc = _miscellaneous(geometry, configuration, settings)
    without = DragBuildup(components, geometry.wing.area_m2, misc)
    leakage = MiscDrag(
        "excrescence_leakage",
        excrescence_drag(without.cd0, settings.excrescence_fraction),
    )
    buildup = DragBuildup(components, geometry.wing.area_m2, (*misc, leakage))
    logger.info(
        "Class II drag, %s configuration: CD0 = %.5f", configuration.name, buildup.cd0
    )
    return buildup
