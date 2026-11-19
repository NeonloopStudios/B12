# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""WP4.1 CSV contracts: the files the scripts exchange.

Each file has comment lines ``# key: value`` (the first is ``# schema:``), one
column-name line, then one row per station. Each class below owns the
reader and the writer of its file. No other module reads or writes these files.
All values are SI. Right half-wing, y from the centreline to the tip.
Layout: wp4/README_41.md, sections 5 to 8.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]

SCHEMA_LIFT: Final[str] = "wp41.lift.v1"
SCHEMA_WEIGHT: Final[str] = "wp41.weight.v1"
SCHEMA_POINT_LOADS: Final[str] = "wp41.point_loads.v1"
SCHEMA_TORQUE: Final[str] = "wp41.torque.v1"

LIFT_COLUMNS: Final[tuple[str, ...]] = (
    "y_m", "chord_m", "cl_d", "cm_c4_d", "n_prime_npm", "m_prime_c4_nmpm",
)
WEIGHT_COLUMNS: Final[tuple[str, ...]] = (
    "y_m", "chord_m", "w_str_npm", "w_fuel_npm", "f_str", "f_fuel",
)
POINT_LOAD_COLUMNS: Final[tuple[str, ...]] = ("name", "y_m", "w_n", "d_ahead_of_sc_m")
TORQUE_COLUMNS: Final[tuple[str, ...]] = ("y_m", "t_prime_nmpm", "torque_nm")


class ContractError(ValueError):
    """A WP4.1 CSV file or table breaks its contract."""


# --- shared helpers ---


def _fmt(value: float) -> str:
    """Shortest text that reads back to the same float (17 significant digits)."""
    return format(float(value), ".17g")


def _write_file(
    path: Path,
    schema: str,
    meta: list[tuple[str, str]],
    columns: tuple[str, ...],
    rows: list[list[str]],
) -> None:
    """Write comment lines, the column line and the rows.

    Raises:
        OSError: If the file cannot be written.
    """
    lines = [f"# schema: {schema}"]
    lines += [f"# {key}: {value}" for key, value in meta]
    lines.append(",".join(columns))
    lines += [",".join(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_file(
    path: Path, schema: str, columns: tuple[str, ...]
) -> tuple[dict[str, str], list[list[str]]]:
    """Read a contract file and check its schema line and column names.

    Returns:
        The comment values by key and the data rows as text.

    Raises:
        ContractError: If the schema or the columns differ, or a row has the
            wrong number of fields.
        OSError: If the file cannot be read.
    """
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if not sep:
                raise ContractError(f"{path}: comment line without ':': {line!r}")
            meta[key.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    if meta.get("schema") != schema:
        found = meta.get("schema")
        raise ContractError(f"{path}: schema {found!r}, expected {schema!r}")
    if not body:
        raise ContractError(f"{path}: no column line")
    header = tuple(next(csv.reader([body[0]])))
    if header != columns:
        raise ContractError(f"{path}: columns {header}, expected {columns}")
    rows = [next(csv.reader([line])) for line in body[1:]]
    for row in rows:
        if len(row) != len(columns):
            raise ContractError(
                f"{path}: row {row} has {len(row)} fields, expected {len(columns)}"
            )
    return meta, rows


def _meta_text(meta: dict[str, str], key: str, path: Path) -> str:
    """Return one comment value or raise ContractError."""
    if key not in meta:
        raise ContractError(f"{path}: missing comment line '# {key}:'")
    return meta[key]


def _meta_float(meta: dict[str, str], key: str, path: Path) -> float:
    """Return one comment value as a float or raise ContractError."""
    text = _meta_text(meta, key, path)
    try:
        return float(text)
    except ValueError as exc:
        raise ContractError(f"{path}: '# {key}: {text}' is not a number") from exc


def _column(rows: list[list[str]], index: int, path: Path) -> FloatArray:
    """Convert one text column to a float64 array."""
    try:
        return np.array([float(row[index]) for row in rows], dtype=np.float64)
    except ValueError as exc:
        raise ContractError(
            f"{path}: column {index} holds a non-number: {exc}"
        ) from exc


def _check_stations(
    name: str, y: FloatArray, *arrays: FloatArray, strict: bool
) -> None:
    """Check that y is 1-D, finite, ordered, and that all arrays share its shape.

    Raises:
        ContractError: On any violation, with the table name.
    """
    if y.ndim != 1 or y.size < 2:
        raise ContractError(f"{name}: y_m needs at least 2 stations, shape {y.shape}")
    for array in (y, *arrays):
        if array.shape != y.shape:
            raise ContractError(
                f"{name}: shape {array.shape} differs from y_m {y.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ContractError(f"{name}: a value is not finite")
    steps = np.diff(y)
    if not (np.all(steps > 0.0) if strict else np.all(steps >= 0.0)):
        raise ContractError(f"{name}: y_m is not increasing")
    if not (y[0] >= 0.0):
        raise ContractError(f"{name}: y_m starts below the centreline: {y[0]}")


# --- lift.csv ---


@dataclass(frozen=True)
class LiftTable:
    """lift.csv: lift and quarter-chord moment of one load case.

    Attributes:
        case: Load case name.
        n: Load factor [-].
        w_n: Weight of the aircraft in the case [N].
        q_pa: Dynamic pressure [Pa].
        cl_total: Required lift coefficient CL_d [-].
        alpha_d_deg: Angle of attack from the two-point interpolation [deg].
        mach_case: Mach number of the load case [-].
        mach_run: Mach number of the VSPAERO run [-].
        extrapolated: True if CL_d lies outside the two VSPAERO angles.
        y_m: Span stations [m].
        chord_m: Local chord [m].
        cl_d: Sectional lift coefficient [-].
        cm_c4_d: Sectional moment coefficient about c/4, nose-up positive [-].
        n_prime_npm: Normal force per unit span, upward positive [N/m].
        m_prime_c4_nmpm: Moment per unit span about c/4, nose-up positive [N m/m].
    """

    case: str
    n: float
    w_n: float
    q_pa: float
    cl_total: float
    alpha_d_deg: float
    mach_case: float
    mach_run: float
    extrapolated: bool
    y_m: FloatArray
    chord_m: FloatArray
    cl_d: FloatArray
    cm_c4_d: FloatArray
    n_prime_npm: FloatArray
    m_prime_c4_nmpm: FloatArray

    def __post_init__(self) -> None:
        _check_stations(
            "lift.csv", self.y_m, self.chord_m, self.cl_d, self.cm_c4_d,
            self.n_prime_npm, self.m_prime_c4_nmpm, strict=True,
        )
        if not (np.all(self.chord_m > 0.0)):
            raise ContractError("lift.csv: chord_m must be > 0")
        if not (self.q_pa > 0.0 and self.w_n > 0.0):
            raise ContractError(
                f"lift.csv: q = {self.q_pa}, W = {self.w_n} must be > 0"
            )

    def to_csv(self, path: Path) -> None:
        """Write the table. Raises OSError if the file cannot be written."""
        meta = [
            ("case", self.case), ("n", _fmt(self.n)), ("W", _fmt(self.w_n)),
            ("q", _fmt(self.q_pa)), ("CL_d", _fmt(self.cl_total)),
            ("alpha_d", _fmt(self.alpha_d_deg)), ("mach_case", _fmt(self.mach_case)),
            ("mach_run", _fmt(self.mach_run)),
            ("extrapolated", "true" if self.extrapolated else "false"),
        ]
        columns = (
            self.y_m, self.chord_m, self.cl_d, self.cm_c4_d,
            self.n_prime_npm, self.m_prime_c4_nmpm,
        )
        rows = [[_fmt(col[i]) for col in columns] for i in range(self.y_m.size)]
        _write_file(path, SCHEMA_LIFT, meta, LIFT_COLUMNS, rows)

    @classmethod
    def from_csv(cls, path: Path) -> LiftTable:
        """Read and validate lift.csv.

        Raises:
            ContractError: If the file breaks the contract.
            OSError: If the file cannot be read.
        """
        meta, rows = _read_file(path, SCHEMA_LIFT, LIFT_COLUMNS)
        flag = _meta_text(meta, "extrapolated", path)
        if flag not in ("true", "false"):
            raise ContractError(
                f"{path}: '# extrapolated: {flag}' must be true or false"
            )
        cols = [_column(rows, i, path) for i in range(len(LIFT_COLUMNS))]
        return cls(
            case=_meta_text(meta, "case", path),
            n=_meta_float(meta, "n", path),
            w_n=_meta_float(meta, "W", path),
            q_pa=_meta_float(meta, "q", path),
            cl_total=_meta_float(meta, "CL_d", path),
            alpha_d_deg=_meta_float(meta, "alpha_d", path),
            mach_case=_meta_float(meta, "mach_case", path),
            mach_run=_meta_float(meta, "mach_run", path),
            extrapolated=flag == "true",
            y_m=cols[0], chord_m=cols[1], cl_d=cols[2], cm_c4_d=cols[3],
            n_prime_npm=cols[4], m_prime_c4_nmpm=cols[5],
        )


# --- weight.csv ---


@dataclass(frozen=True)
class WeightTable:
    """weight.csv: wing structure and fuel weight per unit span at n = 1.

    Attributes:
        case: Load case name.
        y_m: Span stations [m].
        chord_m: Local chord [m].
        w_str_npm: Structure weight per unit span, n = 1 [N/m].
        w_fuel_npm: Fuel weight per unit span, n = 1 [N/m].
        f_str: Chord fraction of the structure centre of gravity [-].
        f_fuel: Chord fraction of the fuel centre of gravity [-].
    """

    case: str
    y_m: FloatArray
    chord_m: FloatArray
    w_str_npm: FloatArray
    w_fuel_npm: FloatArray
    f_str: FloatArray
    f_fuel: FloatArray

    def __post_init__(self) -> None:
        _check_stations(
            "weight.csv", self.y_m, self.chord_m, self.w_str_npm, self.w_fuel_npm,
            self.f_str, self.f_fuel, strict=True,
        )
        if not (np.all(self.chord_m > 0.0)):
            raise ContractError("weight.csv: chord_m must be > 0")
        if not (np.all(self.w_str_npm >= 0.0) and np.all(self.w_fuel_npm >= 0.0)):
            raise ContractError("weight.csv: weight per unit span must be >= 0")
        for name, frac in (("f_str", self.f_str), ("f_fuel", self.f_fuel)):
            if not (np.all(frac >= 0.0) and np.all(frac <= 1.0)):
                raise ContractError(f"weight.csv: {name} must be in [0, 1]")

    def to_csv(self, path: Path) -> None:
        """Write the table. Raises OSError if the file cannot be written."""
        columns = (
            self.y_m, self.chord_m, self.w_str_npm, self.w_fuel_npm,
            self.f_str, self.f_fuel,
        )
        rows = [[_fmt(col[i]) for col in columns] for i in range(self.y_m.size)]
        _write_file(path, SCHEMA_WEIGHT, [("case", self.case)], WEIGHT_COLUMNS, rows)

    @classmethod
    def from_csv(cls, path: Path) -> WeightTable:
        """Read and validate weight.csv.

        Raises:
            ContractError: If the file breaks the contract.
            OSError: If the file cannot be read.
        """
        meta, rows = _read_file(path, SCHEMA_WEIGHT, WEIGHT_COLUMNS)
        cols = [_column(rows, i, path) for i in range(len(WEIGHT_COLUMNS))]
        return cls(
            case=_meta_text(meta, "case", path),
            y_m=cols[0], chord_m=cols[1], w_str_npm=cols[2],
            w_fuel_npm=cols[3], f_str=cols[4], f_fuel=cols[5],
        )


# --- point_loads.csv ---


@dataclass(frozen=True)
class PointLoads:
    """point_loads.csv: one row per engine of the right half-wing.

    Attributes:
        case: Load case name.
        names: Point load names.
        y_m: Span stations of the loads [m].
        w_n: Weights at n = 1 [N].
        d_ahead_of_sc_m: Distance of the centre of gravity ahead of the
            shear centre, forward positive [m].
    """

    case: str
    names: tuple[str, ...]
    y_m: FloatArray
    w_n: FloatArray
    d_ahead_of_sc_m: FloatArray

    def __post_init__(self) -> None:
        count = len(self.names)
        for name, array in (
            ("y_m", self.y_m), ("w_n", self.w_n),
            ("d_ahead_of_sc_m", self.d_ahead_of_sc_m),
        ):
            if array.shape != (count,):
                raise ContractError(
                    f"point_loads.csv: {name} shape {array.shape}, expected ({count},)"
                )
            if not np.all(np.isfinite(array)):
                raise ContractError(f"point_loads.csv: {name} is not finite")
        if not (np.all(self.y_m >= 0.0) and np.all(self.w_n >= 0.0)):
            raise ContractError("point_loads.csv: y_m and w_n must be >= 0")

    def to_csv(self, path: Path) -> None:
        """Write the table. Raises OSError if the file cannot be written."""
        rows = [
            [
                self.names[i], _fmt(self.y_m[i]), _fmt(self.w_n[i]),
                _fmt(self.d_ahead_of_sc_m[i]),
            ]
            for i in range(len(self.names))
        ]
        meta = [("case", self.case)]
        _write_file(path, SCHEMA_POINT_LOADS, meta, POINT_LOAD_COLUMNS, rows)

    @classmethod
    def from_csv(cls, path: Path) -> PointLoads:
        """Read and validate point_loads.csv.

        Raises:
            ContractError: If the file breaks the contract.
            OSError: If the file cannot be read.
        """
        meta, rows = _read_file(path, SCHEMA_POINT_LOADS, POINT_LOAD_COLUMNS)
        return cls(
            case=_meta_text(meta, "case", path),
            names=tuple(row[0] for row in rows),
            y_m=_column(rows, 1, path),
            w_n=_column(rows, 2, path),
            d_ahead_of_sc_m=_column(rows, 3, path),
        )


# --- torque.csv ---


@dataclass(frozen=True)
class TorqueTable:
    """torque.csv: torque per unit span and torque diagram of one load case.

    A point torque makes a step. The step station appears twice, the value
    including the point torque first and the value without it second.

    Attributes:
        case: Load case name.
        n: Load factor [-].
        y_m: Span stations, non-decreasing [m].
        t_prime_nmpm: Torque per unit span, nose-up positive [N m/m].
        torque_nm: Torque of all loads outboard of y, nose-up positive [N m].
    """

    case: str
    n: float
    y_m: FloatArray
    t_prime_nmpm: FloatArray
    torque_nm: FloatArray

    def __post_init__(self) -> None:
        _check_stations(
            "torque.csv", self.y_m, self.t_prime_nmpm, self.torque_nm, strict=False
        )

    def to_csv(self, path: Path) -> None:
        """Write the table. Raises OSError if the file cannot be written."""
        columns = (self.y_m, self.t_prime_nmpm, self.torque_nm)
        rows = [[_fmt(col[i]) for col in columns] for i in range(self.y_m.size)]
        meta = [("case", self.case), ("n", _fmt(self.n))]
        _write_file(path, SCHEMA_TORQUE, meta, TORQUE_COLUMNS, rows)

    @classmethod
    def from_csv(cls, path: Path) -> TorqueTable:
        """Read and validate torque.csv.

        Raises:
            ContractError: If the file breaks the contract.
            OSError: If the file cannot be read.
        """
        meta, rows = _read_file(path, SCHEMA_TORQUE, TORQUE_COLUMNS)
        cols = [_column(rows, i, path) for i in range(len(TORQUE_COLUMNS))]
        return cls(
            case=_meta_text(meta, "case", path),
            n=_meta_float(meta, "n", path),
            y_m=cols[0], t_prime_nmpm=cols[1], torque_nm=cols[2],
        )
