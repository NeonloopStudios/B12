# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mateusz Suszynski
# Created: 2026-10-15
"""Strict YAML input reading shared by the wp3 config loaders.

Each YAML section maps to one frozen dataclass; its keys are the dataclass
field names. The helpers check the structure and the value types, and
collect null (not yet set) inputs so a loader can report all of them in one
error.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """An input YAML file has a wrong structure, a wrong type or nulls."""


def number(name: str, value: object, field_type: object) -> float | int:
    """One YAML value, checked against a dataclass field type (int or float).

    Args:
        name: Input name for the error message.
        value: Value read from the YAML file.
        field_type: Field annotation; "int" requires an integer.

    Returns:
        The value as int (for "int") or float.

    Raises:
        ConfigError: If the value is not a number of the right kind.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number, got {value!r}")
    if field_type == "int":
        if not isinstance(value, int):
            raise ConfigError(f"{name} must be an integer, got {value!r}")
        return value
    return float(value)


def read_mapping(path: Path, expected_keys: Collection[str]) -> Mapping[str, object]:
    """Read a YAML file whose top level must have exactly the expected keys.

    Args:
        path: YAML file.
        expected_keys: Required top-level keys; no other key is allowed.

    Returns:
        The top-level mapping.

    Raises:
        ConfigError: If the file cannot be read or parsed, or the top-level
            keys differ from the expected keys.
    """
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as error:
        logger.error("cannot read input file %s: %s", path, error)
        raise ConfigError(f"cannot read input file {path}") from error
    except yaml.YAMLError as error:
        logger.error("invalid YAML in %s: %s", path, error)
        raise ConfigError(f"invalid YAML in {path}") from error
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: the top level must be a mapping")
    expected = set(expected_keys)
    keys = set(raw)
    if keys != expected:
        raise ConfigError(
            f"{path}: missing sections {sorted(expected - keys)}, "
            f"unknown sections {sorted(keys - expected)}"
        )
    return raw


def build_section[T](
    raw: Mapping[str, object],
    section: str,
    cls: type[T],
    missing: list[str],
) -> T | None:
    """Build one input dataclass from its YAML section.

    Args:
        raw: Top-level mapping of the YAML file.
        section: Section name.
        cls: Frozen dataclass whose fields are the section keys.
        missing: Null keys are appended here as "section.key".

    Returns:
        The dataclass, or None if a key in the section is null.

    Raises:
        ConfigError: If the section is not a mapping, its keys differ from
            the dataclass fields, or a value has the wrong type.
    """
    content = raw[section]
    if not isinstance(content, Mapping):
        raise ConfigError(f"section {section!r} must be a mapping")
    # cls is always a dataclass type: every caller passes one of the input types
    fields = {f.name: f.type for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    keys = set(content)
    if keys != set(fields):
        raise ConfigError(
            f"section {section!r}: missing keys {sorted(set(fields) - keys)}, "
            f"unknown keys {sorted(keys - set(fields))}"
        )
    nulls = [f"{section}.{key}" for key in fields if content[key] is None]
    if nulls:
        missing.extend(nulls)
        return None
    values: dict[str, Any] = {
        key: number(f"{section}.{key}", content[key], field_type)
        for key, field_type in fields.items()
    }
    return cls(**values)
