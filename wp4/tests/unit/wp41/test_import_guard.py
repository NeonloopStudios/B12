# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kajetan R. Gulaj
# Created: 2026-11-19
"""Import guard: wp41 imports no other work package (wp4/README.md, section 2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

WP41_DIR = Path(__file__).resolve().parents[3] / "src" / "wp41"
FORBIDDEN = ("wp2", "wp3", "wp42", "wp43", "b12wp2", "b12wp3")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_import_guard_scans_the_wp41_modules() -> None:
    files = sorted(WP41_DIR.glob("*.py"))

    assert {f.name for f in files} >= {"contracts.py", "integrate.py", "torsion.py"}


@pytest.mark.parametrize("path", sorted(WP41_DIR.glob("*.py")), ids=lambda p: p.name)
def test_wp41_module_imports_no_other_work_package(path: Path) -> None:
    roots = _imported_roots(path)

    assert roots.isdisjoint(FORBIDDEN), f"{path.name} imports {roots & set(FORBIDDEN)}"
