<!--
SPDX-License-Identifier: GPL-3.0-or-later
Copyright (C) 2026
-->

# WP4 — Engineering Structure & Coding Standards

This directory contains the WP4 codebase (`wp41`, `wp42`, `wp43`). Because multiple engineers contribute here concurrently, every module and pull request must follow the layout, isolation boundaries, and Python standards below.

---

## 1. Directory Layout

```text
wp4/
├── scripts/        # Thin, runnable entry-point scripts (one file per step)
├── src/
│   ├── wp41/       # WP4.1 sub-package (self-contained)
│   ├── wp42/       # WP4.2 sub-package (self-contained)
│   └── wp43/       # WP4.3 sub-package (self-contained)
├── tests/          # Unit tests mirroring src/ (no __init__.py at tests/ root)
│   ├── wp41/       # Unit tests for wp41
│   ├── wp42/       # Unit tests for wp42
│   └── wp43/       # Unit tests for wp43
└── out/            # Generated outputs (CSVs, plots, reports)
    ├── wp41/
    ├── wp42/
    └── wp43/
```

### Mirroring Rule
- `tests/` must mirror `src/` one-to-one: `src/wp41/foo.py` is tested in `tests/wp41/test_foo.py` (and similarly for `wp42` and `wp43`).
- Do **not** place an `__init__.py` in `wp4/tests/` (the repository uses `pytest --import-mode=importlib` in `pyproject.toml` to avoid test module name collisions across work packages).

---

## 2. Modularity & Package Boundaries

- **Strict sub-package isolation:** `wp41`, `wp42`, and `wp43` must **never** import from one another, nor from `wp2` or `wp3`. Each sub-package in `src/` is an independent module.
- **No cross-package data via Python imports:** If one stage consumes outputs produced by another, exchange data through explicit, versioned files (e.g., JSON, YAML, or CSV in `out/` or config files) and validate them into frozen dataclasses on load.
- **One-way dependency flow:** `scripts/` may import from `src/wp4*`, never the reverse. Within a sub-package, shared helpers must not import from higher-level feature modules.
- **Enforce with import-guard tests:** Include tests in `tests/` that verify no module in `wp41`, `wp42`, or `wp43` imports another `wp4*` sub-package.

---

## 3. Verification Gate (Required Before Commit / Merge)

Always invoke tools through the project environment's Python interpreter (`<env>\python.exe -m <tool>`), never bare `python`, `pytest`, or `mypy`.

Run in order before submitting changes:

1. **Strict type checking:**
   - `<env>\python.exe -m mypy --strict wp4/src`
2. **Unit tests:**
   - `<env>\python.exe -m pytest wp4/tests`

Both gates must pass with zero warnings/errors. `wp4` starts clean and is gated independently of `wp2`/`wp3` (do not carry over or document preexisting errors/warnings from earlier work packages here).

---

## 4. Python Coding Standards

Summarized from the project's Python rules (Python 3.13+):

### Style & Imports
- Strict PEP 8, **88-character** line limit.
- Include `from __future__ import annotations` at the top of every module.
- Group imports: stdlib → third-party → first-party, sorted alphabetically within each group.

### Typing
- Full type annotations on all function/method signatures, class attributes, and module constants.
- Use `Final`, `ClassVar`, `TypeAlias`, and `NewType` for identifiers that must not be mixed up.
- Use `NDArray[np.float64]` (from `numpy.typing`) instead of bare `np.ndarray`.
- Avoid `Any`. If genuinely unavoidable, add `# type: ignore[<code>]` with an inline explanation.

### Docstrings
- Google-style docstrings are required on every module, class, and public function (with `Args:`, `Returns:`, and `Raises:` sections as applicable).
- Private `_helper` functions require at least a one-line docstring.

### Data, Config & Structure
- **Frozen dataclasses (`@dataclass(frozen=True)`)** for all configuration and structured data interfaces — never pass raw `dict`s across module boundaries.
- Module-level constants: `UPPER_SNAKE_CASE` with `Final`. Non-user constants belong in versioned config files, loaded and validated into frozen dataclasses.
- **Single Responsibility:** keep functions under ~100 lines (excluding dense numerical math) and refactor overly complex logic.
- Remove dead code (unused imports, unreachable branches, superseded helpers) after every edit.

### Errors & Validation
- Raise specific exception types; wrap all file I/O and subprocess calls.
- **Fail loudly** with the offending values — no silent fallbacks.
- Validate numeric bounds using the **positive form**: `if not (x >= 0): raise ...` (this rejects `NaN`, whereas `x < 0` silently passes `NaN`).

### Numerical & Performance
- Vectorize with NumPy/SciPy; avoid Python loops over large arrays.
- Specify explicit `dtype` on every `np.zeros`, `np.ones`, and `np.empty`. Use `float64` where numerical stability requires it (`float32` only when memory dictates and precision is verified).
- Prefer contiguous C-order (`np.ascontiguousarray`) for FFI/BLAS paths, and use in-place operations/views to avoid unnecessary copies.
- Profile (`cProfile`) before optimizing; only optimize measured bottlenecks.

### Forbidden
- Bare `except:` or `except Exception: pass`.
- `TODO`, `FIXME`, or placeholder implementations in committed code.
- Mutable default arguments (e.g., `def f(x=[]):`).
- Wildcard imports (`from module import *`).
- `time.sleep()` in production paths.

---

## 5. General Software Engineering Principles

- **Functional core, imperative shell:** Keep physics, math, and domain logic as pure, deterministic functions/classes in `src/wp4*/`. Keep CLI parsing, file I/O, and plotting orchestration in `scripts/`.
- **Determinism & no global mutable state:** Seed any random number generator explicitly (`np.random.default_rng(seed)`). Never store state in module-level globals; two independent runs or instances must not interfere.
- **Physical & numerical rigor:** Cite equations, units (SI throughout unless explicitly documented), and assumptions in docstrings. Guard domains before calling `sqrt`, `log`, or dividing.
- **Meaningful unit tests:**
  - Test general, non-trivial inputs (avoid `0`, `1`, or symmetric identity values as the primary test case), plus explicit edge/degenerate cases (zero, empty, `NaN`, out-of-range).
  - Assert array `shape` and `dtype` in addition to values, and assert specific exception types on invalid inputs.
  - Keep tests fast and deterministic (no network calls, no sleeping).
- **Scoped commits & code review:** Keep each commit/PR focused on a single sub-package (`wp41`, `wp42`, or `wp43`) with its matching tests. Do not push broken or untested code.
