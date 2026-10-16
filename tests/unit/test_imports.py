"""Unit tests for the import graph: nothing is broken, and the architectures stay apart.

Imports are read from the AST, never from raw text, because docstrings legitimately
name both architectures while explaining these rules.

The boundary rules are the fairness argument in executable form. If A could
import B (or the reverse), or `shared/` could see either one, one architecture
could quietly get something the other does not, and the comparison stops being
like-for-like.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from fyp.shared.helpers.config import PROJECT_ROOT

SRC = PROJECT_ROOT / "src"
PACKAGE = SRC / "fyp"


def _python_files(root: Path):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _imports(path: Path) -> list[tuple[int, str]]:
    """It takes a source file and gives you every module it imports, with the line.

    Relative imports come back as absolute names. The codebase has none today,
    but `from ..architecture_b import x` would otherwise walk straight past the
    boundary tests.
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level and path.is_relative_to(SRC):
            package = list(path.relative_to(SRC).parts[:-1])
            base = package[:len(package) - (node.level - 1)]
            found.append((node.lineno, ".".join(base + ([node.module] if node.module else []))))
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names]
    return found


def _modules_under_src() -> set[str]:
    mods = set()
    for p in _python_files(SRC):
        name = ".".join(p.relative_to(SRC).with_suffix("").parts)
        mods |= {name, name.removesuffix(".__init__")}
    return mods


def test_every_fyp_import_resolves():
    """A move or rename that leaves a stale `fyp.*` import behind fails here,
    not on the first line of a script at the bench. Covers scripts/ and tests/
    as well as src/, because those are what you actually run."""
    mods = _modules_under_src()
    broken = [f"{p.relative_to(PROJECT_ROOT)}:{line} imports {m}"
              for root in (SRC, PROJECT_ROOT / "scripts", PROJECT_ROOT / "tests")
              for p in _python_files(root)
              for line, m in _imports(p)
              if m.split(".")[0] == "fyp" and m not in mods]
    assert not broken, broken


# package -> packages it must never import. One table so every rule has one home.
BOUNDARIES = {
    "shared": ("architecture_a", "architecture_b", "evaluation"),
    "architecture_a": ("architecture_b",),
    "architecture_b": ("architecture_a",),
    "evaluation": ("architecture_a", "architecture_b"),
}


@pytest.mark.parametrize("package, banned", BOUNDARIES.items(), ids=BOUNDARIES.keys())
def test_package_does_not_import_across_its_boundary(package, banned):
    offenders = [f"{p.relative_to(PROJECT_ROOT)}:{line} imports {m}"
                 for p in _python_files(PACKAGE / package)
                 for line, m in _imports(p)
                 if any(m == f"fyp.{b}" or m.startswith(f"fyp.{b}.") for b in banned)]
    assert not offenders, offenders
