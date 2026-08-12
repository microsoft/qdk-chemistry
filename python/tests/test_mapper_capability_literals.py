"""Verify every ``hasattr`` capability probe on a circuit mapper names a real method.

``phase_estimation/circuit_builder`` dispatches on optional mapper capabilities via
``hasattr(circuit_mapper, "some_method")``.  Because the capability is referenced as a
**string literal**, renaming the method on the mapper does not break the reference --
it silently turns the probe ``False`` and takes the fallback branch.  Nothing raises:
no linter, type checker, or import gate can see through a string.

That matters here because the fallbacks are designed to tolerate absence.  A missing
``num_ancillary_qubits`` substitutes an arithmetic qubit-count estimate, and a missing
``get_ancilla_prep_op`` yields ``None``, which the standard and iterative builders then
replace with a no-op state preparation.  A mapper that silently stops advertising a
capability therefore produces a circuit that builds and runs while quietly omitting the
work the capability was there to do.

This test is deliberately source-only (``ast`` over files on disk, no imports), so it
still runs in environments where the compiled extension module is unavailable.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import ast
import difflib
from pathlib import Path

import pytest

# Two names this close are a rename, not two different methods.  Validated against the
# tree: on a correct checkout no probed capability has any near variant at all.
_CONFUSABLE_RATIO = 0.85

_SRC = Path(__file__).parent.parent / "src" / "qdk_chemistry" / "algorithms"
CIRCUIT_BUILDER_DIR = _SRC / "phase_estimation" / "circuit_builder"
MAPPER_DIR = _SRC / "controlled_circuit_mapper"


def _probed_name(node: ast.Call) -> str | None:
    """Return the literal attribute name of a mapper ``hasattr`` probe, if this is one."""
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "hasattr" or len(node.args) != 2:
        return None

    target, attribute = node.args
    if not isinstance(attribute, ast.Constant) or not isinstance(attribute.value, str):
        return None

    # Only probes against something mapper-shaped; other hasattr uses are unrelated.
    if isinstance(target, ast.Name):
        target_name = target.id
    elif isinstance(target, ast.Attribute):
        target_name = target.attr
    else:
        return None

    return attribute.value if "mapper" in target_name.lower() else None


def _capability_probes() -> list[tuple[Path, int, str]]:
    """Every ``hasattr(<...mapper...>, "name")`` probe under the circuit-builder package."""
    probes: list[tuple[Path, int, str]] = []
    for path in sorted(CIRCUIT_BUILDER_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _probed_name(node)
            if name is not None:
                probes.append((path, node.lineno, name))
    return probes


def _mapper_members_by_class() -> dict[str, set[str]]:
    """Members of every controlled circuit mapper, keyed by ``file.py::ClassName``."""
    by_class: dict[str, set[str]] = {}
    for path in sorted(MAPPER_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for klass in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            names: set[str] = set()
            for member in klass.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    names.add(member.name)
                elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    names.add(member.target.id)
                elif isinstance(member, ast.Assign):
                    names.update(t.id for t in member.targets if isinstance(t, ast.Name))
            by_class[f"{path.name}::{klass.name}"] = names
    return by_class


def _mapper_member_names() -> set[str]:
    """Every method and class-level attribute defined by any controlled circuit mapper."""
    return set().union(*_mapper_members_by_class().values()) if _mapper_members_by_class() else set()


def _near_variants(name: str, candidates: set[str]) -> list[str]:
    """Names close enough to ``name`` to be a rename of it rather than a distinct method."""
    return sorted(
        c for c in candidates if c != name and difflib.SequenceMatcher(None, name, c).ratio() >= _CONFUSABLE_RATIO
    )


def test_capability_probe_directories_exist() -> None:
    """Guard the test itself: a moved package must fail loudly, not vacuously pass."""
    assert CIRCUIT_BUILDER_DIR.is_dir(), f"missing {CIRCUIT_BUILDER_DIR}"
    assert MAPPER_DIR.is_dir(), f"missing {MAPPER_DIR}"
    assert _mapper_member_names(), "parsed no members from any mapper -- the scan is broken"


def test_at_least_one_capability_probe_is_scanned() -> None:
    """A refactor that removes every probe should update this test, not silently skip it."""
    assert _capability_probes(), (
        f"no hasattr capability probes found under {CIRCUIT_BUILDER_DIR}; "
        "if the dispatch mechanism changed, delete or rewrite this module"
    )


@pytest.mark.parametrize(
    ("source", "lineno", "capability"),
    [pytest.param(p, n, c, id=f"{p.name}:{n}:{c}") for p, n, c in _capability_probes()],
)
def test_probed_capability_is_defined_by_some_mapper(source: Path, lineno: int, capability: str) -> None:
    """Each probed capability must be defined by at least one mapper class.

    A failure here means the probe can never be satisfied, so the dispatch silently
    takes its fallback branch forever.  The usual cause is renaming the method on the
    mapper without updating this string literal.
    """
    defined = _mapper_member_names()
    assert capability in defined, (
        f'{source.name}:{lineno} probes hasattr(..., "{capability}"), but no class under '
        f"{MAPPER_DIR.name}/ defines it. The probe is permanently False and the fallback "
        f"branch is taken silently. Closest defined names: "
        f"{sorted(n for n in defined if capability.split('_', maxsplit=1)[0] in n) or sorted(defined)[:5]}"
    )


@pytest.mark.parametrize(
    "capability",
    sorted({c for _, _, c in _capability_probes()}),
)
def test_no_mapper_partially_renames_a_probed_capability(capability: str) -> None:
    """No mapper may define a near variant of a probed capability without the capability.

    The previous test only requires *some* mapper to define the literal, so it cannot see
    a rename applied to one mapper but not the others: the surviving definition keeps it
    green while the renamed mapper silently stops advertising the capability.  A class
    holding a close variant and not the real name is that rename.
    """
    offenders = {
        klass: variants
        for klass, members in _mapper_members_by_class().items()
        if capability not in members and (variants := _near_variants(capability, members))
    }
    assert not offenders, (
        f'these mapper classes define a near variant of the probed capability "{capability}" '
        f"but not the capability itself, so hasattr(...) is False for them while other mappers "
        f"keep this test green: {offenders}. Rename the literal in "
        f"{CIRCUIT_BUILDER_DIR.name}/ and every mapper together, or not at all."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
