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

**This module's liveness is tied to the ``hasattr`` dispatch mechanism, and should be
deleted rather than repaired if that mechanism goes away.**  The probes it scans are the
only two in the tree, and both live in ``circuit_builder/base.py``.  If the builder is
ever refactored to obtain the ancilla width from the mapper's returned object instead of
probing for optional methods, ``_capability_probes()`` becomes empty and
``test_at_least_one_capability_probe_is_scanned`` fails.  That failure means the mechanism
was retired, not that something regressed: remove this module.  The alternative -- making
the emptiness check lenient -- would let the module pass vacuously forever, which is the
one outcome it exists to prevent.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import ast
import difflib
from collections.abc import Iterator
from pathlib import Path

import pytest

# Two names this close are a rename, not two different methods.  Validated against the
# tree: on a correct checkout no probed capability has any near variant at all.
_CONFUSABLE_RATIO = 0.85

# Both forms dispatch on an optional capability, so both name a literal that can drift.
_PROBE_ARITIES = {"hasattr": (2,), "getattr": (2, 3)}

# The builder has carried two capability probes since the dispatch was introduced.  This is
# a floor on how many the scan can still *see*, not an inventory of which ones exist: it
# names no capability and so takes no position on the pending ``ancilla``/``ancillary``
# spelling.  See ``test_capability_probe_coverage_has_not_shrunk`` for why it is a count.
_MIN_CAPABILITY_PROBES = 2

_SRC = Path(__file__).parent.parent / "src" / "qdk_chemistry" / "algorithms"
CIRCUIT_BUILDER_DIR = _SRC / "phase_estimation" / "circuit_builder"
MAPPER_DIR = _SRC / "controlled_circuit_mapper"


def _target_name(node: ast.expr) -> str:
    """The identifier a probe interrogates, as far as it is statically visible."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_probe_call(node: ast.AST) -> bool:
    """A ``hasattr``/``getattr`` call with an arity that dispatches on a capability."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and len(node.args) in _PROBE_ARITIES.get(node.func.id, ())
    )


def _probed_name(node: ast.Call) -> str | None:
    """Return the literal attribute name of a mapper capability probe, if this is one.

    Both ``hasattr(mapper, "name")`` and ``getattr(mapper, "name", default)`` dispatch on
    an optional capability, so both are probes and both must be checked.
    """
    if not _is_probe_call(node):
        return None

    target, attribute = node.args[0], node.args[1]
    if not isinstance(attribute, ast.Constant) or not isinstance(attribute.value, str):
        return None

    # Only probes against something mapper-shaped; other lookups are unrelated.
    return attribute.value if "mapper" in _target_name(target).lower() else None


def _capability_probe_calls() -> Iterator[tuple[Path, ast.Call]]:
    """Every ``hasattr``/``getattr`` capability-dispatch call under the circuit-builder package."""
    for path in sorted(CIRCUIT_BUILDER_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if _is_probe_call(node):
                assert isinstance(node, ast.Call)
                yield path, node


def _capability_probes() -> list[tuple[Path, int, str]]:
    """Every ``hasattr``/``getattr`` capability probe against a mapper-shaped target."""
    probes: list[tuple[Path, int, str]] = []
    for path, node in _capability_probe_calls():
        name = _probed_name(node)
        if name is not None:
            probes.append((path, node.lineno, name))
    return probes


def _unscanned_capability_probes() -> list[tuple[Path, int, str]]:
    """Every capability-dispatch call under the circuit-builder package the scan drops.

    ``_probed_name`` recognises a probe only when the target identifier contains ``mapper``
    and the attribute is a string literal.  Both conditions are ordinary refactors away
    from being false: binding the mapper to a local named ``impl``, calling through
    ``self._resolve()``, or moving the capability name into a variable each make a live
    probe invisible to the scan while leaving it working at runtime.

    Detection here is by *capability name* rather than by target shape, because the target
    shape is exactly what the dangerous refactors change.  A dropped call is reported when
    it names a method some mapper actually defines, or when it interrogates a
    mapper-shaped target through a name this module cannot resolve statically.
    """
    known = _mapper_member_names()
    skipped: list[tuple[Path, int, str]] = []
    for path, node in _capability_probe_calls():
        if _probed_name(node) is not None:
            continue
        target, attribute = node.args[0], node.args[1]
        names_a_capability = isinstance(attribute, ast.Constant) and attribute.value in known
        interrogates_a_mapper = "mapper" in _target_name(target).lower()
        if names_a_capability or interrogates_a_mapper:
            skipped.append((path, node.lineno, ast.unparse(node)))
    return skipped


def _uncovered_capability_literals() -> list[tuple[Path, int, str]]:
    """Capability names written as literals that no recognised probe accounts for.

    This is a *diagnostic*, computed only to explain a failure, and is deliberately not
    asserted on: it keys on nothing but "this string is the name of a mapper method", so a
    docstring or an unrelated lookup can appear here.  That imprecision is acceptable for a
    hint and would not be acceptable for a gate.
    """
    known = _mapper_member_names()
    covered = {(path, name) for path, _, name in _capability_probes()}
    found: list[tuple[Path, int, str]] = []
    for path in sorted(CIRCUIT_BUILDER_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in known
                and (path, node.value) not in covered
            ):
                found.append((path, node.lineno, node.value))
    return found


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


def test_no_capability_probe_escapes_the_scanner() -> None:
    """Guard the *discovery* step: a probe the scan cannot see is a probe it cannot check.

    A bare ``assert _capability_probes()`` liveness check fires only when the discovered set
    is empty, so it catches total discovery loss and is blind to partial loss.  That is the
    same existential quantifier that made the per-probe check blind to partial renames --
    ``at least one probe`` hides a lost probe exactly as ``some mapper`` hid a lost
    definition -- and the fix has to be the same shape: constrain the whole set, not its
    cardinality.  (That liveness check used to live here as its own test; it is subsumed by
    ``test_capability_probe_coverage_has_not_shrunk``, which fails on everything it failed
    on and on the partial losses it missed.)

    Rather than pinning the capability names, which would take a position on the pending
    ``ancilla``/``ancillary`` spelling this module deliberately stays neutral on, this pins
    *coverage*: no capability-dispatch call may hide from the scan.  Renaming a local from
    ``circuit_mapper`` to ``impl``, or swapping ``hasattr`` for an equivalent
    ``getattr(..., None)``, then fails here instead of quietly shrinking the parametrised
    set.
    """
    skipped = _unscanned_capability_probes()
    assert not skipped, (
        "these capability probes are invisible to the scan, so the literals they name are "
        "unchecked and a rename of the corresponding mapper method would pass silently:\n"
        + "\n".join(f"  {p.name}:{n}  {src}" for p, n, src in skipped)
        + "\nEither make the probe target mapper-shaped with a literal attribute, or widen "
        "_probed_name to recognise this form."
    )


def test_capability_probe_coverage_has_not_shrunk() -> None:
    """Guard the *enumerator*: a probe written in an unanticipated form is not enumerated.

    ``test_no_capability_probe_escapes_the_scanner`` detects a dropped probe only among
    calls the enumerator already yields, and ``_capability_probe_calls`` yields a node only
    when it is a call to a name in ``_PROBE_ARITIES``.  So detection-by-capability-name is
    applied exclusively to calls already admitted by detection-by-builtin-name, and any
    dispatch that is not literally a ``hasattr``/``getattr`` call is invisible to both.
    Measured, on the tree as it stands:

    ==========================================  ====================================
    ``"cap" in dir(circuit_mapper)``            a ``Compare``, never a ``Call``
    ``_probe = hasattr`` then ``_probe(m, c)``  ``func.id`` is not a known builtin
    ==========================================  ====================================

    Both leave the dispatch working, silently drop a parametrised case, and pass.

    Enumerating more forms cannot close this -- the next refactor invents the next form.
    So this asserts a *magnitude* instead: however a probe is written, a probe the scan
    cannot see is one it does not count.  That also makes the check neutral on naming,
    unlike pinning the discovered set, which would fail on a complete and correct rename
    and so cast a vote on the pending ``ancilla``/``ancillary`` decision.

    It is a floor rather than an equality so that adding a capability is not a failure.
    When the count legitimately drops -- because the merged builder adopts the
    width-carrying-``Circuit`` contract and stops probing at all -- the right response is to
    delete this module, not to lower the number.

    The diagnostic is computed only once the gate has already failed.  It is a hint, and a
    hint must never be able to fail a check whose gate is satisfied.
    """
    probes = _capability_probes()
    if len(probes) >= _MIN_CAPABILITY_PROBES:
        return
    hint = _uncovered_capability_literals()
    pytest.fail(
        f"only {len(probes)} capability probe(s) are visible to the scan, expected at least "
        f"{_MIN_CAPABILITY_PROBES}. Every remaining probe is still checked, but a probe that "
        f"vanished from this count is no longer checked at all.\n"
        f"  visible: {[(p.name, n, c) for p, n, c in probes]}\n"
        f"  capability names in {CIRCUIT_BUILDER_DIR.name}/ with no matching probe: "
        f"{[(p.name, n, c) for p, n, c in hint] or 'none'}\n"
        "If dispatch was rewritten in another form (a `in dir(...)` membership test, an "
        "aliased builtin, a helper), teach _capability_probe_calls that form. If optional "
        "capability dispatch is gone entirely, delete this module rather than lower the floor."
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

    The quantifier is deliberate and worth stating, because it bounds the guarantee:
    this asserts *some* mapper defines the capability, not that any particular one does.
    The default mapper (``pauli_sequence`` -- ``ControlledPauliSequenceMapper``) defines
    neither probed capability, so on a default run both probes are legitimately ``False``
    and both fallbacks fire.  That is polymorphism, not drift.  Consequently this catches
    literal-vs-method drift across the mapper package, and does **not** catch a specific
    mapper quietly ceasing to advertise a capability it used to provide -- the partial
    rename test below is what narrows that gap, and only for close-variant renames.
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
