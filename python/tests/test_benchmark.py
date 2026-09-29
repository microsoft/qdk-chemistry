"""Test for the examples/benchmark/sample_hubbard_resources.py script."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from qdk_chemistry.utils import Logger

pandas = pytest.importorskip("pandas", reason="the sample script writes its table with pandas")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "benchmark" / "sample_hubbard_resources.py"

#: The columns this test pins. The script emits more, including wall-clock timings that
#: are machine dependent; anything not listed here is disregarded.
_PINNED_COLUMNS = (
    "L",
    "sites",
    "system_qubits",
    "electrons",
    "target_precision",
    "qpe_budget",
    "trotter_budget",
    "qpe_bits",
    "base_time",
    "logical_qubits",
    "rotations",
    "rotation_depth",
    "t_gates",
    "ccz_count",
    "ccix_count",
    "toffolis",
    "measurements",
)


#: Rotation synthesis rounds transcendental angles, so these two counts drift by a few
#: units out of millions across platforms (Linux matches exactly, macOS is +1, Windows
#: ARM64 is +5). They are pinned to a relative tolerance; every other column stays exact.
_PLATFORM_SENSITIVE_COLUMNS = frozenset({"rotations", "rotation_depth"})
_PLATFORM_RELATIVE_TOLERANCE = 1e-4


#: Default mode: the whole standard QPE circuit is built and traced.
#: The gate counts scale with the Trotter step count, which follows the exact rule of
#: Algorithm 1 of Apel et al. (arXiv:2609.05316) rather than its small-angle linearization;
#: see ``HubbardPlaquetteTrotter._step_count``. ``logical_qubits`` is independent of the
#: step count and so does not move when that rule changes.
#:
#: L=2 has four sites, which is below the Hamming-weight-phasing break-even of eight terms, so
#: every tower rotates term by term: no adder tree, and therefore no Toffolis at all.
_HUBBARD_L2_FULL_CIRCUIT = {
    "L": 2,
    "sites": 4,
    "system_qubits": 8,
    "electrons": 4,
    "target_precision": 0.0204,
    "qpe_budget": 0.013600000000000001,
    "trotter_budget": 0.0068000000000000005,
    "qpe_bits": 10,
    "base_time": 0.2253660323553513,
    "logical_qubits": 18,
    "rotations": 1047435,
    "rotation_depth": 698414,
    "t_gates": 698155,
    "ccz_count": 0,
    "ccix_count": 0,
    "toffolis": 0,
    "measurements": 10,
}


#: L=4 has sixteen sites, so every tower is above the break-even and takes the
#: Hamming-weight-phasing path: an adder tree compresses sixteen same-angle rotations into a
#: five-bit weight, and each place value takes one synthesized ``Rz``. This is the case that
#: exercises the construction, which is why it is pinned.
_HUBBARD_L4_FULL_CIRCUIT = {
    "L": 4,
    "sites": 16,
    "system_qubits": 32,
    "electrons": 14,
    "target_precision": 0.0816,
    "qpe_budget": 0.054400000000000004,
    "trotter_budget": 0.027200000000000002,
    "qpe_bits": 10,
    "base_time": 0.056341508088837824,
    "logical_qubits": 57,
    "rotations": 261223,
    "rotation_depth": 190086,
    "t_gates": 759067,
    "ccz_count": 355500,
    "ccix_count": 0,
    "toffolis": 355500,
    "measurements": 355510,
}


@pytest.fixture(scope="module")
def script() -> Any:
    """Load the sampling script as a module."""
    spec = importlib.util.spec_from_file_location("sample_hubbard_resources", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # ``main`` silences the logger process-wide; restore the level so that later
    # test modules still observe Logger output.
    previous_level = Logger.get_global_level()
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        Logger.set_global_level(previous_level)
        sys.modules.pop(spec.name, None)


def _compare(row: Any, expected: dict[str, Any]) -> list[str]:
    """Return one message per pinned column the row does not match."""
    mismatches = []
    for column, want in expected.items():
        got = row[column]
        got = got.item() if hasattr(got, "item") else got
        if column in _PLATFORM_SENSITIVE_COLUMNS:
            matches = got == pytest.approx(want, rel=_PLATFORM_RELATIVE_TOLERANCE)
            tolerance = f" (rel={_PLATFORM_RELATIVE_TOLERANCE})"
        else:
            matches = got == pytest.approx(want) if isinstance(want, float) else got == want
            tolerance = ""
        if not matches:
            mismatches.append(f"  L={expected['L']} {column}: expected {want}{tolerance}, got {got}")
    return mismatches


def test_sample_hubbard_L2_and_L4(  # noqa: N802 - L is the lattice side
    script: Any,
    tmp_path: Path,
) -> None:
    """Pin the sampling script's 2x2 and 4x4 lattice results.

    The 4x4 row is the primary pin: its sixteen-term towers are above the Hamming-weight-phasing
    break-even, so it is the case that actually exercises the adder tree and the phase gradient.
    The 2x2 row pins the fallback below the break-even.
    """
    output_path = tmp_path / "hubbard_logical_resources.csv"
    assert script.main(["--size", "2", "4", "-o", str(output_path)]) == 0
    frame = pandas.read_csv(output_path)

    assert len(frame) == 2, "one row per lattice size is expected"
    missing = set(_PINNED_COLUMNS) - set(frame.columns)
    assert not missing, f"pinned columns absent from the table: {sorted(missing)}"

    rows = {int(frame.iloc[index]["L"]): frame.iloc[index] for index in range(len(frame))}
    mismatches = _compare(rows[2], _HUBBARD_L2_FULL_CIRCUIT) + _compare(rows[4], _HUBBARD_L4_FULL_CIRCUIT)
    assert not mismatches, "Mismatches found:\n" + "\n".join(mismatches)

    # A collapse back to the term-by-term fallback would silently erase the adder tree, which is
    # the only source of Toffolis in this circuit. Pin the sign of the count, not just its value.
    assert rows[4]["toffolis"] > 0, "the 4x4 lattice must take the Hamming-weight-phasing path"
    assert rows[2]["toffolis"] == 0, "the 2x2 lattice is below the break-even and phases term by term"
