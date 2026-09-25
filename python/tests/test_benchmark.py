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


#: Default mode: the whole standard QPE circuit is built and traced.
#: The gate counts scale with the Trotter step count, which follows the exact rule of
#: Algorithm 1 of Apel et al. (arXiv:2609.05316) rather than its small-angle linearization;
#: see ``HubbardPlaquetteTrotter._step_count``. ``logical_qubits`` is independent of the
#: step count and so does not move when that rule changes.
_HUBBARD_L2_FULL_CIRCUIT = {
    "L": 2,
    "sites": 4,
    "system_qubits": 8,
    "electrons": 4,
    "target_precision": 0.0204,
    "qpe_budget": 0.0136,
    "trotter_budget": 0.0068,
    "qpe_bits": 10,
    "base_time": 0.2253660323553513,
    "logical_qubits": 25,
    "rotations": 2224986,
    "rotation_depth": 1483448,
    "t_gates": 698155,
    "ccz_count": 610582,
    "ccix_count": 0,
    "toffolis": 610582,
    "measurements": 610592,
}


@pytest.fixture(scope="module")
def script() -> Any:
    """Load the sampling script as a module."""
    spec = importlib.util.spec_from_file_location("sample_hubbard_resources", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_sample_hubbard_L2(  # noqa: N802 - L is the lattice side
    script: Any,
    tmp_path: Path,
) -> None:
    """Pin the sampling script's 2x2 lattice result."""
    output_path = tmp_path / "hubbard_logical_resources.csv"
    assert script.main(["--size", "2", "-o", str(output_path)]) == 0
    frame = pandas.read_csv(output_path)

    assert len(frame) == 1, "one row per lattice size is expected"
    missing = set(_PINNED_COLUMNS) - set(frame.columns)
    assert not missing, f"pinned columns absent from the table: {sorted(missing)}"

    row = frame.iloc[0]
    mismatches = []
    for column, want in _HUBBARD_L2_FULL_CIRCUIT.items():
        got = row[column]
        got = got.item() if hasattr(got, "item") else got
        matches = got == pytest.approx(want) if isinstance(want, float) else got == want
        if not matches:
            mismatches.append(f"  {column}: expected {want}, got {got}")
    assert not mismatches, "Mismatches found:\n" + "\n".join(mismatches)
