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
    "rotations": 2208717,
    "rotation_depth": 1472602,
    "t_gates": 693051,
    "ccz_count": 606116,
    "ccix_count": 0,
    "toffolis": 606116,
    "measurements": 606126,
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
        if column in _PLATFORM_SENSITIVE_COLUMNS:
            matches = got == pytest.approx(want, rel=_PLATFORM_RELATIVE_TOLERANCE)
            tolerance = f" (rel={_PLATFORM_RELATIVE_TOLERANCE})"
        else:
            matches = got == pytest.approx(want) if isinstance(want, float) else got == want
            tolerance = ""
        if not matches:
            mismatches.append(f"  {column}: expected {want}{tolerance}, got {got}")
    assert not mismatches, "Mismatches found:\n" + "\n".join(mismatches)
