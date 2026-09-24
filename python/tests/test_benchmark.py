"""Test for the examples/benchmark/sample_hubbard_resources.py script."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

pandas = pytest.importorskip("pandas", reason="the sample script writes its table with pandas")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "benchmark" / "sample_hubbard_resources.py"

#: Every column the script emits except the wall-clock measurements, which are machine
#: dependent. Both costing modes emit exactly these columns, in this order.
_PINNED_COLUMNS = (
    "L",
    "sites",
    "system_qubits",
    "terms",
    "electrons",
    "lambda",
    "target_precision",
    "qpe_budget",
    "qpe_budget_fraction",
    "trotter_budget",
    "qpe_bits",
    "num_unitary_queries",
    "base_time",
    "t_max",
    "power_strategy",
    "qpe_error_model",
    "qpe_type",
    "hwp_enabled",
    "hwp_max_batch",
    "trotter_steps_per_qpe_bit",
    "one_trotter_step_time",
    "one_trotter_step_ccz_count",
    "one_trotter_step_ccix_count",
    "one_trotter_step_toffolis",
    "logical_qubits",
    "rotations",
    "rotation_depth",
    "t_gates",
    "ccz_count",
    "ccix_count",
    "toffolis",
    "measurements",
)

#: Wall-clock columns the script also emits, excluded from the pinned values above.
_TIMING_COLUMNS = ("logical_estimate_elapsed_s", "elapsed_s")


@dataclass
class BenchmarkResults:
    """Expected results from the benchmark sampling script."""

    results: dict[str, Any]  #: Sampled logical resources

    @staticmethod
    def check_dict(expected_dict: dict[str, Any], actual_dict: dict[str, Any], errors: list[str]) -> None:
        """Check that one actual record matches the expected record.

        Args:
            expected_dict: Expected values.
            actual_dict: Actual values, consumed as they are checked.
            errors: Collects every mismatch so one run reports all of them.

        """
        for key, expected_value in expected_dict.items():
            if key not in actual_dict:
                errors.append(f"Missing key '{key}'")
                continue
            actual_value = actual_dict.pop(key)
            # bool is a subclass of int, so it is compared exactly here too.
            if isinstance(expected_value, (bool, int, str)):
                if actual_value != expected_value:
                    errors.append(f"Mismatch for key '{key}': expected {expected_value}, got {actual_value}")
            elif isinstance(expected_value, float):
                if actual_value != pytest.approx(expected_value):
                    errors.append(f"Mismatch for key '{key}': expected {expected_value}, got {actual_value}")
            elif isinstance(expected_value, dict):
                BenchmarkResults.check_dict(expected_value, dict(actual_value), errors)
            elif isinstance(expected_value, list):
                if len(actual_value) != len(expected_value):
                    errors.append(
                        f"Mismatch for key '{key}': expected {len(expected_value)} entries, got {len(actual_value)}"
                    )
                    continue
                for index, expected_entry in enumerate(expected_value):
                    entry_errors: list[str] = []
                    BenchmarkResults.check_dict(expected_entry, dict(actual_value[index]), entry_errors)
                    errors.extend(f"{key}[{index}]: {error}" for error in entry_errors)
            else:
                raise ValueError(f"Don't know how to handle: {key} of type {type(expected_value)}")

        if actual_dict:
            errors.append(f"Unexpected keys in actual results: {list(actual_dict)}")

    def check_results(self, actual_results: dict[str, Any]) -> None:
        """Check that the actual results match the expected results.

        Args:
            actual_results: Actual results from the script.

        Raises:
            AssertionError: If the actual results do not match the expected results.

        """
        errors: list[str] = []
        BenchmarkResults.check_dict(self.results, dict(actual_results), errors)
        if errors:
            raise AssertionError("\n".join(["Mismatches found:", *errors]))


#: Default mode: the whole standard QPE circuit is built and traced.
_HUBBARD_L2_FULL_CIRCUIT = BenchmarkResults(
    results={
        "L": 2,
        "sites": 4,
        "system_qubits": 8,
        "terms": 29,
        "electrons": 4,
        "lambda": 48.0,
        "target_precision": 0.0204,
        "qpe_budget": 0.0136,
        "qpe_budget_fraction": 0.6666666666666666,
        "trotter_budget": 0.0068,
        "qpe_bits": 10,
        "num_unitary_queries": 1023,
        "base_time": 0.2253660323553513,
        "t_max": 230.77481713187973,
        "power_strategy": "rescale",
        "qpe_error_model": "sine-window-1sigma",
        "qpe_type": "standard-full-circuit",
        "hwp_enabled": True,
        "hwp_max_batch": 0,
        "trotter_steps_per_qpe_bit": "[43, 85, 170, 339, 678, 1355, 2709, 5417, 10833, 21665]",
        "one_trotter_step_time": 0.0053259823939967,
        "one_trotter_step_ccz_count": 14,
        "one_trotter_step_ccix_count": 0,
        "one_trotter_step_toffolis": 14,
        "logical_qubits": 25,
        "rotations": 2208717,
        "rotation_depth": 1472598,
        "t_gates": 693051,
        "ccz_count": 606116,
        "ccix_count": 0,
        "toffolis": 606116,
        "measurements": 606126,
    },
)

#: ``--one-step-scaled``: one traced Trotter step multiplied across the ladder. The
#: schedule and the traced step match the full-circuit run; only the ladder totals differ.
_HUBBARD_L2_ONE_STEP_SCALED = BenchmarkResults(
    results={
        "L": 2,
        "sites": 4,
        "system_qubits": 8,
        "terms": 29,
        "electrons": 4,
        "lambda": 48.0,
        "target_precision": 0.0204,
        "qpe_budget": 0.0136,
        "qpe_budget_fraction": 0.6666666666666666,
        "trotter_budget": 0.0068,
        "qpe_bits": 10,
        "num_unitary_queries": 1023,
        "base_time": 0.2253660323553513,
        "t_max": 230.77481713187973,
        "power_strategy": "rescale",
        "qpe_error_model": "sine-window-1sigma",
        "qpe_type": "standard-one-step-scaled",
        "hwp_enabled": True,
        "hwp_max_batch": 0,
        "trotter_steps_per_qpe_bit": "[43, 85, 170, 339, 678, 1355, 2709, 5417, 10833, 21665]",
        "one_trotter_step_time": 0.0053259823939967,
        "one_trotter_step_ccz_count": 14,
        "one_trotter_step_ccix_count": 0,
        "one_trotter_step_toffolis": 14,
        "logical_qubits": 25,
        "rotations": 2554346,
        "rotation_depth": 1688466,
        "t_gates": 2078112,
        "ccz_count": 606116,
        "ccix_count": 0,
        "toffolis": 606116,
        "measurements": 606116,
    },
)


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


def _as_python(value: Any) -> Any:
    """Return a numpy scalar as its Python equivalent."""
    return value.item() if hasattr(value, "item") else value


def _run(script: Any, tmp_path: Path, *extra_args: str, sizes: Sequence[str] = ("2",)) -> Any:
    """Run the script and return its table.

    Args:
        script: The loaded sampling module.
        tmp_path: Directory to write the table into.
        extra_args: Additional command-line flags.
        sizes: Lattice sizes to sample.

    Returns:
        The table the script wrote.

    """
    output_path = tmp_path / "hubbard_logical_resources.csv"
    assert script.main(["--size", *sizes, "-o", str(output_path), *extra_args]) == 0
    return pandas.read_csv(output_path)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        pytest.param((), _HUBBARD_L2_FULL_CIRCUIT, id="full-circuit"),
        pytest.param(("--one-step-scaled",), _HUBBARD_L2_ONE_STEP_SCALED, id="one-step-scaled"),
    ],
)
def test_sample_hubbard_L2(  # noqa: N802 - L is the lattice side
    script: Any,
    tmp_path: Path,
    extra_args: tuple[str, ...],
    expected: BenchmarkResults,
) -> None:
    """Pin the sampling script's 2x2 lattice result in both costing modes."""
    frame = _run(script, tmp_path, *extra_args)

    assert len(frame) == 1, "one row per lattice size is expected"
    assert set(frame.columns) == set(_PINNED_COLUMNS) | set(_TIMING_COLUMNS)

    first = frame.iloc[0]
    expected.check_results({column: _as_python(first[column]) for column in _PINNED_COLUMNS})


def test_one_step_scaling_is_exact(script: Any, tmp_path: Path) -> None:
    """Every scaled count is the traced step multiplied by the total step count."""
    frame = _run(script, tmp_path, "--one-step-scaled")
    row = frame.iloc[0]

    total_steps = sum(ast.literal_eval(row["trotter_steps_per_qpe_bit"]))
    assert row["ccz_count"] == row["one_trotter_step_ccz_count"] * total_steps
    assert row["ccix_count"] == row["one_trotter_step_ccix_count"] * total_steps


def test_modes_share_a_schema(script: Any, tmp_path: Path) -> None:
    """Both modes emit the same columns so their tables can be concatenated."""
    full = _run(script, tmp_path / "full", sizes=("2",))
    scaled = _run(script, tmp_path / "scaled", "--one-step-scaled", sizes=("2",))

    assert list(full.columns) == list(scaled.columns)
    # The schedule and the traced step do not depend on the costing mode.
    for column in (
        "trotter_steps_per_qpe_bit",
        "one_trotter_step_time",
        "one_trotter_step_ccz_count",
        "base_time",
    ):
        assert full.iloc[0][column] == scaled.iloc[0][column]


def test_multiple_sizes_accumulate(script: Any, tmp_path: Path) -> None:
    """A sweep writes one row per size into a single combined table."""
    frame = _run(script, tmp_path, "--one-step-scaled", sizes=("2", "4"))

    assert list(frame["L"]) == [2, 4]


@pytest.mark.parametrize("size", ["3", "0", "-2"])
def test_rejects_invalid_sizes(script: Any, tmp_path: Path, size: str) -> None:
    """Odd sizes and sizes below two are rejected before any sampling runs."""
    output_path = tmp_path / "hubbard_logical_resources.csv"
    with pytest.raises(SystemExit):
        script.main(["--size", size, "-o", str(output_path)])
    assert not output_path.exists()
