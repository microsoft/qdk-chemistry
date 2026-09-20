"""Test for the examples/benchmark/sample_hubbard_resources.py script."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "examples" / "benchmark" / "sample_hubbard_resources.py"

#: Run parameters the script derives itself, independent of the resource estimator.
_ALGORITHM_COLUMNS = (
    "L",
    "sites",
    "system_qubits",
    "terms",
    "electrons",
    "lambda",
    "sigma",
    "target_precision",
    "base_time",
    "t_max",
    "qpe_type",
    "max_power",
    "num_unitary_queries",
    "power_strategy",
    "effective_evolution_time",
    "trotter_budget",
    "num_divisions",
    "resolved_num_divisions",
    "num_divisions_for_largest_step",
    "steps_per_bit",
    "total_trotter_steps",
    "trotter_step_time",
    "step_max_error",
    "num_bits",
)

#: Estimator outputs, one entry per Pareto frontier point. The remaining columns are
#: wall-clock and memory measurements, which are machine dependent and not pinned.
_FRONTIER_COLUMNS = (
    "name",
    "qubits",
    "physical_compute_qubits",
    "physical_factory_qubits",
    "physical_memory_qubits",
    "factories",
    "step_runtime_s",
    "ladder_runtime_s",
    "ladder_runtime_days",
    "step_error",
    "error",
)


@dataclass
class BenchmarkResults:
    """Expected results from the benchmark sampling script."""

    results: dict[str, Any]  #: Sampled resource estimates

    @staticmethod
    def check_dict(
        expected_dict: dict[str, Any], actual_dict: dict[str, Any], errors: list[str]
    ) -> None:
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
            if isinstance(expected_value, (int, str)):
                if actual_value != expected_value:
                    errors.append(
                        f"Mismatch for key '{key}': expected {expected_value}, "
                        f"got {actual_value}"
                    )
            elif isinstance(expected_value, float):
                if actual_value != pytest.approx(expected_value):
                    errors.append(
                        f"Mismatch for key '{key}': expected {expected_value}, "
                        f"got {actual_value}"
                    )
            elif isinstance(expected_value, dict):
                BenchmarkResults.check_dict(expected_value, dict(actual_value), errors)
            elif isinstance(expected_value, list):
                if len(actual_value) != len(expected_value):
                    errors.append(
                        f"Mismatch for key '{key}': expected {len(expected_value)} "
                        f"entries, got {len(actual_value)}"
                    )
                    continue
                for index, expected_entry in enumerate(expected_value):
                    entry_errors: list[str] = []
                    BenchmarkResults.check_dict(
                        expected_entry, dict(actual_value[index]), entry_errors
                    )
                    errors.extend(f"{key}[{index}]: {error}" for error in entry_errors)
            else:
                raise ValueError(
                    f"Don't know how to handle: {key} of type {type(expected_value)}"
                )

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


_HUBBARD_L2 = BenchmarkResults(
    results={
        "algorithm": {
            "L": 2,
            "sites": 4,
            "system_qubits": 8,
            "terms": 29,
            "electrons": 4,
            "lambda": 48.0,
            "sigma": 0.0204,
            "target_precision": 0.0204,
            "base_time": 0.1503902733221216,
            "t_max": 153.9996398818526,
            "qpe_type": "standard-one-step-scaled",
            "max_power": 512,
            "num_unitary_queries": 1023,
            "power_strategy": "rescale",
            "effective_evolution_time": 76.9998199409263,
            "trotter_budget": 0.0204,
            "num_divisions": 0,
            "resolved_num_divisions": 8347,
            "num_divisions_for_largest_step": 8347,
            "steps_per_bit": "[17, 33, 66, 131, 261, 522, 1044, 2087, 4174, 8347]",
            "total_trotter_steps": 16682,
            "trotter_step_time": 0.0092248496395023,
            "step_max_error": 5.994485073732166e-07,
            "num_bits": 10,
        },
        "frontier": [
            {
                "name": "2x2-step",
                "qubits": 36605,
                "physical_compute_qubits": 14445,
                "physical_factory_qubits": 22160,
                "physical_memory_qubits": 0,
                "factories": "10×T",  # noqa: RUF001 - the estimator emits U+00D7
                "step_runtime_s": 0.044296,
                "ladder_runtime_s": 738.945872,
                "ladder_runtime_days": 0.0085526142592592,
                "step_error": 5.537454922547237e-07,
                "error": 0.0092375823017933,
            },
            {
                "name": "2x2-step",
                "qubits": 37493,
                "physical_compute_qubits": 8685,
                "physical_factory_qubits": 28808,
                "physical_memory_qubits": 0,
                "factories": "13×T",  # noqa: RUF001 - the estimator emits U+00D7
                "step_runtime_s": 0.035256,
                "ladder_runtime_s": 588.1405920000001,
                "ladder_runtime_days": 0.0068071827777777,
                "step_error": 5.537484982277561e-07,
                "error": 0.0092376324474354,
            },
            {
                "name": "2x2-step",
                "qubits": 42037,
                "physical_compute_qubits": 4365,
                "physical_factory_qubits": 37672,
                "physical_memory_qubits": 0,
                "factories": "17×T",  # noqa: RUF001 - the estimator emits U+00D7
                "step_runtime_s": 0.026216,
                "ladder_runtime_s": 437.335312,
                "ladder_runtime_days": 0.0050617512962962,
                "step_error": 5.690789606929511e-07,
                "error": 0.0094933752222798,
            },
            {
                "name": "2x2-step",
                "qubits": 50901,
                "physical_compute_qubits": 4365,
                "physical_factory_qubits": 46536,
                "physical_memory_qubits": 0,
                "factories": "21×T",  # noqa: RUF001 - the estimator emits U+00D7
                "step_runtime_s": 0.021696,
                "ladder_runtime_s": 361.932672,
                "ladder_runtime_days": 0.0041890355555555,
                "step_error": 5.608203727070247e-07,
                "error": 0.0093556054574985,
            },
        ],
    },
)


def _as_python(value: Any) -> Any:
    """Return a numpy scalar as its Python equivalent."""
    return value.item() if hasattr(value, "item") else value


def test_sample_hubbard_L2(tmp_path: Path) -> None:  # noqa: N802 - L is the lattice side
    """Pin the sampling script's 2x2 lattice result."""
    pytest.importorskip("qdk.qre", reason="the sample script estimates physical resources")
    pandas = pytest.importorskip("pandas")

    spec = importlib.util.spec_from_file_location("sample_hubbard_resources", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = script
    try:
        spec.loader.exec_module(script)
        output_path = tmp_path / "hubbard_resources.csv"
        assert script.main(["--size", "2", "-o", str(output_path)]) == 0
        frame = pandas.read_csv(output_path)
    finally:
        sys.modules.pop(spec.name, None)

    assert not frame.empty, "the estimator admitted no configuration for L=2"
    first = frame.iloc[0]
    actual_results = {
        "algorithm": {column: _as_python(first[column]) for column in _ALGORITHM_COLUMNS},
        "frontier": [
            {column: _as_python(row[column]) for column in _FRONTIER_COLUMNS}
            for _, row in frame.iterrows()
        ],
    }

    _HUBBARD_L2.check_results(actual_results)
