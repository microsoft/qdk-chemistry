"""Smoke tests for the reproducible Cholesky basis-transformation benchmark."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCHMARK = Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / "hamiltonian_basis_transformer.py"


def _run_benchmark(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a fresh interpreter so numerical thread settings take effect at import."""
    return subprocess.run(
        [sys.executable, str(BENCHMARK), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=180,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "QSHARP_PYTHON_TELEMETRY": "false"},
    )


@pytest.mark.parametrize(("warmups", "repeats"), [(0, 1), (1, 2)])
def test_benchmark_reports_timings_and_numerical_agreement(warmups, repeats):
    """Verify output shape and scientific agreement, not a wall-clock speedup."""
    completed = _run_benchmark("--case", "lih", "--threads", "1", "--warmups", str(warmups), "--repeats", str(repeats))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    configurations = [
        json.loads(line.removeprefix("BENCHMARK_CONFIG "))
        for line in completed.stdout.splitlines()
        if line.startswith("BENCHMARK_CONFIG ")
    ]
    results = [
        json.loads(line.removeprefix("BENCHMARK_RESULT "))
        for line in completed.stdout.splitlines()
        if line.startswith("BENCHMARK_RESULT ")
    ]
    assert len(configurations) == 1
    assert configurations[0]["package_version"]
    assert configurations[0]["python_version"]
    assert configurations[0]["platform"]
    assert configurations[0]["requested_threads"] == 1
    assert configurations[0]["warmups"] == warmups
    assert configurations[0]["repeats"] == repeats
    assert set(configurations[0]["thread_environment"].values()) == {"1"}
    assert len(results) == 1
    result = results[0]
    assert result["case"] == "lih"
    assert result["basis"] == "cc-pvdz"
    assert result["num_atomic_orbitals"] == 19
    assert result["num_active_orbitals"] == 6
    assert result["num_inactive_orbitals"] == 1
    assert len(result["active_indices"]) == 6
    assert result["inactive_indices"] == [0]
    assert result["cholesky_rank"] > 0
    assert result["cholesky_tolerance"] == 1e-8
    assert result["eri_threshold"] == 1e-12
    assert result["validation_tolerance"] == 1e-10
    assert result["comparison_tolerance"] == 1e-9
    assert result["rotation_seed"] == 588
    assert result["scf_ms"] >= 0
    assert result["source_build_ms"] >= 0
    for operation in ("transform", "rebuild"):
        assert len(result[operation]["samples_ms"]) == repeats
        assert 0 <= result[operation]["min_ms"] <= result[operation]["median_ms"] <= result[operation]["max_ms"]
    assert result["speedup"] == pytest.approx(result["rebuild"]["median_ms"] / result["transform"]["median_ms"])
    assert set(result["max_absolute_errors"]) == {"one_body", "three_center", "inactive_fock", "core_energy"}
    assert max(result["max_absolute_errors"].values()) <= result["comparison_tolerance"]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--threads", "0"),
        ("--warmups", "-1"),
        ("--repeats", "0"),
        ("--case", "unknown"),
    ],
)
def test_benchmark_rejects_invalid_controls(arguments):
    """Invalid controls fail before any molecular work is attempted."""
    completed = _run_benchmark(*arguments)
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "BENCHMARK_RESULT" not in completed.stdout
