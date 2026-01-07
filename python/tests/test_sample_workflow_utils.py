"""Shared utility functions for sample workflow tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


def _run_workflow(cmd, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Execute the workflow CLI with coverage-friendly defaults."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def _skip_for_mpi_failure(result: subprocess.CompletedProcess[str]) -> None:
    """Skip the test when MPI cannot initialize."""
    mpi_err_indicators = [
        "PMIx server's listener thread failed to start",
        "ompi_mpi_init: ompi_rte_init failed",
        "Unable to start a daemon on the local node",
        "MPI_INIT failed",
        "pmix_ifinit: socket() failed",
        "opal_ifinit: socket() failed with errno=1",
    ]
    if any(ind in result.stderr for ind in mpi_err_indicators):
        pytest.skip("Skipping: MPI environment not available for QPE workflow")


def _collect_output_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Return combined stdout/stderr lines for downstream assertions."""
    return (result.stdout + "\n" + result.stderr).splitlines()


def _extract_float(pattern: str, text: str) -> float:
    """Extract the first floating-point value matching ``pattern`` from ``text``."""
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Pattern '{pattern}' not found in output.\n{text}")
    return float(match.group(1))


def _find_line(predicate: Callable[[str], bool], lines: list[str]) -> str:
    """Return the first line satisfying ``predicate`` or raise."""
    for line in lines:
        if predicate(line):
            return line
    raise AssertionError("Expected line not found in workflow output.")


def _extract_sparse_ci_summary(lines: list[str]) -> tuple[int, float, float]:
    """Parse the sparse-CI summary line and return determinant count, energy, and ΔE."""
    summary_line = _find_line(lambda line: "Sparse CI finder (" in line, lines)
    match = re.search(
        r"Sparse CI finder \((\d+) dets\) = ([\-0-9.]+) Hartree \(ΔE = ([\-0-9.]+) mHartree\)",
        summary_line,
    )
    if match is None:
        raise AssertionError(f"Unable to parse sparse CI finder line: {summary_line}")
    det_count = int(match.group(1))
    energy = float(match.group(2))
    delta_mhartree = float(match.group(3))
    return det_count, energy, delta_mhartree


def _assert_warning_constraints(lines: list[str], expected_warning: str | None, expect_no_warnings: bool) -> None:
    """Validate warning presence/absence expectations for a workflow run."""
    if expected_warning is not None:
        warning_line = _find_line(lambda line: expected_warning in line, lines)
        assert "[warning]" in warning_line, "Expected warning line missing logging prefix."
    if expect_no_warnings:
        assert all("[warning]" not in line for line in lines), (
            "Unexpected warning emitted by workflow.\nOutput:\n" + "\n".join(lines)
        )
