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

try:
    import nbformat
    from nbclient import NotebookClient

    _HAS_NOTEBOOK_DEPS = True
except ImportError:
    _HAS_NOTEBOOK_DEPS = False

try:
    from jupyter_client.kernelspec import find_kernel_specs

    _HAS_JUPYTER_CLIENT = True
except ImportError:
    _HAS_JUPYTER_CLIENT = False

_requires_notebook_deps = pytest.mark.xfail(
    not _HAS_NOTEBOOK_DEPS,
    reason="nbclient and nbformat are optional dependencies",
    raises=NameError,
)


def _has_jupyter_kernel(kernel_name: str = "python3") -> bool:
    """Check whether a Jupyter kernel is available."""
    if not _HAS_JUPYTER_CLIENT:
        return False
    try:
        return kernel_name in find_kernel_specs()
    except OSError:
        return False


_HAS_JUPYTER_KERNEL = _has_jupyter_kernel()

_VISUALIZATION_PATTERNS = [
    "MoleculeViewer",
    "Histogram",
    "Circuit",
    "display_html_table",
    "display_warning",
]

_VISUALIZATION_IMPORT_PATTERNS = [
    "from qdk.widgets import MoleculeViewer",
    "from qdk.widgets import Histogram",
    "from qdk.widgets import Circuit",
]


def _contains_visualization(lines: list[str], start_idx: int) -> bool:
    """Check whether a multiline statement contains visualization code."""
    depth = 0
    for line in lines[start_idx:]:
        depth += line.count("(") - line.count(")")
        if any(pattern in line for pattern in _VISUALIZATION_PATTERNS):
            return True
        if depth <= 0:
            break
    return False


def _get_indent_level(line: str) -> int:
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip())


def _strip_visualization_lines(cell_source: str) -> str:
    """Remove visualization statements while preserving other cell logic."""
    lines = cell_source.split("\n")
    filtered_lines = []
    skip_depth = 0
    skip_func_indent: int | None = None

    for line_index, line in enumerate(lines):
        if skip_func_indent is not None:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
                continue
            if _get_indent_level(line) > skip_func_indent:
                filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
                continue
            skip_func_indent = None

        if skip_depth > 0:
            skip_depth += line.count("(") - line.count(")")
            filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
            continue

        should_skip = any(pattern in line for pattern in _VISUALIZATION_PATTERNS)
        if not should_skip:
            should_skip = any(pattern in line for pattern in _VISUALIZATION_IMPORT_PATTERNS)
        if not should_skip:
            open_parens = line.count("(") - line.count(")")
            if open_parens > 0 and _contains_visualization(lines, line_index + 1):
                should_skip = True

        if should_skip:
            if line.strip().startswith("def "):
                skip_func_indent = _get_indent_level(line)
            skip_depth = line.count("(") - line.count(")")
            filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
        else:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _execute_notebook_skip_visualizations(
    notebook_path: Path,
    timeout: int = 1800,
    cell_patches: dict[int, dict[str, str]] | None = None,
) -> None:
    """Execute a notebook after removing visualization-only code.

    Args:
        notebook_path: Path to the notebook file.
        timeout: Maximum seconds allowed for each cell.
        cell_patches: Optional cell-indexed source replacements used to reduce
            expensive parameters without modifying the notebook on disk.

    """
    with open(notebook_path, encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.strip():
            cell.source = _strip_visualization_lines(cell.source)

    if cell_patches:
        for cell_index, replacements in cell_patches.items():
            assert cell_index < len(notebook.cells), (
                f"cell_patches: cell index {cell_index} out of range (notebook has {len(notebook.cells)} cells)"
            )
            assert notebook.cells[cell_index].cell_type == "code", f"cell_patches: cell {cell_index} is not a code cell"
            for old, new in replacements.items():
                assert old in notebook.cells[cell_index].source, (
                    f"cell_patches: string {old!r} not found in cell {cell_index}"
                )
                notebook.cells[cell_index].source = notebook.cells[cell_index].source.replace(old, new)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()


def _run_workflow(cmd, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Execute the workflow CLI with coverage-friendly defaults."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
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
