"""End-to-end tests for sample notebooks and other sample workflows.

This module contains tests for notebooks and interoperability samples
(Pennylane, Q#) that are not covered by dedicated test modules.

See Also:
- test_sample_workflow_sci.py - Sparse-CI workflow tests
- test_sample_workflow_rdkit.py - RDKit geometry tests
- test_sample_workflow_qiskit.py - Qiskit IQPE tests

To run the slow tests (including notebook e2e tests), set the environment variable:
    QDK_CHEMISTRY_RUN_SLOW_TESTS=1 pytest

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
from pathlib import Path

import pytest

# Optional dependencies for notebook execution
try:
    import nbformat
    from nbclient import NotebookClient

    _HAS_NOTEBOOK_DEPS = True
except ImportError:
    _HAS_NOTEBOOK_DEPS = False

_requires_notebook_deps = pytest.mark.xfail(
    not _HAS_NOTEBOOK_DEPS,
    reason="nbclient and nbformat are optional dependencies",
    raises=NameError,
)

try:
    from jupyter_client.kernelspec import find_kernel_specs

    _HAS_JUPYTER_CLIENT = True
except ImportError:
    _HAS_JUPYTER_CLIENT = False

# Environment variable to enable slow tests (including notebook e2e tests)
_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}


def _has_jupyter_kernel(kernel_name: str = "python3") -> bool:
    """Check if a Jupyter kernel is available."""
    if not _HAS_JUPYTER_CLIENT:
        return False
    try:
        return kernel_name in find_kernel_specs()
    except OSError:
        return False


_HAS_JUPYTER_KERNEL = _has_jupyter_kernel()

# Patterns that indicate visualization code that should be skipped
VISUALIZATION_PATTERNS = [
    "MoleculeViewer",
    "Histogram",
    "Circuit",
    "display_html_table",
    "display_warning",
]

# Import patterns that should be removed (visualization-only imports)
VISUALIZATION_IMPORT_PATTERNS = [
    "from qdk.widgets import MoleculeViewer",
    "from qdk.widgets import Histogram",
    "from qdk.widgets import Circuit",
]


def _contains_visualization(lines: list[str], start_idx: int) -> bool:
    """Check if a multi-line statement contains visualization code."""
    depth = 0
    for i in range(start_idx, len(lines)):
        line = lines[i]
        depth += line.count("(") - line.count(")")
        if any(pattern in line for pattern in VISUALIZATION_PATTERNS):
            return True
        if depth <= 0:
            break
    return False


def _get_indent_level(line: str) -> int:
    """Get the indentation level of a line (number of leading spaces)."""
    return len(line) - len(line.lstrip())


def _strip_visualization_lines(cell_source: str) -> str:
    """Remove visualization-related lines from cell source code.

    This preserves the rest of the cell's logic while removing only
    lines that contain visualization code. Handles multi-line statements
    by tracking parenthesis depth, and function definitions by tracking
    indentation.
    """
    lines = cell_source.split("\n")
    filtered_lines = []
    skip_depth = 0  # Track parenthesis depth when skipping multi-line statements
    skip_func_indent: int | None = None  # Track indentation when skipping function body

    for i, line in enumerate(lines):
        # If we're skipping a function body, continue until we hit a line with
        # the same or lesser indentation (that's not blank or a comment)
        if skip_func_indent is not None:
            stripped = line.strip()
            # Blank lines or comments inside the function body should be skipped
            if not stripped or stripped.startswith("#"):
                filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
                continue
            # If this line has greater indentation, it's still part of the function
            if _get_indent_level(line) > skip_func_indent:
                filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
                continue
            # Otherwise, we've exited the function body
            skip_func_indent = None

        # If we're in a skip block, continue skipping until parentheses balance
        if skip_depth > 0:
            skip_depth += line.count("(") - line.count(")")
            filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
            continue

        # Check if this line contains visualization code directly
        should_skip = any(pattern in line for pattern in VISUALIZATION_PATTERNS)

        # Also check for visualization-only imports
        if not should_skip:
            should_skip = any(pattern in line for pattern in VISUALIZATION_IMPORT_PATTERNS)

        # Check if this line starts a multi-line statement that contains visualization
        if not should_skip:
            open_parens = line.count("(") - line.count(")")
            if open_parens > 0 and _contains_visualization(lines, i + 1):
                should_skip = True

        if should_skip:
            # Check if this is a function definition - need to skip the entire body
            stripped = line.strip()
            if stripped.startswith("def "):
                skip_func_indent = _get_indent_level(line)
            # Start tracking parenthesis depth for multi-line statements
            skip_depth = line.count("(") - line.count(")")
            filtered_lines.append(f"# [test] Skipped: {line.strip()[:50]}")
        else:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _execute_notebook_skip_visualizations(notebook_path: Path, timeout: int = 600) -> None:
    """Execute a notebook, stripping visualization code from cells.

    Args:
        notebook_path: Path to the notebook file.
        timeout: Maximum time in seconds to wait for each cell execution.

    Raises:
        CellExecutionError: If a cell fails to execute.

    """
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Process cells to strip visualization lines
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        cell_source = cell.source

        # Skip empty cells
        if not cell_source.strip():
            continue

        # Strip visualization lines from the cell
        cell.source = _strip_visualization_lines(cell_source)

    # Set the working directory to the notebook's directory for relative paths
    notebook_dir = notebook_path.parent

    # Create a notebook client with appropriate kernel and execute
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_dir)}},
    )

    # Execute the entire notebook
    client.execute()


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


@_requires_notebook_deps
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_factory_list():
    """Test the examples/factory_list.ipynb notebook executes without errors."""
    notebook_path = EXAMPLES_DIR / "factory_list.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    _execute_notebook_skip_visualizations(notebook_path)


@_requires_notebook_deps
@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_SLOW_TESTS,
    reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
)
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_state_prep_energy():
    """Test the examples/state_prep_energy.ipynb notebook executes without errors."""
    notebook_path = EXAMPLES_DIR / "state_prep_energy.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    _execute_notebook_skip_visualizations(notebook_path)


@_requires_notebook_deps
@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_SLOW_TESTS,
    reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
)
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_qpe_stretched_n2():
    """Test the examples/qpe_stretched_n2.ipynb notebook executes without errors."""
    notebook_path = EXAMPLES_DIR / "qpe_stretched_n2.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    _execute_notebook_skip_visualizations(notebook_path)
