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

import importlib.metadata
import os
import runpy
import sys
from importlib.util import module_from_spec, spec_from_file_location
from math import comb
from pathlib import Path

import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils import Logger

# Optional dependencies for notebook execution
try:
    import nbformat
    from nbclient import NotebookClient

    _HAS_NOTEBOOK_DEPS = True
except ImportError:
    _HAS_NOTEBOOK_DEPS = False

try:
    import qdk.qre  # noqa: F401

    _HAS_QRE = True
except ImportError:
    _HAS_QRE = False

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

try:
    import pyscf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

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


def _execute_notebook_skip_visualizations(
    notebook_path: Path,
    timeout: int = 1800,
    cell_patches: dict[int, dict[str, str]] | None = None,
) -> None:
    """Execute a notebook, stripping visualization code from cells.

    Args:
        notebook_path: Path to the notebook file.
        timeout: Maximum time in seconds to wait for each cell execution.
        cell_patches: Optional dict mapping cell indices to ``{old: new}``
            string replacements applied before execution.  Use this to
            inject lighter parameters at test time without modifying the
            notebook itself.

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

    # Apply cell-level text patches (e.g., lighter parameters for testing)
    if cell_patches:
        for cell_idx, replacements in cell_patches.items():
            assert cell_idx < len(nb.cells), (
                f"cell_patches: cell index {cell_idx} out of range (notebook has {len(nb.cells)} cells)"
            )
            assert nb.cells[cell_idx].cell_type == "code", f"cell_patches: cell {cell_idx} is not a code cell"
            for old, new in replacements.items():
                assert old in nb.cells[cell_idx].source, f"cell_patches: string {old!r} not found in cell {cell_idx}"
                nb.cells[cell_idx].source = nb.cells[cell_idx].source.replace(old, new)

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
DOCS_PYTHON_EXAMPLES_DIR = Path(__file__).parent.parent.parent / "docs" / "source" / "_static" / "examples" / "python"
TUTORIAL_VERSIONS_FILE = Path(__file__).parent.parent.parent / "docs" / "source" / "tutorials" / "_versions.py"
GROUND_STATE_TUTORIAL_VERSION = str(runpy.run_path(str(TUTORIAL_VERSIONS_FILE))["GROUND_STATE_TUTORIAL_VERSION"])
_INSTALLED_QDK_CHEMISTRY_VERSION = importlib.metadata.version("qdk-chemistry")
_RUN_GROUND_STATE_TUTORIAL_BASELINES = _INSTALLED_QDK_CHEMISTRY_VERSION == GROUND_STATE_TUTORIAL_VERSION
_requires_ground_state_tutorial_version = pytest.mark.skipif(
    not _RUN_GROUND_STATE_TUTORIAL_BASELINES,
    reason=(
        f"Ground-state tutorial baselines require qdk-chemistry=={GROUND_STATE_TUTORIAL_VERSION}; "
        f"installed {_INSTALLED_QDK_CHEMISTRY_VERSION}."
    ),
)


def _load_tutorial_module(module_name: str):
    """Load a course script whose neighboring scripts may be imported."""
    script_path = DOCS_PYTHON_EXAMPLES_DIR / f"{module_name}.py"
    module_spec = spec_from_file_location(module_name, script_path)
    assert module_spec is not None
    assert module_spec.loader is not None
    tutorial_module = module_from_spec(module_spec)
    sys.modules[module_spec.name] = tutorial_module
    sys.path.insert(0, str(DOCS_PYTHON_EXAMPLES_DIR))
    try:
        module_spec.loader.exec_module(tutorial_module)
    except BaseException:
        sys.modules.pop(module_spec.name, None)
        raise
    finally:
        sys.path.pop(0)
    return tutorial_module


def test_load_tutorial_module_removes_failed_import(tmp_path, monkeypatch):
    """Do not cache a partially initialized tutorial module after import failure."""
    module_name = "tutorial_failing_import"
    (tmp_path / f"{module_name}.py").write_text("raise RuntimeError('failed import')\n")
    monkeypatch.setattr(sys.modules[__name__], "DOCS_PYTHON_EXAMPLES_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="failed import"):
        _load_tutorial_module(module_name)

    assert module_name not in sys.modules


def test_tutorial_qpe_setup_accepts_local_build_version():
    """Accept local metadata only when the public version matches the pin."""
    setup_script = DOCS_PYTHON_EXAMPLES_DIR / "tutorial_qpe_setup.py"
    setup_globals = runpy.run_path(str(setup_script))
    public_version = setup_globals["public_version"]

    assert public_version(f"{GROUND_STATE_TUTORIAL_VERSION}+local") == GROUND_STATE_TUTORIAL_VERSION
    assert public_version("2.0.1") != GROUND_STATE_TUTORIAL_VERSION


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="Tutorial workflow requires PySCF")
def test_tutorial_module_imports_preserve_global_logging():
    """Reusable tutorial imports must not change process-wide logging."""
    previous_level = Logger.get_global_level()
    Logger.set_global_level(Logger.LogLevel.warn)
    try:
        for module_name in (
            "tutorial_choose_active_space",
            "tutorial_map_n2_to_qubits",
            "tutorial_prepare_trial_state",
        ):
            _load_tutorial_module(module_name)
        assert Logger.get_global_level() == "warn"
    finally:
        Logger.set_global_level(previous_level)


@pytest.mark.tutorial_baseline
@_requires_ground_state_tutorial_version
def test_tutorial_choose_active_space_results():
    """Check the stable numerical and structural results used by Chapter 3."""
    tutorial_module = _load_tutorial_module("tutorial_choose_active_space")

    result = tutorial_module.run_active_space_workflow()
    assert abs(result.hartree_fock_energy - (-108.418633697214)) < 1e-8
    assert abs(result.valence_energy - (-108.778369520882)) < 1e-8
    assert abs(result.natural_orbital_energy - result.valence_energy) < 1e-10
    assert abs(result.refined_energy - (-108.771051792900)) < 1e-8
    assert result.valence_indices == list(range(2, 10))
    assert result.num_valence_determinants == comb(8, 5) ** 2 == 3136
    assert result.inactive_indices == list(range(4))
    assert result.refined_indices == list(range(4, 10))
    assert result.num_refined_electrons == 6
    assert result.num_virtual_orbitals == 18
    assert result.num_refined_determinants == comb(6, 3) ** 2 == 400
    assert result.orbital_entropies == pytest.approx(
        [
            0.021695655,
            0.029962803,
            0.547855061,
            0.963884097,
            0.963884097,
            0.966011090,
            0.966011090,
            0.554008809,
        ],
        abs=1e-8,
    )
    assert result.valence_energy < result.refined_energy < result.hartree_fock_energy

    cube_data = tutorial_module.generate_active_orbital_cube_data(
        result,
        grid_size=(8, 8, 8),
        margin=4.0,
    )
    assert len(cube_data) == 8
    assert sum(orbital["info"]["Selected by autoCAS"] == "yes" for orbital in cube_data.values()) == 6
    assert all(
        set(orbital["info"]) == {"Occupation", "Entropy", "Selected by autoCAS"} for orbital in cube_data.values()
    )


@pytest.mark.tutorial_baseline
@_requires_ground_state_tutorial_version
def test_tutorial_map_n2_to_qubits_results():
    """Check the stable Jordan-Wigner mapping results used by Chapter 4."""
    _load_tutorial_module("tutorial_choose_active_space")
    tutorial_module = _load_tutorial_module("tutorial_map_n2_to_qubits")

    result = tutorial_module.run_qubit_mapping_workflow()
    assert result.active_space_result.refined_indices == list(range(4, 10))
    assert result.num_active_spatial_orbitals == 6
    assert result.num_active_spin_orbitals == 12
    assert result.num_compute_qubits == 12
    assert result.num_pauli_terms == 383
    assert result.qubit_hamiltonian.encoding == "jordan-wigner"
    assert result.qubit_hamiltonian.fermion_mode_order.value == "blocked"
    assert abs(result.core_energy - (-99.117775949333)) < 1e-8
    assert result.num_fixed_electron_states == 400
    assert abs(result.mapped_active_energy - (-9.653275843566)) < 1e-8
    assert abs(result.mapped_total_energy - result.active_space_result.refined_energy) < 1e-10
    assert abs(result.mapping_energy_difference) < 1e-10

    preview_terms = tutorial_module.representative_pauli_terms(result.qubit_hamiltonian)
    assert len(preview_terms) == 8
    assert preview_terms[0][0] == "I" * result.num_compute_qubits
    assert all(set(pauli_string).issubset({"I", "Z"}) for pauli_string, _ in preview_terms[1:4])
    assert all("X" in pauli_string or "Y" in pauli_string for pauli_string, _ in preview_terms[4:])
    assert tutorial_module.format_pauli_string("IXYI") == "Y(qubit 1) X(qubit 2)"


@pytest.mark.tutorial_baseline
@_requires_ground_state_tutorial_version
def test_tutorial_prepare_trial_state_results():
    """Check the trial-state quality and cost results used by Chapter 5."""
    _load_tutorial_module("tutorial_choose_active_space")
    tutorial_module = _load_tutorial_module("tutorial_prepare_trial_state")

    result = tutorial_module.run_trial_state_workflow()
    assert result.active_space_result.num_refined_determinants == 400
    assert len(result.reference_determinants) == 8
    assert result.reference_determinants[0].occupation == "222000"
    assert abs(result.reference_determinants[0].amplitude - (-0.694657450061)) < 1e-10
    assert abs(result.reference_determinants[0].weight - 0.482548972925) < 1e-10
    assert result.reference_determinants[1].occupation == "202020"
    assert result.reference_determinants[2].occupation == "220200"
    assert abs(result.reference_determinants[2].cumulative_weight - 0.691383326350) < 1e-10
    assert len(result.trial_states) == 3

    one_determinant, two_determinants, four_determinants = result.trial_states
    assert one_determinant.num_determinants == 1
    assert abs(one_determinant.fidelity - 0.482548972925) < 1e-10
    assert one_determinant.num_compute_qubits == 12
    assert one_determinant.num_logical_gates == 6
    assert one_determinant.logical_gate_counts == {"X": 6}

    assert two_determinants.num_determinants == 2
    assert abs(two_determinants.fidelity - 0.578050111416) < 1e-10
    assert two_determinants.num_compute_qubits == 12
    assert two_determinants.num_logical_gates == 14
    assert two_determinants.logical_gate_counts == {
        "CNOT": 6,
        "H": 2,
        "Rz": 2,
        "S": 2,
        "X": 2,
    }

    assert four_determinants.num_determinants == 4
    assert abs(four_determinants.fidelity - 0.717362840569) < 1e-10
    assert four_determinants.num_compute_qubits == 12
    assert four_determinants.num_logical_gates == 30
    assert four_determinants.logical_gate_counts == {
        "CNOT": 16,
        "H": 4,
        "Rz": 4,
        "S": 4,
        "X": 2,
    }


@pytest.mark.tutorial_baseline
@_requires_ground_state_tutorial_version
def test_tutorial_run_iqpe_configuration(capsys):
    """Check the fast IQPE configuration and phase-grid values used by Chapter 6."""
    _load_tutorial_module("tutorial_choose_active_space")
    _load_tutorial_module("tutorial_map_n2_to_qubits")
    _load_tutorial_module("tutorial_prepare_trial_state")
    tutorial_module = _load_tutorial_module("tutorial_run_iqpe")

    problem = tutorial_module.prepare_iqpe_problem()
    assert problem.mapping.num_compute_qubits == 12
    assert problem.trial_state.num_determinants == 4
    assert abs(problem.trial_state.fidelity - 0.717362840569) < 1e-10
    assert problem.num_phase_bits == 6
    assert problem.shots_per_bit == 3
    assert len(problem.iteration_circuits) == 6
    assert abs(problem.mapping.qubit_hamiltonian.schatten_norm - 20.185236582112) < 1e-10
    assert problem.evolution_time.grid_bitstring == "110001"
    assert abs(problem.evolution_time.bound_time_hartree_inverse - 0.155638138835) < 1e-12
    assert abs(problem.evolution_time.bound_reference_phase_fraction - 0.760882767) < 1e-9
    assert abs(problem.evolution_time.time_hartree_inverse - 0.152567288817) < 1e-12
    assert abs(problem.evolution_time.reference_phase_fraction - 0.765600718162) < 1e-10
    assert problem.evolution_time.grid_phase_fraction == 49 / 64
    assert abs(problem.evolution_time.grid_active_energy_hartree - problem.mapping.mapped_active_energy - 1e-3) < 1e-12

    first_run = tutorial_module.IqpeRun(
        seed=1,
        bitstring="110001",
        phase_fraction=49 / 64,
        active_energy_hartree=-9.652275843566,
        total_energy_hartree=-108.770051792900,
        error_hartree=1e-3,
        runtime_seconds=1.0,
    )
    neighboring_run = tutorial_module.IqpeRun(
        seed=2,
        bitstring="110010",
        phase_fraction=50 / 64,
        active_energy_hartree=-9.008790787328,
        total_energy_hartree=-108.126566736662,
        error_hartree=0.644485056238,
        runtime_seconds=1.0,
    )
    counts, mode = tutorial_module.select_unique_mode([first_run, neighboring_run, first_run])
    assert counts == {"110001": 2, "110010": 1}
    assert mode is first_run
    with pytest.raises(RuntimeError, match="no unique mode"):
        tutorial_module.select_unique_mode([first_run, neighboring_run])

    tutorial_module.print_iqpe_settings(
        problem,
        num_complete_runs=20,
        first_seed=42,
    )
    settings_output = capsys.readouterr().out
    for expected_text in (
        "Trial determinants: 4",
        "Readout ancillas: 1",
        "Phase bits: 6",
        "Shots per bit: 3",
        "Complete runs: 20",
        "Simulator seeds: 42-61",
        "first-order Trotter product formula",
        "Trotter divisions: 1",
        "repeated approximate base unitary",
        "0.760882766860",
        "0.152567288817 Hartree^-1",
    ):
        assert expected_text in settings_output


@pytest.mark.slow
@pytest.mark.tutorial_baseline
@_requires_ground_state_tutorial_version
@pytest.mark.skipif(
    not _RUN_SLOW_TESTS,
    reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
)
def test_tutorial_run_iqpe_simulation():
    """Check one complete simulator run from the final Chapter 6 workflow."""
    _load_tutorial_module("tutorial_choose_active_space")
    _load_tutorial_module("tutorial_map_n2_to_qubits")
    _load_tutorial_module("tutorial_prepare_trial_state")
    tutorial_module = _load_tutorial_module("tutorial_run_iqpe")

    problem = tutorial_module.prepare_iqpe_problem()
    run = tutorial_module.run_complete_iqpe(problem, seed=42)
    assert run.bitstring == "110001"
    assert abs(run.error_hartree - 1e-3) < 1e-10


@_requires_notebook_deps
@pytest.mark.tutorial_baseline
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_tutorial_choose_active_space_notebook():
    """Test the Chapter 3 notebook chemistry and visualization-data cells."""
    notebook_path = DOCS_PYTHON_EXAMPLES_DIR / "tutorial_choose_active_space.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    with open(notebook_path, encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)
    nbformat.validate(notebook)

    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert "/Users/" not in notebook_text
    assert "\\Users\\" not in notebook_text
    assert "kernelspec" not in notebook.metadata
    assert notebook.metadata.get("language_info", {}) == {"name": "python"}
    for cell in notebook.cells:
        expected_language = "python" if cell.cell_type == "code" else "markdown"
        assert cell.metadata.get("language") == expected_language
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs

    if not _RUN_GROUND_STATE_TUTORIAL_BASELINES:
        pytest.skip(
            f"Notebook execution requires qdk-chemistry=={GROUND_STATE_TUTORIAL_VERSION}; "
            f"installed {_INSTALLED_QDK_CHEMISTRY_VERSION}."
        )

    _execute_notebook_skip_visualizations(
        notebook_path,
        timeout=360,
        cell_patches={
            5: {
                "grid_size=(30, 30, 30)": "grid_size=(8, 8, 8)",
            },
        },
    )


@_requires_notebook_deps
@pytest.mark.tutorial_baseline
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_tutorial_prepare_trial_state_notebook():
    """Test the Chapter 5 notebook circuit data and validation cells."""
    notebook_path = DOCS_PYTHON_EXAMPLES_DIR / "tutorial_prepare_trial_state.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    with open(notebook_path, encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)
    nbformat.validate(notebook)

    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert "/Users/" not in notebook_text
    assert "\\Users\\" not in notebook_text
    assert "kernelspec" not in notebook.metadata
    assert notebook.metadata.get("language_info", {}) == {"name": "python"}
    for cell in notebook.cells:
        expected_language = "python" if cell.cell_type == "code" else "markdown"
        assert cell.metadata.get("language") == expected_language
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs

    if not _RUN_GROUND_STATE_TUTORIAL_BASELINES:
        pytest.skip(
            f"Notebook execution requires qdk-chemistry=={GROUND_STATE_TUTORIAL_VERSION}; "
            f"installed {_INSTALLED_QDK_CHEMISTRY_VERSION}."
        )

    _execute_notebook_skip_visualizations(notebook_path, timeout=360)


@_requires_notebook_deps
@pytest.mark.tutorial_baseline
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_tutorial_visualize_iqpe_circuit_notebook():
    """Test the Chapter 6 notebook circuit construction and validation cells."""
    notebook_path = DOCS_PYTHON_EXAMPLES_DIR / "tutorial_visualize_iqpe_circuit.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    with open(notebook_path, encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)
    nbformat.validate(notebook)

    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert "/Users/" not in notebook_text
    assert "\\Users\\" not in notebook_text
    assert "kernelspec" not in notebook.metadata
    assert notebook.metadata.get("language_info", {}) == {"name": "python"}
    for cell in notebook.cells:
        expected_language = "python" if cell.cell_type == "code" else "markdown"
        assert cell.metadata.get("id") == cell.id
        assert cell.metadata.get("language") == expected_language
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs

    if not _RUN_GROUND_STATE_TUTORIAL_BASELINES:
        pytest.skip(
            f"Notebook execution requires qdk-chemistry=={GROUND_STATE_TUTORIAL_VERSION}; "
            f"installed {_INSTALLED_QDK_CHEMISTRY_VERSION}."
        )

    _execute_notebook_skip_visualizations(notebook_path, timeout=360)


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
@pytest.mark.skipif(
    not QDK_CHEMISTRY_HAS_QISKIT,
    reason="Qiskit dependencies not available",
)
@pytest.mark.skipif(
    not PYSCF_AVAILABLE,
    reason="PySCF not available",
)
def test_state_prep_energy():
    """Test the examples/state_prep_energy.ipynb notebook executes without errors."""
    notebook_path = EXAMPLES_DIR / "state_prep_energy.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    _execute_notebook_skip_visualizations(
        notebook_path,
        cell_patches={
            25: {
                "total_shots=600000": "total_shots=50000",
            },
        },
    )


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
@pytest.mark.skipif(
    not QDK_CHEMISTRY_HAS_QISKIT,
    reason="Qiskit dependencies not available",
)
@pytest.mark.skipif(
    not _HAS_QRE,
    reason="qdk.qre not available",
)
@pytest.mark.skipif(
    not PYSCF_AVAILABLE,
    reason="PySCF not available",
)
def test_qpe_stretched_n2():
    """Test the examples/qpe_stretched_n2.ipynb notebook executes without errors."""
    notebook_path = EXAMPLES_DIR / "qpe_stretched_n2.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    _execute_notebook_skip_visualizations(
        notebook_path,
        cell_patches={
            34: {
                "NUM_TRIALS = 20": "NUM_TRIALS = 3",
            },
        },
    )
