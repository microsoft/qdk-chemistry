"""End-to-end tests for sample notebooks without dedicated test modules.

See Also:
- test_sample_workflow_sci.py - Sparse-CI workflow tests
- test_sample_workflow_rdkit.py - RDKit geometry tests
- test_sample_workflow_qiskit.py - Qiskit IQPE tests
- test_sample_tutorial_gs_qpe.py - Ground-state QPE tutorial tests

To run slow notebook tests, set:
    QDK_CHEMISTRY_RUN_SLOW_TESTS=1 pytest

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
from pathlib import Path

import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

from .test_sample_workflow_utils import (
    _HAS_JUPYTER_KERNEL,
    _execute_notebook_skip_visualizations,
    _requires_notebook_deps,
)

try:
    import qdk.qre  # noqa: F401

    _HAS_QRE = True
except ImportError:
    _HAS_QRE = False

try:
    import pyscf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

#: Index of the configuration cell in ``examples/sossa_qre.ipynb``, which holds both
#: ``RUN_MODE`` and ``SIMULATION_SHOTS``.
_SOSSA_CONFIG_CELL = 4


@_requires_notebook_deps
@pytest.mark.skipif(
    not _HAS_JUPYTER_KERNEL,
    reason="Jupyter kernel 'python3' not available. Install ipykernel and register the kernel.",
)
def test_factory_list():
    """Test the factory-list notebook executes without errors."""
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
    """Test the state-preparation energy notebook executes without errors."""
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
    """Test the stretched-N2 QPE notebook executes without errors."""
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
    not _HAS_QRE,
    reason="qdk.qre not available",
)
@pytest.mark.parametrize(
    "run_mode",
    [
        "stored",
        pytest.param(
            "double_factorization",
            marks=pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available"),
        ),
        "synthetic",
    ],
)
def test_sossa_qre(run_mode):
    """Test every run mode of the SOSSA notebook executes without errors.

    The notebook is one pass through the SOSSA pipeline, selected by ``RUN_MODE``
    in its configuration cell. Each mode reaches the shared steps by a different
    route, so each is executed: ``"stored"`` is the only one that validates
    against a simulation, ``"double_factorization"`` is the only one that needs
    PySCF, and ``"synthetic"`` is the only one that runs the trade-off sweep.

    The shot patch cuts sampling breadth rather than the science: the shot count
    only sharpens the phase histogram, which nothing downstream reads. Narrowing
    the PSSPC, lattice-surgery or molecule sweeps instead would risk leaving the
    estimator with no feasible point.

    Both patches target the configuration cell, so this test fails loudly if that
    cell moves: ``_execute_notebook_skip_visualizations`` asserts the index is a
    code cell containing the text being replaced.
    """
    notebook_path = EXAMPLES_DIR / "sossa_qre.ipynb"
    assert notebook_path.exists(), f"Notebook not found: {notebook_path}"
    _execute_notebook_skip_visualizations(
        notebook_path,
        cell_patches={
            _SOSSA_CONFIG_CELL: {
                "SIMULATION_SHOTS = 50": "SIMULATION_SHOTS = 16",
                'RUN_MODE = "stored"': f'RUN_MODE = "{run_mode}"',
            },
        },
    )
