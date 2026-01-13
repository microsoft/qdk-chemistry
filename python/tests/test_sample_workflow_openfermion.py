"""End-to-end tests for the OpenFermion sample workflows.

These tests ensure the public OpenFermion examples continue to emit the expected
summary values when executed as scripts. The test is performed by running the same molecule
with QDK chemistry and checking whether the Jordan-Wigner transformed Hamiltonian produces
the same ground state energy as OpenFermion.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure

from .reference_tolerances import float_comparison_absolute_tolerance, scf_energy_tolerance
from .test_sample_workflow_utils import (
    _extract_float,
    _run_workflow,
    _skip_for_mpi_failure,
)


def test_openfermion_molecular_hamiltonian_jordan_wigner():
    """Execute the OpenFermion Jordan-Wigner sample and validate reported energies."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "examples/openFermion/molecular_hamiltonian_jordan_wigner.py"]

    result = _run_workflow(cmd, repo_root)
    if result.returncode != 0 and "ModuleNotFoundError: No module named 'openfermion'" in result.stderr:
        pytest.skip("Skipping: OpenFermion not installed")
    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            "molecular_hamiltonian_jordan_wigner.py exited with "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    # Rebuild the QDK/Chemistry workflow to validate against the example output

    n_active_electrons = 2
    n_active_orbitals = 2

    structure = Structure(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.45 * ANGSTROM_TO_BOHR]], dtype=float),
        ["Li", "H"],
    )

    scf_solver = create("scf_solver")
    ref_scf_energy, scf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

    # Verify SCF energy is correct
    scf_energy = _extract_float(r"SCF total energy:\s+([+\-0-9.]+) Hartree", result.stdout + result.stderr)

    assert np.isclose(scf_energy, ref_scf_energy, atol=scf_energy_tolerance)  # make sure the same molecule is used

    selector = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=n_active_electrons,
        num_active_orbitals=n_active_orbitals,
    )
    active_orbitals = selector.run(scf_wavefunction).get_orbitals()

    constructor = create("hamiltonian_constructor")
    active_hamiltonian = constructor.run(active_orbitals)

    # Obtain qubit Hamiltonian assuming block ordering - spin up first then spin down
    # Note if printed directly, the Pauli operators will not match with OpenFermion output
    qubit_mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
    qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)

    # Obtain the ground state energy by diagonalizing the qubit Hamiltonian matrix
    jordan_wigner_matrix = qubit_hamiltonian.pauli_ops.to_matrix()
    eigenvalues = np.linalg.eigvalsh(jordan_wigner_matrix)
    ground_state_energy = eigenvalues[0]

    # Verify that the ground state energy matches that obtained from OpenFermion's Jordan-Wigner Hamiltonian
    openfermion_jordan_wigner_energy = _extract_float(
        r"Ground state energy is\s+([+\-0-9.]+) Hartree", result.stdout + result.stderr
    )

    assert np.isclose(
        ground_state_energy + active_hamiltonian.get_core_energy(),
        openfermion_jordan_wigner_energy,
        atol=float_comparison_absolute_tolerance,
    )
