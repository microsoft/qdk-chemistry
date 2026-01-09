"""End-to-end tests for the OpenFermion sample workflows.

These tests ensure the public OpenFermion examples continue to emit the expected
summary values when executed as scripts.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure

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

    # Verify SCF and CASCI energies are correct
    scf_energy = _extract_float(r"SCF total energy:\s+([+\-0-9.]+) Hartree", result.stdout + result.stderr)
    casci_energy = _extract_float(r"CASCI total energy:\s+([+\-0-9.]+) Hartree", result.stdout + result.stderr)

    assert np.isclose(scf_energy, -7.86256780, atol=1e-7)
    assert np.isclose(casci_energy, -7.86277317, atol=1e-7)

    # Build QDK/Chemistry qubit Hamiltonian

    active_electrons = 2
    active_orbitals = 2

    structure = Structure(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.45 * ANGSTROM_TO_BOHR]], dtype=float),
        ["Li", "H"],
    )

    scf_solver = create("scf_solver")
    scf_energy, scf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

    selector = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=active_electrons,
        num_active_orbitals=active_orbitals,
    )
    active_orbitals = selector.run(scf_wavefunction).get_orbitals()

    constructor = create("hamiltonian_constructor")
    active_hamiltonian = constructor.run(active_orbitals)

    n_alpha = n_beta = active_electrons // 2
    mc_calculator = create("multi_configuration_calculator")
    casci_energy, casci_wavefunction = mc_calculator.run(active_hamiltonian, n_alpha, n_beta)

    assert "qiskit" in available("qubit_mapper")
    qubit_mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")

    # Obtain qubit Hamiltonian assuming block ordering - spin up first then spin down
    # Note if printed directly, the Pauli operators will not match with openFermion output
    qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)

    # Obtain that the ground state energy by diagonailizing the qubit Hamiltonian matrix
    jwt_matrix = qubit_hamiltonian.pauli_ops.to_matrix()
    eigenvalues = np.linalg.eigvalsh(jwt_matrix)
    ground_state_energy = np.min(eigenvalues)

    # Verify that the ground state energy matches that obtained from OpenFermion's Jordan-Wigner Hamiltonian
    of_jwt_energy = _extract_float(r"Ground state energy is\s+([+\-0-9.]+) Hartree", result.stdout + result.stderr)

    assert np.isclose(ground_state_energy + active_hamiltonian.get_core_energy(), of_jwt_energy, atol=1e-4)
