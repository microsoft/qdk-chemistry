"""Tests for MP2 energy validation against PySCF reference."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Ansatz, CanonicalFourCenterHamiltonianContainer, Hamiltonian, Structure
from qdk_chemistry.data._spin_channels import spin_channel_vector
from qdk_chemistry.data.symmetry import axes

from .reference_tolerances import mp2_energy_tolerance
from .test_helpers import fermionic_reference_matrix, fourfold_average, non_fourfold_perturbation


def create_o2_molecule(bond_length=2.3):
    """Create O2 molecule for testing.

    Args:
        bond_length: O-O bond length in Bohr atomic units.

    Returns:
        Structure: O2 molecule structure.

    """
    coords = np.array([[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]])
    atomic_numbers = [8, 8]
    return Structure(coords, atomic_numbers)


class TestMP2Validation:
    """Test class for MP2 energy validation against PySCF."""

    def test_o2_rmp2_energy_validation(self):
        """Test restricted MP2 energy for singlet O2 against PySCF."""
        pyscf_mp2_corr_energy = -0.38428662586339435
        # Create O2 molecule
        o2_molecule = create_o2_molecule(bond_length=2.3)

        # QDK Chemistry HF calculation
        qdk_scf_solver = create("scf_solver")
        qdk_scf_solver.settings().set("method", "hf")

        _, hf_wavefunction = qdk_scf_solver.run(o2_molecule, 0, 1, "cc-pvdz")
        qdk_orbitals = hf_wavefunction.get_orbitals()

        # Create Hamiltonian for energy evaluation
        ham_constructor = create("hamiltonian_constructor")
        qdk_hamiltonian = ham_constructor.run(qdk_orbitals)

        # Create ansatz from Hamiltonian and wavefunction
        ansatz = Ansatz(qdk_hamiltonian, hf_wavefunction)

        # QDK MP2 calculation
        mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
        qdk_mp2_total_energy, _, _ = mp2_calculator.run(ansatz)
        reference_energy = ansatz.calculate_energy()
        qdk_mp2_corr_energy = qdk_mp2_total_energy - reference_energy

        assert abs(qdk_mp2_corr_energy - pyscf_mp2_corr_energy) < mp2_energy_tolerance, (
            f"MP2 correlation energy mismatch: QDK={qdk_mp2_corr_energy:.8f}, "
            f"PySCF={pyscf_mp2_corr_energy:.8f}, "
            f"diff={abs(qdk_mp2_corr_energy - pyscf_mp2_corr_energy):.2e}"
        )

    def test_o2_ump2_energy_validation(self):
        """Test unrestricted MP2 energies for O2."""
        ref = -0.3509471012518759
        # Create O2 molecule
        o2_molecule = create_o2_molecule(bond_length=2.3)

        # QDK Chemistry UHF calculation
        qdk_scf_solver = create("scf_solver")
        qdk_scf_solver.settings().set("method", "hf")

        _, hf_wavefunction = qdk_scf_solver.run(o2_molecule, 0, 3, "cc-pvdz")
        qdk_orbitals = hf_wavefunction.get_orbitals()

        # Create Hamiltonian for energy evaluation
        ham_constructor = create("hamiltonian_constructor")
        qdk_hamiltonian = ham_constructor.run(qdk_orbitals)

        # Create ansatz and use MP2Calculator
        ansatz = Ansatz(qdk_hamiltonian, hf_wavefunction)

        # QDK UMP2 calculation
        mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
        qdk_ump2_total_energy, _, _ = mp2_calculator.run(ansatz)
        reference_energy = ansatz.calculate_energy()
        qdk_ump2_corr_energy = qdk_ump2_total_energy - reference_energy

        # Check energy equality for unrestricted
        assert abs(qdk_ump2_corr_energy - ref) < mp2_energy_tolerance, (
            f"Unrestricted MP2 correlation energy mismatch: QDK={qdk_ump2_corr_energy:.8f}, "
            f"Ref={ref:.8f}, "
            f"diff={abs(qdk_ump2_corr_energy - ref):.2e}"
        )


@pytest.fixture(scope="module")
def h4_chain():
    """RHF reference and Hamiltonian of an H4 chain, with more than one occupied-virtual pair."""
    scf_solver = create("scf_solver")
    scf_solver.settings().set("method", "hf")
    structure = Structure(np.array([[0.0, 0.0, 1.6 * i] for i in range(4)]), [1] * 4)
    _, wavefunction = scf_solver.run(structure, 0, 1, "sto-3g")
    return wavefunction, create("hamiltonian_constructor").run(wavefunction.get_orbitals())


def _mp2_ansatz(h4_chain, two_body):
    """Ansatz for the H4 chain with its two-body integrals replaced."""
    wavefunction, hamiltonian = h4_chain
    container = CanonicalFourCenterHamiltonianContainer(
        hamiltonian.get_one_body_integrals()[0],
        two_body.ravel(),
        wavefunction.get_orbitals(),
        hamiltonian.get_core_energy(),
        np.empty((0, 0)),
    )
    return Ansatz(Hamiltonian(container), wavefunction)


@pytest.mark.parametrize(
    ("kind", "message"),
    [("exchange", "bra-ket average"), ("hermiticity", "non-Hermitian")],
    ids=["exchange", "hermiticity"],
)
def test_mp2_rejects_non_fourfold_two_body(h4_chain, kind, message):
    """MP2 reads (ia|jb) directly, so tensors outside the fourfold group would give wrong energies."""
    two_body = np.asarray(h4_chain[1].get_two_body_integrals()[0])
    n = round(len(two_body) ** 0.25)
    perturbation = non_fourfold_perturbation(np.random.default_rng(706).normal(size=(n,) * 4), kind)
    mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
    with pytest.raises(ValueError, match=message):
        mp2_calculator.run(_mp2_ansatz(h4_chain, two_body.reshape((n,) * 4) + 0.05 * perturbation))


def test_mp2_accepts_fourfold_two_body(h4_chain):
    """A 4-fold, non-8-fold tensor gives the second-order energy of the literal operator."""
    wavefunction, hamiltonian = h4_chain
    one_body = np.asarray(hamiltonian.get_one_body_integrals()[0])
    n = one_body.shape[0]
    two_body = np.asarray(hamiltonian.get_two_body_integrals()[0]).reshape((n,) * 4)
    two_body = two_body + 0.05 * fourfold_average(np.random.default_rng(706).normal(size=(n,) * 4))
    assert not np.allclose(two_body, two_body.swapaxes(0, 1))
    ansatz = _mp2_ansatz(h4_chain, two_body)
    total_energy, _, _ = create("dynamical_correlation_calculator", "qdk_mp2_calculator").run(ansatz)

    # Sum over doubly excited determinants of the literal operator, blocked alpha then beta modes.
    matrix = fermionic_reference_matrix(one_body, one_body, two_body, two_body, two_body).real
    energies = np.tile(spin_channel_vector(wavefunction.get_orbitals().energies(), axes.alpha()), 2)
    occupied = [0, 1, n, n + 1]
    reference_state = sum(1 << mode for mode in occupied)
    expected = 0.0
    for state in range(len(matrix)):
        removed = [mode for mode in occupied if not state >> mode & 1]
        added = [mode for mode in range(2 * n) if state >> mode & 1 and mode not in occupied]
        if len(removed) == len(added) == 2:
            denominator = energies[removed].sum() - energies[added].sum()
            expected += matrix[reference_state, state] * matrix[state, reference_state] / denominator
    assert total_energy - ansatz.calculate_energy() == pytest.approx(expected, abs=1e-10)
