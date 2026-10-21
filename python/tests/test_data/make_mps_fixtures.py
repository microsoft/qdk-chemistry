"""Generate the ``mps_h2`` and ``mps_spin`` MPS/Hamiltonian fixtures used by ``test_state_preparation_mps.py``.

Each fixture stores ONE fixed physical qubit Hamiltonian and its exact ground state (found by direct
diagonalization, an independent classical eigensolver), plus one genuinely entangled, right-canonical
MPS (site 0 holding the orthogonality center) per ``site_to_orbital_order`` convention (identity and
reversed) used by the Q# state preparation circuit. The Hamiltonian/eigenvalue never change with
order; only the per-order MPS tensors change, so that -- once mapped back to the physical qubit basis
by that order's own ``site_to_orbital_order`` -- every order reconstructs the identical eigenstate.

``mps_h2`` is the *real* molecular H2/STO-3G Hamiltonian (bond length 0.74 Angstrom, the same
geometry/basis used by this repository's own ``test_qdk_qubit_mapper_factorized.py`` molecular
fixtures), restricted-Hartree-Fock orbitals, Jordan-Wigner mapped via this repository's own ``qdk``
qubit mapper. Its two spatial orbitals are each one ``physical_dimension=4`` chemistry site (the
``('0', 'u', 'd', '2')`` basis of two qubits per site), exercising the chemistry-orbital path.
``mps_spin`` is a small anisotropic XY spin chain with a staggered field
(``Jx*XX + Jy*YY + hA*Z0 + hB*Z1``, all four coefficients distinct, one qubit per site) so that a
site-order mistake changes the eigenstate's chain-order representation rather than leaving it invariant.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pyscf.ao2mo
import pyscf.gto
import pyscf.scf

sys.path.insert(0, str(Path(__file__).parent.parent))  # for helpers, used only by _h2_fixture()

from helpers import create_test_orbitals

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Configuration,
    Hamiltonian,
    MajoranaMapping,
)


def _reorder_matrix(num_sites, qubits_per_site, site_to_orbital_order):
    """Map the chain-order basis to the physical blocked Jordan-Wigner basis, including fermionic signs."""
    num_qubits = qubits_per_site * num_sites
    labels, parse = (
        ("01", Configuration.from_bitstring) if qubits_per_site == 1 else ("0ud2", Configuration.from_spin_half_string)
    )
    reorder = np.zeros((1 << num_qubits, 1 << num_qubits))
    for index in range(1 << num_qubits):
        orbital_states = ["0"] * num_sites
        created = []
        for site, orbital in enumerate(site_to_orbital_order):
            physical = (index >> (qubits_per_site * (num_sites - 1 - site))) & ((1 << qubits_per_site) - 1)
            orbital_states[orbital] = labels[physical]
            created += [channel * num_sites + orbital for channel in range(qubits_per_site) if physical >> channel & 1]
        bits = parse("".join(orbital_states)).to_bits(num_qubits)
        assert sorted(created) == [qubit for qubit, bit in enumerate(bits) if bit]
        inversions = sum(first > second for i, first in enumerate(created) for second in created[i + 1 :])
        reorder[int("".join(map(str, bits)), 2), index] = (-1) ** inversions
    return reorder


def _pauli_decompose(matrix, atol=1e-10):
    """Hilbert-Schmidt decomposition of a Hermitian matrix into Pauli strings and coefficients."""
    num_qubits = matrix.shape[0].bit_length() - 1
    paulis = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    pauli_strings, coefficients = [], []
    for labels in itertools.product("IXYZ", repeat=num_qubits):
        operator = np.array([[1.0]])
        for label in labels:
            operator = np.kron(operator, paulis[label])
        coefficient = np.trace(operator.conj().T @ matrix).real / matrix.shape[0]
        if abs(coefficient) > atol:
            pauli_strings.append("".join(labels))
            coefficients.append(coefficient)
    return pauli_strings, coefficients


def _svd_split_chain(chain_state, num_sites, physical_dim):
    """Right-canonicalize a chain-order state vector into ``num_sites`` MPS site tensors.

    Standard right-to-left SVD sweep: at each step the rightmost not-yet-factored physical index
    (merged with the bond already found to its right) is split off as an isometric site tensor,
    leaving the orthogonality center folded into the remaining (un-normalized) left factor, which
    becomes site 0 once every other site has been factored out.
    """
    tensors: list[np.ndarray] = [np.zeros(0)] * num_sites
    remaining = np.asarray(chain_state, dtype=float).reshape(-1, 1)
    for site in range(num_sites - 1, 0, -1):
        rows, bond_right = remaining.shape
        matrix = remaining.reshape(rows // physical_dim, physical_dim * bond_right)
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        bond_left = len(singular_values)
        tensors[site] = right.reshape(bond_left, physical_dim, bond_right)
        remaining = left * singular_values
    tensors[0] = remaining.reshape(1, physical_dim, remaining.shape[1])
    return [tensor.tolist() for tensor in tensors]


def _make_fixture(name, hamiltonian, ground_state, num_sites, qubits_per_site, reference_eigenvalue):
    """Store one fixed physical ``(hamiltonian, ground_state)`` plus per-order MPS tensors.

    ``hamiltonian``/``ground_state`` are the qubit Hamiltonian and its exact ground state in the
    physical qubit basis that ``QubitOperator``/QPE act on directly; this basis does not depend on
    ``site_to_orbital_order``. For each candidate order, ``_reorder_matrix`` is the orthogonal map
    (a signed permutation, so its transpose is its inverse) from that order's own MPS chain-order
    basis to the physical basis above; pulling ``ground_state`` back by that transpose gives the
    chain-order state whose per-order, right-canonical MPS tensors -- once mapped forward again by
    the Q# circuit using that same ``site_to_orbital_order`` -- reconstruct the identical physical
    ``ground_state``, regardless of order.
    """
    np.testing.assert_allclose(hamiltonian @ ground_state, reference_eigenvalue * ground_state, atol=1e-8)
    pauli_strings, coefficients = _pauli_decompose(hamiltonian)

    orders = {}
    for order in (list(range(num_sites)), list(reversed(range(num_sites)))):
        reorder = _reorder_matrix(num_sites, qubits_per_site, order)
        chain_state = reorder.T @ ground_state
        np.testing.assert_allclose(reorder @ chain_state, ground_state, atol=1e-8)
        orders[",".join(map(str, order))] = {
            "site_tensors": _svd_split_chain(chain_state, num_sites, 1 << qubits_per_site),
        }

    payload = {
        "num_sites": num_sites,
        "qubits_per_site": qubits_per_site,
        "reference_eigenvalue": reference_eigenvalue,
        "pauli_strings": pauli_strings,
        "coefficients": coefficients,
        "orders": orders,
    }
    (Path(__file__).parent / f"{name}.mps.json").write_text(json.dumps(payload, indent=1) + "\n")


def _h2_fixture():
    """Real H2/STO-3G Hamiltonian (RHF orbitals, Jordan-Wigner), ground state found by ``eigh``.

    Same geometry/basis (``H 0 0 0; H 0 0 0.74``, ``sto-3g``) as the ``h2_sto3g`` molecular case
    in ``test_qdk_qubit_mapper_factorized.py``. ``reference_eigenvalue`` is the eigenvalue of the
    stored qubit Hamiltonian, matching ``StandardPhaseEstimation.raw_energy`` without an added
    classical energy correction.
    """
    mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mean_field = pyscf.scf.RHF(mol)
    mean_field.conv_tol = 1e-12
    mean_field.kernel()
    assert mean_field.converged, "H2/STO-3G RHF did not converge"

    coeff = mean_field.mo_coeff
    num_orbitals = coeff.shape[1]
    one_body = coeff.T @ mean_field.get_hcore() @ coeff
    two_body = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, coeff), num_orbitals)
    hamiltonian = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            one_body,
            np.ascontiguousarray(two_body.ravel()),
            create_test_orbitals(num_orbitals),
            float(mean_field.energy_nuc()),
            np.eye(0),
        )
    )
    mapping = MajoranaMapping.jordan_wigner(num_modes=2 * num_orbitals)
    qubit_hamiltonian = create("qubit_mapper", "qdk").run(hamiltonian, mapping)

    matrix = qubit_hamiltonian.get_container().to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    assert eigenvalues[1] - eigenvalues[0] > 1e-6, (
        "H2 ground state must be non-degenerate for a well-defined MPS target"
    )
    ground_state = eigenvectors[:, 0].real
    _make_fixture(
        "mps_h2",
        matrix.real,
        ground_state,
        num_sites=num_orbitals,
        qubits_per_site=2,
        reference_eigenvalue=float(eigenvalues[0]),
    )


def _spin_fixture():
    """Anisotropic XY chain with a staggered field, ``H = Jx*XX + Jy*YY + hA*Z0 + hB*Z1``.

    Distinct field strengths make an incorrect site permutation observable against the fixed Hamiltonian.
    """
    pauli_x, pauli_y, pauli_z, eye2 = (
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
        np.eye(2),
    )
    hamiltonian = (
        1.0 * np.kron(pauli_x, pauli_x)
        + 0.6 * np.kron(pauli_y, pauli_y)
        + 0.3 * np.kron(pauli_z, eye2)
        - 0.2 * np.kron(eye2, pauli_z)
    ).real
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    assert eigenvalues[1] - eigenvalues[0] > 1e-6, (
        "spin ground state must be non-degenerate for a well-defined MPS target"
    )
    chain_state, reference_eigenvalue = eigenvectors[:, 0], float(eigenvalues[0])
    np.testing.assert_allclose(hamiltonian @ chain_state, reference_eigenvalue * chain_state, atol=1e-12)
    _make_fixture(
        "mps_spin", hamiltonian, chain_state, num_sites=2, qubits_per_site=1, reference_eigenvalue=reference_eigenvalue
    )


if __name__ == "__main__":
    _h2_fixture()
    _spin_fixture()
    print("Wrote mps_h2.mps.json and mps_spin.mps.json")
