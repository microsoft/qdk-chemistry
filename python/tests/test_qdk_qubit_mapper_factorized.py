"""Tests for specialized fast mapping paths (Cholesky, Sparse) in QdkQubitMapper."""

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CholeskyHamiltonianContainer,
    Hamiltonian,
    LatticeGraph,
    MajoranaMapping,
    Orbitals,
    QubitHamiltonian,
)
from qdk_chemistry.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
    create_ppp_hamiltonian,
    ohno_potential,
)
from .test_helpers import create_test_basis_set


def assert_qubit_hamiltonians_equal(qh1: QubitHamiltonian, qh2: QubitHamiltonian, atol: float = 1e-12):
    """Compare two QubitHamiltonians term-by-term within absolute tolerance."""
    dict1 = {k: v for k, v in zip(qh1.pauli_strings, qh1.coefficients) if abs(v) > atol}
    dict2 = {k: v for k, v in zip(qh2.pauli_strings, qh2.coefficients) if abs(v) > atol}

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Ensure all significantly large coefficients are present in both
    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    for k in missing_in_2:
        assert abs(dict1[k]) < 1e-11, f"Term {k} is missing in dense path but present in fast path with value {dict1[k]}"
    for k in missing_in_1:
        assert abs(dict2[k]) < 1e-11, f"Term {k} is missing in fast path but present in dense path with value {dict2[k]}"

    # Check overlapping keys
    for k in keys1.intersection(keys2):
        val1 = dict1[k]
        val2 = dict2[k]
        assert np.isclose(val1, val2, atol=atol, rtol=1e-8), f"Coefficients mismatch for term {k}: fast={val1}, dense={val2}"


def make_dense_counterpart(h_fast: Hamiltonian) -> Hamiltonian:
    """Reconstruct a dense CanonicalFourCenterHamiltonianContainer from the fast one."""
    h1_alpha, h1_beta = h_fast.get_one_body_integrals()
    h2_aaaa, h2_aabb, h2_bbbb = h_fast.get_two_body_integrals()
    orbitals = h_fast.get_orbitals()
    core_energy = h_fast.get_core_energy()
    norb = h1_alpha.shape[0]

    if orbitals.is_restricted():
        fock = np.zeros((norb, norb))
        container_dense = CanonicalFourCenterHamiltonianContainer(
            h1_alpha, h2_aaaa, orbitals, core_energy, fock
        )
    else:
        fock_a = np.zeros((norb, norb))
        fock_b = np.zeros((norb, norb))
        container_dense = CanonicalFourCenterHamiltonianContainer(
            h1_alpha, h1_beta,
            h2_aaaa, h2_aabb, h2_bbbb,
            orbitals, core_energy, fock_a, fock_b
        )
    return Hamiltonian(container_dense)


def make_random_restricted_cholesky(norb: int, naux: int = 15, seed: int = 42):
    """Construct a mock restricted Cholesky Hamiltonian."""
    rng = np.random.default_rng(seed)

    one_body = rng.standard_normal((norb, norb))
    one_body = 0.5 * (one_body + one_body.T)

    three_center = rng.standard_normal((norb * norb, naux))

    coeffs = np.eye(norb)
    basis_set = create_test_basis_set(norb, f"test-restricted-{norb}")
    orbitals = Orbitals(coeffs, None, None, basis_set)

    fock = rng.standard_normal((norb, norb))
    fock = 0.5 * (fock + fock.T)

    container = CholeskyHamiltonianContainer(
        one_body, three_center, orbitals, 1.5, fock
    )
    return Hamiltonian(container)


def make_random_unrestricted_cholesky(norb: int, naux: int = 15, seed: int = 42):
    """Construct a mock unrestricted Cholesky Hamiltonian."""
    rng = np.random.default_rng(seed)

    one_body_a = rng.standard_normal((norb, norb))
    one_body_a = 0.5 * (one_body_a + one_body_a.T)
    one_body_b = rng.standard_normal((norb, norb))
    one_body_b = 0.5 * (one_body_b + one_body_b.T)

    three_center_aa = rng.standard_normal((norb * norb, naux))
    three_center_bb = rng.standard_normal((norb * norb, naux))

    coeffs_alpha = np.eye(norb)
    coeffs_beta = np.eye(norb)
    if norb >= 2:
        coeffs_beta[0, 0] = 0.8
        coeffs_beta[0, 1] = 0.6
        coeffs_beta[1, 0] = 0.6
        coeffs_beta[1, 1] = -0.8

    basis_set = create_test_basis_set(norb, f"test-unrestricted-{norb}")
    orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

    fock_a = rng.standard_normal((norb, norb))
    fock_a = 0.5 * (fock_a + fock_a.T)
    fock_b = rng.standard_normal((norb, norb))
    fock_b = 0.5 * (fock_b + fock_b.T)

    container = CholeskyHamiltonianContainer(
        one_body_a, one_body_b,
        three_center_aa, three_center_bb,
        orbitals, 2.0, fock_a, fock_b
    )
    return Hamiltonian(container)


# Sweeps: systems H2 (norb=2), H2O (norb=7), N2 (norb=14)
def test_cholesky_fast_path():
    """Verify the Cholesky fast path matches the dense counterpart term-by-term."""
    for norb in [2, 7, 14]:
        for is_restricted in [True, False]:
            for encoding in ["jordan_wigner", "bravyi_kitaev", "parity"]:
                # Construct the Cholesky Hamiltonian
                if is_restricted:
                    h_fast = make_random_restricted_cholesky(norb)
                else:
                    h_fast = make_random_unrestricted_cholesky(norb)

                # Get mapping
                num_modes = 2 * norb
                if encoding == "jordan_wigner":
                    mapping = MajoranaMapping.jordan_wigner(num_modes)
                elif encoding == "bravyi_kitaev":
                    mapping = MajoranaMapping.bravyi_kitaev(num_modes)
                else:
                    mapping = MajoranaMapping.parity(num_modes)

                # Map with fast path
                mapper = create("qubit_mapper", "qdk")
                qh_fast = mapper.run(h_fast, mapping)

                # Map with dense fallback
                h_dense = make_dense_counterpart(h_fast)
                qh_dense = mapper.run(h_dense, mapping)

                # Assert equivalence
                assert_qubit_hamiltonians_equal(qh_fast, qh_dense, atol=1e-12)


# Sweeps: 2x2 Hubbard, 1x4 Hubbard, PPP chain
def test_sparse_fast_path():
    """Verify the Sparse fast path matches the dense counterpart term-by-term."""
    for model_name in ["hubbard_2x2", "hubbard_1x4", "ppp_chain"]:
        for encoding in ["jordan_wigner", "bravyi_kitaev", "parity"]:
            # Construct the Lattice Graph and Hamiltonian
            if model_name == "hubbard_2x2":
                lattice = LatticeGraph.square(2, 2)
                h_fast = create_hubbard_hamiltonian(lattice, epsilon=-0.5, t=1.0, U=0.3)
            elif model_name == "hubbard_1x4":
                lattice = LatticeGraph.chain(4)
                h_fast = create_hubbard_hamiltonian(lattice, epsilon=-0.5, t=1.0, U=0.3)
            else:
                # PPP chain
                n = 4
                epsilon_r = 0.9
                u_val = 0.4
                r_val = 1.2
                lattice = LatticeGraph.chain(n)
                v = ohno_potential(lattice, U=u_val, R=r_val, epsilon_r=epsilon_r)
                epsilon_vec = np.zeros(n)
                t_mat = np.ones((n, n))
                u_vec = np.full(n, u_val)
                z_vec = np.ones(n)
                h_fast = create_ppp_hamiltonian(lattice, epsilon=epsilon_vec, t=t_mat, U=u_vec, V=v, z=z_vec)

            norb = lattice.num_sites
            num_modes = 2 * norb

            # Get mapping
            if encoding == "jordan_wigner":
                mapping = MajoranaMapping.jordan_wigner(num_modes)
            elif encoding == "bravyi_kitaev":
                mapping = MajoranaMapping.bravyi_kitaev(num_modes)
            else:
                mapping = MajoranaMapping.parity(num_modes)

            # Map with fast path
            mapper = create("qubit_mapper", "qdk")
            qh_fast = mapper.run(h_fast, mapping)

            # Map with dense fallback
            h_dense = make_dense_counterpart(h_fast)
            qh_dense = mapper.run(h_dense, mapping)

            # Assert equivalence
            assert_qubit_hamiltonians_equal(qh_fast, qh_dense, atol=1e-12)


# ─── PySCF-based molecular test suite ─────────────────────────────────────

try:
    import pyscf
    import pyscf.ao2mo
    import pyscf.gto
    import pyscf.scf
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

MOLECULES = [
    ("h2_sto3g", "H 0 0 0; H 0 0 0.74", "sto-3g"),
    ("h2o_sto3g", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "sto-3g"),
    ("n2_sto3g", "N 0 0 0; N 0 0 1.10", "sto-3g"),
]


def _molecular_cholesky_pair(atom: str, basis: str):
    """Build a restricted Cholesky Hamiltonian using PySCF."""
    mol = pyscf.gto.M(atom=atom, basis=basis, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, f"SCF did not converge for {atom} / {basis}"

    coeff = mf.mo_coeff
    norb = coeff.shape[1]
    h1 = coeff.T @ mf.get_hcore() @ coeff

    # Reconstruct two-body integrals using eigh Cholesky-like decomposition
    eri_mo = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, coeff), norb)
    pair_matrix = eri_mo.reshape(norb * norb, norb * norb)
    pair_matrix = 0.5 * (pair_matrix + pair_matrix.T)
    eigvals, eigvecs = np.linalg.eigh(pair_matrix)
    keep = eigvals > 1e-12
    factors = eigvecs[:, keep] * np.sqrt(eigvals[keep])  # (norb^2, naux)

    three_center = np.ascontiguousarray(factors)
    orbitals = create_test_basis_set(norb, f"test-pyscf-{norb}")
    # Wrap in Orbitals class
    coeffs = np.eye(norb)
    mo_orbitals = Orbitals(coeffs, None, None, orbitals)
    fock = np.zeros((norb, norb))

    h_fast = Hamiltonian(
        CholeskyHamiltonianContainer(
            h1, three_center, mo_orbitals, float(mf.energy_nuc()), fock
        )
    )
    return h_fast, norb


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not installed")
def test_cholesky_molecular_fast_path():
    """Verify that molecular Cholesky matches dense mapped version on real orbitals."""
    for name, atom, basis in MOLECULES:
        for encoding in ["jordan_wigner", "bravyi_kitaev", "parity"]:
            h_fast, norb = _molecular_cholesky_pair(atom, basis)
            num_modes = 2 * norb

            if encoding == "jordan_wigner":
                mapping = MajoranaMapping.jordan_wigner(num_modes)
            elif encoding == "bravyi_kitaev":
                mapping = MajoranaMapping.bravyi_kitaev(num_modes)
            else:
                mapping = MajoranaMapping.parity(num_modes)

            mapper = create("qubit_mapper", "qdk")
            qh_fast = mapper.run(h_fast, mapping)

            h_dense = make_dense_counterpart(h_fast)
            qh_dense = mapper.run(h_dense, mapping)

            assert_qubit_hamiltonians_equal(qh_fast, qh_dense, atol=1e-12)


