"""Equivalence tests for QdkQubitMapper sparse / factorized fast paths.

These tests verify that mapping a Hamiltonian stored in a
:class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` or
:class:`~qdk_chemistry.data.SparseHamiltonianContainer` produces a
``QubitHamiltonian`` that is numerically equivalent, term-by-term, to the
dense (:class:`~qdk_chemistry.data.CanonicalFourCenterHamiltonianContainer`)
path for the *same* underlying integrals.  The fast paths never materialize a
dense N^4 two-body tensor (see ``QdkQubitMapper._run_impl`` and the C++
``majorana_map_hamiltonian_factorized`` dispatcher), so this is the primary
correctness bar from issue #474.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

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
)
from qdk_chemistry.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
    create_ppp_hamiltonian,
)

from .test_helpers import create_test_basis_set, create_test_orbitals

try:
    import pyscf  # noqa: F401
    import pyscf.ao2mo  # noqa: F401
    import pyscf.gto  # noqa: F401
    import pyscf.scf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

ENCODINGS = ["jordan_wigner", "bravyi_kitaev", "parity"]


# ─── Equivalence helper ───────────────────────────────────────────────────


def _assert_term_by_term_equivalent(qh_reference, qh_candidate, atol: float = 1e-12) -> None:
    """Assert two QubitHamiltonians match term-by-term after canonical sorting.

    Terms present in one operator but not the other are treated as having a
    zero coefficient in the missing operator; such a term passes only if its
    coefficient magnitude is below ``atol`` (i.e. it was legitimately pruned by
    the shared coefficient threshold).
    """
    ref = dict(zip(qh_reference.pauli_strings, qh_reference.coefficients, strict=True))
    cand = dict(zip(qh_candidate.pauli_strings, qh_candidate.coefficients, strict=True))

    assert qh_reference.num_qubits == qh_candidate.num_qubits

    for key in set(ref) | set(cand):
        a = ref.get(key, 0.0 + 0.0j)
        b = cand.get(key, 0.0 + 0.0j)
        assert abs(a - b) < atol, f"Coefficient mismatch for {key!r}: dense={a}, fast={b} (|diff|={abs(a - b):.3e})"


# ─── Cholesky factor construction (no external dependencies) ───────────────


def _symmetric_three_center(n: int, naux: int, rng: np.random.Generator) -> np.ndarray:
    """Random three-center factors symmetric in the orbital pair index.

    Symmetry ``L[p, q, Q] = L[q, p, Q]`` guarantees the reconstructed
    four-center integrals ``(pq|rs) = sum_Q L[p,q,Q] L[r,s,Q]`` carry the
    permutational symmetry of real-orbital electron-repulsion integrals.
    """
    raw = rng.standard_normal((n, n, naux)) * 0.3
    return 0.5 * (raw + raw.transpose(1, 0, 2))


def _four_center_from_factors(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Contract two three-center factor tensors into a dense ``(pq|rs)`` tensor."""
    return np.einsum("pqQ,rsQ->pqrs", left, right)


def _build_restricted_cholesky_pair(n: int, naux: int, seed: int):
    """Build (dense, cholesky) restricted Hamiltonians with identical integrals."""
    rng = np.random.default_rng(seed)
    factors = _symmetric_three_center(n, naux, rng)
    three_center = np.ascontiguousarray(factors.reshape(n * n, naux))
    eri = np.ascontiguousarray(_four_center_from_factors(factors, factors).ravel())

    raw = rng.standard_normal((n, n)) * 0.3
    one_body = (raw + raw.T) / 2 + np.diag(np.linspace(1.0, -0.5, n))

    orbitals = create_test_orbitals(n)
    core_energy = 0.7
    empty_fock = np.eye(0)

    dense = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, eri, orbitals, core_energy, empty_fock))
    cholesky = Hamiltonian(
        CholeskyHamiltonianContainer(one_body, three_center, orbitals, core_energy, empty_fock)
    )
    return dense, cholesky


def _build_unrestricted_cholesky_pair(n: int, naux: int, seed: int):
    """Build (dense, cholesky) unrestricted Hamiltonians with identical integrals."""
    rng = np.random.default_rng(seed)
    factors_a = _symmetric_three_center(n, naux, rng)
    factors_b = _symmetric_three_center(n, naux, rng)
    tc_aa = np.ascontiguousarray(factors_a.reshape(n * n, naux))
    tc_bb = np.ascontiguousarray(factors_b.reshape(n * n, naux))

    eri_aaaa = np.ascontiguousarray(_four_center_from_factors(factors_a, factors_a).ravel())
    eri_aabb = np.ascontiguousarray(_four_center_from_factors(factors_a, factors_b).ravel())
    eri_bbbb = np.ascontiguousarray(_four_center_from_factors(factors_b, factors_b).ravel())

    raw_a = rng.standard_normal((n, n)) * 0.3
    h1_alpha = (raw_a + raw_a.T) / 2 + np.diag(np.linspace(1.0, -0.5, n))
    raw_b = rng.standard_normal((n, n)) * 0.3
    h1_beta = (raw_b + raw_b.T) / 2 + np.diag(np.linspace(0.8, -0.3, n))

    coeffs_alpha = np.eye(n)
    coeffs_beta = np.eye(n) + rng.standard_normal((n, n)) * 0.1
    basis_set = create_test_basis_set(n, "chol-uhf")
    orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)
    assert not orbitals.is_restricted()

    core_energy = 0.7
    empty = np.eye(0)

    dense = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            h1_alpha, h1_beta, eri_aaaa, eri_aabb, eri_bbbb, orbitals, core_energy, empty, empty
        )
    )
    cholesky = Hamiltonian(
        CholeskyHamiltonianContainer(h1_alpha, h1_beta, tc_aa, tc_bb, orbitals, core_energy, empty, empty)
    )
    return dense, cholesky


# ─── Densify helper for sparse model Hamiltonians ──────────────────────────


def _densify(hamiltonian: Hamiltonian) -> Hamiltonian:
    """Return a CanonicalFourCenter copy of a Hamiltonian (forces dense path)."""
    h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
    aaaa, aabb, bbbb = hamiltonian.get_two_body_integrals()
    orbitals = hamiltonian.get_orbitals()
    core_energy = hamiltonian.get_core_energy()
    empty = np.eye(0)

    if orbitals.is_restricted():
        return Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
                np.array(h1_alpha), np.array(aaaa).ravel(), orbitals, core_energy, empty
            )
        )
    return Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            np.array(h1_alpha),
            np.array(h1_beta),
            np.array(aaaa).ravel(),
            np.array(aabb).ravel(),
            np.array(bbbb).ravel(),
            orbitals,
            core_energy,
            empty,
            empty,
        )
    )


# ─── Cholesky fast path: restricted ────────────────────────────────────────


class TestCholeskyFastPathRestricted:
    """Restricted Cholesky path equals the dense path for every encoding."""

    # Differing sizes exercise both single-word and multi-orbital dispatch.
    SYSTEMS = [(1, 3), (2, 5), (3, 8), (4, 6)]

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("n", "naux"), SYSTEMS)
    def test_cholesky_matches_dense(self, n: int, naux: int, encoding: str) -> None:
        dense, cholesky = _build_restricted_cholesky_pair(n, naux, seed=100 + n)
        assert cholesky.get_container_type() == "cholesky"
        assert dense.get_container_type() == "canonical_four_center"

        mapping = getattr(MajoranaMapping, encoding)(num_modes=2 * n)
        mapper = create("qubit_mapper", "qdk")
        qh_dense = mapper.run(dense, mapping)
        qh_cholesky = mapper.run(cholesky, mapping)

        _assert_term_by_term_equivalent(qh_dense, qh_cholesky)


# ─── Cholesky fast path: unrestricted ──────────────────────────────────────


class TestCholeskyFastPathUnrestricted:
    """Unrestricted Cholesky path equals the dense path for every encoding."""

    SYSTEMS = [(2, 5), (3, 7)]

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("n", "naux"), SYSTEMS)
    def test_cholesky_matches_dense(self, n: int, naux: int, encoding: str) -> None:
        dense, cholesky = _build_unrestricted_cholesky_pair(n, naux, seed=200 + n)
        assert cholesky.get_container_type() == "cholesky"

        mapping = getattr(MajoranaMapping, encoding)(num_modes=2 * n)
        mapper = create("qubit_mapper", "qdk")
        qh_dense = mapper.run(dense, mapping)
        qh_cholesky = mapper.run(cholesky, mapping)

        _assert_term_by_term_equivalent(qh_dense, qh_cholesky)


# ─── Cholesky fast path: real molecular systems (PySCF) ────────────────────

MOLECULES = [
    ("h2_sto3g", "H 0 0 0; H 0 0 0.74", "sto-3g"),
    ("h2o_sto3g", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "sto-3g"),
    ("n2_sto3g", "N 0 0 0; N 0 0 1.10", "sto-3g"),
]


def _molecular_cholesky_pair(atom: str, basis: str):
    """Build (dense, cholesky) restricted Hamiltonians from real MO integrals.

    The molecular electron-repulsion integral matrix is positive semidefinite,
    so its eigendecomposition yields exact three-center Cholesky-like factors.
    Both containers are constructed from the same factorization so equivalence
    holds to within tight floating-point tolerance.
    """
    mol = pyscf.gto.M(atom=atom, basis=basis, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    coeff = mf.mo_coeff
    norb = coeff.shape[1]
    h1 = coeff.T @ mf.get_hcore() @ coeff

    eri_mo = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, coeff), norb)
    pair_matrix = eri_mo.reshape(norb * norb, norb * norb)
    pair_matrix = 0.5 * (pair_matrix + pair_matrix.T)
    eigvals, eigvecs = np.linalg.eigh(pair_matrix)
    keep = eigvals > 1e-12
    factors = eigvecs[:, keep] * np.sqrt(eigvals[keep])  # (norb^2, naux)

    eri_recon = (factors @ factors.T).reshape(norb, norb, norb, norb)

    orbitals = create_test_orbitals(norb)
    enuc = float(mf.energy_nuc())
    empty = np.eye(0)

    dense = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(h1, np.ascontiguousarray(eri_recon.ravel()), orbitals, enuc, empty)
    )
    cholesky = Hamiltonian(
        CholeskyHamiltonianContainer(h1, np.ascontiguousarray(factors), orbitals, enuc, empty)
    )
    return dense, cholesky, norb


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestCholeskyFastPathMolecular:
    """Cholesky path equals the dense path on real molecular Hamiltonians."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("name", "atom", "basis"), MOLECULES, ids=[m[0] for m in MOLECULES])
    def test_molecular_cholesky_matches_dense(
        self, name: str, atom: str, basis: str, encoding: str
    ) -> None:
        dense, cholesky, norb = _molecular_cholesky_pair(atom, basis)
        assert cholesky.get_container_type() == "cholesky"

        mapping = getattr(MajoranaMapping, encoding)(num_modes=2 * norb)
        mapper = create("qubit_mapper", "qdk")
        qh_dense = mapper.run(dense, mapping)
        qh_cholesky = mapper.run(cholesky, mapping)

        _assert_term_by_term_equivalent(qh_dense, qh_cholesky)


# ─── Sparse fast path: lattice / model Hamiltonians ────────────────────────


def _hubbard_2x2() -> Hamiltonian:
    return create_hubbard_hamiltonian(LatticeGraph.square(2, 2), epsilon=0.0, t=1.0, U=4.0)


def _hubbard_1x4() -> Hamiltonian:
    return create_hubbard_hamiltonian(LatticeGraph.chain(4), epsilon=0.0, t=1.0, U=4.0)


def _ppp_1x4() -> Hamiltonian:
    return create_ppp_hamiltonian(LatticeGraph.chain(4), epsilon=0.0, t=1.0, U=4.0, V=1.5, z=1.0)


SPARSE_SYSTEMS = [
    ("hubbard_2x2", _hubbard_2x2),
    ("hubbard_1x4", _hubbard_1x4),
    ("ppp_1x4", _ppp_1x4),
]


class TestSparseFastPath:
    """Sparse path equals the dense path for lattice/model Hamiltonians."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("name", "factory"), SPARSE_SYSTEMS, ids=[s[0] for s in SPARSE_SYSTEMS])
    def test_sparse_matches_dense(self, name: str, factory, encoding: str) -> None:
        sparse = factory()
        assert sparse.get_container_type() == "sparse"

        dense = _densify(sparse)
        assert dense.get_container_type() == "canonical_four_center"

        n_spatial = sparse.get_one_body_integrals()[0].shape[0]
        mapping = getattr(MajoranaMapping, encoding)(num_modes=2 * n_spatial)
        mapper = create("qubit_mapper", "qdk")

        qh_sparse = mapper.run(sparse, mapping)
        qh_dense = mapper.run(dense, mapping)

        _assert_term_by_term_equivalent(qh_dense, qh_sparse)
