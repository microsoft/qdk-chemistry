"""Equivalence tests for QdkQubitMapper sparse / factorized fast paths.

These tests verify that mapping a Hamiltonian stored in a
:class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` or
:class:`~qdk_chemistry.data.SparseHamiltonianContainer` produces a
``QubitOperator`` that is numerically equivalent, term-by-term, to the
dense (:class:`~qdk_chemistry.data.CanonicalFourCenterHamiltonianContainer`)
path for the *same* underlying integrals.  The fast paths never materialize a
dense N^4 two-body tensor (the C++ ``majorana_map_hamiltonian`` overload taking
a ``Hamiltonian`` dispatches on the container type), so this is the primary
correctness bar from issue #474.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CholeskyHamiltonianContainer,
    Hamiltonian,
    HamiltonianType,
    LatticeGraph,
    MajoranaMapping,
    Orbitals,
    SparseHamiltonianContainer,
    Symmetries,
)
from qdk_chemistry.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
    create_ppp_hamiltonian,
)

from .helpers import (
    create_dense_test_hamiltonian,
    create_test_basis_set,
    create_test_orbitals,
    eightfold_orbit,
    fermionic_reference_matrix,
    fourfold_orbit,
)

try:
    import pyscf
    import pyscf.ao2mo
    import pyscf.gto
    import pyscf.scf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

ENCODINGS = ["jordan_wigner", "bravyi_kitaev", "parity"]


# ─── Equivalence helper ───────────────────────────────────────────────────


def _sum_coefficients_by_pauli(qh) -> dict:
    """Collapse a QubitOperator into a {pauli_string: summed_coefficient} map.

    Coefficients are summed per Pauli string so the comparison is correct even
    if an operator ever emits the same Pauli string in more than one term
    (rather than silently overwriting, as a plain ``dict(zip(...))`` would).
    ``strict=True`` enforces equal lengths (and is required by the repo's
    ``B905`` lint rule; the project already targets Python >= 3.10).
    """
    acc: dict = {}
    for label, coeff in zip(qh.pauli_strings, qh.coefficients, strict=True):
        acc[label] = acc.get(label, 0.0 + 0.0j) + coeff
    return acc


def _assert_term_by_term_equivalent(qh_reference, qh_candidate, atol: float = 1e-12) -> None:
    """Assert two QubitHamiltonians match term-by-term after canonical sorting.

    Terms present in one operator but not the other are treated as having a
    zero coefficient in the missing operator; such a term passes only if its
    coefficient magnitude is below ``atol`` (i.e. it was legitimately pruned by
    the shared coefficient threshold). The ``1e-12`` absolute tolerance is the
    bar mandated by issue #474; it is not flaky here because every fast path
    drives the *same* C++ mapping arithmetic as the dense path.
    """
    ref = _sum_coefficients_by_pauli(qh_reference)
    cand = _sum_coefficients_by_pauli(qh_candidate)

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
    cholesky = Hamiltonian(CholeskyHamiltonianContainer(one_body, three_center, orbitals, core_energy, empty_fock))
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
    SYSTEMS: ClassVar = [(1, 3), (2, 5), (3, 8), (4, 6)]

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("n", "naux"), SYSTEMS)
    def test_cholesky_matches_dense(self, n: int, naux: int, encoding: str) -> None:
        """Cholesky and dense containers map to the same operator."""
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

    SYSTEMS: ClassVar = [(2, 5), (3, 7)]

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("n", "naux"), SYSTEMS)
    def test_cholesky_matches_dense(self, n: int, naux: int, encoding: str) -> None:
        """Unrestricted Cholesky and dense containers map identically."""
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
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, f"SCF did not converge for {atom} / {basis}"

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
    cholesky = Hamiltonian(CholeskyHamiltonianContainer(h1, np.ascontiguousarray(factors), orbitals, enuc, empty))
    return dense, cholesky, norb


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestCholeskyFastPathMolecular:
    """Cholesky path equals the dense path on real molecular Hamiltonians."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize(("atom", "basis"), [(m[1], m[2]) for m in MOLECULES], ids=[m[0] for m in MOLECULES])
    def test_molecular_cholesky_matches_dense(self, atom: str, basis: str, encoding: str) -> None:
        """Molecular Cholesky and dense containers map identically."""
        dense, cholesky, norb = _molecular_cholesky_pair(atom, basis)
        assert cholesky.get_container_type() == "cholesky"

        mapping = getattr(MajoranaMapping, encoding)(num_modes=2 * norb)
        mapper = create("qubit_mapper", "qdk")
        qh_dense = mapper.run(dense, mapping)
        qh_cholesky = mapper.run(cholesky, mapping)

        _assert_term_by_term_equivalent(qh_dense, qh_cholesky)


# ─── Sparse fast path: lattice / model Hamiltonians ────────────────────────


def _hubbard_2x2() -> Hamiltonian:
    """Hubbard model on a 2x2 square lattice (sparse container)."""
    return create_hubbard_hamiltonian(LatticeGraph.square(2, 2), epsilon=0.0, t=1.0, U=4.0)


def _hubbard_1x4() -> Hamiltonian:
    """Hubbard model on a 4-site chain (sparse container)."""
    return create_hubbard_hamiltonian(LatticeGraph.chain(4), epsilon=0.0, t=1.0, U=4.0)


def _ppp_1x4() -> Hamiltonian:
    """Pariser-Parr-Pople model on a 4-site chain (sparse container)."""
    return create_ppp_hamiltonian(LatticeGraph.chain(4), epsilon=0.0, t=1.0, U=4.0, V=1.5, z=1.0)


SPARSE_SYSTEMS = [
    ("hubbard_2x2", _hubbard_2x2),
    ("hubbard_1x4", _hubbard_1x4),
    ("ppp_1x4", _ppp_1x4),
]


class TestSparseFastPath:
    """Sparse path equals the dense path for lattice/model Hamiltonians."""

    @pytest.mark.parametrize("encoding", ENCODINGS)
    @pytest.mark.parametrize("factory", [s[1] for s in SPARSE_SYSTEMS], ids=[s[0] for s in SPARSE_SYSTEMS])
    def test_sparse_matches_dense(self, factory, encoding: str) -> None:
        """Sparse and densified containers map to the same operator."""
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


# ─── Fourfold storage against an independent reference ─────────────────────


def _native(hamiltonian: Hamiltonian, mapping: MajoranaMapping | None = None):
    """Map with the native mapper, using Jordan-Wigner by default."""
    if mapping is None:
        mapping = MajoranaMapping.jordan_wigner(2 * len(hamiltonian.get_one_body_integrals()[0]))
    return create("qubit_mapper", "qdk").run(hamiltonian, mapping)


def _fourfold_cholesky_pair(restricted: bool, layout: str = "rotated"):
    """Cholesky Hamiltonian with pair-antisymmetric factors, its dense twin, and the reference matrix."""
    symmetric_alpha = np.array([[0.8, 0.2], [0.2, 0.3]])
    symmetric_beta = np.array([[0.1, -0.4], [-0.4, 0.7]])
    antisymmetric_alpha = np.array([[0.0, 0.5], [-0.5, 0.0]])
    antisymmetric_beta = np.array([[0.0, -0.3], [0.3, 0.0]])
    if layout == "pure-antisymmetric":
        symmetric_alpha = np.zeros((2, 2))
        symmetric_beta = np.zeros((2, 2))
    rotation = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
    if layout in ("rank-deficient", "wide"):
        num_auxiliary = 3 if layout == "rank-deficient" else 7
        rotation = np.linalg.qr(np.random.default_rng(706).normal(size=(num_auxiliary, 2)))[0].T
    factors_alpha = np.column_stack((symmetric_alpha.ravel(), antisymmetric_alpha.ravel())) @ rotation
    factors_beta = np.column_stack((symmetric_beta.ravel(), antisymmetric_beta.ravel())) @ rotation
    one_alpha = np.array([[1.0, 0.3], [0.3, 0.5]])
    one_beta = np.array([[0.4, -0.2], [-0.2, -0.1]])
    if restricted:
        factors_beta, one_beta = factors_alpha, one_alpha
    two_aa = (factors_alpha @ factors_alpha.T).reshape((2,) * 4)
    two_ab = (factors_alpha @ factors_beta.T).reshape((2,) * 4)
    two_bb = (factors_beta @ factors_beta.T).reshape((2,) * 4)
    for tensor in (two_aa, two_ab, two_bb):
        np.testing.assert_allclose(tensor, tensor.transpose(1, 0, 3, 2), atol=1e-12, rtol=0)
    assert not np.allclose(two_aa, two_aa.swapaxes(0, 1))
    if not restricted:
        assert not np.allclose(two_aa, two_bb)
    if not restricted and layout != "pure-antisymmetric":
        assert not np.allclose(two_ab, two_ab.transpose(2, 3, 0, 1))
    empty = np.empty((0, 0))
    if restricted:
        dense = create_dense_test_hamiltonian(one_alpha, two_aa)
        container = CholeskyHamiltonianContainer(one_alpha, factors_alpha, dense.get_orbitals(), 0.0, empty)
    else:
        dense = create_dense_test_hamiltonian(one_alpha, two_aa, one_beta, two_ab, two_bb)
        container = CholeskyHamiltonianContainer(
            one_alpha, one_beta, factors_alpha, factors_beta, dense.get_orbitals(), 0.0, empty, empty
        )
    stored = Hamiltonian(container)
    for actual, expected in zip(stored.get_two_body_integrals(), (two_aa, two_ab, two_bb), strict=True):
        np.testing.assert_allclose(actual, expected.ravel(), atol=1e-12, rtol=0)
    reference = fermionic_reference_matrix(one_alpha, one_beta, two_aa, two_ab, two_bb)
    return stored, dense, reference


def _sparse_pair(entries: dict, tensor: np.ndarray):
    """Sparse Hamiltonian from ``entries``, its dense twin from ``tensor``, and the reference matrix."""
    num_orbitals = tensor.shape[0]
    one_body = np.diag(np.linspace(1.0, -0.5, num_orbitals)) + 0.1 * np.ones((num_orbitals, num_orbitals))
    stored = Hamiltonian(SparseHamiltonianContainer(csc_matrix(one_body), entries, 0.0))
    dense = create_dense_test_hamiltonian(one_body, tensor)
    reference = fermionic_reference_matrix(one_body, one_body, tensor, tensor, tensor)
    return stored, dense, reference


def _full_sparse_pair(seed: tuple, zero_suborbit: bool = False, mixed: bool = False):
    """Sparse pair storing the whole eightfold class of ``seed`` with unequal fourfold suborbits."""
    num_orbitals = 3 if mixed else max(seed) + 1
    primary = fourfold_orbit(seed)
    entries = {
        indices: 0.7 if indices in primary else (0.0 if zero_suborbit else -0.2) for indices in eightfold_orbit(seed)
    }
    tensor = np.zeros((num_orbitals,) * 4)
    for indices, value in entries.items():
        tensor[indices] = value
    if mixed:
        representative = (0, 1, 1, 2)
        assert not (eightfold_orbit(seed) & eightfold_orbit(representative))
        entries[representative] = 0.125
        for indices in eightfold_orbit(representative):
            tensor[indices] = 0.125
    return _sparse_pair(entries, tensor)


class TestCholeskyFourFold:
    """Cholesky factors with pair-antisymmetric parts map to the literal fourfold operator."""

    @pytest.mark.parametrize("restricted", [True, False], ids=["restricted", "unrestricted"])
    @pytest.mark.parametrize("layout", ["rotated", "pure-antisymmetric", "rank-deficient", "wide"])
    def test_full_matrix(self, restricted: bool, layout: str) -> None:
        """Cholesky and dense containers both reproduce the reference matrix."""
        stored, dense, reference = _fourfold_cholesky_pair(restricted, layout)
        np.testing.assert_allclose(reference, reference.conj().T, atol=1e-12, rtol=0)
        np.testing.assert_allclose(_native(dense).to_matrix(), reference, atol=1e-11, rtol=0)
        np.testing.assert_allclose(_native(stored).to_matrix(), reference, atol=1e-11, rtol=0)

    def test_restricted_cross_components_rejected(self) -> None:
        """Factors whose symmetric and antisymmetric parts mix give a non-Hermitian tensor."""
        symmetric = np.array([[0.8, 0.2], [0.2, 0.3]])
        antisymmetric = np.array([[0.0, 0.5], [-0.5, 0.0]])
        factors = (symmetric + antisymmetric).reshape(4, 1)
        tensor = (factors @ factors.T).reshape((2,) * 4)
        assert not np.allclose(tensor, tensor.transpose(1, 0, 3, 2))
        container = CholeskyHamiltonianContainer(np.eye(2), factors, create_test_orbitals(2), 0.0, np.empty((0, 0)))
        with pytest.raises(ValueError, match="non-Hermitian"):
            _native(Hamiltonian(container))


class TestSparseLiteralEntries:
    """Sparse two-body entries are literal: absent positions are zero."""

    @pytest.mark.parametrize(
        "seed", [(0, 1, 0, 1), (0, 1, 0, 2), (0, 0, 0, 0), (0, 0, 1, 1)], ids=["repeated-four", "eight", "one", "two"]
    )
    def test_complete_class_full_matrix(self, seed: tuple) -> None:
        """A fully stored class maps literally and matches its dense view."""
        stored, dense, reference = _full_sparse_pair(seed)
        np.testing.assert_allclose(
            stored.get_two_body_integrals()[0], dense.get_two_body_integrals()[0], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(_native(dense).to_matrix(), reference, atol=1e-11, rtol=0)
        np.testing.assert_allclose(_native(stored).to_matrix(), reference, atol=1e-11, rtol=0)

    @pytest.mark.parametrize("seed", [(0, 1, 0, 1), (0, 1, 0, 2)], ids=["repeated-four", "eight"])
    def test_full_class_mixed_with_eightfold_representative(self, seed: tuple) -> None:
        """A literal fourfold class and a deprecated single representative can coexist."""
        with pytest.warns(DeprecationWarning, match="deprecated"):
            stored, dense, reference = _full_sparse_pair(seed, zero_suborbit=True, mixed=True)
        np.testing.assert_allclose(_native(dense).to_matrix(), reference, atol=1e-11, rtol=0)
        np.testing.assert_allclose(_native(stored).to_matrix(), reference, atol=1e-11, rtol=0)

    @pytest.mark.parametrize("seed", [(0, 1, 0, 1), (0, 1, 0, 2)], ids=["repeated-four", "eight"])
    def test_legacy_representative_expands_with_warning(self, seed: tuple) -> None:
        """A deprecated single representative stands for its whole eightfold class."""
        tensor = np.zeros((max(seed) + 1,) * 4)
        for indices in eightfold_orbit(seed):
            tensor[indices] = 0.7
        with pytest.warns(DeprecationWarning, match="deprecated"):
            stored, dense, reference = _sparse_pair({seed: 0.7}, tensor)
        np.testing.assert_array_equal(stored.get_two_body_integrals()[0], dense.get_two_body_integrals()[0])
        np.testing.assert_allclose(_native(stored).to_matrix(), reference, atol=1e-11, rtol=0)

    def test_non_hermitian_map_is_not_expanded(self) -> None:
        """The shorthand assumes 8-fold symmetry, so a non-Hermitian map stays literal, with no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            container = SparseHamiltonianContainer(
                csc_matrix(np.eye(2)), {(0, 1, 0, 1): 0.7}, 0.0, HamiltonianType.NonHermitian
            )
        expected = np.zeros((2,) * 4)
        expected[0, 1, 0, 1] = 0.7
        np.testing.assert_array_equal(
            np.reshape(Hamiltonian(container).get_two_body_integrals()[0], (2,) * 4), expected
        )

    @pytest.mark.parametrize("seed", [(0, 1, 0, 1), (0, 1, 0, 2)], ids=["repeated-four", "eight"])
    @pytest.mark.parametrize("invalid", ["nonhermitian", "partial"])
    def test_invalid_classes_rejected(self, seed: tuple, invalid: str) -> None:
        """Classes that are not fourfold when read literally are rejected."""
        primary = fourfold_orbit(seed)
        entries = {indices: 0.7 if indices in primary else 0.0 for indices in eightfold_orbit(seed)}
        if invalid == "partial":
            del entries[sorted(entries)[-1]]
        else:
            entries[seed] += 0.25
        stored = Hamiltonian(SparseHamiltonianContainer(csc_matrix(np.eye(max(seed) + 1)), entries))
        with pytest.raises(ValueError, match="non-Hermitian"):
            _native(stored)

    @pytest.mark.parametrize("seed", [(0, 1, 0, 1), (0, 1, 0, 2)], ids=["repeated-four", "eight"])
    def test_fourfold_suborbit_without_zeros_full_matrix(self, seed: tuple) -> None:
        """A complete fourfold suborbit needs no explicit zeros for the rest of its class."""
        entries = dict.fromkeys(fourfold_orbit(seed), 0.7)
        tensor = np.zeros((max(seed) + 1,) * 4)
        for indices, value in entries.items():
            tensor[indices] = value
        stored, dense, reference = _sparse_pair(entries, tensor)
        np.testing.assert_allclose(_native(dense).to_matrix(), reference, atol=1e-11, rtol=0)
        np.testing.assert_allclose(_native(stored).to_matrix(), reference, atol=1e-11, rtol=0)


class TestFourFoldStorageEncodings:
    """Fourfold Cholesky and sparse storage match the dense path beyond Jordan-Wigner."""

    @pytest.mark.parametrize("storage", ["restricted-cholesky", "unrestricted-cholesky", "sparse"])
    @pytest.mark.parametrize("tapered", [False, True], ids=["bravyi-kitaev", "parity-tapered"])
    def test_storage_matches_dense(self, storage: str, tapered: bool) -> None:
        """Stored and dense containers map to the same coefficients."""
        if storage == "sparse":
            stored, dense, _reference = _full_sparse_pair((0, 1, 0, 1), zero_suborbit=True)
        else:
            stored, dense, _reference = _fourfold_cholesky_pair(storage == "restricted-cholesky")
        num_modes = 2 * len(dense.get_one_body_integrals()[0])
        if tapered:
            mapping = MajoranaMapping.parity(num_modes, Symmetries(n_alpha=1, n_beta=1))
        else:
            mapping = MajoranaMapping.bravyi_kitaev(num_modes)
        expected = _native(dense, mapping)
        actual = _native(stored, mapping)
        assert actual.encoding == expected.encoding
        _assert_term_by_term_equivalent(expected, actual, atol=1e-11)
