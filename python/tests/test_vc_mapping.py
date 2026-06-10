"""Tests for the Verstraete-Cirac fermion-to-qubit encoding.

Acceptance criteria from the feature request:
- Factory accepts 2D lattices of sizes 2x2, 2x3, 3x3, 4x4 and the result is
  consumable by QubitMapper.
- Fermi-Hubbard 2x2 (t=1, U=4, half filling): four lowest codespace
  eigenvalues match Jordan-Wigner to 1e-10.
- Max Pauli weight over all NN hopping terms is the same finite integer for
  open LxL lattices with L in {2, 3, 4}.
- JSON and HDF5 round-trips reproduce the mapping, and the round-tripped
  mapping produces an identical QubitHamiltonian term-by-term.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile

import h5py
import numpy as np
import pytest

from qdk_chemistry._core.data import PauliTermAccumulator, sparse_pauli_word_to_label
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
)
from qdk_chemistry.utils.pauli_matrix import pauli_to_sparse_matrix

from .test_helpers import create_test_orbitals

# ─── helpers ─────────────────────────────────────────────────────────────


def _pauli_weight(pauli_str: str) -> int:
    """Return the number of non-identity Pauli operators in the string."""
    return sum(c != "I" for c in pauli_str)


def _hubbard_lattice(rows: int, cols: int, t: float, u: float) -> Hamiltonian:
    """Open-boundary Fermi-Hubbard on a rows x cols lattice, row-major sites.

    QDK doubles the n spatial orbitals to 2n spin-orbitals internally
    (blocked order), which matches the VC mapping's two spin blocks.  The
    on-site U enters as the (s,s|s,s) two-body integral, giving the
    alpha-beta repulsion U * n_{s,up} n_{s,down} after spin doubling.
    """
    n = rows * cols
    h1 = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            if c + 1 < cols:
                h1[s, s + 1] = h1[s + 1, s] = -t
            if r + 1 < rows:
                h1[s, s + cols] = h1[s + cols, s] = -t
    h2 = np.zeros(n**4)
    if u != 0.0:
        for s in range(n):
            h2[s * n**3 + s * n**2 + s * n + s] = u
    orbitals = create_test_orbitals(n)
    return Hamiltonian(CanonicalFourCenterHamiltonianContainer(h1, h2, orbitals, 0.0, np.eye(0)))


def _mu(k: int) -> list[tuple[int, int]]:
    """Pauli word of the combined-JW Majorana k in the VC interleaved layout."""
    m, b = divmod(k, 2)
    word = [(q, 3) for q in range(m)]
    word.append((m, 1 if b == 0 else 2))
    return word


def _stabilizer(sigma: int, s: int, t: int, n_sites: int) -> tuple[complex, list[tuple[int, int]]]:
    """S = i * c_s * d_t: the VC auxiliary stabilizer for sites s, t in spin block sigma."""
    c = _mu(2 * (sigma * 2 * n_sites + 2 * s + 1))
    d = _mu(2 * (sigma * 2 * n_sites + 2 * t + 1) + 1)
    phase, word = PauliTermAccumulator.multiply_uncached(c, d)
    return 1j * phase, word


def _all_stabilizers(rows: int, cols: int) -> list[tuple[complex, list[tuple[int, int]]]]:
    """Edge stabilizers plus column closers; their joint +1 space has dim 2^(2*rows*cols)."""
    n_sites = rows * cols
    stabs = []
    for sigma in (0, 1):
        for s in range(n_sites - cols):
            stabs.append(_stabilizer(sigma, s, s + cols, n_sites))
        for col in range(cols):
            stabs.append(_stabilizer(sigma, (rows - 1) * cols + col, col, n_sites))
    return stabs


def _sparse_pauli(label: str, coeff: complex):
    """Sparse matrix of coeff * (Pauli given by label)."""
    return coeff * pauli_to_sparse_matrix([label], np.array([1.0 + 0j]))


def _qh_sparse(qh):
    """Sparse matrix of a QubitHamiltonian."""
    return pauli_to_sparse_matrix(list(qh.pauli_strings), np.asarray(qh.coefficients))


# ─── factory tests ───────────────────────────────────────────────────────


class TestVerstraeteCiracFactory:
    """Tests for the MajoranaMapping.verstraete_cirac factory."""

    @pytest.mark.parametrize(("rows", "cols"), [(2, 2), (2, 3), (3, 3), (4, 4)])
    def test_factory_lattice_sizes(self, rows: int, cols: int) -> None:
        """Factory constructs mappings for the lattice sizes in the acceptance criteria."""
        vc = MajoranaMapping.verstraete_cirac(rows, cols)
        n_sites = rows * cols
        assert vc.num_modes == 2 * n_sites
        assert vc.num_qubits == 4 * n_sites
        assert vc.name == "verstraete-cirac"
        assert vc.is_majorana_atomic is False

    @pytest.mark.parametrize(("rows", "cols"), [(1, 2), (2, 1), (0, 0), (1, 1)])
    def test_too_small_lattice_raises(self, rows: int, cols: int) -> None:
        """Lattices smaller than 2x2 are rejected."""
        with pytest.raises(ValueError, match="rows >= 2 and cols >= 2"):
            MajoranaMapping.verstraete_cirac(rows, cols)

    def test_majorana_raises(self) -> None:
        """Individual Majorana images are unavailable for the bilinear-only VC mapping."""
        vc = MajoranaMapping.verstraete_cirac(2, 2)
        with pytest.raises(ValueError, match="bilinear-only"):
            vc.majorana(0)

    @pytest.mark.parametrize(("rows", "cols"), [(2, 2), (2, 3)])
    def test_bilinear_antisymmetry(self, rows: int, cols: int) -> None:
        """bilinear(k, j) = -bilinear(j, k)."""
        vc = MajoranaMapping.verstraete_cirac(rows, cols)
        m = 2 * vc.num_modes
        for j in range(0, m, 3):
            for k in range(j + 1, m, 3):
                c_fwd, w_fwd = vc.bilinear(j, k)
                c_rev, w_rev = vc.bilinear(k, j)
                assert w_fwd == w_rev
                assert abs(c_fwd + c_rev) < 1e-12

    def test_bilinears_square_to_identity(self) -> None:
        """(i gamma_j gamma_k)^2 = I for the VC bilinears."""
        vc = MajoranaMapping.verstraete_cirac(2, 2)
        m = 2 * vc.num_modes
        for j in range(0, m, 2):
            for k in range(j + 1, m, 2):
                coeff, word = vc.bilinear(j, k)
                phase, prod = PauliTermAccumulator.multiply_uncached(word, word)
                assert prod == [], f"({j},{k}): word^2 not identity"
                assert abs(coeff * coeff * phase - 1.0) < 1e-12


# ─── stabilizer structure ────────────────────────────────────────────────


class TestVerstraeteCiracStabilizers:
    """The mapped Hamiltonian commutes with every VC stabilizer."""

    def test_hamiltonian_commutes_with_stabilizers(self) -> None:
        """[H_vc, S_e] = 0 exactly for all edge and column-closing stabilizers."""
        rows, cols = 2, 2
        ham = _hubbard_lattice(rows, cols, t=1.0, u=4.0)
        vc = MajoranaMapping.verstraete_cirac(rows, cols)
        qh = create("qubit_mapper", "qdk").run(ham, vc)
        nq = qh.num_qubits
        H = _qh_sparse(qh)  # noqa: N806

        for s_coeff, s_word in _all_stabilizers(rows, cols):
            label = sparse_pauli_word_to_label(s_word, nq)
            S = _sparse_pauli(label, s_coeff)  # noqa: N806
            comm = H @ S - S @ H
            assert abs(comm).max() < 1e-12, f"[H, S] != 0 for stabilizer {label}"

    def test_stabilizers_square_to_identity(self) -> None:
        """S_e^2 = +1 for every stabilizer."""
        rows, cols = 2, 3
        for s_coeff, s_word in _all_stabilizers(rows, cols):
            phase, prod = PauliTermAccumulator.multiply_uncached(s_word, s_word)
            assert prod == []
            assert abs(s_coeff * s_coeff * phase - 1.0) < 1e-12


# ─── Pauli weight locality ────────────────────────────────────────────────


class TestVerstraeteCiracLocality:
    """Constant Pauli weight for nearest-neighbour hopping."""

    def test_weight_independent_of_lattice_size(self) -> None:
        """Max NN-hopping Pauli weight is the same finite integer for L in {2, 3, 4}."""
        mapper = create("qubit_mapper", "qdk")
        weights = set()
        for size in (2, 3, 4):
            ham = _hubbard_lattice(size, size, t=1.0, u=0.0)
            vc = MajoranaMapping.verstraete_cirac(size, size)
            qh = mapper.run(ham, vc)
            weights.add(max(_pauli_weight(ps) for ps in qh.pauli_strings))
        assert weights == {4}, f"Expected constant weight 4, got: {weights}"

    def test_jw_weight_grows_for_comparison(self) -> None:
        """Under JW the same vertical hops produce strings that grow with L."""
        mapper = create("qubit_mapper", "qdk")
        jw_weights = []
        for size in (2, 3, 4):
            ham = _hubbard_lattice(size, size, t=1.0, u=0.0)
            qh = mapper.run(ham, MajoranaMapping.jordan_wigner(num_modes=2 * size * size))
            jw_weights.append(max(_pauli_weight(ps) for ps in qh.pauli_strings))
        assert jw_weights == sorted(jw_weights)
        assert jw_weights[0] < jw_weights[-1]


# ─── eigenvalue correctness ───────────────────────────────────────────────


class TestVerstraeteCiracEigenvalues:
    """Codespace spectrum equals the Jordan-Wigner spectrum."""

    def test_fermi_hubbard_2x2_codespace_matches_jw(self) -> None:
        """2x2 Fermi-Hubbard (t=1, U=4): codespace eigenvalues match JW to 1e-10.

        The codespace is the joint +1 eigenspace of all stabilizers (vertical
        edges plus column closers); with all 2N auxiliary Majoranas fixed it
        has dimension 2^(2N), exactly the JW Hilbert space.
        """
        rows, cols = 2, 2
        n_sites = rows * cols
        ham = _hubbard_lattice(rows, cols, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        qh_jw = mapper.run(ham, MajoranaMapping.jordan_wigner(num_modes=2 * n_sites))
        qh_vc = mapper.run(ham, MajoranaMapping.verstraete_cirac(rows, cols))

        eigs_jw = np.sort(np.linalg.eigvalsh(_qh_sparse(qh_jw).toarray()))

        nq = qh_vc.num_qubits
        dim = 2**nq
        H_vc = _qh_sparse(qh_vc)  # noqa: N806

        # Project a random block onto the joint +1 stabilizer eigenspace,
        # orthonormalize, and diagonalize the restricted Hamiltonian.
        code_dim = 2 ** (2 * n_sites)
        rng = np.random.default_rng(7)
        V = rng.standard_normal((dim, code_dim + 40)) + 1j * rng.standard_normal((dim, code_dim + 40))  # noqa: N806
        for s_coeff, s_word in _all_stabilizers(rows, cols):
            label = sparse_pauli_word_to_label(s_word, nq)
            S = _sparse_pauli(label, s_coeff)  # noqa: N806
            V = 0.5 * (V + S @ V)  # noqa: N806
        Q, R = np.linalg.qr(V)  # noqa: N806
        basis = Q[:, np.abs(np.diag(R)) > 1e-8]
        assert basis.shape[1] == code_dim

        H_res = basis.conj().T @ (H_vc @ basis)  # noqa: N806
        eigs_vc = np.sort(np.linalg.eigvalsh(H_res))

        np.testing.assert_allclose(eigs_vc[:4], eigs_jw[:4], atol=1e-10)
        np.testing.assert_allclose(eigs_vc, eigs_jw, atol=1e-10)


# ─── serialization ───────────────────────────────────────────────────────


class TestVerstraeteCiracSerialization:
    """JSON and HDF5 round-trip tests."""

    @staticmethod
    def _bilinears_equal(a: MajoranaMapping, b: MajoranaMapping) -> bool:
        m = 2 * a.num_modes
        for j in range(m):
            for k in range(j + 1, m):
                ca, wa = a.bilinear(j, k)
                cb, wb = b.bilinear(j, k)
                if list(wa) != list(wb) or abs(ca - cb) > 1e-14:
                    return False
        return True

    def test_json_round_trip(self) -> None:
        """JSON round-trip preserves every bilinear."""
        vc = MajoranaMapping.verstraete_cirac(2, 2)
        loaded = MajoranaMapping.from_json(vc.to_json())
        assert loaded.name == vc.name
        assert loaded.num_modes == vc.num_modes
        assert loaded.num_qubits == vc.num_qubits
        assert not loaded.is_majorana_atomic
        assert self._bilinears_equal(vc, loaded)

    def test_hdf5_round_trip(self) -> None:
        """HDF5 round-trip preserves every bilinear."""
        vc = MajoranaMapping.verstraete_cirac(2, 2)
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                vc.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded = MajoranaMapping.from_hdf5(hf)
        assert loaded.name == vc.name
        assert loaded.num_modes == vc.num_modes
        assert loaded.num_qubits == vc.num_qubits
        assert self._bilinears_equal(vc, loaded)

    def test_round_tripped_hamiltonian_terms_match(self) -> None:
        """Round-tripped mapping produces a term-by-term identical QubitHamiltonian."""
        ham = _hubbard_lattice(2, 2, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        vc = MajoranaMapping.verstraete_cirac(2, 2)
        qh_orig = mapper.run(ham, vc)

        loaded = MajoranaMapping.from_json(vc.to_json())
        qh_loaded = mapper.run(ham, loaded)

        orig = dict(zip(qh_orig.pauli_strings, qh_orig.coefficients, strict=True))
        rt = dict(zip(qh_loaded.pauli_strings, qh_loaded.coefficients, strict=True))
        assert set(orig) == set(rt)
        for ps, coeff in orig.items():
            assert np.isclose(coeff, rt[ps], atol=1e-14)
