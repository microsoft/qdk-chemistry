"""Tests for the Verstraete-Cirac fermion-to-qubit encoding.

Covers issue #482 acceptance criteria: factory on connected lattice graphs,
codespace spectrum vs Jordan-Wigner, constant hopping Pauli weight,
serialization round-trips, and stabilizer commutation on diverse topologies.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_huckel_hamiltonian

LATTICE_SIZES = [(2, 2), (2, 3), (3, 3), (4, 4)]


def _hubbard(nx: int, ny: int, t: float = 1.0, u: float = 4.0, **kwargs):
    """Return a square-lattice Fermi-Hubbard model and its ``LatticeGraph``."""
    lattice = LatticeGraph.square(nx, ny, **kwargs)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=t, U=u)
    return lattice, hamiltonian


def _lowest_eigenvalues_dense(qh: QubitHamiltonian, k: int) -> np.ndarray:
    """Return the ``k`` smallest eigenvalues via dense diagonalization."""
    matrix = qh.to_matrix(sparse=False)
    return np.sort(np.linalg.eigvalsh(matrix).real)[:k]


def _lowest_eigenvalues_sparse(qh: QubitHamiltonian, k: int) -> np.ndarray:
    """Return the ``k`` smallest eigenvalues via sparse Lanczos."""
    matrix = qh.to_matrix(sparse=True).tocsr()
    matrix = 0.5 * (matrix + matrix.getH())
    vals = spla.eigsh(
        matrix,
        k=k,
        which="SA",
        return_eigenvectors=False,
        tol=1e-12,
        maxiter=500_000,
    )
    return np.sort(vals.real)[:k]


def _pauli_strings_commute(p1: str, p2: str) -> bool:
    """Return True when two Pauli strings commute."""
    anti = sum(1 for c1, c2 in zip(p1, p2, strict=False) if c1 != "I" and c2 not in {"I", c1})
    return (anti % 2) == 0


class TestVerstraeteCiracFactory:
    """Structural checks on the factory output."""

    @pytest.mark.parametrize(("nx", "ny"), LATTICE_SIZES)
    def test_factory_shapes(self, nx: int, ny: int) -> None:
        """VC mapping has expected modes, qubits, and stabilizers."""
        n_sites = nx * ny
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny))

        assert mapping.name == "verstraete-cirac"
        assert mapping.base_encoding == "verstraete-cirac"
        assert mapping.num_modes == 2 * n_sites
        assert mapping.num_qubits >= 2 * mapping.num_modes
        assert len(mapping.stabilizers) > 0
        assert not mapping.is_majorana_atomic

    @pytest.mark.parametrize(("nx", "ny"), LATTICE_SIZES)
    def test_qubit_mapper_consumes_without_error(self, nx: int, ny: int) -> None:
        """Qubit mapper accepts VC mapping on square lattices."""
        _, hamiltonian = _hubbard(nx, ny)
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny))
        qh = create("qubit_mapper", "qdk").run(hamiltonian, mapping)

        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == mapping.num_qubits
        assert len(qh.pauli_strings) == len(qh.coefficients)
        assert qh.is_hermitian()

    def test_too_few_sites_rejected(self) -> None:
        """Fewer than three sites raises ``ValueError``."""
        with pytest.raises(ValueError, match="at least 3 sites"):
            MajoranaMapping.verstraete_cirac(LatticeGraph.square(1, 2))

    def test_majorana_accessor_raises(self) -> None:
        """``majorana()`` is unavailable for bilinear-only encodings."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 2))
        with pytest.raises(ValueError, match="bilinear-only"):
            mapping.majorana(0)

    @pytest.mark.parametrize(
        ("factory", "args", "kwargs"),
        [
            ("square", (2, 2), {}),
            ("triangular", (2, 2), {}),
            ("triangular", (3, 3), {}),
            ("honeycomb", (2, 2), {"periodic_x": True, "periodic_y": True}),
            ("kagome", (2, 2), {"periodic_x": True, "periodic_y": True}),
            ("kagome", (3, 3), {}),
        ],
        ids=[
            "square-2x2",
            "triangular-2x2",
            "triangular-3x3",
            "honeycomb-2x2-periodic",
            "kagome-2x2-periodic",
            "kagome-3x3",
        ],
    )
    def test_general_lattice_accepted(self, factory: str, args: tuple, kwargs: dict) -> None:
        """Factory accepts square, triangular, honeycomb, and kagome graphs."""
        lattice = getattr(LatticeGraph, factory)(*args, **kwargs)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        assert mapping.num_modes == 2 * lattice.num_sites
        assert len(mapping.stabilizers) > 0

    def test_chain_graph_is_pure_backbone(self) -> None:
        """Open chains need no auxiliary modes or stabilizers."""
        lattice = LatticeGraph.chain(5)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        assert mapping.num_modes == 2 * lattice.num_sites
        assert mapping.num_qubits == mapping.num_modes
        assert mapping.stabilizers == []

    def test_custom_graph_from_adjacency_matrix(self) -> None:
        """Factory accepts arbitrary connected graphs, not only lattice presets."""
        # 5-cycle (same topology as C++ custom-edge test).
        adj = np.zeros((5, 5))
        for u, v in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)):
            adj[u, v] = adj[v, u] = 1.0
        lattice = LatticeGraph.from_dense_matrix(adj)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        assert mapping.num_modes == 10
        assert len(mapping.stabilizers) > 0

    def test_stabilizers_are_hermitian_pauli(self) -> None:
        """Each stabilizer is a real +/-1 Pauli operator."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(3, 3))
        for coeff, _word in mapping.stabilizers:
            assert np.isclose(coeff.imag, 0.0)
            assert np.isclose(abs(coeff), 1.0)

    def test_content_hash_depends_on_lattice(self) -> None:
        """Distinct lattices produce distinct mapping hashes."""
        a = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 2))
        b = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        assert a.content_hash() != b.content_hash()

    @pytest.mark.parametrize(
        ("factory", "args", "kwargs"),
        [
            ("square", (2, 2), {}),
            ("square", (3, 3), {}),
            ("square", (3, 4), {}),
            ("honeycomb", (2, 2), {"periodic_x": True, "periodic_y": True}),
            ("triangular", (2, 2), {}),
            ("kagome", (2, 2), {"periodic_x": True, "periodic_y": True}),
            ("kagome", (3, 3), {}),
        ],
        ids=[
            "square-2x2",
            "square-3x3",
            "square-3x4",
            "honeycomb-2x2-periodic",
            "triangular-2x2",
            "kagome-2x2-periodic",
            "kagome-3x3",
        ],
    )
    def test_stabilizers_commute_with_hamiltonian(self, factory: str, args: tuple, kwargs: dict) -> None:
        """Stabilizers pairwise commute and commute with mapped Hückel terms."""
        lattice = getattr(LatticeGraph, factory)(*args, **kwargs)
        ham = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh = create("qubit_mapper", "qdk").run(ham, mapping)

        assert mapping.num_qubits - len(mapping.stabilizers) == 2 * lattice.num_sites

        stabs = [sparse_pauli_word_to_label(word, qh.num_qubits) for _, word in mapping.stabilizers]
        for i in range(len(stabs)):
            for j in range(i + 1, len(stabs)):
                assert _pauli_strings_commute(stabs[i], stabs[j])
        for stab in stabs:
            for term in qh.pauli_strings:
                assert _pauli_strings_commute(stab, term)


class TestVerstraeteCiracSpectrum:
    """Codespace spectrum must reproduce Jordan-Wigner."""

    def test_hubbard_2x2_matches_jordan_wigner(self) -> None:
        """2x2 Hubbard lowest eigenvalues match Jordan-Wigner."""
        nx, ny = 2, 2
        n_sites = nx * ny
        _, hamiltonian = _hubbard(nx, ny, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        jw_eigs = _lowest_eigenvalues_dense(
            mapper.run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes=2 * n_sites)),
            k=4,
        )
        vc_eigs = _lowest_eigenvalues_sparse(
            mapper.run(
                hamiltonian,
                MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny)),
            ),
            k=4,
        )
        assert np.max(np.abs(jw_eigs - vc_eigs)) < 1e-10

    def test_disconnected_graph_rejected(self) -> None:
        """Two-component graphs raise ``ValueError``."""
        adj = np.zeros((4, 4))
        adj[0, 1] = adj[1, 0] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        lattice = LatticeGraph.from_dense_matrix(adj)
        with pytest.raises(ValueError, match="connected"):
            MajoranaMapping.verstraete_cirac(lattice)

    def test_hubbard_2x2_spectrum_survives_threshold(self) -> None:
        """2x2 Hubbard spectrum unchanged under coefficient thresholding."""
        nx, ny = 2, 2
        n_sites = nx * ny
        _, hamiltonian = _hubbard(nx, ny, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        jw_eigs = _lowest_eigenvalues_dense(
            mapper.run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes=2 * n_sites)),
            k=4,
        )
        vc_eigs = _lowest_eigenvalues_sparse(
            mapper.run(
                hamiltonian,
                MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny)),
                threshold=1.0,
            ),
            k=4,
        )
        assert np.max(np.abs(jw_eigs - vc_eigs)) < 1e-10

    def test_hubbard_3x3_matches_jordan_wigner(self) -> None:
        """Spectral check on 3x3 — not just 'runs without error'."""
        nx, ny = 3, 3
        n_sites = nx * ny
        _, hamiltonian = _hubbard(nx, ny, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        jw_eigs = _lowest_eigenvalues_sparse(
            mapper.run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes=2 * n_sites)),
            k=6,
        )
        vc_eigs = _lowest_eigenvalues_sparse(
            mapper.run(
                hamiltonian,
                MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny)),
            ),
            k=6,
        )
        assert np.max(np.abs(jw_eigs[:4] - vc_eigs[:4])) < 1e-10

    def test_periodic_2x2_hubbard_matches_jordan_wigner(self) -> None:
        """Periodic 2x2 Hubbard lowest eigenvalues match Jordan-Wigner."""
        lattice = LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True)
        n_modes = 2 * lattice.num_sites
        hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-2.0, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")

        jw_eigs = _lowest_eigenvalues_sparse(
            mapper.run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes=n_modes)),
            k=4,
        )
        vc_eigs = _lowest_eigenvalues_sparse(
            mapper.run(hamiltonian, MajoranaMapping.verstraete_cirac(lattice)),
            k=4,
        )
        np.testing.assert_allclose(jw_eigs[:2], vc_eigs[:2], atol=1e-10)

    def test_engine_identity_matches_jordan_wigner(self) -> None:
        """Stabilizer penalty leaves the encoding-independent identity coefficient."""
        lattice = LatticeGraph.square(2, 2)
        vc = MajoranaMapping.verstraete_cirac(lattice)
        jw = MajoranaMapping.jordan_wigner(vc.num_modes)
        _, hamiltonian = _hubbard(2, 2, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        def identity_coeff(qh: QubitHamiltonian) -> float:
            terms = dict(qh.get_real_coefficients())
            return float(terms.get("I" * qh.num_qubits, 0.0))

        qh_jw = mapper.run(hamiltonian, jw)
        qh_vc = mapper.run(hamiltonian, vc)
        assert identity_coeff(qh_jw) == pytest.approx(identity_coeff(qh_vc), rel=0, abs=1e-10)

    @pytest.mark.slow
    def test_huckel_3x2_codespace_matches_jordan_wigner(self) -> None:
        """3x2 Hückel codespace eigenvalues match Jordan-Wigner."""
        lattice = LatticeGraph.square(3, 2)
        n_modes = 2 * lattice.num_sites
        hamiltonian = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)
        mapper = create("qubit_mapper", "qdk")

        jw_eigs = _lowest_eigenvalues_sparse(
            mapper.run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes=n_modes)),
            k=8,
        )
        vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh_vc = mapper.run(hamiltonian, vc_mapping)
        h_vc = qh_vc.to_matrix(sparse=True)
        eigs_vc, vecs_vc = spla.eigsh(h_vc, k=8, which="SA", tol=1e-12, maxiter=500_000)

        stabs = [
            QubitHamiltonian(
                [sparse_pauli_word_to_label(word, qh_vc.num_qubits)],
                np.array([coeff]),
            ).to_matrix(sparse=True)
            for coeff, word in vc_mapping.stabilizers
        ]
        code_space_eigs = []
        for idx in range(len(eigs_vc)):
            vec = vecs_vc[:, idx]
            if all(np.isclose(np.real(vec.conj().T @ (s @ vec)), 1.0, atol=1e-4) for s in stabs):
                code_space_eigs.append(eigs_vc[idx])

        assert len(code_space_eigs) >= 2
        unique_vc = np.unique(np.round(np.sort(code_space_eigs), 6))
        unique_jw = np.unique(np.round(np.sort(jw_eigs), 6))
        np.testing.assert_allclose(unique_vc[:2], unique_jw[:2], atol=1e-10)


class TestVerstraeteCiracLocality:
    """Hopping terms keep a finite, size-independent Pauli weight on square grids."""

    @staticmethod
    def _max_hopping_weight(nx: int, ny: int) -> int:
        """Return the maximum Pauli weight among nearest-neighbor bilinears."""
        lattice = LatticeGraph.square(nx, ny)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        adjacency = np.asarray(lattice.adjacency_matrix())
        n_sites = nx * ny
        max_weight = 0
        for p in range(n_sites):
            for q in range(p + 1, n_sites):
                if adjacency[p, q] == 0.0:
                    continue
                for a in range(2):
                    for b in range(2):
                        _, word = mapping.bilinear(2 * p + a, 2 * q + b)
                        max_weight = max(max_weight, len(word))
        return max_weight

    def test_hopping_weight_independent_of_size(self) -> None:
        """Nearest-neighbor bilinear Pauli weight is constant in L for LxL grids."""
        weights = {L: self._max_hopping_weight(L, L) for L in (2, 3, 4)}
        assert len(set(weights.values())) == 1, weights
        assert next(iter(weights.values())) > 0

    def test_hopping_weight_from_bilinear_table(self) -> None:
        """Pauli weight is read from bilinears, not inferred from mapped Hamiltonian."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(4, 4))
        lattice = LatticeGraph.square(4, 4)
        adjacency = np.asarray(lattice.adjacency_matrix())
        max_bilinear = 0
        for p in range(lattice.num_sites):
            for q in range(p + 1, lattice.num_sites):
                if adjacency[p, q] == 0.0:
                    continue
                for a in range(2):
                    for b in range(2):
                        _, word = mapping.bilinear(2 * p + a, 2 * q + b)
                        max_bilinear = max(max_bilinear, len(word))
        assert max_bilinear > 0
        assert max_bilinear == self._max_hopping_weight(4, 4)


class TestVerstraeteCiracSerialization:
    """The mapping survives JSON/HDF5 round-trips term-by-term."""

    @staticmethod
    def _assert_same_qubit_hamiltonian(reloaded: MajoranaMapping) -> None:
        """Reloaded mapping maps a reference Hamiltonian identically."""
        _, hamiltonian = _hubbard(2, 3)
        mapper = create("qubit_mapper", "qdk")
        original = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        assert mapper.run(hamiltonian, reloaded).equiv(mapper.run(hamiltonian, original), atol=1e-12)

    def test_json_roundtrip(self) -> None:
        """JSON serialization preserves mapping and mapped Hamiltonian."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        reloaded = MajoranaMapping.from_json(mapping.to_json())
        assert reloaded.num_modes == mapping.num_modes
        assert reloaded.num_qubits == mapping.num_qubits
        assert len(reloaded.stabilizers) == len(mapping.stabilizers)
        self._assert_same_qubit_hamiltonian(reloaded)

    def test_hdf5_roundtrip(self) -> None:
        """HDF5 serialization preserves mapping and mapped Hamiltonian."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mapping.h5")
            with h5py.File(path, "w") as hf:
                mapping.to_hdf5(hf)
            with h5py.File(path, "r") as hf:
                reloaded = MajoranaMapping.from_hdf5(hf)
        assert reloaded.num_qubits == mapping.num_qubits
        self._assert_same_qubit_hamiltonian(reloaded)

    def test_without_tapering_is_identity(self) -> None:
        """``without_tapering()`` leaves VC stabilizers unchanged."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 2))
        base = mapping.without_tapering()
        assert base.name == mapping.name
        assert len(base.stabilizers) == len(mapping.stabilizers)

    def test_json_rejects_too_small_num_qubits(self) -> None:
        """JSON load rejects ``num_qubits`` smaller than ``num_modes``."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 2))
        data = mapping.to_json()
        data["num_qubits"] = 1
        with pytest.raises(ValueError, match="num_qubits"):
            MajoranaMapping.from_json(data)
