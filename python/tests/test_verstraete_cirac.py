"""Tests for the Verstraete-Cirac fermion-to-qubit encoding.

Covers the acceptance criteria of issue #482:

* the factory accepts 2x2, 2x3, 3x3 and 4x4 lattices and ``QubitMapper``
  consumes the result without error,
* the codespace spectrum of a 2x2 Fermi-Hubbard model matches Jordan-Wigner,
* nearest-neighbour hopping terms keep a constant (size-independent) Pauli
  weight, and
* the mapping survives JSON and HDF5 round-trips and reproduces the same
  qubit Hamiltonian term-by-term.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import tempfile

import h5py
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian

LATTICE_SIZES = [(2, 2), (2, 3), (3, 3), (4, 4)]


def _hubbard(nx: int, ny: int, t: float = 1.0, u: float = 4.0):
    """Open-boundary Fermi-Hubbard model on an ``nx`` x ``ny`` square lattice."""
    lattice = LatticeGraph.square(nx, ny)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=t, U=u)
    return lattice, hamiltonian


def _lowest_eigenvalues_dense(qh: QubitHamiltonian, k: int) -> np.ndarray:
    """Return the ``k`` smallest eigenvalues via a dense diagonalization."""
    matrix = qh.to_matrix(sparse=False)
    return np.sort(np.linalg.eigvalsh(matrix).real)[:k]


def _lowest_eigenvalues_sparse(qh: QubitHamiltonian, k: int) -> np.ndarray:
    """Return the ``k`` smallest eigenvalues via a sparse Lanczos solve."""
    matrix = qh.to_matrix(sparse=True).tocsr()
    matrix = 0.5 * (matrix + matrix.getH())
    vals = spla.eigsh(matrix, k=k, which="SA", return_eigenvectors=False)
    return np.sort(vals.real)[:k]


class TestVerstraeteCiracFactory:
    """Structural checks on the factory output."""

    @pytest.mark.parametrize(("nx", "ny"), LATTICE_SIZES)
    def test_factory_shapes(self, nx: int, ny: int) -> None:
        """Mode, qubit, and stabilizer counts match the doubled-block layout."""
        n_sites = nx * ny
        lattice = LatticeGraph.square(nx, ny)
        mapping = MajoranaMapping.verstraete_cirac(lattice)

        assert mapping.name == "verstraete-cirac"
        assert mapping.base_encoding == "verstraete-cirac"
        # One Verstraete-Cirac block per spin sector.
        assert mapping.num_modes == 2 * n_sites
        # Each physical mode carries one auxiliary qubit.
        assert mapping.num_qubits == 4 * n_sites
        # Stabilizer count equals the number of physical modes.
        assert len(mapping.stabilizers) == 2 * n_sites
        assert not mapping.is_majorana_atomic

    @pytest.mark.parametrize(("nx", "ny"), LATTICE_SIZES)
    def test_qubit_mapper_consumes_without_error(self, nx: int, ny: int) -> None:
        """QubitMapper maps a Hubbard model on each lattice size without error."""
        _, hamiltonian = _hubbard(nx, ny)
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny))
        mapper = create("qubit_mapper", "qdk")

        qh = mapper.run(hamiltonian, mapping)

        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 4 * nx * ny
        assert len(qh.pauli_strings) == len(qh.coefficients)
        assert qh.is_hermitian()

    def test_non_grid_lattice_rejected(self) -> None:
        """A 1-D chain is not a 2-D rectangular grid and must be rejected."""
        chain = LatticeGraph.chain(3)
        with pytest.raises(ValueError, match="rectangular grid"):
            MajoranaMapping.verstraete_cirac(chain)

    def test_extra_long_range_edge_rejected(self) -> None:
        """A rectangular site set with a non-nearest-neighbour edge is rejected."""
        adj = LatticeGraph.square(2, 2).adjacency_matrix().copy()
        adj[0, 3] = 1.0
        adj[3, 0] = 1.0
        bad = LatticeGraph.from_dense_matrix(adj)
        with pytest.raises(ValueError, match="nearest neighbours"):
            MajoranaMapping.verstraete_cirac(bad)


class TestVerstraeteCiracSpectrum:
    """Codespace spectrum must reproduce Jordan-Wigner."""

    def test_hubbard_2x2_matches_jordan_wigner(self) -> None:
        """Four lowest codespace eigenvalues match Jordan-Wigner within 1e-10."""
        nx, ny = 2, 2
        n_sites = nx * ny
        _, hamiltonian = _hubbard(nx, ny, t=1.0, u=4.0)
        mapper = create("qubit_mapper", "qdk")

        jw = MajoranaMapping.jordan_wigner(num_modes=2 * n_sites)
        jw_qh = mapper.run(hamiltonian, jw)
        jw_eigs = _lowest_eigenvalues_dense(jw_qh, k=4)

        vc = MajoranaMapping.verstraete_cirac(LatticeGraph.square(nx, ny))
        vc_qh = mapper.run(hamiltonian, vc)
        assert vc_qh.is_hermitian()
        vc_eigs = _lowest_eigenvalues_sparse(vc_qh, k=4)

        assert np.max(np.abs(jw_eigs - vc_eigs)) < 1e-10


class TestVerstraeteCiracLocality:
    """Hopping terms keep a finite, size-independent Pauli weight."""

    @staticmethod
    def _max_hopping_weight(nx: int, ny: int) -> int:
        """Largest Pauli weight among the four bilinears of every lattice bond."""
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
        """Max hopping Pauli weight is the same finite integer for L in 2,3,4."""
        weights = {L: self._max_hopping_weight(L, L) for L in (2, 3, 4)}
        assert len(set(weights.values())) == 1, weights
        assert next(iter(weights.values())) > 0


class TestVerstraeteCiracSerialization:
    """The mapping survives JSON/HDF5 round-trips term-by-term."""

    @staticmethod
    def _assert_same_qubit_hamiltonian(reloaded: MajoranaMapping) -> None:
        """Assert a reloaded mapping reproduces the original qubit Hamiltonian."""
        _, hamiltonian = _hubbard(2, 3)
        mapper = create("qubit_mapper", "qdk")
        original = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        qh_original = mapper.run(hamiltonian, original)
        qh_reloaded = mapper.run(hamiltonian, reloaded)
        assert qh_reloaded.equiv(qh_original, atol=1e-12)

    def test_json_roundtrip(self) -> None:
        """The mapping round-trips through JSON term-by-term."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        reloaded = MajoranaMapping.from_json(mapping.to_json())

        assert reloaded.num_modes == mapping.num_modes
        assert reloaded.num_qubits == mapping.num_qubits
        assert len(reloaded.stabilizers) == len(mapping.stabilizers)
        self._assert_same_qubit_hamiltonian(reloaded)

    def test_hdf5_roundtrip(self) -> None:
        """The mapping round-trips through HDF5 term-by-term."""
        mapping = MajoranaMapping.verstraete_cirac(LatticeGraph.square(2, 3))
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                mapping.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                reloaded = MajoranaMapping.from_hdf5(hf)

        assert reloaded.num_modes == mapping.num_modes
        assert len(reloaded.stabilizers) == len(mapping.stabilizers)
        self._assert_same_qubit_hamiltonian(reloaded)
