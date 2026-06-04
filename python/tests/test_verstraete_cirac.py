"""Tests for the Verstraete-Cirac fermion-to-qubit encoding.

Tests cover:
- Correctness: VC-encoded Fermi-Hubbard eigenvalues match Jordan-Wigner (2x2, t=1, U=4, half-filling)
- Locality: max Pauli weight of nearest-neighbour hopping terms is constant across lattice sizes
- Serialization: JSON and HDF5 round-trips produce identical QubitHamiltonian terms
- Construction: factory works for 2x2, 2x3, 3x3, 4x4 and QubitMapper consumes them
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile

import h5py
import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.qubit_mapper.verstraete_cirac import (
    _VCLatticeLayout,
    codespace_effective_hamiltonian,
    verstraete_cirac,
)
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian


# --- Helpers -----------------------------------------------------------------


def _hubbard_hamiltonian(rows: int, cols: int, t: float, U: float):  # noqa: N803
    """Create a Fermi-Hubbard Hamiltonian on an open-boundary square lattice."""
    from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian  # noqa: PLC0415

    lattice = LatticeGraph.square(cols, rows, periodic_x=False, periodic_y=False)
    return create_hubbard_hamiltonian(lattice, epsilon=0.0, t=t, U=U)


def _pauli_weight(pauli_str: str) -> int:
    """Count the number of non-identity Paulis in a label string."""
    return sum(1 for c in pauli_str if c != "I")


def _qh_term_dict(qh: QubitHamiltonian) -> dict[str, complex]:
    """Build a {pauli_string: coefficient} dict from a QubitHamiltonian."""
    return dict(zip(qh.pauli_strings, qh.coefficients, strict=True))


# --- Criterion 1: factory + QubitMapper consumption --------------------------


class TestConstruction:
    """VC factory produces valid MajoranaMappings that QubitMapper consumes."""

    @pytest.mark.parametrize(
        "rows,cols",
        [(2, 2), (2, 3), (3, 3), (4, 4)],
        ids=["2x2", "2x3", "3x3", "4x4"],
    )
    def test_factory_single_species(self, rows: int, cols: int) -> None:
        """Factory produces correct MajoranaMapping dimensions (single species)."""
        mapping = verstraete_cirac(rows, cols, num_species=1)
        n_sites = rows * cols
        assert mapping.num_modes == n_sites
        assert mapping.name == "verstraete-cirac"
        assert mapping.is_majorana_atomic
        assert len(mapping.table) == 2 * n_sites

    @pytest.mark.parametrize(
        "rows,cols",
        [(2, 2), (2, 3), (3, 3), (4, 4)],
        ids=["2x2", "2x3", "3x3", "4x4"],
    )
    def test_qubit_mapper_consumes_without_error(self, rows: int, cols: int) -> None:
        """QubitMapper.run() succeeds with the VC mapping for each lattice size."""
        ham = _hubbard_hamiltonian(rows, cols, t=1.0, U=4.0)
        mapping = verstraete_cirac(rows, cols, num_species=2)
        mapper = create("qubit_mapper", "qdk")
        qh = mapper.run(ham, mapping)
        assert isinstance(qh, QubitHamiltonian)
        assert len(qh.pauli_strings) > 0
        assert qh.num_qubits == mapping.num_qubits

    def test_two_species_qubit_count(self) -> None:
        """Two-species mapping uses 2x the single-species qubit count."""
        for rows, cols in [(2, 2), (3, 3)]:
            m1 = verstraete_cirac(rows, cols, num_species=1)
            m2 = verstraete_cirac(rows, cols, num_species=2)
            assert m2.num_qubits == 2 * m1.num_qubits

    def test_invalid_lattice_rejected(self) -> None:
        """Lattice smaller than 2x2 is rejected."""
        with pytest.raises(ValueError, match="at least 2"):
            verstraete_cirac(1, 3)
        with pytest.raises(ValueError, match="at least 2"):
            verstraete_cirac(2, 1)

    def test_invalid_species_rejected(self) -> None:
        """num_species < 1 is rejected."""
        with pytest.raises(ValueError, match="num_species"):
            verstraete_cirac(2, 2, num_species=0)


# --- Criterion 2: eigenvalue correctness ------------------------------------


class TestEigenvalueCorrectness:
    """VC codespace eigenvalues match JW for 2x2 Fermi-Hubbard."""

    def test_2x2_hubbard_eigenvalues(self) -> None:
        """Four lowest eigenvalues match JW to within 1e-10 (2x2, t=1, U=4)."""
        rows, cols = 2, 2
        n_modes = 2 * rows * cols  # two spin species

        ham = _hubbard_hamiltonian(rows, cols, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")

        # Jordan-Wigner reference
        jw_mapping = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        qh_jw = mapper.run(ham, jw_mapping)
        eigs_jw = np.sort(np.real(np.linalg.eigvalsh(qh_jw.to_matrix())))

        # Verstraete-Cirac: map then project onto codespace
        vc_mapping = verstraete_cirac(rows, cols, num_species=2)
        qh_vc = mapper.run(ham, vc_mapping)
        H_eff = codespace_effective_hamiltonian(qh_vc, rows, cols, num_species=2)  # noqa: N806
        eigs_vc = np.sort(np.real(np.linalg.eigvalsh(H_eff)))

        # Both omit core energy, so compare directly
        np.testing.assert_allclose(
            eigs_vc[:4],
            eigs_jw[:4],
            atol=1e-10,
            err_msg="VC lowest 4 eigenvalues do not match JW",
        )

    def test_codespace_dimension(self) -> None:
        """Codespace has the expected dimension 2^(num_modes)."""
        rows, cols = 2, 2
        ham = _hubbard_hamiltonian(rows, cols, t=1.0, U=4.0)
        mapping = verstraete_cirac(rows, cols, num_species=2)
        mapper = create("qubit_mapper", "qdk")
        qh_vc = mapper.run(ham, mapping)

        H_eff = codespace_effective_hamiltonian(qh_vc, rows, cols, num_species=2)  # noqa: N806
        expected_dim = 2 ** (2 * rows * cols)
        assert H_eff.shape == (expected_dim, expected_dim)


# --- Criterion 3: locality --------------------------------------------------


class TestLocality:
    """Max nearest-neighbour hopping Pauli weight is constant across sizes."""

    @staticmethod
    def _max_hopping_weight(rows: int, cols: int) -> int:
        """Max Pauli weight of hopping terms for a pure-hopping Hamiltonian."""
        lattice = LatticeGraph.square(cols, rows, periodic_x=False, periodic_y=False)
        from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian  # noqa: PLC0415

        ham = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=0.0)
        mapping = verstraete_cirac(rows, cols, num_species=2)
        mapper = create("qubit_mapper", "qdk")
        qh = mapper.run(ham, mapping)
        return max(_pauli_weight(ps) for ps in qh.pauli_strings)

    def test_constant_pauli_weight(self) -> None:
        """Max nearest-neighbour hopping weight is identical for L=2, 3, 4."""
        weights = {L: self._max_hopping_weight(L, L) for L in (2, 3, 4)}
        assert weights[2] == weights[3] == weights[4], (
            f"Pauli weights differ across lattice sizes: {weights}"
        )
        assert weights[2] > 0


# --- Criterion 4: serialization round-trips ----------------------------------


class TestSerialization:
    """VC mapping survives JSON/HDF5 round-trips with identical QubitHamiltonian."""

    @pytest.mark.parametrize("rows,cols", [(2, 2), (2, 3)], ids=["2x2", "2x3"])
    def test_json_mapping_roundtrip(self, rows: int, cols: int) -> None:
        """MajoranaMapping Majorana table survives JSON round-trip."""
        mapping = verstraete_cirac(rows, cols, num_species=1)
        loaded = MajoranaMapping.from_json(mapping.to_json())

        assert loaded.num_modes == mapping.num_modes
        assert loaded.num_qubits == mapping.num_qubits
        assert loaded.name == mapping.name
        for k in range(2 * mapping.num_modes):
            assert list(loaded.majorana(k)) == list(mapping.majorana(k))

    @pytest.mark.parametrize("rows,cols", [(2, 2), (2, 3)], ids=["2x2", "2x3"])
    def test_hdf5_mapping_roundtrip(self, rows: int, cols: int) -> None:
        """MajoranaMapping Majorana table survives HDF5 round-trip."""
        mapping = verstraete_cirac(rows, cols, num_species=1)
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                mapping.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded = MajoranaMapping.from_hdf5(hf)

        assert loaded.num_modes == mapping.num_modes
        assert loaded.num_qubits == mapping.num_qubits
        for k in range(2 * mapping.num_modes):
            assert list(loaded.majorana(k)) == list(mapping.majorana(k))

    def test_json_roundtrip_produces_identical_hamiltonian(self) -> None:
        """JSON round-tripped mapping yields term-by-term identical QubitHamiltonian."""
        mapping = verstraete_cirac(2, 2, num_species=2)
        loaded = MajoranaMapping.from_json(mapping.to_json())

        ham = _hubbard_hamiltonian(2, 2, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")
        qh_orig = mapper.run(ham, mapping)
        qh_loaded = mapper.run(ham, loaded)

        d_orig = _qh_term_dict(qh_orig)
        d_loaded = _qh_term_dict(qh_loaded)
        assert len(d_orig) == len(d_loaded)
        for ps, coeff in d_orig.items():
            assert ps in d_loaded, f"Missing Pauli string after JSON round-trip: {ps}"
            np.testing.assert_allclose(d_loaded[ps], coeff, atol=1e-14)

    def test_hdf5_roundtrip_produces_identical_hamiltonian(self) -> None:
        """HDF5 round-tripped mapping yields term-by-term identical QubitHamiltonian."""
        mapping = verstraete_cirac(2, 2, num_species=2)
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                mapping.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded = MajoranaMapping.from_hdf5(hf)

        ham = _hubbard_hamiltonian(2, 2, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")
        qh_orig = mapper.run(ham, mapping)
        qh_loaded = mapper.run(ham, loaded)

        d_orig = _qh_term_dict(qh_orig)
        d_loaded = _qh_term_dict(qh_loaded)
        assert len(d_orig) == len(d_loaded)
        for ps, coeff in d_orig.items():
            assert ps in d_loaded, f"Missing Pauli string after HDF5 round-trip: {ps}"
            np.testing.assert_allclose(d_loaded[ps], coeff, atol=1e-14)


# --- Lattice layout unit tests -----------------------------------------------


class TestLatticeLayout:
    """Unit tests for _VCLatticeLayout internals."""

    def test_uniform_degree(self) -> None:
        """Every site has exactly 4 incident edge qubits (padded with phantoms)."""
        for rows, cols in [(2, 2), (2, 3), (3, 3), (4, 4)]:
            layout = _VCLatticeLayout(rows, cols)
            for y in range(rows):
                for x in range(cols):
                    edges = layout.incident_edge_qubits(x, y)
                    assert len(edges) == 4, f"({x},{y}) on {rows}x{cols}: {len(edges)} edges"
                    assert len(set(edges)) == 4, f"({x},{y}) has duplicate edge qubits"

    def test_shared_edge_between_neighbours(self) -> None:
        """Adjacent sites share exactly one real edge qubit."""
        layout = _VCLatticeLayout(3, 3)
        # Horizontal neighbour (0,0)-(1,0)
        shared_h = set(layout.incident_edge_qubits(0, 0)) & set(layout.incident_edge_qubits(1, 0))
        assert len(shared_h) == 1
        # Vertical neighbour (0,0)-(0,1)
        shared_v = set(layout.incident_edge_qubits(0, 0)) & set(layout.incident_edge_qubits(0, 1))
        assert len(shared_v) == 1

    def test_no_qubit_index_overlap(self) -> None:
        """Site qubits and auxiliary qubits never share indices."""
        for rows, cols in [(2, 2), (3, 3)]:
            layout = _VCLatticeLayout(rows, cols)
            site_qs = {layout.site_qubit(x, y) for y in range(rows) for x in range(cols)}
            aux_qs = set()
            for y in range(rows):
                for x in range(cols):
                    aux_qs.update(layout.incident_edge_qubits(x, y))
            assert site_qs.isdisjoint(aux_qs)
