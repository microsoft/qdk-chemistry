"""Tests for Verstraete-Cirac fermion-to-qubit mapping."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile

import h5py
import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_huckel_hamiltonian


class TestVerstraeteCiracMapping:
    """Tests covering the Verstraete-Cirac mapping factory, dimensions, and properties."""

    def test_factory_and_dimensions(self) -> None:
        """Test that VC factory accepts correct grid dimensions and rejects invalid ones."""
        # Valid even x_dimension grids
        lattice_2x2 = LatticeGraph.square(2, 2)
        mapping_2x2 = MajoranaMapping.verstraete_cirac(lattice_2x2)
        assert mapping_2x2.grid_nx == 2
        assert mapping_2x2.grid_ny == 2
        assert mapping_2x2.name == "verstraete-cirac"
        assert mapping_2x2.base_encoding == "verstraete-cirac"
        assert not mapping_2x2.is_majorana_atomic

        lattice_2x3 = LatticeGraph.square(2, 3)
        mapping_2x3 = MajoranaMapping.verstraete_cirac(lattice_2x3)
        assert mapping_2x3.grid_nx == 2
        assert mapping_2x3.grid_ny == 3

        lattice_4x4 = LatticeGraph.square(4, 4)
        mapping_4x4 = MajoranaMapping.verstraete_cirac(lattice_4x4)
        assert mapping_4x4.grid_nx == 4
        assert mapping_4x4.grid_ny == 4

        lattice_3x3 = LatticeGraph.square(3, 3)
        mapping_3x3 = MajoranaMapping.verstraete_cirac(lattice_3x3)
        assert mapping_3x3.grid_nx == 3
        assert mapping_3x3.grid_ny == 3

        # Invalid spin species count should raise ValueError / invalid_argument
        with pytest.raises(ValueError, match="spin"):
            MajoranaMapping.verstraete_cirac(lattice_2x2, num_spin_species=0)

    def test_general_lattices_single_spin(self) -> None:
        """Verify that QubitMapper consumes VC mapping for general 2D lattices with single spin species."""
        mapper = create("qubit_mapper", "qdk")
        for nx, ny in [(2, 2), (2, 3), (3, 3), (4, 4)]:
            lattice = LatticeGraph.square(nx, ny)
            ham = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)
            mapping = MajoranaMapping.verstraete_cirac(lattice, num_spin_species=1)
            qh = mapper.run(ham, mapping)
            assert qh.num_qubits == 2 * nx * ny

    def test_majorana_raises_error(self) -> None:
        """Verstraete-Cirac mapping is bilinear-only and majorana(k) should raise ValueError."""
        lattice = LatticeGraph.square(2, 2)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        with pytest.raises(ValueError, match="bilinear-only"):
            mapping.majorana(0)

    def test_without_tapering(self) -> None:
        """without_tapering should return the mapping itself (as it has no tapering)."""
        lattice = LatticeGraph.square(2, 2)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        base = mapping.without_tapering()
        assert base.grid_nx == 2
        assert base.grid_ny == 2
        assert base.name == "verstraete-cirac"

    def test_serialization_round_trip(self) -> None:
        """Verify that Verstraete-Cirac mapping survives JSON and HDF5 serialization round-trips."""
        lattice = LatticeGraph.square(2, 2)
        mapping = MajoranaMapping.verstraete_cirac(lattice)

        # JSON Round-trip
        json_data = mapping.to_json()
        loaded_json = MajoranaMapping.from_json(json_data)
        assert loaded_json.name == mapping.name
        assert loaded_json.grid_nx == mapping.grid_nx
        assert loaded_json.grid_ny == mapping.grid_ny
        assert loaded_json.num_modes == mapping.num_modes
        assert loaded_json.num_qubits == mapping.num_qubits
        assert not loaded_json.is_majorana_atomic

        # HDF5 Round-trip
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                mapping.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded_hdf5 = MajoranaMapping.from_hdf5(hf)
        assert loaded_hdf5.name == mapping.name
        assert loaded_hdf5.grid_nx == mapping.grid_nx
        assert loaded_hdf5.grid_ny == mapping.grid_ny
        assert loaded_hdf5.num_modes == mapping.num_modes
        assert loaded_hdf5.num_qubits == mapping.num_qubits
        assert not loaded_hdf5.is_majorana_atomic


class TestVerstraeteCiracHubbardSpectral:
    """Spectral validation of Verstraete-Cirac mapping for Fermi-Hubbard model."""

    def test_spectral_validation_2x2_hubbard(self) -> None:
        """Compare eigenvalues of 2x2 Hubbard model under VC and Jordan-Wigner mappings.

        Model parameters: t = 1.0, U = 4.0, half-filling (epsilon = -U/2 = -2.0).
        """
        # 1. Construct 2x2 square lattice
        lattice = LatticeGraph.square(2, 2)
        # 4 sites in lattice, each with 2 spin species -> 8 spin-orbitals / modes
        n_modes = 8

        # 2. Construct Hubbard Hamiltonian
        hubbard_ham = create_hubbard_hamiltonian(lattice, epsilon=-2.0, t=1.0, U=4.0)

        # 3. Map using Jordan-Wigner (JW)
        mapper = create("qubit_mapper", "qdk")
        jw_mapping = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        qh_jw = mapper.run(hubbard_ham, jw_mapping)

        # Solve JW eigenvalues using sparse solver
        h_jw = qh_jw.to_matrix(sparse=True)
        eigs_jw, _ = eigsh(h_jw, k=10, which="SA")
        eigs_jw = np.sort(eigs_jw)
        unique_jw = np.unique(np.round(eigs_jw, 6))

        # 4. Map using Verstraete-Cirac (VC)
        vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh_vc = mapper.run(hubbard_ham, vc_mapping)

        # Verify VC number of qubits.
        assert qh_vc.num_qubits == 2 * n_modes

        # Solve VC eigenvalues using sparse solver
        h_vc = qh_vc.to_matrix(sparse=True)
        # Solve for 128 eigenvalues to cover the 64-fold degeneracy of the ground state and first excited state
        eigs_vc, _ = eigsh(h_vc, k=128, which="SA")
        eigs_vc = np.sort(eigs_vc)
        unique_vc = np.unique(np.round(eigs_vc, 6))

        # The unique lowest eigenvalues in the code space of VC must match the unique lowest eigenvalues of JW.
        # We check the 2 lowest unique eigenvalues.
        np.testing.assert_allclose(unique_vc[:2], unique_jw[:2], atol=1e-10)

        # Test serialization of the resulting QubitHamiltonian
        qh_json = qh_vc.to_json()
        qh_loaded = QubitHamiltonian.from_json(qh_json)
        assert qh_loaded.num_qubits == qh_vc.num_qubits
        assert len(qh_loaded.pauli_strings) == len(qh_vc.pauli_strings)


class TestVerstraeteCiracPauliWeightScaling:
    """Verify that nearest-neighbor hop Pauli weight remains constant for grid lengths L in {2, 3, 4}."""

    def test_pauli_weight_scaling(self) -> None:
        """Check nearest-neighbor hopping terms in the mapped qubit Hamiltonian.

        Max Pauli weight of nearest-neighbor hops should be constant for grids of dimensions:
        - 2x2 (L=2)
        - 2x3 (L=3)
        - 2x4 (L=4)
        """
        mapper = create("qubit_mapper", "qdk")

        max_weights = []
        for grid_length in [2, 3, 4]:
            lattice = LatticeGraph.square(2, grid_length)
            # Create a simple hopping-only Hamiltonian (U=0) on the 2xL lattice
            ham = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=0.0)

            vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
            qh_vc = mapper.run(ham, vc_mapping)

            weights = []
            for pauli_str in qh_vc.pauli_strings:
                if pauli_str == "I" * len(pauli_str):
                    continue
                # Weight is the number of non-I characters
                weight = sum(1 for c in pauli_str if c != "I")
                weights.append(weight)

            max_weights.append(max(weights))

        # Check that the maximum Pauli weight does not grow with L
        assert max_weights[0] == max_weights[1] == max_weights[2], (
            f"Maximum Pauli weights grew or varied with grid length: {max_weights}"
        )
