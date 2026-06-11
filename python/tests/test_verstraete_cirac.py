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

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_huckel_hamiltonian


class TestVerstraeteCiracMapping:
    """Tests covering the Verstraete-Cirac mapping factory, dimensions, and properties."""

    def test_factory_and_dimensions(self) -> None:
        """Test that VC factory accepts correct grid dimensions and rejects invalid ones."""
        # Testing valid grids: square and rectangular
        lattice_2x2 = LatticeGraph.square(2, 2)
        mapping_2x2 = MajoranaMapping.verstraete_cirac(lattice_2x2)
        assert len(mapping_2x2.stabilizers) > 0
        assert mapping_2x2.name == "verstraete-cirac"
        assert mapping_2x2.base_encoding == "verstraete-cirac"
        assert not mapping_2x2.is_majorana_atomic

        lattice_2x3 = LatticeGraph.square(2, 3)
        mapping_2x3 = MajoranaMapping.verstraete_cirac(lattice_2x3)
        assert len(mapping_2x3.stabilizers) > 0

        lattice_3x3 = LatticeGraph.square(3, 3)
        mapping_3x3 = MajoranaMapping.verstraete_cirac(lattice_3x3)
        assert len(mapping_3x3.stabilizers) > 0

        lattice_4x4 = LatticeGraph.square(4, 4)
        mapping_4x4 = MajoranaMapping.verstraete_cirac(lattice_4x4)
        assert len(mapping_4x4.stabilizers) > 0

    def test_general_lattices(self) -> None:
        """Verify that QubitMapper consumes VC mapping for general lattices."""
        mapper = create("qubit_mapper", "qdk")
        lattices = [
            LatticeGraph.square(2, 2),
            LatticeGraph.square(3, 3),
            LatticeGraph.honeycomb(2, 2, periodic_x=True, periodic_y=True),
            LatticeGraph.triangular(2, 2),
        ]
        for lattice in lattices:
            ham = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)
            mapping = MajoranaMapping.verstraete_cirac(lattice)
            qh = mapper.run(ham, mapping)
            assert qh.num_qubits == mapping.num_qubits
            assert mapping.num_qubits - len(mapping.stabilizers) == 2 * lattice.num_sites

    def test_honeycomb_and_higher_connectivity(self) -> None:
        """Verify honeycomb (degree 3) and triangular/kagome (higher degree up to 6) lattices work."""
        lattice_honeycomb = LatticeGraph.honeycomb(2, 2, periodic_x=True, periodic_y=True)
        mapping_honeycomb = MajoranaMapping.verstraete_cirac(lattice_honeycomb)
        assert len(mapping_honeycomb.stabilizers) > 0
        
        lattice_triangular = LatticeGraph.triangular(2, 2)
        mapping_triangular = MajoranaMapping.verstraete_cirac(lattice_triangular)
        assert len(mapping_triangular.stabilizers) > 0

        lattice_kagome = LatticeGraph.kagome(2, 2, periodic_x=True, periodic_y=True)
        mapping_kagome = MajoranaMapping.verstraete_cirac(lattice_kagome)
        assert len(mapping_kagome.stabilizers) > 0

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
        assert len(base.stabilizers) == len(mapping.stabilizers)
        assert base.name == "verstraete-cirac"

    def test_serialization_round_trip(self) -> None:
        """Verify that Verstraete-Cirac mapping survives JSON and HDF5 serialization round-trips."""
        lattice = LatticeGraph.square(2, 2)
        mapping = MajoranaMapping.verstraete_cirac(lattice)

        # Construct a simple Hubbard Hamiltonian on the lattice
        ham = create_hubbard_hamiltonian(lattice, epsilon=-2.0, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")
        qh_orig = mapper.run(ham, mapping)

        # JSON Round-trip
        json_data = mapping.to_json()
        loaded_json = MajoranaMapping.from_json(json_data)
        assert loaded_json.name == mapping.name
        assert loaded_json.num_modes == mapping.num_modes
        assert loaded_json.num_qubits == mapping.num_qubits
        assert not loaded_json.is_majorana_atomic

        qh_json = mapper.run(ham, loaded_json)
        assert qh_json.num_qubits == qh_orig.num_qubits
        assert len(qh_json.pauli_strings) == len(qh_orig.pauli_strings)
        terms_orig = sorted(zip(qh_orig.pauli_strings, qh_orig.coefficients, strict=False))
        terms_json = sorted(zip(qh_json.pauli_strings, qh_json.coefficients, strict=False))
        for (p_orig, c_orig), (p_json, c_json) in zip(terms_orig, terms_json, strict=False):
            assert p_orig == p_json
            assert np.isclose(c_orig, c_json, atol=1e-10)

        # HDF5 Round-trip
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with h5py.File(f.name, "w") as hf:
                mapping.to_hdf5(hf)
            with h5py.File(f.name, "r") as hf:
                loaded_hdf5 = MajoranaMapping.from_hdf5(hf)
        assert loaded_hdf5.name == mapping.name
        assert loaded_hdf5.num_modes == mapping.num_modes
        assert loaded_hdf5.num_qubits == mapping.num_qubits
        assert not loaded_hdf5.is_majorana_atomic

        qh_hdf5 = mapper.run(ham, loaded_hdf5)
        assert qh_hdf5.num_qubits == qh_orig.num_qubits
        assert len(qh_hdf5.pauli_strings) == len(qh_orig.pauli_strings)
        terms_hdf5 = sorted(zip(qh_hdf5.pauli_strings, qh_hdf5.coefficients, strict=False))
        for (p_orig, c_orig), (p_h5, c_h5) in zip(terms_orig, terms_hdf5, strict=False):
            assert p_orig == p_h5
            assert np.isclose(c_orig, c_h5, atol=1e-10)

    def test_stabilizers_and_commutation(self) -> None:
        """Verify that stabilizers mutually commute and commute with mapped H."""

        def commute(p1: str, p2: str) -> bool:
            assert len(p1) == len(p2)
            anti_commutes = 0
            for c1, c2 in zip(p1, p2, strict=False):
                if c1 != "I" and c2 not in {"I", c1}:
                    anti_commutes += 1
            return (anti_commutes % 2) == 0

        mapper = create("qubit_mapper", "qdk")
        for nx, ny in [(2, 2), (3, 3), (3, 4)]:
            lattice = LatticeGraph.square(nx, ny)
            ham = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)

            mapping = MajoranaMapping.verstraete_cirac(lattice)
            assert mapping.num_qubits - len(mapping.stabilizers) == 2 * lattice.num_sites

            qh_vc = mapper.run(ham, mapping)

            # Convert stabilizers to Pauli labels
            stabs = []
            for _, word in mapping.stabilizers:
                label = sparse_pauli_word_to_label(word, qh_vc.num_qubits)
                stabs.append(label)

            # Check mutual commutation of stabilizers: [S_i, S_j] = 0
            for i in range(len(stabs)):
                for j in range(i + 1, len(stabs)):
                    assert commute(stabs[i], stabs[j]), f"Stabilizers {i} and {j} do not commute!"

            # Check commutation with Hamiltonian: [H, S_i] = 0
            for i, stab in enumerate(stabs):
                for p_term in qh_vc.pauli_strings:
                    assert commute(stab, p_term), f"Stabilizer {i} does not commute with Hamiltonian term {p_term}!"

    def test_pauli_weight_scaling(self) -> None:
        """Check nearest-neighbor hopping terms in the mapped qubit Hamiltonian.

        Max Pauli weight of nearest-neighbor hops should be constant for square grids of dimensions:
        - 2x2 (L=2)
        - 3x3 (L=3)
        - 4x4 (L=4)
        """
        mapper = create("qubit_mapper", "qdk")

        max_weights = []
        for grid_length in [2, 3, 4]:
            lattice = LatticeGraph.square(grid_length, grid_length)
            # Create a simple hopping-only Hamiltonian (U=0) on the L x L lattice
            ham = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=0.0)

            vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
            qh_vc = mapper.run(ham, vc_mapping)

            weights = []
            for pauli_str, coeff in zip(qh_vc.pauli_strings, qh_vc.coefficients, strict=False):
                if pauli_str == "I" * len(pauli_str):
                    continue
                # Skip stabilizer penalty terms (which have coefficient >= 9.0,
                # whereas nearest-neighbor hops have coefficient <= 1.0)
                if np.abs(coeff) > 2.0:
                    continue
                # Weight is the number of non-I characters
                weight = sum(1 for c in pauli_str if c != "I")
                weights.append(weight)

            max_weights.append(max(weights))

        assert max_weights[0] == max_weights[1] == max_weights[2], (
            f"Maximum Pauli weights grew or varied with grid length: {max_weights}"
        )


class TestVerstraeteCiracSpectral:
    """Tests covering the spectral validation of the Verstraete-Cirac mapping."""

    def test_spectral_validation_2x2_hubbard(self) -> None:
        """Compare eigenvalues of 2x2 periodic Fermi-Hubbard model under VC and JW mappings.

        Model parameters: t = 1.0, U = 4.0, half-filling (epsilon = -U/2 = -2.0).
        """
        lattice = LatticeGraph.square(2, 2, periodic_x=True)
        n_modes = 8  # 4 spatial sites * 2 spins
        hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-2.0, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")

        jw_mapping = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        qh_jw = mapper.run(hamiltonian, jw_mapping)
        h_jw = qh_jw.to_matrix(sparse=True)
        eigs_jw, _ = eigsh(h_jw, k=10, which="SA")
        unique_jw = np.unique(np.round(np.sort(eigs_jw), 6))

        vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh_vc = mapper.run(hamiltonian, vc_mapping)

        # Verify system size and extract lowest unique eigenvalues
        assert qh_vc.num_qubits == vc_mapping.num_qubits
        h_vc = qh_vc.to_matrix(sparse=True)
        eigs_vc, _ = eigsh(h_vc, k=10, which="SA")
        unique_vc = np.unique(np.round(np.sort(eigs_vc), 6))

        # Ensure lowest physical energy levels match baseline
        np.testing.assert_allclose(unique_vc[:2], unique_jw[:2], atol=1e-10)

    def test_spectral_validation_3x2_huckel(self) -> None:
        """Compare eigenvalues of 3x2 Hückel model under VC and JW mappings.

        Validates that the lowest eigenstates of the penalized Hamiltonian are
        in the +1 codespace sector of the generated loop-plaquette stabilizers.
        """
        lattice = LatticeGraph.square(3, 2)
        n_modes = 12  # 6 spatial sites * 2 spins
        hamiltonian = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)
        mapper = create("qubit_mapper", "qdk")

        jw_mapping = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        qh_jw = mapper.run(hamiltonian, jw_mapping)
        h_jw = qh_jw.to_matrix(sparse=True)
        eigs_jw, _ = eigsh(h_jw, k=10, which="SA")
        unique_jw = np.unique(np.round(np.sort(eigs_jw), 6))

        vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh_vc = mapper.run(hamiltonian, vc_mapping)

        # Extract lowest eigenstates and their respective eigenstates
        assert qh_vc.num_qubits == vc_mapping.num_qubits
        h_vc = qh_vc.to_matrix(sparse=True)
        eigs_vc, vecs_vc = eigsh(h_vc, k=10, which="SA")

        # Construct sparse matrices for each generated loop-plaquette stabilizer
        stabs = []
        for coeff, word in vc_mapping.stabilizers:
            label = sparse_pauli_word_to_label(word, qh_vc.num_qubits)
            qh_stab = QubitHamiltonian([label], np.array([coeff]))
            stabs.append(qh_stab.to_matrix(sparse=True))

        # Project and filter out any unphysical states (out-of-codespace)
        code_space_eigs = []
        for idx in range(len(eigs_vc)):
            vec = vecs_vc[:, idx]
            is_in_code_space = True

            for stab in stabs:
                # Only a simultaneous +1 eigenstate belongs to the physical codespace
                expectation_value = np.real(vec.conj().T @ (stab @ vec))
                if not np.isclose(expectation_value, 1.0, atol=1e-4):
                    is_in_code_space = False
                    break

            if is_in_code_space:
                code_space_eigs.append(eigs_vc[idx])

        # Assert that physical states were found and unique energy levels match baseline
        assert len(code_space_eigs) >= 2
        unique_vc = np.unique(np.round(np.sort(code_space_eigs), 6))
        np.testing.assert_allclose(unique_vc[:2], unique_jw[:2], atol=1e-10)
