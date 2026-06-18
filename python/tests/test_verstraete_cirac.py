"""Tests for Verstraete-Cirac fermion-to-qubit mapping."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import tempfile

import h5py
import numpy as np
import pytest
import scipy.sparse
from scipy.sparse.linalg import eigsh

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_huckel_hamiltonian


def _particle_number_operator(mapping: MajoranaMapping) -> QubitHamiltonian:
    """Build the total physical particle-number operator for a Majorana mapping."""
    labels = ["I" * mapping.num_qubits]
    coefficients = [0.5 * mapping.num_modes]

    for mode in range(mapping.num_modes):
        coeff, word = mapping.bilinear(2 * mode, 2 * mode + 1)
        labels.append(sparse_pauli_word_to_label(word, mapping.num_qubits))
        coefficients.append(0.5 * coeff)

    return QubitHamiltonian(labels, np.array(coefficients, dtype=complex))


def _particle_number_basis_indices(mapping: MajoranaMapping, n_particles: int) -> np.ndarray:
    """Return computational-basis indices in the requested physical particle-number sector."""
    number_matrix = _particle_number_operator(mapping).to_matrix(sparse=True).tocsr()
    diagonal = number_matrix.diagonal()
    off_diagonal = number_matrix - scipy.sparse.diags(diagonal, format="csr")
    if off_diagonal.nnz:
        assert np.max(np.abs(off_diagonal.data)) < 1e-12
    return np.flatnonzero(np.isclose(diagonal, n_particles, atol=1e-10))


def _stabilizer_codespace_projector(mapping: MajoranaMapping) -> scipy.sparse.csr_matrix:
    """Project onto the simultaneous +1 eigenspace of the mapping stabilizers."""
    dim = 1 << mapping.num_qubits
    projector = scipy.sparse.identity(dim, dtype=complex, format="csr")

    for coeff, word in mapping.stabilizers:
        label = sparse_pauli_word_to_label(word, mapping.num_qubits)
        stabilizer = QubitHamiltonian([label], np.array([coeff], dtype=complex)).to_matrix(sparse=True).tocsr()
        projector = 0.5 * (projector + stabilizer @ projector)

    return projector.tocsr()


def _orthonormal_projected_basis(
    projector: scipy.sparse.csr_matrix,
    candidate_indices: np.ndarray,
    expected_dim: int,
) -> np.ndarray:
    """Build an orthonormal basis for projected candidate basis states."""
    basis: list[np.ndarray] = []

    for idx in candidate_indices:
        vec = np.asarray(projector[:, idx].toarray()).ravel()
        if np.linalg.norm(vec) < 1e-12:
            continue

        for basis_vec in basis:
            vec -= basis_vec * np.vdot(basis_vec, vec)

        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue

        basis.append(vec / norm)
        if len(basis) == expected_dim:
            break

    assert len(basis) == expected_dim
    return np.column_stack(basis)


def _restricted_eigenvalues(hamiltonian: QubitHamiltonian, basis: np.ndarray) -> np.ndarray:
    """Diagonalize a Hamiltonian restricted to the span of an orthonormal basis."""
    matrix = hamiltonian.to_matrix(sparse=True)
    restricted = basis.conj().T @ (matrix @ basis)
    restricted = 0.5 * (restricted + restricted.conj().T)
    return np.linalg.eigvalsh(restricted)


def _lattice_edges(lattice: LatticeGraph) -> list[tuple[int, int, float]]:
    """Return undirected lattice edges with their graph weights."""
    edges = []
    for i in range(lattice.num_sites):
        for j in range(i + 1, lattice.num_sites):
            weight = lattice.weight(i, j)
            if weight != 0.0:
                edges.append((i, j, weight))
    return edges


def _hopping_pauli_terms(
    lattice: LatticeGraph,
    mapping: MajoranaMapping,
    hopping: float,
) -> dict[str, complex]:
    """Expand only the physical nearest-neighbor hopping terms for a lattice mapping."""
    excitation_coefficients = (
        (1.0 + 0.0j, 0.0 + 1.0j),
        (0.0 - 1.0j, 1.0 + 0.0j),
    )
    n_spatial = lattice.num_sites
    terms: dict[str, complex] = {}

    for i, j, weight in _lattice_edges(lattice):
        h_ij = -hopping * weight
        for spin_offset in (0, n_spatial):
            for p, q in (
                (spin_offset + i, spin_offset + j),
                (spin_offset + j, spin_offset + i),
            ):
                for a in range(2):
                    for b in range(2):
                        coeff, word = mapping.bilinear(2 * p + a, 2 * q + b)
                        label = sparse_pauli_word_to_label(word, mapping.num_qubits)
                        term_coeff = (
                            -1j
                            * coeff
                            * h_ij
                            * 0.25
                            * excitation_coefficients[a][b]
                        )
                        terms[label] = terms.get(label, 0.0 + 0.0j) + term_coeff

    return {label: coeff for label, coeff in terms.items() if coeff != 0.0}


class TestVerstraeteCiracMapping:
    """Tests covering the Verstraete-Cirac mapping factory, dimensions, and properties."""

    def test_factory_and_dimensions(self) -> None:
        """Test that VC factory accepts valid lattices (>= 3 sites) and rejects lattices with < 3 sites."""
        # Testing valid lattices with >= 3 sites
        lattice_2x2 = LatticeGraph.square(2, 2)
        mapping_2x2 = MajoranaMapping.verstraete_cirac(lattice_2x2)
        assert len(mapping_2x2.stabilizers) > 0
        assert len(mapping_2x2.auxiliary_penalty_terms) > 0
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

        # Testing invalid lattices with < 3 sites
        lattice_1x2 = LatticeGraph.square(1, 2)
        with pytest.raises(ValueError, match="requires a lattice graph with at least 3 sites"):
            MajoranaMapping.verstraete_cirac(lattice_1x2)

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
        assert len(base.auxiliary_penalty_terms) == len(mapping.auxiliary_penalty_terms)
        assert base.name == "verstraete-cirac"

    def test_json_serialization(self) -> None:
        """Verify that Verstraete-Cirac mapping survives JSON serialization."""
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
        assert len(loaded_json.stabilizers) == len(mapping.stabilizers)
        assert len(loaded_json.auxiliary_penalty_terms) == len(mapping.auxiliary_penalty_terms)

        qh_json = mapper.run(ham, loaded_json)
        assert qh_json.num_qubits == qh_orig.num_qubits
        assert len(qh_json.pauli_strings) == len(qh_orig.pauli_strings)
        terms_orig = sorted(zip(qh_orig.pauli_strings, qh_orig.coefficients, strict=False))
        terms_json = sorted(zip(qh_json.pauli_strings, qh_json.coefficients, strict=False))
        for (p_orig, c_orig), (p_json, c_json) in zip(terms_orig, terms_json, strict=False):
            assert p_orig == p_json
            assert np.isclose(c_orig, c_json, atol=1e-10)

    def test_hdf5_serialization(self) -> None:
        """Verify that Verstraete-Cirac mapping survives HDF5 serialization."""
        lattice = LatticeGraph.square(2, 2)
        mapping = MajoranaMapping.verstraete_cirac(lattice)

        # Construct a simple Hubbard Hamiltonian on the lattice
        ham = create_hubbard_hamiltonian(lattice, epsilon=-2.0, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")
        qh_orig = mapper.run(ham, mapping)
        terms_orig = sorted(zip(qh_orig.pauli_strings, qh_orig.coefficients, strict=False))

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
        assert len(loaded_hdf5.stabilizers) == len(mapping.stabilizers)
        assert len(loaded_hdf5.auxiliary_penalty_terms) == len(mapping.auxiliary_penalty_terms)

        qh_hdf5 = mapper.run(ham, loaded_hdf5)
        assert qh_hdf5.num_qubits == qh_orig.num_qubits
        assert len(qh_hdf5.pauli_strings) == len(qh_orig.pauli_strings)
        terms_hdf5 = sorted(zip(qh_hdf5.pauli_strings, qh_hdf5.coefficients, strict=False))
        for (p_orig, c_orig), (p_h5, c_h5) in zip(terms_orig, terms_hdf5, strict=False):
            assert p_orig == p_h5
            assert np.isclose(c_orig, c_h5, atol=1e-10)

    @pytest.mark.parametrize(
        ("lattice_type", "args", "kwargs"),
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
    def test_stabilizers_and_commutation(self, lattice_type: str, args: tuple, kwargs: dict) -> None:
        """Verify that stabilizers mutually commute and commute with mapped H for various lattices."""

        def commute(p1: str, p2: str) -> bool:
            assert len(p1) == len(p2)
            anti_commutes = 0
            for c1, c2 in zip(p1, p2, strict=False):
                if c1 != "I" and c2 not in {"I", c1}:
                    anti_commutes += 1
            return (anti_commutes % 2) == 0

        mapper = create("qubit_mapper", "qdk")
        lattice = getattr(LatticeGraph, lattice_type)(*args, **kwargs)

        ham = create_huckel_hamiltonian(lattice, epsilon=-2.0, t=1.0)
        mapping = MajoranaMapping.verstraete_cirac(lattice)

        assert mapping.num_qubits - len(mapping.stabilizers) == 2 * lattice.num_sites
        assert len(mapping.stabilizers) > 0

        qh_vc = mapper.run(ham, mapping)
        assert qh_vc.num_qubits == mapping.num_qubits

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
            lattice = LatticeGraph.square(grid_length, grid_length, dfs_ordering=True)
            ham = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=0.0)
            vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
            qh_vc = mapper.run(ham, vc_mapping)

            expected_hopping_labels = set(_hopping_pauli_terms(lattice, vc_mapping, hopping=1.0))
            penalty_labels = {
                sparse_pauli_word_to_label(word, vc_mapping.num_qubits)
                for _, word in vc_mapping.auxiliary_penalty_terms
            }
            observed_hopping_labels = {
                pauli_str
                for pauli_str in qh_vc.pauli_strings
                if pauli_str != "I" * qh_vc.num_qubits and pauli_str not in penalty_labels
            }
            assert observed_hopping_labels == expected_hopping_labels

            weights = [
                sum(1 for c in pauli_str if c != "I")
                for pauli_str in observed_hopping_labels
            ]
            assert weights

            max_weights.append(max(weights))

        assert max_weights[0] == max_weights[1] == max_weights[2], (
            f"Maximum Pauli weights grew or varied with grid length: {max_weights}"
        )

    def test_stabilizer_penalty_includes_two_body_terms(self) -> None:
        """Pure Hubbard interactions should contribute to the stabilizer penalty scale."""
        lattice = LatticeGraph.square(2, 2)
        hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=0.0, U=20.0)
        mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh_vc = create("qubit_mapper", "qdk").run(hamiltonian, mapping)

        coeff_by_label = dict(zip(qh_vc.pauli_strings, qh_vc.coefficients, strict=True))
        penalty_labels = {
            sparse_pauli_word_to_label(word, mapping.num_qubits)
            for _, word in mapping.auxiliary_penalty_terms
        }
        assert penalty_labels

        penalty_coefficients = []
        for label in penalty_labels:
            penalty_coefficients.append(abs(coeff_by_label[label]))

        raw_stabilizer_labels = {
            sparse_pauli_word_to_label(word, mapping.num_qubits) for _, word in mapping.stabilizers
        }
        for label in raw_stabilizer_labels - penalty_labels:
            assert abs(coeff_by_label.get(label, 0.0)) < 1e-10

        for _, word in mapping.auxiliary_penalty_terms:
            label = sparse_pauli_word_to_label(word, mapping.num_qubits)
            assert label in coeff_by_label

        assert min(penalty_coefficients) > 10.0


class TestVerstraeteCiracSpectral:
    """Tests covering the spectral validation of the Verstraete-Cirac mapping."""

    def test_spectral_validation_2x2_hubbard(self) -> None:
        """Compare the four lowest eigenvalues of the 2x2 open Fermi-Hubbard model.

        Model parameters: t = 1.0, U = 4.0, half-filling (epsilon = -U/2 = -2.0).
        """
        lattice = LatticeGraph.square(2, 2, dfs_ordering=True)
        n_modes = 8  # 4 spatial sites * 2 spins
        hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-2.0, t=1.0, U=4.0)
        mapper = create("qubit_mapper", "qdk")
        n_half_filling = lattice.num_sites
        expected_sector_dim = math.comb(n_modes, n_half_filling)

        jw_mapping = MajoranaMapping.jordan_wigner(num_modes=n_modes)
        qh_jw = mapper.run(hamiltonian, jw_mapping)
        h_jw = qh_jw.to_matrix(sparse=True)
        jw_half_filling = _particle_number_basis_indices(jw_mapping, n_half_filling)
        assert len(jw_half_filling) == expected_sector_dim
        jw_sector = h_jw[jw_half_filling][:, jw_half_filling].toarray()
        eigs_jw = np.linalg.eigvalsh(0.5 * (jw_sector + jw_sector.conj().T))

        vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
        qh_vc = mapper.run(hamiltonian, vc_mapping)

        # Restrict explicitly to the physical half-filled stabilizer codespace.
        assert qh_vc.num_qubits == vc_mapping.num_qubits
        assert qh_vc.num_qubits - len(vc_mapping.stabilizers) == n_modes
        vc_half_filling = _particle_number_basis_indices(vc_mapping, n_half_filling)
        assert len(vc_half_filling) > expected_sector_dim

        vc_codespace_projector = _stabilizer_codespace_projector(vc_mapping)
        vc_basis = _orthonormal_projected_basis(vc_codespace_projector, vc_half_filling, expected_sector_dim)
        expected_identity = np.eye(expected_sector_dim, dtype=complex)

        vc_number_matrix = _particle_number_operator(vc_mapping).to_matrix(sparse=True)
        np.testing.assert_allclose(
            vc_basis.conj().T @ (vc_number_matrix @ vc_basis),
            n_half_filling * expected_identity,
            atol=1e-10,
        )
        for coeff, word in vc_mapping.stabilizers:
            label = sparse_pauli_word_to_label(word, vc_mapping.num_qubits)
            stabilizer = QubitHamiltonian([label], np.array([coeff], dtype=complex)).to_matrix(sparse=True)
            np.testing.assert_allclose(
                vc_basis.conj().T @ (stabilizer @ vc_basis),
                expected_identity,
                atol=1e-10,
            )

        eigs_vc = _restricted_eigenvalues(qh_vc, vc_basis)

        # Ensure the four lowest half-filled physical energy levels match the Jordan-Wigner baseline.
        np.testing.assert_allclose(eigs_vc[:4], eigs_jw[:4], atol=1e-10)

    @pytest.mark.slow
    def test_spectral_validation_3x2_huckel(self) -> None:
        """Compare eigenvalues of 3x2 Hückel model under VC and JW mappings.

        Validates that the lowest eigenstates of the penalized Hamiltonian are
        in the +1 codespace sector of the generated loop-plaquette stabilizers.
        """
        lattice = LatticeGraph.square(3, 2, dfs_ordering=True)
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
