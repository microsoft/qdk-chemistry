"""Tests for Verstraete-Cirac fermion-to-qubit mapping."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile

import h5py
import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_huckel_hamiltonian

_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}


class TestVerstraeteCiracMapping:
    """Tests covering the Verstraete-Cirac mapping factory, dimensions, and properties."""

    def test_factory_and_dimensions(self) -> None:
        """Test that VC factory accepts valid lattices (>= 3 sites) and rejects lattices with < 3 sites."""
        # Testing valid lattices with >= 3 sites
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
            """Return True if Pauli strings p1 and p2 commute."""
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

    @pytest.mark.parametrize(
        ("lattice_type", "kwargs", "max_allowed_weight"),
        [
            ("square", {}, 8),
            ("triangular", {}, 8),
            ("honeycomb", {}, 12),
            ("kagome", {}, 12),
        ],
        ids=[
            "square",
            "triangular",
            "honeycomb",
            "kagome",
        ],
    )
    def test_pauli_weight_scaling(self, lattice_type: str, kwargs: dict, max_allowed_weight: int) -> None:
        """Check if Pauli weights stay below a certain threshold.

        Max Pauli weight of nearest-neighbor hops should be constant for grids of dimensions
        L = 2, 3, 4.

        We check across multiple lattice topologies (square, triangular, honeycomb, kagome):
        - Mapped hopping terms retain constant-weight scaling (e.g. max weight 4 or 5)
        - Mapped stabilizer products S_i * S_{i+1} have weights that are bounded by the
          analytical Jordan-Wigner overlap formula: w <= delta_start + delta_end + 2,
          where delta_start and delta_end are the qubit offsets between the stabilizer endpoints
        - Physical loop-plaquettes are bounded by the topology's respective maximum weight limit
        """
        mapper = create("qubit_mapper", "qdk")

        def multiply_pauli_labels(p1: str, p2: str) -> str:
            """Return the Pauli string resulting from the product of p1 and p2 (ignoring phase)."""
            table = {
                ("I", "I"): "I",
                ("I", "X"): "X",
                ("I", "Y"): "Y",
                ("I", "Z"): "Z",
                ("X", "I"): "X",
                ("X", "X"): "I",
                ("X", "Y"): "Z",
                ("X", "Z"): "Y",
                ("Y", "I"): "Y",
                ("Y", "X"): "Z",
                ("Y", "Y"): "I",
                ("Y", "Z"): "X",
                ("Z", "I"): "Z",
                ("Z", "X"): "Y",
                ("Z", "Y"): "X",
                ("Z", "Z"): "I",
            }
            return "".join(table[(c1, c2)] for c1, c2 in zip(p1, p2, strict=False))

        def get_expected_weight(p1: str, p2: str) -> int:
            """Return the analytical upper bound on the weight of the product of p1 and p2."""
            indices1 = [idx for idx, char in enumerate(p1) if char != "I"]
            indices2 = [idx for idx, char in enumerate(p2) if char != "I"]
            if not indices1 or not indices2:
                return 0
            a1, b1 = indices1[0], indices1[-1]
            a2, b2 = indices2[0], indices2[-1]
            if a1 > a2:
                a1, b1, a2, b2 = a2, b2, a1, b1
            if a2 <= b1:
                union_size = max(b1, b2) - a1 + 1
                overlap_int = max(0, min(b1, b2) - a2 - 1)
                return union_size - overlap_int
            return (b1 - a1 + 1) + (b2 - a2 + 1)

        def check_local_plaquette(
            stabs_list: list[str],
            u_idx: int,
            v_idx: int,
            w: int,
            p_str: str,
            thresh: int,
            max_weight: int,
        ) -> None:
            """Assert that the product weight of a local stabilizer pair does not exceed max_weight."""
            indices1 = [idx for idx, char in enumerate(stabs_list[u_idx]) if char != "I"]
            indices2 = [idx for idx, char in enumerate(stabs_list[v_idx]) if char != "I"]
            if indices1 and indices2:
                a1, b1 = indices1[0], indices1[-1]
                a2, b2 = indices2[0], indices2[-1]
                if abs(a1 - a2) <= thresh and abs(b1 - b2) <= thresh:
                    assert w <= max_weight, (
                        f"Local plaquette stabilizer {p_str} has weight {w} > {max_weight}, violating local scaling"
                    )

        def get_mst_edges(stabs_list: list[str], start_idx: int, count: int, root: int) -> list[tuple[int, int]]:
            """Return MST edges over a subset of stabilizers using Prim's algorithm."""
            if count <= 1:
                return []

            in_mst = [False] * count
            min_weight = [float("inf")] * count
            parent = [root] * count

            min_weight[root] = 0

            for _ in range(count):
                u = -1
                min_val = float("inf")
                for i in range(count):
                    if not in_mst[i] and min_weight[i] < min_val:
                        min_val = min_weight[i]
                        u = i

                in_mst[u] = True

                for v in range(count):
                    if not in_mst[v]:
                        prod_str = multiply_pauli_labels(stabs_list[start_idx + u], stabs_list[start_idx + v])
                        w = sum(1 for c in prod_str if c != "I")
                        if w < min_weight[v]:
                            min_weight[v] = w
                            parent[v] = u

            edges = []
            for v in range(count):
                if v != root:
                    edges.append((start_idx + parent[v], start_idx + v))
            return edges

        max_weights = []
        for grid_length in [2, 3, 4]:
            lattice = getattr(LatticeGraph, lattice_type)(grid_length, grid_length, dfs_ordering=True, **kwargs)
            # Create a simple hopping-only Hamiltonian (U=0) on the L x L lattice
            ham = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=0.0)

            vc_mapping = MajoranaMapping.verstraete_cirac(lattice)
            qh_vc = mapper.run(ham, vc_mapping)

            # Identify and construct the set of stabilizer penalty terms using the product of stabilizers logic
            stabs = [sparse_pauli_word_to_label(word, qh_vc.num_qubits) for _, word in vc_mapping.stabilizers]
            half_stabs = len(stabs) // 2

            penalty_strings = set()
            if half_stabs >= 1:
                threshold = (max_allowed_weight - 2) // 2

                # Find the lightest stabilizer as root for alpha
                root_alpha = 0
                root_alpha_weight = sum(1 for c in stabs[0] if c != "I")
                for i in range(1, half_stabs):
                    w = sum(1 for c in stabs[i] if c != "I")
                    if w < root_alpha_weight:
                        root_alpha = i
                        root_alpha_weight = w

                penalty_strings.add(stabs[root_alpha])
                alpha_edges = get_mst_edges(stabs, 0, half_stabs, root_alpha)
                for u, v in alpha_edges:
                    prod = multiply_pauli_labels(stabs[u], stabs[v])
                    penalty_strings.add(prod)
                    weight = sum(1 for c in prod if c != "I")
                    expected_weight = get_expected_weight(stabs[u], stabs[v])
                    assert weight <= expected_weight, (
                        f"Weight {weight} exceeds analytical expected bound {expected_weight} "
                        f"for product of stabilizer {u} and {v}"
                    )
                    check_local_plaquette(stabs, u, v, weight, prod, threshold, max_allowed_weight)

                # Find the lightest stabilizer as root for beta
                root_beta = 0
                root_beta_weight = sum(1 for c in stabs[half_stabs] if c != "I")
                for i in range(1, half_stabs):
                    w = sum(1 for c in stabs[half_stabs + i] if c != "I")
                    if w < root_beta_weight:
                        root_beta = i
                        root_beta_weight = w

                penalty_strings.add(stabs[half_stabs + root_beta])
                beta_edges = get_mst_edges(stabs, half_stabs, half_stabs, root_beta)
                for u, v in beta_edges:
                    prod = multiply_pauli_labels(stabs[u], stabs[v])
                    penalty_strings.add(prod)
                    weight = sum(1 for c in prod if c != "I")
                    expected_weight = get_expected_weight(stabs[u], stabs[v])
                    assert weight <= expected_weight, (
                        f"Weight {weight} exceeds analytical expected bound {expected_weight} "
                        f"for product of stabilizer {u} and {v}"
                    )
                    check_local_plaquette(stabs, u, v, weight, prod, threshold, max_allowed_weight)
            else:
                penalty_strings.update(stabs)

            weights = []
            for pauli_str in qh_vc.pauli_strings:
                if pauli_str == "I" * len(pauli_str):
                    continue
                # Skip local plaquette penalty terms
                if pauli_str in penalty_strings:
                    continue
                # Weight is the number of non-I characters
                weight = sum(1 for c in pauli_str if c != "I")
                weights.append(weight)

            max_weights.append(max(weights))

        # Verify that physical terms have constant-weight scaling.
        # For the triangular lattice, the 2x2 grid is too small to reach the bulk max hopping weight
        # of 5, so we only assert that the weights are equal/constant for the larger sizes L >= 3.
        if lattice_type == "triangular":
            assert max_weights[1] == max_weights[2], (
                f"Maximum Pauli weights grew with grid length for {lattice_type}: {max_weights}"
            )
        else:
            assert all(w == max_weights[0] for w in max_weights), (
                f"Maximum Pauli weights grew or varied with grid length for {lattice_type}: {max_weights}"
            )


class TestVerstraeteCiracSpectral:
    """Tests covering the spectral validation of the Verstraete-Cirac mapping."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _RUN_SLOW_TESTS,
        reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
    )
    def test_spectral_validation_2x2_hubbard(self) -> None:
        """Compare eigenvalues of 2x2 periodic Fermi-Hubbard model under VC and JW mappings.

        Model parameters: t = 1.0, U = 4.0, half-filling (epsilon = -U/2 = -2.0).
        """
        lattice = LatticeGraph.square(2, 2, periodic_x=True, dfs_ordering=True)
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

    @pytest.mark.slow
    @pytest.mark.skipif(
        not _RUN_SLOW_TESTS,
        reason="Skipping slow test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
    )
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
