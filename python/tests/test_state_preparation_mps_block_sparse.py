"""Tests for matrix product state preparation with block-sparse unitary synthesis.

Tests both the classical preprocessing (decomposition correctness) and
the full Q# circuit (state preparation fidelity via statevector simulation).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation.matrix_product_state import (
    MatrixProductStatePreparationData,
)
from qdk_chemistry.data import Circuit, Configuration, MPSContainer, MPSSite, Orbitals, Wavefunction
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context
from qdk_chemistry.utils.unitary_synthesis import block_sparse_unitary_synthesis, matrix_product_state_synthesis

from .mps_test_helpers import (
    JORDAN_WIGNER_CONVENTION_CASES,
    REFERENCE_MPS_EXPECTED_STATE,
    REFERENCE_MPS_TENSORS,
    SPINLESS_JORDAN_WIGNER_CONVENTION_CASES,
    assert_same_givens,
    blocked_jordan_wigner_state,
    contract_mps,
    dense_site,
    dense_target,
    make_mps,
    make_site,
    particle_number_blocked_sites,
    preparation_data,
    random_mps,
    random_orthogonal,
    random_particle_number_tensors,
    reconstruct_givens,
    right_normalized_mps,
    right_normalized_tensors,
    simulate_mps_preparation,
    site_isometry,
)
from .test_helpers import create_test_basis_set, create_test_wavefunction

_OPERATION = "QDKChemistry.Utils.MPSSparse.MPSSparse"
_SITE_STRUCT = "QDKChemistry.Utils.MPSSparse.SparseSiteSynthesis"


def assert_same_preparation_data(
    actual: MatrixProductStatePreparationData, expected: MatrixProductStatePreparationData
) -> None:
    """Require identical permutations and numerically identical rotation data."""
    assert actual.num_sites == expected.num_sites
    assert actual.num_qubits_per_site == expected.num_qubits_per_site
    assert actual.ancilla_bits == expected.ancilla_bits
    np.testing.assert_allclose(actual.initial_state_vec, expected.initial_state_vec, atol=1e-14)
    assert len(actual.sites) == len(expected.sites)
    for actual_site, expected_site in zip(actual.sites, expected.sites, strict=True):
        assert actual_site.column_permutation == expected_site.column_permutation
        assert actual_site.row_permutation == expected_site.row_permutation
        assert_same_givens(actual_site.block_givens, expected_site.block_givens)


def assert_prepares_state(params: dict, num_sites: int, ancilla_bits: int, target_state: np.ndarray) -> None:
    """Simulate block-sparse MPS preparation and compare the post-selected state with the target.

    ``target_state`` is indexed like :func:`contract_mps` and is mapped to the blocked
    Jordan-Wigner qubit basis with the site-to-orbital order in ``params``.
    """
    ancilla_zero_prob, prepared = simulate_mps_preparation(_OPERATION, _SITE_STRUCT, params, num_sites, ancilla_bits)
    assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"
    target = blocked_jordan_wigner_state(target_state, params["siteToOrbitalOrder"], params["numQubitsPerSite"])
    fidelity = np.abs(np.vdot(target, prepared)) ** 2
    assert fidelity > 0.90, f"Fidelity {fidelity:.4f} too low for num_sites={num_sites}"


class TestBlockSparseQSharpFidelity:
    """Test that the MPSSparse Q# circuit produces the correct state."""

    def test_fidelity_random_mps(self):
        """Test sparse state preparation fidelity on a random MPS."""
        mps = random_mps(num_sites=2, bond_dim=4, rng=np.random.default_rng(42))
        data = preparation_data(mps, "block_sparse")
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), 2, data.ancilla_bits, contract_mps(mps))

    def test_fidelity_reference_mps(self):
        """Test sparse preparation fidelity on a fixed four-site MPS."""
        data = preparation_data(right_normalized_mps(REFERENCE_MPS_TENSORS), "block_sparse")
        params = data.to_qsharp_params(rotation_bits=6)
        assert_prepares_state(params, 4, data.ancilla_bits, REFERENCE_MPS_EXPECTED_STATE)

    def test_fidelity_permuted_site_order(self):
        """A non-identity site_to_orbital_order must place each chain site on its mapped orbital."""
        # Scramble the chain -> orbital placement (a permutation of range(num_sites)).
        site_to_orbital_order = [2, 0, 3, 1]
        data = preparation_data(
            make_mps(right_normalized_tensors(REFERENCE_MPS_TENSORS), site_to_orbital_order=site_to_orbital_order),
            "block_sparse",
        )
        params = data.to_qsharp_params(rotation_bits=6)
        assert_prepares_state(params, 4, data.ancilla_bits, REFERENCE_MPS_EXPECTED_STATE)

    @pytest.mark.parametrize(("num_sites", "bond_dim"), [(2, 2), (4, 4)])
    def test_fidelity_random_spinless_mps(self, num_sites, bond_dim):
        """Random right-canonical spinless MPSs are prepared with one qubit per site."""
        mps = random_mps(num_sites=num_sites, bond_dim=bond_dim, site_dim=2, rng=np.random.default_rng(13))
        data = preparation_data(mps, "block_sparse")
        assert data.num_qubits_per_site == 1
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), num_sites, data.ancilla_bits, contract_mps(mps))

    def test_fidelity_permuted_spinless_site_order(self):
        """Spinless chain sites land on their mapped orbitals with the reordering signs."""
        mps = random_mps(num_sites=4, bond_dim=4, site_dim=2, rng=np.random.default_rng(21))
        data = preparation_data(make_mps(mps.sites, site_to_orbital_order=[1, 3, 0, 2]), "block_sparse")
        params = data.to_qsharp_params(rotation_bits=6)
        assert_prepares_state(params, 4, data.ancilla_bits, contract_mps(mps))

    @pytest.mark.parametrize("site_dim", [2, 4])
    def test_fidelity_particle_number_blocked_mps(self, site_dim):
        """Particle-number-blocked sites, whose dense arrays have zero blocks, are prepared exactly."""
        tensors = random_particle_number_tensors(4, 2, max_bond=3, site_dim=site_dim, rng=np.random.default_rng(4))
        mps = make_mps(particle_number_blocked_sites(tensors))
        data = preparation_data(mps, "block_sparse")
        assert_prepares_state(data.to_qsharp_params(rotation_bits=6), 4, data.ancilla_bits, contract_mps(mps))

    @pytest.mark.parametrize(
        ("tensors", "site_to_orbital_order", "expected"),
        [*JORDAN_WIGNER_CONVENTION_CASES, *SPINLESS_JORDAN_WIGNER_CONVENTION_CASES],
    )
    def test_fidelity_follows_blocked_jordan_wigner_convention(self, tensors, site_to_orbital_order, expected):
        """Sites land on blocked Jordan-Wigner qubits with the fermionic reordering signs."""
        data = preparation_data(make_mps(tensors, site_to_orbital_order=site_to_orbital_order), "block_sparse")
        params = data.to_qsharp_params(rotation_bits=6)
        ancilla_zero_prob, prepared = simulate_mps_preparation(_OPERATION, _SITE_STRUCT, params, 2, data.ancilla_bits)
        assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"
        fidelity = np.abs(np.vdot(dense_target(expected, 2 * params["numQubitsPerSite"]), prepared)) ** 2
        assert fidelity > 0.90, f"Fidelity {fidelity:.4f} too low"

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_permutation_via_qroam_with_measurement_uncompute(self, num_bits):
        """The lookup, SWAP, and measurement-based unlookup apply the permutation exactly.

        Each repetition samples new X-basis outcomes in the unlookup, so the phase fixup is
        exercised on different measurement records.
        """

        def table(values: list[int]) -> str:
            rows = (", ".join("true" if value >> bit & 1 else "false" for bit in range(num_bits)) for value in values)
            return "[" + ", ".join(f"[{row}]" for row in rows) + "]"

        context = get_qsharp_context()
        rng = np.random.default_rng(num_bits)
        for _ in range(6):
            permutation = rng.permutation(1 << num_bits).tolist()
            angles = rng.uniform(0.2, np.pi - 0.2, num_bits)
            amplitudes = np.ones(1)
            for angle in angles:
                # Little-endian: qubit k is bit k of the register value.
                amplitudes = np.kron([np.cos(angle / 2), np.sin(angle / 2)], amplitudes)
            expected = np.zeros_like(amplitudes)
            expected[permutation] = amplitudes

            # Target qubit k is prepared by Ry(angles[k]) before the permutation.
            context.eval(f"use target = Qubit[{num_bits}];")
            for qubit, angle in enumerate(angles):
                context.eval(f"Ry({float(angle):.15f}, target[{qubit}]);")
            inverse = np.argsort(permutation).tolist()
            context.eval(
                f"QDKChemistry.Utils.MPSSparse.PermutationViaQROAM({table(permutation)}, {table(inverse)}, target);"
            )
            dump = context.dump_machine()
            context.eval("ResetAll(target);")

            # DumpMachine shows target[0] as the most-significant bit; scratch qubits must be released clean.
            state = np.zeros(1 << num_bits, dtype=complex)
            for index in dump:
                assert index >> num_bits == 0 or abs(dump[index]) < 1e-10, "scratch qubits were not uncomputed"
                if index >> num_bits == 0:
                    state[int(format(index, f"0{num_bits}b")[::-1], 2)] = dump[index]
            assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)
            assert np.abs(np.vdot(expected, state)) ** 2 > 1 - 1e-10


class TestBlockSparsePreprocessing:
    """Test the classical site decompositions independently of Q# simulation."""

    def test_reference_mps_contracts_to_expected_state(self):
        """The packed MPSSite export reshapes to the reference tensors."""
        mps = right_normalized_mps(REFERENCE_MPS_TENSORS)
        np.testing.assert_allclose(contract_mps(mps), REFERENCE_MPS_EXPECTED_STATE, atol=1e-7)

    def test_array_and_site_inputs_agree(self):
        """Dense arrays take the same native MPSSite path as explicit sites."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        from_arrays = preparation_data(make_mps(tensors), "block_sparse")
        from_sites = preparation_data(right_normalized_mps(REFERENCE_MPS_TENSORS), "block_sparse")
        assert_same_preparation_data(from_arrays, from_sites)
        assert from_arrays.num_sites == 4
        assert len(from_arrays.sites) == 3
        assert len(from_arrays.initial_state_vec) == 4 * (1 << from_arrays.ancilla_bits)
        np.testing.assert_allclose(np.linalg.norm(from_arrays.initial_state_vec), 1.0)

    def test_particle_number_blocked_sites_match_dense(self):
        """Missing symmetry blocks are zeros in the decomposed target, matching dense input."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        blocked = particle_number_blocked_sites(tensors)
        assert any(len(site.left_sector_order) > 1 for site in blocked)
        for site, tensor in zip(blocked, tensors, strict=True):
            np.testing.assert_array_equal(dense_site(site), tensor)
        blocked_mps = make_mps(blocked)
        np.testing.assert_allclose(contract_mps(blocked_mps), REFERENCE_MPS_EXPECTED_STATE, atol=1e-7)
        assert_same_preparation_data(
            preparation_data(blocked_mps, "block_sparse"),
            preparation_data(make_mps(tensors), "block_sparse"),
        )

    @pytest.mark.parametrize("site_dim", [2, 4])
    def test_site_permutations_are_bijections(self, site_dim):
        """Every decomposed site permutation acts on the full site-plus-ancilla register."""
        mps = random_mps(num_sites=4, bond_dim=4, site_dim=site_dim, rng=np.random.default_rng(2))
        data = preparation_data(mps, "block_sparse")
        assert data.num_qubits_per_site == site_dim // 2
        active_dim = site_dim * (1 << data.ancilla_bits)
        assert len(data.initial_state_vec) == active_dim
        for site in data.sites:
            assert sorted(site.column_permutation) == list(range(active_dim))
            assert sorted(site.row_permutation) == list(range(active_dim))
            assert len(site.block_givens.layer_angles) == len(site.block_givens.layer_shifted)
            assert len(site.block_givens.phases) == active_dim

    @pytest.mark.parametrize("physical", [2, 4])
    def test_sparse_site_decompositions_reconstruct_sites(self, physical):
        """P_row · V · P_col maps each left-bond state to its column of the site isometry for every site."""
        tensors = random_particle_number_tensors(4, 2, max_bond=3, site_dim=physical, rng=np.random.default_rng(1))
        chi = 1 << int(np.ceil(np.log2(max(max(tensor.shape[0], tensor.shape[2]) for tensor in tensors))))
        assert tensors[1].shape[0] > 1
        assert np.count_nonzero(tensors[1]) < tensors[1].size

        results = matrix_product_state_synthesis(make_mps(tensors), chi, "block_sparse")
        tensors = tensors[1:]

        assert len(results) == len(tensors)
        for tensor, synthesis in zip(tensors, results, strict=True):
            col_perm, row_perm = synthesis.column_permutation, synthesis.row_permutation
            assert sorted(col_perm) == sorted(row_perm) == list(range(physical * chi))
            block = reconstruct_givens(synthesis.block_givens)
            inverse_row = np.argsort(row_perm)
            unitary = block[np.ix_(inverse_row, col_perm)]
            np.testing.assert_allclose(unitary.T @ unitary, np.eye(physical * chi), atol=1e-11)
            np.testing.assert_allclose(unitary[:, : tensor.shape[0]], site_isometry(tensor, chi), atol=1e-10)

    def test_sparse_site_decomposition_rejects_invalid_sites(self):
        """Native validation errors surface as ValueError for blocked tensors."""
        tensor = random_orthogonal(8, np.random.default_rng(5))[:, :2].reshape(4, 2, 2).transpose(2, 0, 1)
        site = make_site(tensor)
        with pytest.raises(ValueError, match="bond"):
            block_sparse_unitary_synthesis(site, 1)
        complex_site = make_site(tensor.astype(complex))
        with pytest.raises(ValueError, match="real"):
            block_sparse_unitary_synthesis(complex_site, 2)
        scaled = make_site(2.0 * tensor)
        with pytest.raises(ValueError, match="isometric"):
            block_sparse_unitary_synthesis(scaled, 2)
        with pytest.raises(ValueError, match="sector order"):
            MPSSite(site.tensor, [], site.physical_sector_order, site.right_sector_order)


class TestBlockSparseValidation:
    """Test that unsupported MPS inputs are rejected before circuit construction."""

    def test_requires_mps_container(self):
        """Non-MPS wavefunctions are rejected."""
        with pytest.raises(TypeError, match="requires an MPSContainer"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(
                create_test_wavefunction(2)
            )

    @pytest.mark.parametrize("orthogonality_center", [None, 1])
    def test_requires_center_zero(self, orthogonality_center):
        """An unknown or nonzero orthogonality center is not accepted as right-canonical."""
        mps = make_mps(right_normalized_tensors(REFERENCE_MPS_TENSORS), orthogonality_center=orthogonality_center)
        assert mps.orthogonality_center == orthogonality_center
        with pytest.raises(ValueError, match="right-canonical MPS with center zero"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(Wavefunction(mps))

    def test_requires_canonical_physical_basis_on_every_site(self):
        """A permuted local basis on any site is rejected."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        swapped_basis = [Configuration.from_spin_half_string(state) for state in ("0", "d", "u", "2")]
        sites = [make_site(tensor) for tensor in tensors]
        sites[2] = make_site(tensors[2], swapped_basis)
        assert sites[2].physical_basis == swapped_basis
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', 'u', 'd', '2'\)"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(
                Wavefunction(make_mps(sites))
            )
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', 'u', 'd', '2'\)"):
            preparation_data(make_mps(sites), "block_sparse")

    def test_requires_canonical_spinless_basis(self):
        """A permuted binary local basis is rejected."""
        sites = random_mps(num_sites=2, bond_dim=2, site_dim=2, rng=np.random.default_rng(3)).sites
        swapped_basis = [Configuration.from_bitstring(state) for state in ("1", "0")]
        sites = [sites[0], make_site(dense_site(sites[1]), swapped_basis)]
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', '1'\)"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(
                Wavefunction(make_mps(sites))
            )
        with pytest.raises(ValueError, match=r"physical basis ordering \('0', '1'\)"):
            preparation_data(make_mps(sites), "block_sparse")

    def test_requires_two_or_four_physical_states_per_site(self):
        """Sites with any other local dimension are rejected."""
        three_states = [Configuration.from_spin_half_string(state) for state in ("0", "u", "d")]
        site = make_site(np.array([[[1.0], [0.0], [0.0]]]), three_states)
        with pytest.raises(ValueError, match="two or four physical states per site"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(
                Wavefunction(make_mps([site]))
            )
        with pytest.raises(ValueError, match="two or four physical states per site"):
            preparation_data(make_mps([site]), "block_sparse")

    def test_requires_uniform_physical_dimension(self):
        """Spinless and spatial-orbital sites cannot be mixed in one chain."""
        sites = [make_site(np.array([[[1.0, 0.0], [0.0, 0.0]]])), make_site(np.ones((2, 4, 1)) / np.sqrt(8))]
        with pytest.raises(ValueError, match="same physical dimension on every site"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(
                Wavefunction(make_mps(sites))
            )
        with pytest.raises(ValueError, match="same physical dimension on every site"):
            preparation_data(make_mps(sites), "block_sparse")

    def test_requires_real_tensors(self):
        """Complex MPS tensors are rejected."""
        tensors = [tensor.astype(complex) for tensor in right_normalized_tensors(REFERENCE_MPS_TENSORS)]
        with pytest.raises(ValueError, match="only real-valued"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(
                Wavefunction(make_mps(tensors))
            )
        with pytest.raises(ValueError, match="only real-valued"):
            preparation_data(make_mps(tensors), "block_sparse")

    def test_requires_one_site_per_molecular_orbital(self):
        """An active-space MPS that omits molecular orbitals is rejected."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS[2:])
        tensors[0] = tensors[0][:1] / np.linalg.norm(tensors[0][:1])
        orbitals = Orbitals(
            np.eye(3),
            None,
            None,
            create_test_basis_set(3),
            sym.spin_index_set(3, [0, 1], [0, 1]),
            sym.spin_index_set(3, [], []),
        )
        mps = MPSContainer([make_site(tensor) for tensor in tensors], orbitals, orthogonality_center=0)
        with pytest.raises(ValueError, match="exactly one MPS site per molecular orbital"):
            create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(Wavefunction(mps))

    def test_rejects_zero_initial_state(self):
        """A zero initial site cannot be normalized as a quantum state."""
        with pytest.raises(ValueError, match="finite amplitudes with nonzero norm"):
            preparation_data(make_mps([np.zeros((1, 4, 1))]), "block_sparse")

    def test_site_to_orbital_order_comes_from_container(self):
        """Preparation preserves the container's site order."""
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        data = preparation_data(make_mps(tensors), "block_sparse")
        assert data.site_to_orbital_order == [0, 1, 2, 3]
        container = make_mps(tensors, site_to_orbital_order=[2, 0, 3, 1])
        params = preparation_data(container, "block_sparse").to_qsharp_params(6)
        assert params["siteToOrbitalOrder"] == [2, 0, 3, 1]


class TestBlockSparseStatePreparationRun:
    """Test the registered algorithm end to end on native MPS containers."""

    def test_run_builds_qsharp_factory_from_container(self):
        """run() forwards the site decomposition, settings, and site order to Q#."""
        site_to_orbital_order = [2, 0, 3, 1]
        tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS)
        wavefunction = Wavefunction(make_mps(tensors, site_to_orbital_order=site_to_orbital_order))
        prep = create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse", rotation_bits=8)
        circuit = prep.run(wavefunction)

        assert isinstance(circuit, Circuit)
        assert circuit.encoding == "jordan-wigner"
        assert circuit._qsharp_op is not None
        factory = circuit._qsharp_factory
        assert factory is not None
        assert factory.program is QSHARP_UTILS.MPSSparse.MakeMPSSparseCircuit
        expected = preparation_data(wavefunction.get_container(), "block_sparse").to_qsharp_params(8)
        assert factory.parameter["siteToOrbitalOrder"] == site_to_orbital_order
        assert circuit.num_qubits == expected["numQubitsPerSite"] * expected["numSites"]
        assert factory.parameter.keys() == expected.keys()
        for key in ("numSites", "numQubitsPerSite", "siteToOrbitalOrder", "rotationBits", "numAncillaQubits"):
            assert factory.parameter[key] == expected[key]
        np.testing.assert_allclose(factory.parameter["initialStateVec"], expected["initialStateVec"])

    def test_run_defaults_to_identity_site_order(self):
        """An omitted site order maps chain site k to orbital k."""
        wavefunction = Wavefunction(right_normalized_mps(REFERENCE_MPS_TENSORS))
        circuit = create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse").run(wavefunction)
        assert circuit._qsharp_factory.parameter["siteToOrbitalOrder"] == [0, 1, 2, 3]
        assert circuit._qsharp_factory.parameter["rotationBits"] == 10
        assert circuit.metadata.num_phase_gradient_ancillas == 0

    def test_shared_gradient_widens_the_register_it_declares(self):
        """Opting out of internal allocation appends a gradient the caller must own."""
        prep = create(
            "state_prep",
            "matrix_product_state",
            unitary_synthesis="block_sparse",
            rotation_bits=6,
            allocate_phase_gradient=False,
        )
        circuit = prep.run(Wavefunction(right_normalized_mps(REFERENCE_MPS_TENSORS)))
        assert circuit.num_qubits == 8 + 6
        assert circuit.metadata.num_phase_gradient_ancillas == 6

    def test_resource_estimate(self):
        """The generated Adaptive circuit compiles for logical resource estimation."""
        num_sites = 2
        mps = random_mps(num_sites=num_sites, bond_dim=4, rng=np.random.default_rng(42))
        circuit = create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse", rotation_bits=6).run(
            Wavefunction(mps)
        )
        logical_counts = circuit.estimate()["logicalCounts"]
        assert logical_counts["numQubits"] >= 2 * num_sites
        assert logical_counts["cczCount"] + logical_counts["tCount"] + logical_counts["rotationCount"] > 0

    def test_spinless_resource_estimate_uses_one_qubit_per_site(self):
        """A spinless MPS needs fewer qubits and Toffolis than a spatial-orbital MPS of equal bond dimension."""
        spinless = random_mps(num_sites=4, bond_dim=4, site_dim=2, rng=np.random.default_rng(42))
        spatial = random_mps(num_sites=4, bond_dim=4, rng=np.random.default_rng(42))
        prep = create("state_prep", "matrix_product_state", unitary_synthesis="block_sparse", rotation_bits=6)
        spinless_circuit = prep.run(Wavefunction(spinless))
        assert spinless_circuit._qsharp_factory.parameter["numQubitsPerSite"] == 1
        spinless_counts = spinless_circuit.estimate()["logicalCounts"]
        spatial_counts = prep.run(Wavefunction(spatial)).estimate()["logicalCounts"]
        assert spinless_counts["numQubits"] < spatial_counts["numQubits"]
        assert 0 < spinless_counts["cczCount"] < spatial_counts["cczCount"]
