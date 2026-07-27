"""Tests for MPS sparse state preparation algorithm.

Tests both the classical preprocessing (decomposition correctness) and
the full Q# circuit (state preparation fidelity via statevector simulation).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np

from qdk_chemistry.algorithms.state_preparation.mps_sparse import generate_mps_sparse_preparation_data
from qdk_chemistry.data import AbelianMPSContainer, AbelianMPSSite
from qdk_chemistry.data.symmetry import SymmetryBlockedScalarCount, SymmetryLabel, SymmetryProduct
from qdk_chemistry.utils.qsharp import get_qsharp_context

from .test_helpers import create_test_orbitals


def _particle_count(count: int) -> SymmetryBlockedScalarCount:
    symmetry = SymmetryProduct([])
    label = SymmetryLabel([])
    return SymmetryBlockedScalarCount([symmetry], [((label,), count)])


def make_mps(
    tensors: Sequence[np.ndarray],
    orthogonality_center: int | None = 0,
) -> AbelianMPSContainer:
    """Construct a native MPS wavefunction from dense test tensors."""
    num_sites = len(tensors)
    for index in range(num_sites - 1):
        if tensors[index].shape[2] != tensors[index + 1].shape[0]:
            raise ValueError("Adjacent MPS sites have incompatible bond spaces.")

    sites = [
        AbelianMPSSite.from_dense_abelian(
            tensor,
            {0: tensor.shape[0]},
            {0: tensor.shape[2]},
            [0] * tensor.shape[1],
            0,
        )
        for tensor in tensors
    ]
    return AbelianMPSContainer(
        sites,
        create_test_orbitals(max(1, num_sites)),
        _particle_count(0),
        _particle_count(0),
        orthogonality_center=orthogonality_center,
    )


def right_normalized_mps(tensors: Sequence[np.ndarray]) -> AbelianMPSContainer:
    """Construct a right-canonical MPS preserving the normalized state."""
    normalized = [np.array(tensor, copy=True) for tensor in tensors]
    for site in range(len(normalized) - 1, 0, -1):
        chi_left, physical, chi_right = normalized[site].shape
        matrix = normalized[site].reshape(chi_left, physical * chi_right)
        q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
        normalized[site] = q_matrix.T.reshape(chi_left, physical, chi_right)
        previous_left, previous_physical, _ = normalized[site - 1].shape
        previous = normalized[site - 1].reshape(previous_left * previous_physical, chi_left)
        normalized[site - 1] = (previous @ r_matrix.T).reshape(previous_left, previous_physical, chi_left)
    normalized[0] /= np.linalg.norm(normalized[0])
    return make_mps(normalized)


def random_mps(
    num_sites: int,
    bond_dim: int,
    site_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> AbelianMPSContainer:
    """Construct a right-normalized random native MPS for algorithm tests."""
    rng = np.random.default_rng() if rng is None else rng
    bond_dims = [1]
    for site in range(1, num_sites):
        max_left = bond_dims[-1] * site_dim
        max_right = site_dim ** min(site, num_sites - site)
        bond_dims.append(min(bond_dim, max_left, max_right))
    bond_dims.append(1)

    tensors = [rng.standard_normal((bond_dims[site], site_dim, bond_dims[site + 1])) for site in range(num_sites)]
    return right_normalized_mps(tensors)


def contract_mps(wavefunction: AbelianMPSContainer) -> np.ndarray:
    """Contract a native MPS into a normalized dense state vector."""
    state = wavefunction.sites[0].to_dense()
    for site in wavefunction.sites[1:]:
        tensor = site.to_dense()
        left, num_states, previous_bond = state.shape
        incoming_bond, physical, outgoing_bond = tensor.shape
        state = (state.reshape(left * num_states, previous_bond) @ tensor.reshape(incoming_bond, -1)).reshape(
            left, num_states * physical, outgoing_bond
        )
    vector = state.sum(axis=0).flatten()
    norm = np.linalg.norm(vector)
    return vector if norm <= 1e-15 else vector / norm


# Fixed MPS data used across preprocessing, fidelity, and resource tests.
_reference_mps_tensors = (
    np.array(
        [
            [
                [0.01650572, 0.0, 0.0, 0.0],
                [0.0, -0.52929781, 0.0, 0.0],
                [0.0, 0.0, -0.84462254, 0.0],
                [0.0, 0.0, 0.0, -0.07863941],
            ]
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.05969264, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9973967, 0.04045497, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.08381532, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.98376348, 0.15869598, 0.0],
            ],
            [
                [-0.0421477, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.46961402, 0.0265522, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.41109095, 0.03268939, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.77904869],
            ],
        ]
    ),
    np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.19640516, 0.0, 0.0, 0.0], [0.0, -0.98052283, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.98052283, 0.0, 0.0, 0.0], [0.0, 0.19640516, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [-0.02411236, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.99970925, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [-0.99970925, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.02411236, 0.0]],
            [
                [-0.17695837, 0.0, 0.0, 0.0],
                [0.0, -0.58052668, 0.0, 0.0],
                [0.0, 0.0, -0.53176612, 0.0],
                [0.0, 0.0, 0.0, -0.59067698],
            ],
        ]
    ),
    np.array(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[1.0], [0.0], [0.0], [0.0]],
        ]
    ),
)

_reference_mps_expected_state = np.array(
    [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.01650572, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.03159519, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.12468186, 0.        , 0.        ,
     0.51343194, 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.07079231, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.15403441, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.82743524, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00331447, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00930066, 0.        , 0.        ,
     0.03580077, 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00334943, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.03225657, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.01084116, 0.        , 0.        ,
     0.03556534, 0.        , 0.        , 0.03257808, 0.        , 0.        ,
     0.03618719, 0.        , 0.        , 0.        ])  # fmt: skip


class TestMPSSparseQSharpFidelity:
    """Test that the MPSSparse Q# circuit produces the correct state."""

    def test_fidelity_random_mps(self):
        """Test sparse state preparation fidelity on a random MPS."""
        num_sites = 2
        bond_dim = 4
        rng = np.random.default_rng(42)
        mps = random_mps(num_sites=num_sites, bond_dim=bond_dim, rng=rng)
        target_state = contract_mps(mps)

        data = generate_mps_sparse_preparation_data(mps.sites)
        num_state_qubits = 2 * num_sites
        dump = _dump_prepared_state(data.to_qsharp_params(rotation_bits=6), num_state_qubits, data.ancilla_bits)
        state_amplitudes = _extract_state_amplitudes_sparse(dump, num_state_qubits, data.ancilla_bits)

        # P(ancilla = |0>) should be high
        ancilla_zero_prob = np.sum(np.abs(state_amplitudes) ** 2)
        assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"

        # Normalize and reindex
        state_amplitudes = state_amplitudes / np.sqrt(ancilla_zero_prob)
        state_amplitudes = _reindex_sites(state_amplitudes, num_sites)

        fidelity = np.abs(np.vdot(target_state, state_amplitudes[: len(target_state)])) ** 2
        assert fidelity > 0.90, f"Fidelity {fidelity:.4f} too low for num_sites={num_sites}, bond_dim={bond_dim}"

    def test_fidelity_reference_mps(self):
        """Test sparse preparation fidelity on a fixed four-site MPS."""
        mps = right_normalized_mps(_reference_mps_tensors)
        target_state = _reference_mps_expected_state

        data = generate_mps_sparse_preparation_data(mps.sites)
        num_sites = 4
        num_state_qubits = 2 * num_sites
        dump = _dump_prepared_state(data.to_qsharp_params(rotation_bits=6), num_state_qubits, data.ancilla_bits)
        state_amplitudes = _extract_state_amplitudes_sparse(dump, num_state_qubits, data.ancilla_bits)

        ancilla_zero_prob = np.sum(np.abs(state_amplitudes) ** 2)
        assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"

        state_amplitudes = state_amplitudes / np.sqrt(ancilla_zero_prob)
        state_amplitudes = _reindex_sites(state_amplitudes, num_sites)

        fidelity = np.abs(np.vdot(target_state, state_amplitudes[: len(target_state)])) ** 2
        assert fidelity > 0.90, f"Fidelity {fidelity:.4f} too low"

    def test_fidelity_permuted_site_order(self):
        """A non-identity site_to_orbital_order must place each chain site on its mapped orbital."""
        mps = right_normalized_mps(_reference_mps_tensors)
        num_sites = 4
        # Scramble the chain -> orbital placement (a permutation of range(num_sites)).
        site_to_orbital_order = [2, 0, 3, 1]
        target_state = _permute_sites(_reference_mps_expected_state, site_to_orbital_order)

        data = generate_mps_sparse_preparation_data(mps.sites)
        num_state_qubits = 2 * num_sites
        params = data.to_qsharp_params(rotation_bits=6, site_to_orbital_order=site_to_orbital_order)
        dump = _dump_prepared_state(params, num_state_qubits, data.ancilla_bits)
        state_amplitudes = _extract_state_amplitudes_sparse(dump, num_state_qubits, data.ancilla_bits)

        ancilla_zero_prob = np.sum(np.abs(state_amplitudes) ** 2)
        assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"

        state_amplitudes = state_amplitudes / np.sqrt(ancilla_zero_prob)
        state_amplitudes = _reindex_sites(state_amplitudes, num_sites)

        fidelity = np.abs(np.vdot(target_state, state_amplitudes[: len(target_state)])) ** 2
        assert fidelity > 0.90, f"Fidelity {fidelity:.4f} too low"


# =============================================================================
# Helper functions
# =============================================================================


def _extract_state_amplitudes_sparse(
    dump,
    num_state_qubits: int,
    num_ancilla_qubits: int,
) -> np.ndarray:
    """Extract amplitudes where ancilla = |0> from a sparse DumpMachine result.

    Works with large qubit counts where as_dense_state() would be infeasible.
    Only considers basis states where all internal qubits (beyond state+ancilla)
    are |0>, i.e., properly uncomputed.

    Parameters
    ----------
    dump : StateDump
        Sparse dump from qsharp.dump_machine().
    num_state_qubits : int
        Number of state qubits (lowest-addressed qubits).
    num_ancilla_qubits : int
        Number of ancilla qubits (next after state).

    Returns
    -------
    np.ndarray
        Amplitudes for the state register conditioned on ancilla = |0>.

    """
    num_relevant_qubits = num_state_qubits + num_ancilla_qubits
    ancilla_mask = (1 << num_ancilla_qubits) - 1
    state_dim = 2**num_state_qubits
    state_amplitudes = np.zeros(state_dim, dtype=complex)

    for idx in dump:
        # Only consider states where internal qubits (above state+ancilla) are 0
        if idx >> num_relevant_qubits != 0:
            continue
        # Only consider states where ancilla = |0>
        if (idx & ancilla_mask) == 0:
            state_idx = idx >> num_ancilla_qubits
            if state_idx < state_dim:
                state_amplitudes[state_idx] = dump[idx]

    return state_amplitudes


def _reindex_sites(state_amplitudes: np.ndarray, num_sites: int) -> np.ndarray:
    """Reindex from Q# big-endian to Python MPS convention.

    The Q# circuit uses little-endian within each 2-qubit site, so
    DumpMachine's big-endian bits need to be reversed within each site.
    """
    site_bits = 2
    state_dim = len(state_amplitudes)
    reordered = np.zeros_like(state_amplitudes)
    for dm_idx in range(state_dim):
        py_idx = 0
        for site in range(num_sites):
            shift = (num_sites - 1 - site) * site_bits
            site_val = (dm_idx >> shift) & ((1 << site_bits) - 1)
            # Reverse bits within this site
            rev_val = 0
            for b in range(site_bits):
                if site_val & (1 << b):
                    rev_val |= 1 << (site_bits - 1 - b)
            py_idx |= rev_val << shift
        reordered[py_idx] = state_amplitudes[dm_idx]
    return reordered


def _permute_sites(state: np.ndarray, site_to_orbital_order: Sequence[int]) -> np.ndarray:
    """Relocate each chain site's amplitudes to its mapped orbital position.

    ``state`` is indexed with chain site 0 as the most-significant 2-bit group.
    Orbital ``o`` holds chain site ``k`` where ``site_to_orbital_order[k] == o``.
    The 2-bit physical value of each site is moved atomically, so the result is
    independent of the within-site bit convention.
    """
    num_sites = len(site_to_orbital_order)
    permuted = np.zeros_like(state)
    for idx in range(len(state)):
        chain_vals = [(idx >> (2 * (num_sites - 1 - k))) & 0b11 for k in range(num_sites)]
        orbital_vals = [0] * num_sites
        for chain_site, orbital in enumerate(site_to_orbital_order):
            orbital_vals[orbital] = chain_vals[chain_site]
        new_idx = 0
        for orbital in range(num_sites):
            new_idx |= orbital_vals[orbital] << (2 * (num_sites - 1 - orbital))
        permuted[new_idx] = state[idx]
    return permuted


def _dump_prepared_state(params: dict, num_state_qubits: int, num_ancilla_qubits: int):
    """Run sparse MPS preparation in an isolated Q# context and dump its state."""
    parameter_names = (
        "initialStateVec",
        "numSites",
        "siteToOrbitalOrder",
        "rotationBits",
        "siteDecompositions",
    )
    arguments = ", ".join(_to_qsharp_literal(params[name]) for name in parameter_names)
    context = get_qsharp_context()
    context.eval(f"use state = Qubit[{num_state_qubits}];")
    context.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
    context.eval(f"QDKChemistry.Utils.MPSSparse.MPSSparse({arguments}, state, ancilla)")
    dump = context.dump_machine()
    context.eval("ResetAll(state + ancilla);")
    return dump


def _to_qsharp_literal(value) -> str:
    """Serialize nested numeric and Boolean data as a Q# literal."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.15f}"
    if isinstance(value, dict):
        fields = ", ".join(f"{name} = {_to_qsharp_literal(item)}" for name, item in value.items())
        return f"new QDKChemistry.Utils.MPSSparse.SparseUnitaryDecomposition {{ {fields} }}"
    if isinstance(value, list):
        return f"[{', '.join(_to_qsharp_literal(item) for item in value)}]"
    return str(value)
