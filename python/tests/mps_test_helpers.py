"""Shared fixtures and helpers for the MPS state preparation tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np
import pytest

from qdk_chemistry.algorithms.state_preparation.matrix_product_state import (
    MatrixProductStatePreparation,
    MatrixProductStatePreparationData,
)
from qdk_chemistry.data import Configuration, MPSContainer, MPSSite
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import get_qsharp_context

from .test_helpers import create_test_orbitals

_PHYSICAL_PARTICLE_NUMBERS = {
    2: (0, 1),  # Particle numbers of the ('0', '1') physical basis states.
    4: (0, 1, 1, 2),  # Particle numbers of the ('0', 'u', 'd', '2') physical basis states.
}


def make_site(values: np.ndarray, physical_basis: Sequence[Configuration] | None = None) -> MPSSite:
    """Wrap a dense ``(left, physical, right)`` array as one trivial-symmetry MPS block."""
    left, physical, right = values.shape
    product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
    tensor_type = sym.SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else sym.SymmetryBlockedTensorRank3
    tensor = tensor_type(
        [product] * 3,
        [{label: left}, {label: physical}, {label: right}],
        [((label,) * 3, values.reshape(left * physical, right))],
    )
    return MPSSite(tensor, [label], [label], [label], list(physical_basis or []))


def particle_number_blocked_sites(tensors: Sequence[np.ndarray], tolerance: float = 1e-12) -> list[MPSSite]:
    """Store dense sites as particle-number-blocked MPS sites.

    Bond particle numbers are propagated from left to right using every entry above
    ``tolerance``. Bond indices must already be grouped by particle number. Each block
    copies its dense slice exactly, and exactly-zero blocks are omitted. Sites use the
    ``('0', '1')`` or ``('0', 'u', 'd', '2')`` physical basis by their physical dimension.
    """
    physical_particle_numbers = _PHYSICAL_PARTICLE_NUMBERS[tensors[0].shape[1]]
    max_particles = max(physical_particle_numbers) * len(tensors)
    product = sym.SymmetryProduct([sym.axes.particle_number(max_particles)])

    def label(count: int) -> sym.SymmetryLabel:
        return sym.SymmetryLabel([sym.axes.particle_number_value(count)])

    def sectors(numbers: Sequence[int]) -> list[tuple[int, int, int]]:
        """Return ``(particle_number, start, stop)`` for contiguous bond sectors."""
        result: list[tuple[int, int, int]] = []
        for index, count in enumerate(numbers):
            if result and result[-1][0] == count:
                result[-1] = (count, result[-1][1], index + 1)
            else:
                if any(existing == count for existing, _, _ in result):
                    raise ValueError("Bond indices must be grouped by particle number.")
                result.append((count, index, index + 1))
        return result

    physical_sectors = sectors(physical_particle_numbers)
    left_numbers = [0]
    sites = []
    for tensor in tensors:
        right_numbers: list[int | None] = [None] * tensor.shape[2]
        for left, physical, right in zip(*np.nonzero(np.abs(tensor) > tolerance), strict=True):
            count = left_numbers[left] + physical_particle_numbers[physical]
            if right_numbers[right] not in (None, count):
                raise ValueError("Dense tensor does not conserve particle number.")
            right_numbers[right] = count
        resolved_right_numbers = [count for count in right_numbers if count is not None]
        if len(resolved_right_numbers) != len(right_numbers):
            raise ValueError("Every right-bond index needs a nonzero entry.")
        left_sectors, right_sectors = sectors(left_numbers), sectors(resolved_right_numbers)
        blocks = []
        for left_count, left_start, left_stop in left_sectors:
            for physical_count, physical_start, physical_stop in physical_sectors:
                for right_count, right_start, right_stop in right_sectors:
                    block = tensor[left_start:left_stop, physical_start:physical_stop, right_start:right_stop]
                    if np.any(block != 0.0):
                        key = (label(left_count), label(physical_count), label(right_count))
                        blocks.append((key, block.reshape(-1, right_stop - right_start)))
        extents = [
            {label(count): stop - start for count, start, stop in slot_sectors}
            for slot_sectors in (left_sectors, physical_sectors, right_sectors)
        ]
        sites.append(
            MPSSite(
                sym.SymmetryBlockedTensorRank3([product] * 3, extents, blocks),
                [label(count) for count, _, _ in left_sectors],
                [label(count) for count, _, _ in physical_sectors],
                [label(count) for count, _, _ in right_sectors],
            )
        )
        left_numbers = resolved_right_numbers
    return sites


def make_mps(
    tensors: Sequence[np.ndarray | MPSSite],
    orthogonality_center: int | None = 0,
    site_to_orbital_order: Sequence[int] | None = None,
) -> MPSContainer:
    """Construct a native MPS container from dense test tensors or prebuilt sites."""
    sites = [tensor if isinstance(tensor, MPSSite) else make_site(np.asarray(tensor)) for tensor in tensors]
    return MPSContainer(
        sites,
        create_test_orbitals(max(1, len(sites))),
        orthogonality_center=orthogonality_center,
        site_to_orbital_order=list(site_to_orbital_order or []),
    )


def preparation_data(mps: MPSContainer, unitary_synthesis: str = "dense") -> MatrixProductStatePreparationData:
    """Generate preparation data from a container using the selected algorithm setting."""
    preparer = MatrixProductStatePreparation()
    preparer.settings().update("unitary_synthesis", unitary_synthesis)
    return preparer.generate_matrix_product_state_preparation_data(mps)


def right_normalized_tensors(tensors: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return right-canonical dense tensors preserving the normalized state."""
    normalized = [np.array(tensor, dtype=float, copy=True) for tensor in tensors]
    for site in range(len(normalized) - 1, 0, -1):
        chi_left, physical, chi_right = normalized[site].shape
        matrix = normalized[site].reshape(chi_left, physical * chi_right)
        q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
        normalized[site] = q_matrix.T.reshape(chi_left, physical, chi_right)
        previous_left, previous_physical, _ = normalized[site - 1].shape
        previous = normalized[site - 1].reshape(previous_left * previous_physical, chi_left)
        normalized[site - 1] = (previous @ r_matrix.T).reshape(previous_left, previous_physical, chi_left)
    normalized[0] /= np.linalg.norm(normalized[0])
    return normalized


def right_normalized_mps(tensors: Sequence[np.ndarray]) -> MPSContainer:
    """Construct a right-canonical MPS preserving the normalized state."""
    return make_mps(right_normalized_tensors(tensors))


def random_mps(
    num_sites: int,
    bond_dim: int,
    site_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> MPSContainer:
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


def random_particle_number_tensors(
    num_sites: int,
    num_particles: int,
    max_bond: int,
    site_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Return right-canonical dense tensors of a random state with a fixed particle number.

    Each bond index carries the particle number to its left, and bond indices are grouped by
    that number, so the tensors are block-sparse and :func:`particle_number_blocked_sites`
    can store them as particle-number-blocked sites. Each bond sector is as large as the
    configurations on both sides of the bond allow, capped at ``max_bond``.
    """
    rng = np.random.default_rng() if rng is None else rng
    physical_numbers = _PHYSICAL_PARTICLE_NUMBERS[site_dim]

    # configurations[sites][particles]: physical-index strings of `sites` sites with `particles` particles.
    configurations = [np.zeros(num_particles + 1, dtype=int) for _ in range(num_sites + 1)]
    configurations[0][0] = 1
    for sites in range(1, num_sites + 1):
        for count in physical_numbers:
            configurations[sites][count:] += configurations[sites - 1][: num_particles + 1 - count]

    def bond_sectors(bond: int) -> dict[int, int]:
        """Return the sector dimensions of the bond after ``bond`` sites, keyed by particle number."""
        return {
            count: min(
                max_bond, int(configurations[bond][count]), int(configurations[num_sites - bond][num_particles - count])
            )
            for count in range(num_particles + 1)
            if configurations[bond][count] and configurations[num_sites - bond][num_particles - count]
        }

    def offsets(sectors: dict[int, int]) -> dict[int, int]:
        return dict(zip(sectors, np.cumsum([0, *sectors.values()])[:-1].tolist(), strict=True))

    tensors = []
    for site in range(num_sites):
        left_sectors, right_sectors = bond_sectors(site), bond_sectors(site + 1)
        left_offsets, right_offsets = offsets(left_sectors), offsets(right_sectors)
        tensor = np.zeros((sum(left_sectors.values()), site_dim, sum(right_sectors.values())))
        for left_count, left_dim in left_sectors.items():
            columns = [
                (physical, right_offsets[left_count + count] + index)
                for physical, count in enumerate(physical_numbers)
                if left_count + count in right_sectors
                for index in range(right_sectors[left_count + count])
            ]
            # Orthonormal rows within the sector keep the site right-orthonormal.
            rows, _ = np.linalg.qr(rng.standard_normal((len(columns), left_dim)))
            for row in range(left_dim):
                for column, (physical, right) in enumerate(columns):
                    tensor[left_offsets[left_count] + row, physical, right] = rows[column, row]
        tensors.append(tensor)
    return tensors


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random real orthogonal matrix."""
    matrix, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return matrix


def reconstruct_givens(givens) -> np.ndarray:
    """Reconstruct ``D · L_l · ... · L_1`` from a ``GivensDecomposition``."""
    result = np.eye(len(givens.phases))
    for angles, shifted in zip(givens.layer_angles, givens.layer_shifted, strict=True):
        upper = (1 if shifted else 0) + 2 * np.arange(len(angles))
        cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
        first, second = result[upper], result[upper + 1]
        result[upper] = cosine * first - sine * second
        result[upper + 1] = sine * first + cosine * second
    return np.where(np.asarray(givens.phases, dtype=bool), -1.0, 1.0)[:, None] * result


def assert_same_givens(actual, expected) -> None:
    """Require two ``GivensDecomposition`` objects with identical layers and close angles."""
    assert actual.layer_shifted == expected.layer_shifted
    assert actual.phases == expected.phases
    assert len(actual.layer_angles) == len(expected.layer_angles)
    for actual_layer, expected_layer in zip(actual.layer_angles, expected.layer_angles, strict=True):
        np.testing.assert_allclose(actual_layer, expected_layer, atol=1e-12)


def site_isometry(tensor: np.ndarray, ancilla_dim: int) -> np.ndarray:
    """Return the isometry of a ``(left, physical, right)`` site on the joint register.

    Row ``physical * ancilla_dim + right`` and column ``left`` hold ``tensor[left, physical, right]``,
    with the right bond padded to ``ancilla_dim``.
    """
    left, physical, right = tensor.shape
    padded = np.zeros((physical, ancilla_dim, left))
    padded[:, :right, :] = tensor.transpose(1, 2, 0)
    return padded.reshape(physical * ancilla_dim, left)


def dense_site(site: MPSSite) -> np.ndarray:
    """Return a site's ``(left, physical, right)`` array from its packed dense matrix."""
    return site.to_dense().reshape(site.shape)


def contract_mps(wavefunction: MPSContainer) -> np.ndarray:
    """Contract a native MPS into a normalized dense state vector."""
    state = dense_site(wavefunction.sites[0])
    for site in wavefunction.sites[1:]:
        tensor = dense_site(site)
        left, num_states, previous_bond = state.shape
        incoming_bond, physical, outgoing_bond = tensor.shape
        state = (state.reshape(left * num_states, previous_bond) @ tensor.reshape(incoming_bond, -1)).reshape(
            left, num_states * physical, outgoing_bond
        )
    vector = state.sum(axis=0).flatten()
    norm = np.linalg.norm(vector)
    return vector if norm <= 1e-15 else vector / norm


# Fixed four-site MPS from the Qualtran MPSPreparation tests (Apache-2.0), used across
# preprocessing, fidelity, and resource tests.
REFERENCE_MPS_TENSORS = (
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

REFERENCE_MPS_EXPECTED_STATE = np.array(
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


_SQRT_HALF = float(np.sqrt(0.5))
_SINGLET_TENSORS = (
    np.array([[[0.0, 0.0], [_SQRT_HALF, 0.0], [0.0, -_SQRT_HALF], [0.0, 0.0]]]),
    np.array([[[0.0], [0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0], [0.0]]]),
)
_DOUBLE_OCCUPANCY_TENSORS = (
    np.array([[[0.0, 0.0], [0.0, _SQRT_HALF], [0.0, 0.0], [_SQRT_HALF, 0.0]]]),
    np.array([[[0.0], [1.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [1.0]]]),
)

# Two-site MPSs with hand-derived blocked Jordan-Wigner amplitudes, keyed by the qubit
# occupations (alpha_0, alpha_1, beta_0, beta_1) with qubit 0 most significant.
JORDAN_WIGNER_CONVENTION_CASES = (
    # (|ud> - |du>)/sqrt(2) in chain order is the singlet: reordering b0 a1 into a1 b0 flips |du>.
    pytest.param(_SINGLET_TENSORS, [0, 1], {0b1001: _SQRT_HALF, 0b0110: _SQRT_HALF}, id="singlet"),
    # Chain site 0 on orbital 1: |ud> creates a1 b0 (ordered); |du> creates b1 a0 (one swap).
    pytest.param(_SINGLET_TENSORS, [1, 0], {0b0110: _SQRT_HALF, 0b1001: _SQRT_HALF}, id="singlet-swapped"),
    # (|2u> + |u2>)/sqrt(2): |2u> creates a0 b0 a1 (one swap); |u2> creates a0 a1 b1 (ordered).
    pytest.param(_DOUBLE_OCCUPANCY_TENSORS, [0, 1], {0b1110: -_SQRT_HALF, 0b1101: _SQRT_HALF}, id="double"),
    # Chain site 0 on orbital 1: |2u> creates a1 b1 a0 (two swaps); |u2> creates a1 a0 b0 (one swap).
    pytest.param(_DOUBLE_OCCUPANCY_TENSORS, [1, 0], {0b1101: _SQRT_HALF, 0b1110: -_SQRT_HALF}, id="double-swapped"),
)

_SPINLESS_PAIR_TENSORS = (
    np.array([[[_SQRT_HALF, 0.0], [0.0, _SQRT_HALF]]]),
    np.array([[[1.0], [0.0]], [[0.0], [1.0]]]),
)
_SPINLESS_HOP_TENSORS = (
    np.array([[[0.0, -_SQRT_HALF], [_SQRT_HALF, 0.0]]]),
    np.array([[[1.0], [0.0]], [[0.0], [1.0]]]),
)

# Two-site spinless MPSs with hand-derived Jordan-Wigner amplitudes, keyed by the qubit
# occupations (mode_0, mode_1) with qubit 0 most significant.
SPINLESS_JORDAN_WIGNER_CONVENTION_CASES = (
    # (|00> + |11>)/sqrt(2) in chain order: |11> creates c0 c1, already in qubit order.
    pytest.param(_SPINLESS_PAIR_TENSORS, [0, 1], {0b00: _SQRT_HALF, 0b11: _SQRT_HALF}, id="spinless-pair"),
    # Chain site 0 on orbital 1: |11> creates c1 c0 (one swap).
    pytest.param(_SPINLESS_PAIR_TENSORS, [1, 0], {0b00: _SQRT_HALF, 0b11: -_SQRT_HALF}, id="spinless-pair-swapped"),
    # (|10> - |01>)/sqrt(2): one particle per term, so only the placement changes.
    pytest.param(_SPINLESS_HOP_TENSORS, [0, 1], {0b10: _SQRT_HALF, 0b01: -_SQRT_HALF}, id="spinless-hop"),
    pytest.param(_SPINLESS_HOP_TENSORS, [1, 0], {0b01: _SQRT_HALF, 0b10: -_SQRT_HALF}, id="spinless-hop-swapped"),
)


def dense_target(amplitudes: dict[int, float], num_qubits: int) -> np.ndarray:
    """Return a dense state vector with the given nonzero amplitudes."""
    state = np.zeros(1 << num_qubits)
    for index, amplitude in amplitudes.items():
        state[index] = amplitude
    return state


def blocked_jordan_wigner_state(
    chain_state: np.ndarray,
    site_to_orbital_order: Sequence[int] | None = None,
    qubits_per_site: int = 2,
) -> np.ndarray:
    """Map a contracted MPS state to the blocked Jordan-Wigner qubit basis.

    ``chain_state`` is indexed like :func:`contract_mps`: chain site 0 is the most-significant
    group of ``qubits_per_site`` bits holding the physical index. With two qubits per site the
    physical index is in the ``('0', 'u', 'd', '2')`` basis and its basis states create the alpha
    and then the beta mode of each site in chain order; with one qubit per site it is in the
    ``('0', '1')`` basis of one spinless mode. The result is indexed by the qubit occupations of
    :meth:`Configuration.to_bits`, qubit 0 most significant, and carries the sign of reordering
    the creation operators into qubit order.
    """
    num_sites = (len(chain_state).bit_length() - 1) // qubits_per_site
    order = list(range(num_sites)) if site_to_orbital_order is None else list(site_to_orbital_order)
    num_qubits = qubits_per_site * num_sites
    labels, parse = (
        ("01", Configuration.from_bitstring) if qubits_per_site == 1 else ("0ud2", Configuration.from_spin_half_string)
    )
    blocked = np.zeros(1 << num_qubits, dtype=chain_state.dtype)
    for index in np.flatnonzero(chain_state):
        orbital_states = ["0"] * num_sites
        created: list[int] = []
        for site, orbital in enumerate(order):
            physical = (int(index) >> (qubits_per_site * (num_sites - 1 - site))) & ((1 << qubits_per_site) - 1)
            orbital_states[orbital] = labels[physical]
            # Physical bit k is the mode on qubit k * num_sites + orbital (alpha, then beta).
            created += [channel * num_sites + orbital for channel in range(qubits_per_site) if physical >> channel & 1]
        bits = parse("".join(orbital_states)).to_bits(num_qubits)
        assert sorted(created) == [qubit for qubit, bit in enumerate(bits) if bit]
        inversions = sum(first > second for i, first in enumerate(created) for second in created[i + 1 :])
        blocked[int("".join(map(str, bits)), 2)] = (-1) ** inversions * chain_state[index]
    return blocked


def simulate_mps_preparation(
    operation: str,
    site_struct: str,
    params: dict,
    num_sites: int,
    num_ancilla_qubits: int,
) -> tuple[float, np.ndarray]:
    """Run an MPS preparation operation and post-select the bond register on ``|0>``.

    Args:
        operation: Fully qualified Q# operation with signature
            ``(initialStateVec, numSites, siteToOrbitalOrder, rotationBits, siteDecompositions, state, ancilla)``.
        site_struct: Fully qualified Q# struct of one site decomposition.
        params: Parameters produced by ``to_qsharp_params``.
        num_sites: Number of MPS sites.
        num_ancilla_qubits: Width of the bond register.

    Returns:
        ``P(ancilla = 0)`` and the normalized post-selected state in the blocked Jordan-Wigner
        qubit basis, qubit 0 most significant (see :func:`blocked_jordan_wigner_state`).

    """
    parameter_names = ("initialStateVec", "numSites", "siteToOrbitalOrder", "rotationBits", "siteDecompositions")
    arguments = ", ".join(_to_qsharp_literal(params[name], site_struct) for name in parameter_names)
    num_state_qubits = params["numQubitsPerSite"] * num_sites
    context = get_qsharp_context()
    context.eval(f"use state = Qubit[{num_state_qubits}];")
    context.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
    context.eval(f"{operation}({arguments}, state, ancilla)")
    dump = context.dump_machine()
    context.eval("ResetAll(state + ancilla);")

    # DumpMachine orders qubits state[0]..state[N-1], ancilla[0].. from the most-significant
    # bit, so the bond register occupies the lowest bits. Basis states with any other qubit
    # set (scratch qubits that were not uncomputed) are excluded.
    num_relevant_qubits = num_state_qubits + num_ancilla_qubits
    ancilla_mask = (1 << num_ancilla_qubits) - 1
    amplitudes = np.zeros(1 << num_state_qubits, dtype=complex)
    for idx in dump:
        if idx >> num_relevant_qubits == 0 and idx & ancilla_mask == 0:
            amplitudes[idx >> num_ancilla_qubits] = dump[idx]
    ancilla_zero_probability = float(np.sum(np.abs(amplitudes) ** 2))
    return ancilla_zero_probability, amplitudes / np.sqrt(ancilla_zero_probability)


_GIVENS_STRUCT = "GivensDecomposition.GivensDecomposition"
_GIVENS_FIELDS = {"layerAngles", "layerShifted", "phases"}


def _to_qsharp_literal(value, site_struct: str) -> str:
    """Serialize nested numeric and Boolean data as a Q# literal.

    Dictionaries with the fields of the Q# ``GivensDecomposition`` struct become that struct, and
    every other dictionary becomes ``site_struct``.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.15f}"
    if isinstance(value, dict):
        fields = ", ".join(f"{name} = {_to_qsharp_literal(item, site_struct)}" for name, item in value.items())
        struct = _GIVENS_STRUCT if value.keys() == _GIVENS_FIELDS else site_struct
        return f"new {struct} {{ {fields} }}"
    if isinstance(value, list):
        return f"[{', '.join(_to_qsharp_literal(item, site_struct) for item in value)}]"
    return str(value)
