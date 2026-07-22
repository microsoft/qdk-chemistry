"""Test-only helpers for constructing and contracting native MPS data."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np

from qdk_chemistry.data import AbelianMPSContainer, AbelianMPSSite

from .test_helpers import create_test_orbitals

# Per-physical-state particle counts for spin-half orbitals:
# |0⟩ -> 0, |↑⟩ -> 1, |↓⟩ -> 1, |2⟩ -> 2
DELTA_N = [0, 1, 1, 2]


def _compute_sector_dims(
    bond_dim: int, left_max_n: int, right_max_n: int
) -> tuple[dict[int, int], dict[int, int]]:
    """Compute sector dimensions for a bond, distributing evenly.

    The left bond can carry sectors from 0..left_max_n and the right bond
    from 0..right_max_n. Dimensions are distributed as evenly as possible,
    with each sector getting at least 1.
    """
    def _distribute(dim: int, max_n: int) -> dict[int, int]:
        num_sectors = max_n + 1
        if dim <= num_sectors:
            # At most one per sector; assign to the first `dim` sectors.
            return dict.fromkeys(range(dim), 1)
        base = dim // num_sectors
        remainder = dim % num_sectors
        sizes = {}
        for n in range(num_sectors):
            sizes[n] = base + (1 if n < remainder else 0)
        return sizes

    left_sizes = _distribute(bond_dim, left_max_n)
    right_sizes = _distribute(bond_dim, right_max_n)
    return left_sizes, right_sizes


def _sector_sizes_for_chain(
    bond_dims: list[int], num_sites: int
) -> list[dict[int, int]]:
    """Compute sector sizes for each bond in an MPS chain.

    bond_dims[k] is the dimension of the bond between site k-1 and site k
    (with bond_dims[0] = left boundary, bond_dims[num_sites] = right boundary).

    Returns a list of length num_sites+1 where entry k is the sector-size map
    for bond k.
    """
    sector_sizes = []
    for k in range(num_sites + 1):
        # Maximum particle number reachable at bond k is min(2*k, total_sites*2)
        max_n = min(2 * k, 2 * num_sites)
        num_sectors = max_n + 1
        dim = bond_dims[k]
        if dim <= num_sectors:
            sizes = dict.fromkeys(range(dim), 1)
        else:
            base = dim // num_sectors
            remainder = dim % num_sectors
            sizes = {n: base + (1 if n < remainder else 0) for n in range(num_sectors)}
        sector_sizes.append(sizes)
    return sector_sizes


def dense_to_abelian_site(
    tensor: np.ndarray,
    left_sector_sizes: dict[int, int],
    right_sector_sizes: dict[int, int],
    max_particle_number: int | None = None,
) -> AbelianMPSSite:
    """Convert a dense tensor to a particle-number-blocked AbelianMPSSite.

    Args:
        tensor: Dense array of shape (chi_left, d, chi_right).
        left_sector_sizes: Map from particle number to sector dimension (left bond).
        right_sector_sizes: Map from particle number to sector dimension (right bond).
        max_particle_number: Maximum particle number for the axis. Defaults to
            max key in either sector map.

    Returns:
        A particle-number-blocked AbelianMPSSite.

    """
    if max_particle_number is None:
        max_particle_number = max(
            max(left_sector_sizes.keys(), default=0),
            max(right_sector_sizes.keys(), default=0),
        )
    return AbelianMPSSite.from_dense_abelian(
        tensor,
        left_sector_sizes,
        right_sector_sizes,
        DELTA_N[: tensor.shape[1]],
        max_particle_number,
    )


def make_mps(
    tensors: Sequence[np.ndarray],
    orthogonality_center: int | None = 0,
) -> AbelianMPSContainer:
    """Construct a native MPS wavefunction from dense test tensors.

    Creates particle-number-blocked sites with sectors distributed evenly
    across the bond dimension.
    """
    num_sites = len(tensors)
    bond_dims = [t.shape[0] for t in tensors] + [tensors[-1].shape[2]]
    sector_sizes = _sector_sizes_for_chain(bond_dims, num_sites)
    max_n = 2 * num_sites

    sites = []
    for k, tensor in enumerate(tensors):
        sites.append(dense_to_abelian_site(tensor, sector_sizes[k], sector_sizes[k + 1], max_n))

    return AbelianMPSContainer(
        sites,
        create_test_orbitals(max(1, num_sites)),
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
