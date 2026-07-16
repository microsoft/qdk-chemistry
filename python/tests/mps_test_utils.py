"""Test-only helpers for constructing and contracting native MPS data."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np

from qdk_chemistry.data import MPSSite, MPSWavefunction

from .test_helpers import create_test_orbitals


def make_mps(tensors: Sequence[np.ndarray]) -> MPSWavefunction:
    """Construct a native MPS wavefunction from dense test tensors."""
    sites = [MPSSite.from_dense(tensor) for tensor in tensors]
    return MPSWavefunction(sites, create_test_orbitals(max(1, len(sites))))


def random_mps(
    num_sites: int,
    bond_dim: int,
    site_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> MPSWavefunction:
    """Construct a right-normalized random native MPS for algorithm tests."""
    rng = np.random.default_rng() if rng is None else rng
    bond_dims = [1]
    for site in range(1, num_sites):
        max_left = bond_dims[-1] * site_dim
        max_right = site_dim ** min(site, num_sites - site)
        bond_dims.append(min(bond_dim, max_left, max_right))
    bond_dims.append(1)

    tensors = [rng.standard_normal((bond_dims[site], site_dim, bond_dims[site + 1])) for site in range(num_sites)]
    for site in range(num_sites - 1, 0, -1):
        chi_left, physical, chi_right = tensors[site].shape
        matrix = tensors[site].reshape(chi_left, physical * chi_right)
        q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
        tensors[site] = q_matrix.T.reshape(chi_left, physical, chi_right)
        previous_left, previous_physical, _ = tensors[site - 1].shape
        previous = tensors[site - 1].reshape(previous_left * previous_physical, chi_left)
        tensors[site - 1] = (previous @ r_matrix.T).reshape(previous_left, previous_physical, chi_left)
    return make_mps(tensors)


def contract_mps(wavefunction: MPSWavefunction) -> np.ndarray:
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
