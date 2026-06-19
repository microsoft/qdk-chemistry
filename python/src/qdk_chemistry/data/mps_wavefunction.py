"""MPS Wavefunction container for Matrix Product State representations.

This module provides a Python container for wavefunctions stored as Matrix Product
States (MPS). The MPS tensors are stored directly and can be used with the
MPS-based sequential state preparation algorithm.

The MPS decomposition and Givens-based circuit synthesis is based on the work of:
  - Berry, Tong, et al. arXiv:2409.11748 (site unitary construction, Appendix B)
  - Rupprecht & Wölk (2025), published on Zenodo
    (https://zenodo.org/records/15587498)
    Original implementation rewritten and adapted for integration into qdk-chemistry.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["MPSWavefunction"]


class MPSWavefunction:
    """Container for wavefunctions represented as Matrix Product States (MPS).

    Stores the MPS tensors and provides methods for computing the full state vector
    and metadata needed by the MPS state preparation algorithm.

    Parameters
    ----------
    tensors : sequence of np.ndarray
        MPS tensors. ``tensors[i]`` has shape ``(chi_left_i, d, chi_right_i)``
        where ``d`` is the local Hilbert space dimension (typically 4 for
        two-qubit sites in Jordan-Wigner encoding) and ``chi`` are the bond
        dimensions.
    site_dim : int, optional
        Local site dimension (default 4, i.e. 2 qubits per site).

    Attributes
    ----------
    tensors : list[np.ndarray]
        The MPS tensors.
    num_sites : int
        Number of MPS sites.
    site_dim : int
        Local Hilbert space dimension per site.
    bond_dims : list[int]
        Bond dimensions ``[chi_0, chi_1, ..., chi_N]`` where ``chi_0 = chi_N = 1``
        for open boundary conditions.

    Notes
    -----
    The MPS tensors should be in right-canonical form for optimal circuit synthesis.
    The first tensor should have ``chi_left = 1`` (open boundary on the left) and
    the last tensor should have ``chi_right = 1`` (open boundary on the right).

    Attribution
    -----------
    The Berry decomposition and Givens rotation circuit synthesis used by the
    MPS state preparation algorithm is based on code originally published by
    Felix Rupprecht (DLR) on Zenodo (https://zenodo.org/records/15587498),
    rewritten and adapted for integration into the QDK Chemistry library.
    """

    def __init__(self, tensors: Sequence[np.ndarray], site_dim: int = 4):
        """Initialize the MPS wavefunction container.

        Args:
            tensors: Sequence of MPS tensors with shapes (chi_left, d, chi_right).
            site_dim: Local Hilbert space dimension per site (default 4).

        Raises:
            ValueError: If tensors are empty, have inconsistent dimensions,
                or violate open boundary conditions.
        """
        if not tensors:
            raise ValueError("MPS tensors must not be empty.")

        self.site_dim = site_dim
        self.tensors = [np.asarray(t, dtype=np.float64) for t in tensors]
        self.num_sites = len(self.tensors)

        # Validate tensor shapes
        for i, t in enumerate(self.tensors):
            if t.ndim != 3:
                raise ValueError(
                    f"Tensor {i} must be 3-dimensional, got shape {t.shape}."
                )
            if t.shape[1] != site_dim:
                raise ValueError(
                    f"Tensor {i} has site dimension {t.shape[1]}, expected {site_dim}."
                )

        # Validate bond dimension consistency
        for i in range(self.num_sites - 1):
            chi_right = self.tensors[i].shape[2]
            chi_left_next = self.tensors[i + 1].shape[0]
            if chi_right != chi_left_next:
                raise ValueError(
                    f"Bond dimension mismatch between site {i} (chi_right={chi_right}) "
                    f"and site {i+1} (chi_left={chi_left_next})."
                )

        # Validate open boundary conditions
        if self.tensors[0].shape[0] != 1:
            raise ValueError(
                f"First tensor must have chi_left=1 (open boundary), got {self.tensors[0].shape[0]}."
            )
        if self.tensors[-1].shape[2] != 1:
            raise ValueError(
                f"Last tensor must have chi_right=1 (open boundary), got {self.tensors[-1].shape[2]}."
            )

        self.bond_dims = [self.tensors[0].shape[0]]
        for t in self.tensors:
            self.bond_dims.append(t.shape[2])

    @property
    def max_bond_dim(self) -> int:
        """Maximum bond dimension (chi_max) across all bonds."""
        return max(self.bond_dims)

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (2 per site for d=4)."""
        qubits_per_site = int(np.log2(self.site_dim))
        return self.num_sites * qubits_per_site

    def contract(self) -> np.ndarray:
        """Contract the MPS to obtain the full state vector.

        Returns
        -------
        np.ndarray
            Normalized state vector of length ``site_dim ** num_sites``.
        """
        state = self.tensors[0]  # (1, d, chi_1)
        for tensor in self.tensors[1:]:
            # state: (1, d^k, chi_prev), tensor: (chi_prev, d, chi_next)
            left = state.shape[0]
            num_states = state.shape[1]
            chi_prev = state.shape[2]
            chi_in, d, chi_next = tensor.shape

            state_flat = state.reshape(left * num_states, chi_prev)
            tensor_flat = tensor.reshape(chi_in, d * chi_next)
            result = state_flat @ tensor_flat
            state = result.reshape(left, num_states * d, chi_next)

        vec = state.flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-15:
            vec = vec / norm
        return vec

    @classmethod
    def from_state_vector(
        cls,
        state_vector: np.ndarray,
        num_sites: int,
        max_bond_dim: int | None = None,
        site_dim: int = 4,
    ) -> MPSWavefunction:
        """Construct an MPS from a full state vector via SVD truncation.

        Parameters
        ----------
        state_vector : np.ndarray
            Full state vector of length ``site_dim ** num_sites``.
        num_sites : int
            Number of MPS sites.
        max_bond_dim : int or None
            Maximum bond dimension. If None, no truncation is applied.
        site_dim : int
            Local site dimension (default 4).

        Returns
        -------
        MPSWavefunction
            The MPS representation of the state vector.
        """
        state_vector = np.asarray(state_vector, dtype=np.float64)
        total_dim = site_dim ** num_sites
        if len(state_vector) != total_dim:
            raise ValueError(
                f"State vector length {len(state_vector)} doesn't match "
                f"site_dim^num_sites = {total_dim}."
            )

        # Normalize
        norm = np.linalg.norm(state_vector)
        if norm > 1e-15:
            state_vector = state_vector / norm

        tensors = []
        remaining = state_vector.copy()
        chi_left = 1

        for i in range(num_sites - 1):
            dim_right = site_dim ** (num_sites - i - 1)
            mat = remaining.reshape(chi_left * site_dim, dim_right)

            u_svd, s_svd, vt_svd = np.linalg.svd(mat, full_matrices=False)

            chi_right = len(s_svd)
            if max_bond_dim is not None and chi_right > max_bond_dim:
                chi_right = max_bond_dim
                u_svd = u_svd[:, :chi_right]
                s_svd = s_svd[:chi_right]
                vt_svd = vt_svd[:chi_right, :]

            tensor = u_svd.reshape(chi_left, site_dim, chi_right)
            tensors.append(tensor)

            remaining = (np.diag(s_svd) @ vt_svd).flatten()
            chi_left = chi_right

        # Last tensor: (chi_left, d, 1)
        last_tensor = remaining.reshape(chi_left, site_dim, 1)
        tensors.append(last_tensor)

        return cls(tensors, site_dim=site_dim)

    @classmethod
    def random(
        cls,
        num_sites: int,
        bond_dim: int,
        site_dim: int = 4,
        rng: np.random.Generator | None = None,
        right_canonical: bool = True,
    ) -> MPSWavefunction:
        """Generate a random MPS with specified bond dimension.

        Parameters
        ----------
        num_sites : int
            Number of sites.
        bond_dim : int
            Internal bond dimension (capped by dimensional constraints).
        site_dim : int
            Local site dimension (default 4).
        rng : np.random.Generator or None
            Random number generator. If None, uses default.
        right_canonical : bool
            If True, put the MPS in right-canonical form (default True).

        Returns
        -------
        MPSWavefunction
            A random MPS wavefunction.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Determine bond dimensions respecting boundary constraints
        bond_dims = [1]
        for i in range(1, num_sites):
            max_left = bond_dims[-1] * site_dim
            max_right = site_dim ** min(i, num_sites - i)
            chi = min(bond_dim, max_left, max_right)
            bond_dims.append(chi)
        bond_dims.append(1)

        tensors = []
        for i in range(num_sites):
            chi_l = bond_dims[i]
            chi_r = bond_dims[i + 1]
            t = rng.standard_normal((chi_l, site_dim, chi_r))
            tensors.append(t)

        if right_canonical:
            tensors = _make_right_canonical(tensors)

        return cls(tensors, site_dim=site_dim)


def _make_right_canonical(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Put MPS tensors in right-canonical form via successive QR decompositions.

    Sweeps from right to left. After this, all tensors except the first
    satisfy A†A = I (right-isometric).
    """
    result = [t.copy() for t in tensors]
    num_sites = len(result)

    for i in range(num_sites - 1, 0, -1):
        chi_l, d, chi_r = result[i].shape
        # Reshape to (chi_l, d * chi_r) and do QR from the right
        mat = result[i].reshape(chi_l, d * chi_r)
        # SVD-based right canonicalization: mat = U @ S @ Vt
        # Vt becomes the new tensor, U @ S is absorbed left
        q_mat, r_mat = np.linalg.qr(mat.T, mode='reduced')
        result[i] = q_mat.T.reshape(chi_l, d, chi_r)
        chi_l_prev, d_prev, _ = result[i - 1].shape
        left_mat = result[i - 1].reshape(chi_l_prev * d_prev, chi_l)
        result[i - 1] = (left_mat @ r_mat.T).reshape(chi_l_prev, d_prev, chi_l)

    return result
