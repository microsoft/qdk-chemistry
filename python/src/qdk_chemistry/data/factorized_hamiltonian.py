"""FactorizedHamiltonian: stores DFTHC factorization output for SOSSA circuits.

The factorized Hamiltonian stores the classical preprocessing results from
Double-Factorized Tensor Hypercontraction (DFTHC) optimization, which serve
as inputs to the SOS spectrum-amplified block encoding circuit.

The stored matrices correspond to the decomposition (arXiv:2502.15882):

    H ≈ Σ_σ (Σ_j w⁺_j a†_{u⁺_j σ} a_{u⁺_j σ} + Σ_j w⁻_j a_{u⁻_j σ} a†_{u⁻_j σ})
      + ½ Σ_{r,c} (Σ_b w^{rc}_b Σ_{pq} U^r_{bp} U^r_{bq} E_{pq})²
      + E_SOS

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data._hashing import _hash_array, _hash_float, _hash_str, _hash_uint
from qdk_chemistry.data.base import DataClass

__all__: list[str] = []


class FactorizedHamiltonian(DataClass):
    """Stores DFTHC factorization results needed for SOSSA block encoding circuits.

    Attributes:
        num_orbitals: Number of spatial orbitals (N).
        num_ranks: Number of ranks in the factorization (R).
        num_bases: Number of basis functions per rank (B).
        num_copies: Number of copies per rank (C).
        h1_majorana: Adjusted one-body matrix in Majorana basis, shape [N, N].
        basis_vectors: Orbital rotation matrices U, shape [R, B, N].
        two_body_weights: Two-body SF weights W, shape [R, B, C].
        identity_weight: Identity scalar WB, shape [R, C].
        core_energy: Nuclear repulsion + inactive core energy (Ha).

    """

    _data_type_name = "factorized_hamiltonian"
    _serialization_version = "0.1.0"

    def __init__(
        self,
        h1_majorana: np.ndarray,
        basis_vectors: np.ndarray,
        two_body_weights: np.ndarray,
        identity_weight: np.ndarray,
        core_energy: float = 0.0,
    ) -> None:
        """Initialize a FactorizedHamiltonian.

        Args:
            h1_majorana: Adjusted one-body matrix in Majorana basis, shape [N, N].
            basis_vectors: Orbital rotation matrices U, shape [R, B, N].
            two_body_weights: Two-body SF weights W, shape [R, B, C].
            identity_weight: Identity scalar WB, shape [R, C].
            core_energy: Nuclear repulsion + inactive core energy (Ha).

        Raises:
            ValueError: If array shapes are inconsistent.

        """
        h1_majorana = np.asarray(h1_majorana, dtype=np.float64)
        basis_vectors = np.asarray(basis_vectors, dtype=np.float64)
        two_body_weights = np.asarray(two_body_weights, dtype=np.float64)
        identity_weight = np.asarray(identity_weight, dtype=np.float64)

        if h1_majorana.ndim != 2 or h1_majorana.shape[0] != h1_majorana.shape[1]:
            raise ValueError(f"h1_majorana must be a square matrix, got shape {h1_majorana.shape}")

        N = h1_majorana.shape[0]

        if basis_vectors.ndim != 3:
            raise ValueError(f"basis_vectors must have shape [R, B, N], got {basis_vectors.shape}")
        R, B, N_u = basis_vectors.shape
        if N_u != N:
            raise ValueError(
                f"basis_vectors last dimension ({N_u}) must match "
                f"h1_majorana dimension ({N})"
            )

        if two_body_weights.ndim != 3:
            raise ValueError(f"two_body_weights must have shape [R, B, C], got {two_body_weights.shape}")
        R_w, B_w, C = two_body_weights.shape
        if R_w != R or B_w != B:
            raise ValueError(
                f"two_body_weights shape ({two_body_weights.shape}) inconsistent "
                f"with basis_vectors shape ({basis_vectors.shape})"
            )

        if identity_weight.ndim != 2:
            raise ValueError(f"identity_weight must have shape [R, C], got {identity_weight.shape}")
        R_wb, C_wb = identity_weight.shape
        if R_wb != R or C_wb != C:
            raise ValueError(
                f"identity_weight shape ({identity_weight.shape}) inconsistent "
                f"with two_body_weights shape ({two_body_weights.shape})"
            )

        self.num_orbitals = N
        self.num_ranks = R
        self.num_bases = B
        self.num_copies = C
        self.h1_majorana = h1_majorana
        self.basis_vectors = basis_vectors
        self.two_body_weights = two_body_weights
        self.identity_weight = identity_weight
        self.core_energy = float(core_energy)

        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "factorized_hamiltonian")
        _hash_uint(h, self.num_orbitals)
        _hash_uint(h, self.num_ranks)
        _hash_uint(h, self.num_bases)
        _hash_uint(h, self.num_copies)
        _hash_float(h, self.core_energy)
        _hash_array(h, self.h1_majorana)
        _hash_array(h, self.basis_vectors)
        _hash_array(h, self.two_body_weights)
        _hash_array(h, self.identity_weight)

    def get_summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"FactorizedHamiltonian(N={self.num_orbitals}, R={self.num_ranks}, "
            f"B={self.num_bases}, C={self.num_copies}, "
            f"core_energy={self.core_energy:.6f})"
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "num_orbitals": self.num_orbitals,
            "num_ranks": self.num_ranks,
            "num_bases": self.num_bases,
            "num_copies": self.num_copies,
            "core_energy": self.core_energy,
            "h1_majorana": self.h1_majorana.tolist(),
            "basis_vectors": self.basis_vectors.tolist(),
            "two_body_weights": self.two_body_weights.tolist(),
            "identity_weight": self.identity_weight.tolist(),
        }
        return self._add_json_version(data)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "FactorizedHamiltonian":
        """Deserialize from a JSON dictionary."""
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            h1_majorana=np.array(json_data["h1_majorana"]),
            basis_vectors=np.array(json_data["basis_vectors"]),
            two_body_weights=np.array(json_data["two_body_weights"]),
            identity_weight=np.array(json_data["identity_weight"]),
            core_energy=json_data["core_energy"],
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["num_orbitals"] = self.num_orbitals
        group.attrs["num_ranks"] = self.num_ranks
        group.attrs["num_bases"] = self.num_bases
        group.attrs["num_copies"] = self.num_copies
        group.attrs["core_energy"] = self.core_energy
        group.create_dataset("h1_majorana", data=self.h1_majorana)
        group.create_dataset("basis_vectors", data=self.basis_vectors)
        group.create_dataset("two_body_weights", data=self.two_body_weights)
        group.create_dataset("identity_weight", data=self.identity_weight)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "FactorizedHamiltonian":
        """Load from an HDF5 group."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls(
            h1_majorana=np.array(group["h1_majorana"]),
            basis_vectors=np.array(group["basis_vectors"]),
            two_body_weights=np.array(group["two_body_weights"]),
            identity_weight=np.array(group["identity_weight"]),
            core_energy=float(group.attrs["core_energy"]),
        )
