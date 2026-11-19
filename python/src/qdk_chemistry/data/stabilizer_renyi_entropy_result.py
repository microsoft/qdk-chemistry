"""Immutable stabilizer Renyi entropy results and truncation metadata."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import isfinite
from typing import TYPE_CHECKING, Any

from qdk_chemistry.data._hashing import _hash_arg, _numpy_scalar_to_python
from qdk_chemistry.data.base import DataClass

if TYPE_CHECKING:
    import hashlib

    import h5py

__all__ = ["StabilizerRenyiEntropyResult"]


class StabilizerRenyiEntropyResult(DataClass):
    """Second-order stabilizer Renyi entropy in bits and immutable provenance.

    Configuration counts exclude exact zeros. ``coefficient_cutoff`` is ``None``
    when truncation is disabled. The discarded probability is measured in the
    original state, before renormalizing the retained coefficients.
    Unchanged configuration counts require zero discarded probability.
    Zero entropy is stored as positive zero, and enabled cutoffs as floats.
    """

    _serialization_version = "0.1.0"

    def __init__(
        self,
        *,
        entropy: float,
        num_qubits: int,
        num_configurations: int,
        num_retained_configurations: int,
        truncation_enabled: bool = False,
        coefficient_cutoff: float | None = None,
        discarded_weight: float = 0.0,
        alpha: int = 2,
        log_base: int = 2,
        encoding: str = "jordan-wigner",
        fermion_mode_order: str = "blocked",
    ) -> None:
        """Initialize an entropy result with validated, scalar metadata.

        Args:
            entropy: Second-order stabilizer Renyi entropy in bits.
            num_qubits: Active occupation register width, including empty modes.
            num_configurations: Number of nonzero configurations before truncation.
            num_retained_configurations: Number of nonzero configurations used in the calculation.
            truncation_enabled: Whether coefficient truncation was requested.
            coefficient_cutoff: Strict amplitude threshold, or None when disabled.
            discarded_weight: Removed probability weight before renormalization.
            alpha: Renyi order; currently only 2 is supported.
            log_base: Logarithm base; currently only 2 is supported.
            encoding: Occupation encoding; currently only Jordan-Wigner is supported.
            fermion_mode_order: Spin-half mode ordering; currently only blocked is supported.

        Raises:
            ValueError: If numerical values, configuration counts, or provenance are inconsistent.

        """
        for name, value in (
            ("num_qubits", num_qubits),
            ("num_configurations", num_configurations),
            ("num_retained_configurations", num_retained_configurations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if num_retained_configurations > num_configurations or num_configurations > 2**num_qubits:
            raise ValueError("Configuration counts must satisfy 1 <= retained <= original <= 2**num_qubits.")
        if not isfinite(entropy) or not 0 <= entropy <= num_qubits:
            raise ValueError("entropy must be finite and between zero and num_qubits.")
        if not isfinite(discarded_weight) or not 0 <= discarded_weight < 1:
            raise ValueError("discarded_weight must be finite and in [0, 1).")
        if not isinstance(truncation_enabled, bool):
            raise ValueError("truncation_enabled must be a boolean.")
        if truncation_enabled:
            if coefficient_cutoff is None or not isfinite(coefficient_cutoff) or coefficient_cutoff < 0:
                raise ValueError("Enabled truncation requires a finite, non-negative coefficient_cutoff.")
            if num_retained_configurations == num_configurations and discarded_weight != 0:
                raise ValueError("Unchanged configuration counts require zero discarded_weight.")
        elif (
            coefficient_cutoff is not None or discarded_weight != 0 or num_retained_configurations != num_configurations
        ):
            raise ValueError(
                "Disabled truncation requires no cutoff, no discarded weight, and unchanged configuration counts."
            )
        if alpha != 2 or log_base != 2 or encoding != "jordan-wigner" or fermion_mode_order != "blocked":
            raise ValueError("Only alpha=2, log_base=2, Jordan-Wigner encoding, and blocked ordering are supported.")

        self.entropy = 0.0 if entropy == 0 else float(entropy)
        self.num_qubits = num_qubits
        self.num_configurations = num_configurations
        self.num_retained_configurations = num_retained_configurations
        self.truncation_enabled = truncation_enabled
        self.coefficient_cutoff = float(coefficient_cutoff) if coefficient_cutoff is not None else None
        self.discarded_weight = float(discarded_weight)
        self.alpha = alpha
        self.log_base = log_base
        self.encoding = encoding
        self.fermion_mode_order = fermion_mode_order
        super().__init__()

    @staticmethod
    def data_type_name() -> str:
        """Return the result's wire-format identifier."""
        return "stabilizer_renyi_entropy_result"

    def _hash_update(self, h: "hashlib._Hash") -> None:
        """Hash every serialized scalar, including provenance."""
        _hash_arg(h, self.to_json())

    def get_summary(self) -> str:
        """Return a human-readable entropy and truncation summary."""
        return (
            f"Stabilizer Renyi entropy (alpha={self.alpha}): {self.entropy:.12g} bits\n"
            f"Active qubits: {self.num_qubits}; encoding: {self.encoding}; ordering: {self.fermion_mode_order}\n"
            f"N_conf (retained/original): {self.num_retained_configurations}/{self.num_configurations}; "
            f"truncation enabled: {self.truncation_enabled}; cutoff: {self.coefficient_cutoff}\n"
            f"Discarded probability weight: {self.discarded_weight:.12g}"
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize the entropy and its provenance to a versioned dictionary."""
        return self._add_json_version(
            {
                "entropy": self.entropy,
                "num_qubits": self.num_qubits,
                "num_configurations": self.num_configurations,
                "num_retained_configurations": self.num_retained_configurations,
                "truncation_enabled": self.truncation_enabled,
                "coefficient_cutoff": self.coefficient_cutoff,
                "discarded_weight": self.discarded_weight,
                "alpha": self.alpha,
                "log_base": self.log_base,
                "encoding": self.encoding,
                "fermion_mode_order": self.fermion_mode_order,
            }
        )

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "StabilizerRenyiEntropyResult":
        """Reconstruct a result, validating the version and all metadata."""
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            entropy=json_data["entropy"],
            num_qubits=json_data["num_qubits"],
            num_configurations=json_data["num_configurations"],
            num_retained_configurations=json_data["num_retained_configurations"],
            truncation_enabled=json_data["truncation_enabled"],
            coefficient_cutoff=json_data["coefficient_cutoff"],
            discarded_weight=json_data["discarded_weight"],
            alpha=json_data["alpha"],
            log_base=json_data["log_base"],
            encoding=json_data["encoding"],
            fermion_mode_order=json_data["fermion_mode_order"],
        )

    def to_hdf5(self, group: "h5py.Group") -> None:
        """Serialize scalar fields, replacing result attributes in a reused group."""
        self._add_hdf5_version(group)
        for key, value in self.to_json().items():
            if value is not None:
                group.attrs[key] = value
            elif key in group.attrs:
                del group.attrs[key]

    @classmethod
    def from_hdf5(cls, group: "h5py.Group") -> "StabilizerRenyiEntropyResult":
        """Reconstruct a result from versioned HDF5 attributes."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        data = {key: _numpy_scalar_to_python(value) for key, value in group.attrs.items()}
        data["coefficient_cutoff"] = data.get("coefficient_cutoff")
        return cls.from_json(data)
