"""Unitary container for plaquette Trotterization of the 2D Fermi-Hubbard model."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer

if TYPE_CHECKING:
    import h5py

__all__ = ["HubbardPlaquetteContainer"]


class HubbardPlaquetteContainer(UnitaryContainer):
    r"""Container for second-order plaquette Trotter step for the 2D Fermi-Hubbard model.

    The step is :math:`P^{1/2}\,(I^{1/2} G I^{1/2} P)^r\,P^{-1/2}`, where :math:`I` is the
    on-site interaction and :math:`P`, :math:`G` are the pink and gold hopping tilings.
    Each tiling is a set of vertex-disjoint four-cycles, so Q# can apply every plaquette
    of a tiling in parallel and phase the whole tiling through one Hamming-weight
    register.

    Every angle below is a :math:`\theta` entering as :math:`e^{-i\theta P}` for its Pauli
    word :math:`P`. For the Fermi-Hubbard parameters hopping :math:`t` and on-site interaction
    :math:`U` and the per-step duration :math:`\delta = T / r` for a total evolution time
    :math:`T` over :math:`r` steps:

    * ``interaction_angle`` is :math:`(U/4)\delta`, the :math:`Z_i Z_{i+M}` angle of one
      :math:`e^{-i\delta I}` layer, applied once per site.

    * ``hopping_angle`` is :math:`\kappa = 2t\delta`, shared by both tilings. A plaquette's
      hopping matrix is diagonalized with :math:`\pm 2t` nonzero eigenvalues.

    * ``constant_shift`` is the per-step scalar phase from the classical energy
      offset :math:`U\eta/2 - UM/4`. It is not sent to Q#; it is applied only by
      :meth:`eigenvalue_from_phase` when converting a measured phase back to energy.

    Args:
        width: Number of lattice columns.
        height: Number of lattice rows.
        interaction_angle: On-site :math:`Z Z` pair angle for a full interaction layer.
        constant_shift: Classical per-step scalar phase offset used only in phase-to-energy conversion.
        hopping_angle: :math:`\kappa = 2 t \delta`, shared by both tilings.
        step_reps: Number of repetitions of the body.
        scale: Total evolution time the step count was derived for.

    Raises:
        TypeError: If ``width``, ``height``, or ``step_reps`` is not an integer.
        ValueError: If ``step_reps`` is not positive, or the plaquettes cannot tile a ``width`` x ``height``
            lattice; see :meth:`validate_shape`.

    """

    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for plaquette evolution containers.

        Returns:
            ``"hubbard_plaquette_container"``.

        """
        return "hubbard_plaquette_container"

    def __init__(
        self,
        width: int,
        height: int,
        interaction_angle: float,
        constant_shift: float = 0.0,
        hopping_angle: float = 0.0,
        step_reps: int = 1,
        scale: float = 1.0,
    ) -> None:
        """Initialize the container."""
        for name, value in (("width", width), ("height", height), ("step_reps", step_reps)):
            # bool is an int subclass, but True as a count is always a mistake.
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if step_reps <= 0:
            raise ValueError(f"step_reps must be a positive integer, got {step_reps}.")
        self.width = int(width)
        self.height = int(height)
        self.validate_shape(self.width, self.height)
        self.interaction_angle = float(interaction_angle)
        self.constant_shift = float(constant_shift)
        self.hopping_angle = float(hopping_angle)
        self.step_reps = int(step_reps)
        self.scale = float(scale)
        super().__init__()

    @staticmethod
    def validate_shape(width: int, height: int) -> None:
        """Check that the plaquettes tile a periodic ``width`` x ``height`` lattice.

        Args:
            width: Number of lattice columns.
            height: Number of lattice rows.

        Raises:
            ValueError: Unless both sides are even and at least four, or the lattice is 2x2.

        """
        if width % 2 or height % 2:
            raise ValueError(f"Plaquette tiling requires even side lengths, got {width}x{height}.")
        if (width < 4 or height < 4) and (width, height) != (2, 2):
            raise ValueError(
                f"Plaquette tiling requires both sides to be at least four, or exactly 2x2, got {width}x{height}."
            )

    @property
    def num_sites(self) -> int:
        """Return the number of lattice sites."""
        return self.width * self.height

    @property
    def type(self) -> str:
        """Return the container type."""
        return "hubbard_plaquette"

    @property
    def num_qubits(self) -> int:
        """Return the width of the system register, two spin orbitals per site."""
        return 2 * self.width * self.height

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        r"""Recover a Hamiltonian eigenvalue from a time-evolution phase.

        For :math:`U(t) = e^{-iHt}` an eigenstate with energy :math:`E` acquires phase
        :math:`e^{-iEt}`, so QPE measures :math:`\varphi = (-Et / 2\pi) \bmod 1`.
        The circuit evolves under the particle-hole symmetric Hamiltonian; the scalar
        correction stored per Trotter step is added after converting the measured phase.

        Args:
            phase_fraction: Measured phase fraction :math:`\varphi \in [0, 1)`.

        Returns:
            float: The corresponding Hamiltonian eigenvalue.

        """
        angle = (phase_fraction % 1.0) * (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return float((-angle + self.constant_shift * self.step_reps) / self.scale)

    def combine(self, other: UnitaryContainer) -> UnitaryContainer:
        """Combining plaquette evolutions is not supported.

        Args:
            other: The container to append after this one.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("HubbardPlaquetteContainer does not support combine.")

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        return self._add_json_version(
            {
                "container_type": self.type,
                "width": self.width,
                "height": self.height,
                "interaction_angle": self.interaction_angle,
                "constant_shift": self.constant_shift,
                "hopping_angle": self.hopping_angle,
                "step_reps": self.step_reps,
                "scale": self.scale,
            }
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the container to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["payload"] = json.dumps(self.to_json())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> HubbardPlaquetteContainer:
        """Create a container from JSON.

        Args:
            json_data: The serialized container.

        Returns:
            HubbardPlaquetteContainer: The reconstructed container.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            json_data["width"],
            json_data["height"],
            float(json_data["interaction_angle"]),
            float(json_data["constant_shift"]),
            float(json_data["hopping_angle"]),
            json_data["step_reps"],
            float(json_data["scale"]),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> HubbardPlaquetteContainer:
        """Create a container from an HDF5 group.

        Args:
            group: The HDF5 group holding the serialized container.

        Returns:
            HubbardPlaquetteContainer: The reconstructed container.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a human-readable summary of the container."""
        return (
            f"Hubbard plaquette evolution ({self.width}x{self.height} lattice, "
            f"{self.num_qubits} qubits, {self.step_reps} repetitions): "
            f"interaction angle: {self.interaction_angle}, constant shift: {self.constant_shift}, "
            f"hopping angle: {self.hopping_angle}"
        )
