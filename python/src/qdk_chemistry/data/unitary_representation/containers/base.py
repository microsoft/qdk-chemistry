"""QDK/Chemistry unitary container base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import Any

import h5py

from qdk_chemistry.data.base import DataClass

__all__: list[str] = ["UnitaryContainer"]


class UnitaryContainer(DataClass):
    """Abstract class for a unitary container."""

    # Class attribute for filename validation
    _data_type_name = "unitary_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    @property
    @abstractmethod
    def type(self) -> str:
        """Get the type of the unitary container.

        Returns:
            The type of the unitary container.

        """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits the unitary acts on.

        Returns:
            The number of qubits.

        """

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Convert the UnitaryContainer to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the UnitaryContainer

        """

    @abstractmethod
    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the UnitaryContainer to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to

        """

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: dict[str, Any]) -> "UnitaryContainer":
        """Create UnitaryContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            UnitaryContainer

        """

    @classmethod
    @abstractmethod
    def from_hdf5(cls, group: h5py.Group) -> "UnitaryContainer":
        """Load an instance from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            UnitaryContainer

        """

    @abstractmethod
    def get_summary(self) -> str:
        """Get summary of unitary container.

        Returns:
            str: Summary string describing the UnitaryContainer's contents and properties

        """

    @abstractmethod
    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        r"""Recover a Hamiltonian eigenvalue from the measured phase fraction.

        Each unitary encoding maps Hamiltonian eigenvalues to phases on the
        unit circle.  This method inverts that mapping so that a measured
        phase fraction :math:`\varphi \in [0, 1)` is converted back to the
        corresponding eigenvalue :math:`E`.

        Args:
            phase_fraction: Measured phase fraction :math:`\varphi \in [0, 1)`.

        Returns:
            float: The corresponding Hamiltonian eigenvalue.

        """

    @abstractmethod
    def phases_from_eigenvalue(self, eigenvalue: float) -> list[float]:
        r"""Recover every phase fraction a Hamiltonian eigenvalue is measured at.

        The closed-form inverse of :meth:`eigenvalue_from_phase`, so that
        ``eigenvalue_from_phase(phi) == eigenvalue`` for every ``phi`` returned.

        The forward map need not be injective, so this returns *all* of the
        phases carrying the eigenvalue, ascending and without repeats.  Solving
        for them is what lets a caller find where on the phase circle an energy
        condition changes, instead of evaluating the forward map bin by bin.

        Args:
            eigenvalue: The Hamiltonian eigenvalue :math:`E`.

        Returns:
            list[float]: The phase fractions :math:`\varphi \in [0, 1)` QPE
            measures the eigenvalue at, sorted ascending.  Never empty.

        Raises:
            ValueError: If the eigenvalue lies outside the range the encoding can
                represent, so that no phase corresponds to it.

        """

    def phase_from_eigenvalue(self, eigenvalue: float) -> float:
        r"""Recover the principal phase fraction a Hamiltonian eigenvalue is measured at.

        The smallest phase :meth:`phases_from_eigenvalue` solves for, so the two
        can never disagree.  Use it when an encoding is known to be injective, or
        when any one of the phases carrying the eigenvalue will do.

        Args:
            eigenvalue: The Hamiltonian eigenvalue :math:`E`.

        Returns:
            float: The smallest phase fraction :math:`\varphi \in [0, 1)` QPE
            measures the eigenvalue at.

        Raises:
            ValueError: If the eigenvalue lies outside the range the encoding can
                represent, so that no phase corresponds to it.

        """
        return self.phases_from_eigenvalue(eigenvalue)[0]

    @abstractmethod
    def combine(self, other: "UnitaryContainer") -> "UnitaryContainer":
        """Combine this container with another to represent sequential application.

        Args:
            other: The container to append after this one.

        Returns:
            A new container representing the combined evolution.

        """
