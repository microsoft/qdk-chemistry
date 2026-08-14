"""Boson-to-qubit mapper abstractions and utilities.

This module provides the base class :class:`BosonQubitMapper` and the
:class:`BosonQubitMapperFactory` for mapping bosonic Hamiltonians to qubit
operators.

Bosonic mapping is a separate algorithm type from the fermionic
``qubit_mapper`` because it takes a different argument
(:class:`~qdk_chemistry.data.BosonMapping` rather than
:class:`~qdk_chemistry.data.MajoranaMapping`) and answers a different
question. Keeping the types separate means one fixed ``_run_impl`` signature
per algorithm type, and leaves the behaviour of ``create("qubit_mapper", ...)``
completely unchanged.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Settings

if TYPE_CHECKING:  # Only needed for type annotations; avoid importing into module namespace
    from qdk_chemistry.data import BosonMapping, Hamiltonian, QubitOperator

__all__: list[str] = []


class BosonQubitMapperSettings(Settings):
    """Base settings for all BosonQubitMapper implementations.

    Settings are variant-specific (thresholds, etc.). The encoding and the
    occupation cutoff are determined by the
    :class:`~qdk_chemistry.data.BosonMapping` passed to
    :meth:`~qdk_chemistry.algorithms.BosonQubitMapper.run`.
    """

    def __init__(self) -> None:
        """Initialize BosonQubitMapperSettings."""
        super().__init__()


class BosonQubitMapper(Algorithm):
    """Abstract base class for mapping a bosonic Hamiltonian to a QubitOperator.

    A bosonic Hamiltonian is an ordinary
    :class:`~qdk_chemistry.data.Hamiltonian` whose orbital basis is a
    :class:`~qdk_chemistry.data.BosonicModes`; the occupation cutoff lives on
    that basis, not on the mapping. Backends read the cutoff from the supplied
    :class:`~qdk_chemistry.data.BosonMapping` and validate it against the
    Hamiltonian's basis, so a mismatch is a hard error rather than silently
    wrong physics.

    Because the cutoff is padded to a power of two, the encoded subspace is the
    full qubit Hilbert space: there is no unphysical subspace and therefore no
    leakage and no penalty term.
    """

    def __init__(self):
        """Initialize the BosonQubitMapper."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``boson_qubit_mapper`` as the algorithm type name.

        Returns:
            str: The algorithm type name.

        """
        return "boson_qubit_mapper"

    def run(
        self,
        hamiltonian: Hamiltonian,
        mapping: BosonMapping,
    ) -> QubitOperator:
        """Map a bosonic Hamiltonian to a qubit operator.

        Args:
            hamiltonian: The bosonic Hamiltonian, ideally carrying a ``BosonicModes`` basis.
            mapping: The boson-to-qubit encoding.

        Returns:
            QubitOperator: The mapped operator with encoding metadata set.

        """
        self._settings.lock()
        return self._run_impl(hamiltonian, mapping)

    @abstractmethod
    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: BosonMapping,
    ) -> QubitOperator:
        """Construct a QubitOperator from a bosonic Hamiltonian using the given mapping.

        Args:
            hamiltonian: The bosonic Hamiltonian.
            mapping: The boson-to-qubit encoding.

        Returns:
            QubitOperator: An instance of the QubitOperator.

        """


class BosonQubitMapperFactory(AlgorithmFactory):
    """Factory class for creating BosonQubitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return ``boson_qubit_mapper`` as the algorithm type name.

        Returns:
            str: The algorithm type name.

        """
        return "boson_qubit_mapper"

    def default_algorithm_name(self) -> str:
        """Return ``qdk`` as the default algorithm name.

        Returns:
            str: The default algorithm name.

        """
        return "qdk"
