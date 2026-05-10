"""Qiskit-based qubit mappers to map electronic structure Hamiltonians to qubit Hamiltonians.

This module provides a QiskitQubitMapper class to convert Hamiltonians to QubitHamiltonians
using different mapping strategies ("jordan-wigner", "bravyi-kitaev", and "parity").
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)

from qdk_chemistry.algorithms.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data import Hamiltonian, QubitHamiltonian
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Symmetries
    from qdk_chemistry.data.majorana_mapping import MajoranaMapping

__all__ = ["QiskitQubitMapper", "QiskitQubitMapperSettings"]

_SUPPORTED_ENCODINGS: dict[str, type] = {
    "jordan-wigner": JordanWignerMapper,
    "bravyi-kitaev": BravyiKitaevMapper,
    "parity": ParityMapper,
}


class QiskitQubitMapperSettings(QubitMapperSettings):
    """Settings configuration for a QiskitQubitMapper."""

    def __init__(self):
        """Initialize QiskitQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()


class QiskitQubitMapper(QubitMapper):
    """Map an electronic structure Hamiltonian to a QubitHamiltonian using Qiskit.

    The encoding is determined by the :class:`~qdk_chemistry.data.MajoranaMapping`
    passed to :meth:`run`. The plugin uses ``mapping.name`` to select the
    corresponding Qiskit Nature mapper. Custom (unnamed) mappings are not
    supported — use the QDK variant instead.

    Supported ``mapping.name`` values:
        - ``"jordan-wigner"``
        - ``"bravyi-kitaev"``
        - ``"parity"``

    """

    QubitMappers: ClassVar = _SUPPORTED_ENCODINGS

    def __init__(self):
        """Initialize QiskitQubitMapper."""
        Logger.trace_entering()
        super().__init__()
        self._settings = QiskitQubitMapperSettings()

    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
        symmetries: Symmetries | None = None,  # noqa: ARG002
    ) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the selected mapping strategy.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            mapping: The Majorana-to-Pauli encoding. Only built-in encodings are supported.
            symmetries: Optional symmetry information. Not used by this implementation.

        Returns:
            QubitHamiltonian: An instance of the QubitHamiltonian.

        Raises:
            NotImplementedError: If ``mapping.name`` is not a supported Qiskit encoding.

        """
        Logger.trace_entering()
        encoding_name = mapping.name

        if encoding_name not in _SUPPORTED_ENCODINGS:
            raise NotImplementedError(
                f"Qiskit plugin does not support MajoranaMapping with name {encoding_name!r}. "
                f"Supported names: {sorted(_SUPPORTED_ENCODINGS.keys())}. "
                f"Use the QDK variant for custom mappings."
            )

        (h1_a, _) = hamiltonian.get_one_body_integrals()
        (h2_aa, _, _) = hamiltonian.get_two_body_integrals()
        num_orbs = len(hamiltonian.get_orbitals().get_active_space_indices()[0])
        electronic_hamiltonian = ElectronicEnergy.from_raw_integrals(
            h1_a=h1_a, h2_aa=h2_aa.reshape(num_orbs, num_orbs, num_orbs, num_orbs)
        )
        fermionic_op = electronic_hamiltonian.second_q_op()
        qubit_mapper = _SUPPORTED_ENCODINGS[encoding_name]()
        qubit_op = qubit_mapper.map(fermionic_op)
        return QubitHamiltonian(
            pauli_strings=qubit_op.paulis.to_labels(),
            coefficients=qubit_op.coeffs,
            encoding=encoding_name,
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )

    def name(self) -> str:
        """Return the algorithm name ``qiskit``."""
        Logger.trace_entering()
        return "qiskit"
