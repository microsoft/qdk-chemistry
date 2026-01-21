"""Qiskit-based qubit mappers to map electronic structure Hamiltonians to qubit Hamiltonians.

This module provides a QiskitQubitMapper class to convert Hamiltonians to QubitHamiltonians
using different mapping strategies ("jordan-wigner", "bravyi-kitaev", and "parity").
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import ClassVar

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)

from qdk_chemistry.algorithms.qubit_mapper import QubitMapper
from qdk_chemistry.data import Hamiltonian, QubitHamiltonian, Settings
from qdk_chemistry.utils import Logger

__all__ = ["QiskitQubitMapper", "QiskitQubitMapperSettings"]


class QiskitQubitMapperSettings(Settings):
    """Settings configuration for a QiskitQubitMapper.

    QiskitQubitMapper-specific settings:
        encoding (string, default="jordan-wigner"): Qubit mapping strategy to use.

            Valid options: "jordan-wigner", "bravyi-kitaev", "parity"

    """

    def __init__(self):
        """Initialize QiskitQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "encoding",
            "string",
            "jordan-wigner",
            "Qubit mapping strategy to use",
            ["jordan-wigner", "bravyi-kitaev", "parity"],
        )


class QiskitQubitMapper(QubitMapper):
    """Class to map an electronic structure Hamiltonian to a QubitHamiltonian using a Qiskit mapper."""

    QubitMappers: ClassVar = {
        "bravyi-kitaev": BravyiKitaevMapper,
        "jordan-wigner": JordanWignerMapper,
        "parity": ParityMapper,
    }

    def __init__(self, encoding: str = "jordan-wigner"):
        """Initialize QiskitQubitMapper with a specific mapping strategy.

        Args:
            encoding (str): Qubit mapping strategy to use ("jordan-wigner", "bravyi-kitaev", or "parity").
                Default: "jordan-wigner".

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QiskitQubitMapperSettings()
        self._settings.set("encoding", encoding)

    def _run_impl(self, hamiltonian: Hamiltonian) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the selected mapping strategy.

        Args:
            hamiltonian (Hamiltonian): The fermionic Hamiltonian.

        Returns:
            QubitHamiltonian: An instance of the QubitHamiltonian.

        """
        Logger.trace_entering()
        encoding = self._settings.get("encoding")
        if encoding not in self.QubitMappers:
            raise ValueError(
                f"Encoding {encoding} is unknown for QiskitQubitMapper.\n"
                f"Please use one of the following options: {self.QubitMappers.keys()}"
            )

        (h1_a, _) = hamiltonian.get_one_body_integrals()
        (h2_aa, _, _) = hamiltonian.get_two_body_integrals()
        num_orbs = len(hamiltonian.get_orbitals().get_active_space_indices()[0])
        electronic_hamiltonian = ElectronicEnergy.from_raw_integrals(
            h1_a=h1_a, h2_aa=h2_aa.reshape(num_orbs, num_orbs, num_orbs, num_orbs)
        )
        fermionic_op = electronic_hamiltonian.second_q_op()
        qubit_mapper = self.QubitMappers[encoding]()
        qubit_op = qubit_mapper.map(fermionic_op)
        return QubitHamiltonian(
            pauli_strings=qubit_op.paulis.to_labels(),
            coefficients=qubit_op.coeffs,
            encoding=encoding,
        )

    def name(self) -> str:
        """Return the algorithm name ``qiskit``."""
        Logger.trace_entering()
        return "qiskit"
