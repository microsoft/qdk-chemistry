r"""QDK/Chemistry implementation of the LCU (Linear Combination of Unitaries).

References:
    Childs, A. M. and Wiebe, N. "Hamiltonian simulation using linear
    combinations of unitary operations." *Quantum Information &
    Computation* 12.11-12 (2012): 901-924.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    HamiltonianUnitaryBuilder,
    HamiltonianUnitaryBuilderSettings,
)
from qdk_chemistry.data import QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.block_encoding import (
    BlockEncodingContainer,
    ControlledOperation,
    Prepare,
    Select,
)

__all__: list[str] = ["BlockEncodingBuilder", "BlockEncodingSettings"]


class BlockEncodingSettings(HamiltonianUnitaryBuilderSettings):
    """Settings for the block encoding builder."""

    def __init__(self):
        """Initialize BlockEncodingSettings with default values.

        Attributes:
            power: The power to which the Hamiltonian is raised.
            quantum_walk: If True, wrap block encoding with quantum walk operator (use with QPE).
                If False, use plain block encoding (use with Hadamard test).

        """
        super().__init__()
        self._set_default("power", "int", 1, "The power to which the Hamiltonian is raised.")
        self._set_default(
            "quantum_walk",
            "bool",
            False,
            "If True, wrap block encoding with quantum walk operator (use with QPE). "
            "If False, use plain block encoding (use with Hadamard test).",
        )


class BlockEncodingBuilder(HamiltonianUnitaryBuilder):
    r"""LCU (Linear Combination of Unitaries) block encoding builder."""

    def __init__(
        self,
        power: int = 1,
        quantum_walk: bool = False,
    ):
        r"""Initialize the LCU builder.

        Given a qubit Hamiltonian :math:`H = \sum_{j=1}^{L} \alpha_j P_j` expressed as a
        linear combination of Pauli strings :math:`P_j` with scalar coefficients
        :math:`\alpha_j`, this builder constructs an LCU representation that block-encodes
        :math:`H / \lambda`, where :math:`\lambda` is the Schatten norm (L1 norm) of the
        Hamiltonian. The LCU representation follows the PREPARE-SELECT-PREPARE† pattern.

        The PREPARE oracle encodes amplitudes :math:`\sqrt{|\alpha_j| / \lambda}` into
        an ancilla register of :math:`\lceil \log_2 L \rceil` qubits. The SELECT oracle applies
        the corresponding Pauli string :math:`P_j` (with sign correction) controlled on the
        ancilla index. The :math:`\alpha_j` is positive by absorbing the sign into phases of
        the SELECT operation.

        Args:
            power: The power to raise the unitary to. Defaults to 1.
            quantum_walk: If True, the circuit mapper wraps the block encoding with
                a quantum walk operator (use with QPE). If False, use the plain
                block encoding (use with Hadamard test). Defaults to False.

        """
        super().__init__()
        self._settings = BlockEncodingSettings()
        self._settings.set("power", power)
        self._settings.set("quantum_walk", quantum_walk)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct the unitary representation using LCU block encoding.

        Computes normalized amplitudes, signs, and controlled operations from the
        qubit Hamiltonian, then packages them into generalized Prepare/Select/Reflect
        dataclasses stored in an LCUContainer.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.

        Returns:
            UnitaryRepresentation: The unitary representation built for the LCU block encoding.

        """
        power: int = self._settings.get("power")
        quantum_walk: bool = self._settings.get("quantum_walk")
        coefficients = qubit_hamiltonian.coefficients
        num_terms = len(coefficients)
        num_prepare_qubits = int(np.ceil(np.log2(num_terms)))

        prepare = self._build_prepare(qubit_hamiltonian, num_prepare_qubits)
        select = self._build_select(qubit_hamiltonian, num_prepare_qubits)

        container = BlockEncodingContainer(
            power=power,
            prepare=prepare,
            select=select,
            reflect=quantum_walk,
        )

        return UnitaryRepresentation(container=container)

    @staticmethod
    def _build_prepare(qubit_hamiltonian: QubitHamiltonian, num_prepare_qubits: int) -> Prepare:
        """Compute the PREPARE statevector from Hamiltonian coefficients."""
        coefficients = qubit_hamiltonian.coefficients
        schatten_norm = qubit_hamiltonian.schatten_norm
        if schatten_norm < 1e-15:
            raise ValueError("Schatten norm is too small, cannot build LCU block encoding.")

        abs_coeffs = np.abs(coefficients)
        statevector = np.sqrt(abs_coeffs / schatten_norm)

        return Prepare(
            statevector=statevector,
            num_prepare_qubits=num_prepare_qubits,
            prepare_qubits=list(range(num_prepare_qubits)),
        )

    @staticmethod
    def _build_select(qubit_hamiltonian: QubitHamiltonian, num_prepare_qubits: int) -> Select:
        """Compute SELECT controlled operations and phases from Hamiltonian terms."""
        coefficients = qubit_hamiltonian.coefficients
        pauli_strings = qubit_hamiltonian.pauli_strings
        num_terms = len(coefficients)
        num_system_qubits = qubit_hamiltonian.num_qubits
        phases = np.where(coefficients >= 0, 1, -1)

        controlled_ops = [
            ControlledOperation(
                ctrl_state=i,
                operation=pauli_strings[i],
            )
            for i in range(num_terms)
        ]

        return Select(
            controlled_operations=controlled_ops,
            phases=phases,
            num_prepare_qubits=num_prepare_qubits,
            num_target_qubits=num_system_qubits,
            prepare_qubits=list(range(num_prepare_qubits)),
            target_qubits=list(range(num_prepare_qubits, num_prepare_qubits + num_system_qubits)),
        )

    def name(self) -> str:
        """Return the algorithm name."""
        return "block_encoding"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
