"""QDK/Chemistry phase estimation builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    Settings,
)

__all__: list[str] = [
    "IterativeQpeCircuitBuilder",
    "QpeCircuitBuilder",
    "QpeCircuitBuilderFactory",
    "QpeCircuitBuilderSettings",
    "StandardQpeCircuitBuilder",
    "coherent_qpe_measured_indices",
    "split_coherent_qpe_bitstring",
]

#: Final-measurement policies understood by the standard QPE circuit builder.
MEASUREMENT_POLICIES: tuple[str, ...] = ("phase", "eigenvector", "none")


def qpe_measured_indices(phase_indices: list[int], trailing_indices: list[int]) -> list[int]:
    """Order register indices so the executor's key reads phase-register first.

    The circuit executor reverses the Q# ``Result[]`` when it forms a bitstring
    key. Emitting ``reversed(trailing) ++ phase`` therefore yields the key
    ``phase (most-significant-bit first) ++ trailing (register order)``, because
    the inverse QFT leaves the phase register least-significant-bit first.

    Args:
        phase_indices: Register indices of the phase qubits, in register order.
        trailing_indices: Register indices to append after the phase bits in the
            key, in register order.

    Returns:
        Register indices in the order the Q# entry point should measure them.

    """
    return list(reversed(trailing_indices)) + phase_indices


def coherent_qpe_measured_indices(
    num_bits: int,
    num_system_qubits: int,
    num_ancilla_qubits: int,
) -> list[int]:
    """Return the register indices an amplified QPE circuit should measure.

    The register is laid out as ``phase ++ system ++ unitary ancillas``. Only the
    phase register and the unitary (block-encoding signal) ancillas carry
    information; the system qubits are left to be reset. The resulting key is
    consumed by :func:`split_coherent_qpe_bitstring`.

    Args:
        num_bits: Number of phase qubits.
        num_system_qubits: Number of system qubits.
        num_ancilla_qubits: Number of block-encoding ancilla qubits (0 for Trotter).

    Returns:
        Register indices in the order the Q# entry point should measure them.

    """
    phase_indices = list(range(num_bits))
    ancilla_indices = [num_bits + num_system_qubits + index for index in range(num_ancilla_qubits)]
    return qpe_measured_indices(phase_indices, ancilla_indices)


def split_coherent_qpe_bitstring(bitstring: str, num_bits: int) -> tuple[str, str]:
    """Split an executor bitstring into its phase bits and signal-ancilla bits.

    Args:
        bitstring: A key produced by the circuit executor for a circuit built with
            :func:`coherent_qpe_measured_indices`.
        num_bits: Number of phase qubits.

    Returns:
        A tuple of (phase bits most-significant-bit first, signal ancilla bits).

    Raises:
        ValueError: If the bitstring is shorter than the phase register.

    """
    if len(bitstring) < num_bits:
        raise ValueError(f"Bitstring '{bitstring}' is shorter than the {num_bits}-qubit phase register.")
    return bitstring[:num_bits], bitstring[num_bits:]


class QpeCircuitBuilderSettings(Settings):
    """Settings for the Phase Estimation Builder algorithm."""

    def __init__(self):
        """Initialize the settings for the Phase Estimation Builder.

        Includes nested algorithm references for the evolution builder
        and the circuit mapper used to construct phase estimation circuits.

        """
        super().__init__()
        self._set_default("num_bits", "int", -1, "The number of phase bits to estimate.")
        self._set_default(
            "measurement",
            "string",
            "phase",
            "Final measurement: 'phase' measures the phase register in the computational "
            "basis, 'eigenvector' also measures the system register in 'measurement_basis', "
            "and 'none' measures nothing and returns a coherent, adjointable circuit.",
        )
        self._set_default(
            "measurement_basis",
            "string",
            "Z",
            "Pauli basis for the system register when measurement is 'eigenvector'. A single "
            "letter is broadcast to every system qubit; otherwise one letter per system qubit. "
            "'I' resets a qubit without recording a bit.",
        )
        self._set_default(
            "unitary_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
        )
        self._set_default(
            "controlled_circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )


class QpeCircuitBuilder(Algorithm):
    """Abstract base class for phase estimation circuit builders."""

    def __init__(
        self,
        num_bits: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
    ):
        """Initialize the QpeCircuitBuilder with default settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            unitary_builder: Optional algorithm reference for the unitary builder.
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.

        """
        super().__init__()
        self._settings = QpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if controlled_circuit_mapper is not None:
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

    def type_name(self) -> str:
        """Return the algorithm type name as qpe_circuit_builder."""
        return "qpe_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Build phase estimation circuits.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build circuits.

        Returns:
            A list of quantum circuits for phase estimation.

        """

    def _create_controlled_circuit(
        self,
        qubit_hamiltonian: QubitOperator,
        power: int,
    ) -> tuple[Circuit, int]:
        r"""Create the controlled circuit for the given Hamiltonian and power.

        Sets the ``power`` on the unitary builder so it produces :math:`U^{\\text{power}}`
        according to its ``power_strategy``, then maps the result to a controlled circuit.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            power: The power to which the unitary should be raised.

        Returns:
            A tuple of (circuit, num_ancilla_qubits) where circuit implements
            controlled-:math:`U^{\\text{power}}` and num_ancilla_qubits is the number
            of ancilla qubits used by the unitary beyond the system qubits.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_builder.settings().update("power", power)
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        num_ancilla_qubits = unitary_rep.get_num_qubits() - qubit_hamiltonian.num_qubits
        circuit_mapper = self._create_nested("controlled_circuit_mapper")
        circuit_mapper.settings().update("control_indices", [0])
        circuit = circuit_mapper.run(unitary_rep)
        return circuit, num_ancilla_qubits

    def measurement_policy(self) -> str:
        """Return the validated ``measurement`` setting.

        Returns:
            One of ``"phase"``, ``"eigenvector"`` or ``"none"``.

        Raises:
            ValueError: If the setting holds an unknown policy.

        """
        policy = str(self._settings.get("measurement"))
        if policy not in MEASUREMENT_POLICIES:
            raise ValueError(f"measurement must be one of {list(MEASUREMENT_POLICIES)}. Got '{policy}'.")
        return policy

    def measurement_plan(
        self,
        num_bits: int,
        num_system_qubits: int,
    ) -> tuple[list[int], list[str]]:
        """Resolve the ``measurement`` setting into register indices and Pauli bases.

        The register is laid out as ``phase ++ system ++ unitary ancillas``. No
        policy reads the block-encoding ancillas, so only the phase and system
        registers appear here. The returned lists are aligned element-wise and
        ordered so that the executor's bitstring key reads phase register
        most-significant-bit first.

        Args:
            num_bits: Number of phase qubits.
            num_system_qubits: Number of system qubits.

        Returns:
            A tuple of (register indices to measure, Pauli letter per index).

        Raises:
            ValueError: If ``measurement_basis`` does not match the system register.

        """
        policy = self.measurement_policy()
        if policy == "none":
            return [], []

        phase_indices = list(range(num_bits))
        if policy == "phase":
            return phase_indices, ["Z"] * num_bits

        basis = str(self._settings.get("measurement_basis")).upper()
        if len(basis) == 1:
            basis *= num_system_qubits
        if len(basis) != num_system_qubits:
            raise ValueError(
                f"measurement_basis must be a single Pauli letter or one letter per system "
                f"qubit ({num_system_qubits}). Got '{basis}'."
            )
        if any(letter not in "IXYZ" for letter in basis):
            raise ValueError(f"measurement_basis must only contain the letters I, X, Y or Z. Got '{basis}'.")

        system_indices = [num_bits + index for index in range(num_system_qubits)]
        indices = qpe_measured_indices(phase_indices, system_indices)
        bases = [basis[index] for index in reversed(range(num_system_qubits))] + ["Z"] * num_bits
        return indices, bases


class QpeCircuitBuilderFactory(AlgorithmFactory):
    """Factory class for creating QpeCircuitBuilder instances."""

    def __init__(self):
        """Initialize the QpeCircuitBuilderFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as qpe_circuit_builder."""
        return "qpe_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return qdk_iterative as default algorithm name."""
        return "qdk_iterative"


class IterativeQpeCircuitBuilder(QpeCircuitBuilder):
    """Abstract base class for iterative phase estimation circuit builders.

    Serves as a type-checking abstraction for implementations of the iterative
    (Kitaev-style) quantum phase estimation algorithm.

    """


class StandardQpeCircuitBuilder(QpeCircuitBuilder):
    """Abstract base class for standard (QFT-based) phase estimation circuit builders.

    Serves as a type-checking abstraction for implementations of the standard
    (non-iterative) quantum phase estimation algorithm using QFT.

    """
