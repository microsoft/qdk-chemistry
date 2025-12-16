"""QDK/Chemistry implemantation of the first order trotter constructor."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.time_evolution.constructor.base import TimeEvolutionConstructor
from qdk_chemistry.data import QubitHamiltonian, Settings, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    EvolutionOrdering,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

__all__: list[str] = []


class FirstOrderTrotterConstructorSettings(Settings):
    """Settings for first-order Trotterization unitary constructor.

    Attributes:
        num_trotter_steps (int): The number of Trotter steps to use in the construction.
        tolerence (float): The absolute tolerance for filtering small coefficients.

    """

    def __init__(self):
        """Initialize FirstOrderTrotterConstructorSettings with default values."""
        super().__init__()
        self._set_default("num_trotter_steps", "int", 1)
        self._set_default("tolerence", "float", 1e-12)


class FirstOrderTrotterConstructor(TimeEvolutionConstructor):
    """First-order Trotterization unitary constructor."""

    def __init__(self, num_trotter_steps: int = 1, tolerence: float = 1e-12):
        """Initialize FirstOrderTrotterConstructor with given settings.

        Args:
            num_trotter_steps (int): The number of Trotter steps to use in the construction.
            tolerence (float): The absolute tolerance for filtering small coefficients.

        """
        super().__init__()
        self._settings = FirstOrderTrotterConstructorSettings()
        self._settings.set("num_trotter_steps", num_trotter_steps)
        self._settings.set("tolerence", tolerence)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct the control unitary circuit using first-order Trotterization.

        Args:
            qubit_hamiltonian (QubitHamiltonian): The qubit Hamiltonian to be used in the construction.
            time (float): The total evolution time.

        Returns:
            QuantumCircuit: The constructed control unitary circuit.

        """
        # Calculate evolution time per Trotter step
        dt = time / self._settings.get("num_trotter_steps")
        tolerence = self._settings.get("tolerence")

        terms, ordering = _decompose_trotter_step(qubit_hamiltonian, time=dt, atol=tolerence)

        num_qubits = qubit_hamiltonian.num_qubits

        container = PauliProductFormulaContainer(
            step_terms=terms,
            evolution_ordering=ordering,
            step_reps=self._settings.get("num_trotter_steps"),
            num_qubits=num_qubits,
        )

        return TimeEvolutionUnitary(container=container)

    def name(self) -> str:
        """Return the name of the controlled unitary constructor."""
        return "first_order_trotter"


def _pauli_label_to_map(label: str) -> dict[int, str]:
    """Translate a Pauli label to a mapping ``qubit -> {X, Y, Z}``.

    Args:
        label: Pauli string label in little-endian ordering.

    Returns:
        Dictionary assigning each non-identity qubit index to its Pauli axis.

    """
    mapping: dict[int, str] = {}
    for index, char in enumerate(reversed(label)):  # reversed: right-most char -> qubit 0
        if char != "I":
            mapping[index] = char
    return mapping


def _decompose_trotter_step(
    qubit_hamiltonian: QubitHamiltonian, time: float, *, atol: float = 1e-12
) -> tuple[list[ExponentiatedPauliTerm], EvolutionOrdering]:
    """Decompose a single Trotter step into exponentiated Pauli terms.

    Args:
        qubit_hamiltonian: The qubit Hamiltonian to be decomposed.
        time: The evolution time for the single step.
        atol: Absolute tolerance for filtering small coefficients.

    Returns:
        A tuple containing:
            - A list of ``ExponentiatedPauliTerm`` representing the decomposed terms.
            - An ``EvolutionOrdering`` representing the sequence of evolution.

    """
    terms: list[ExponentiatedPauliTerm] = []

    for pauli, coeff in zip(
        qubit_hamiltonian.pauli_ops.paulis,
        qubit_hamiltonian.pauli_ops.coeffs,
        strict=True,
    ):
        if abs(coeff) < atol:
            continue

        if abs(coeff.imag) > atol:
            raise ValueError(
                f"Non-Hermitian Hamiltonian: coefficient {coeff} for term "
                f"{pauli.to_label()} has nonzero imaginary part."
            )

        mapping = _pauli_label_to_map(pauli.to_label())

        angle = float(coeff.real) * time
        terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

    ordering = EvolutionOrdering(indices=list(range(len(terms))))

    return terms, ordering
