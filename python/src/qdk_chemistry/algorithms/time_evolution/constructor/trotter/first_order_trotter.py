"""QDK/Chemistry implementation of the first order trotter constructor."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.time_evolution.constructor.base import TimeEvolutionConstructor
from qdk_chemistry.data import QubitHamiltonian, Settings, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

__all__: list[str] = ["FirstOrderTrotter", "FirstOrderTrotterSettings"]


class FirstOrderTrotterSettings(Settings):
    """Settings for first-order Trotterization unitary constructor."""

    def __init__(self):
        """Initialize FirstOrderTrotterSettings with default values.

        Attributes:
            num_trotter_steps: The number of Trotter steps to use in the construction.
            tolerance: The absolute tolerance for filtering small coefficients.

        """
        super().__init__()
        self._set_default("num_trotter_steps", "int", 1, "The number of Trotter steps.")
        self._set_default("tolerance", "float", 1e-12, "The absolute tolerance for filtering small coefficients.")


class FirstOrderTrotter(TimeEvolutionConstructor):
    """First-order Trotterization unitary constructor."""

    def __init__(self, num_trotter_steps: int = 1, tolerance: float = 1e-12):
        r"""Initialize FirstOrderTrotter with specified Trotter decomposition parameters.

        The First Order Trotter method approximates the time evolution operator :math:`e^{-iHt}`
        by decomposing the Hamiltonian H into a sum of terms and using the product formula:
        :math:`e^{-iHt} \approx \left[\prod_i e^{-iH_i t/n}\right]^n`, where n is the number of Trotter steps.

        Args:
            num_trotter_steps: Number of Trotter steps for the decomposition. Higher values improve accuracy
                but increase circuit depth. Defaults to 1.
            tolerance: Absolute threshold for filtering small Hamiltonian coefficients. Defaults to 1e-12.

        """
        super().__init__()
        self._settings = FirstOrderTrotterSettings()
        self._settings.set("num_trotter_steps", num_trotter_steps)
        self._settings.set("tolerance", tolerance)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct the time evolution unitary using first-order Trotterization.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.

        Returns:
            TimeEvolutionUnitary: The constructed time evolution unitary.

        """
        # Calculate evolution time per Trotter step
        dt = time / self._settings.get("num_trotter_steps")
        tolerance = self._settings.get("tolerance")

        terms = self._decompose_trotter_step(qubit_hamiltonian, time=dt, atol=tolerance)

        num_qubits = qubit_hamiltonian.num_qubits

        container = PauliProductFormulaContainer(
            step_terms=terms,
            step_reps=self._settings.get("num_trotter_steps"),
            num_qubits=num_qubits,
        )

        return TimeEvolutionUnitary(container=container)

    @staticmethod
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
        self, qubit_hamiltonian: QubitHamiltonian, time: float, *, atol: float = 1e-12
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose a single Trotter step into exponentiated Pauli terms.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be decomposed.
            time: The evolution time for the single step.
            atol: Absolute tolerance for filtering small coefficients.

        Returns:
        A list of ``ExponentiatedPauliTerm`` representing the decomposed terms.

        """
        terms: list[ExponentiatedPauliTerm] = []

        for pauli, coeff in zip(
            qubit_hamiltonian.pauli_ops.paulis,
            qubit_hamiltonian.pauli_ops.coeffs,
            strict=True,
        ):
            if abs(coeff) < atol:
                continue

            coeff_complex = complex(coeff)
            if abs(coeff_complex.imag) > atol:
                raise ValueError(
                    f"Non-Hermitian Hamiltonian: coefficient {coeff} for term "
                    f"{pauli.to_label()} has nonzero imaginary part."
                )
            mapping = self._pauli_label_to_map(pauli.to_label())

            angle = coeff_complex.real * time
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

        return terms

    def name(self) -> str:
        """Return the name of the time evolution unitary constructor."""
        return "first_order_trotter"

    def type_name(self) -> str:
        """Return time_evolution_constructor as the algorithm type name."""
        return "time_evolution_constructor"
