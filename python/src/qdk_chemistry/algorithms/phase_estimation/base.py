"""QDK/Chemistry phase estimation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.hamiltonian_input import (
    HamiltonianInput,
    describe_hamiltonian_input,
    is_lattice_hamiltonian,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    Settings,
    UnitaryRepresentation,
)

__all__: list[str] = ["PhaseEstimation", "PhaseEstimationFactory", "PhaseEstimationSettings"]


class PhaseEstimationSettings(Settings):
    """Settings for the Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Phase Estimation.

        Includes nested algorithm references for the circuit builder
        and circuit executor.

        """
        super().__init__()
        self._set_default(
            "qpe_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("qpe_circuit_builder", "qdk_iterative"),
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )


class PhaseEstimation(Algorithm):
    """Abstract base class for phase estimation algorithms.

    Phase estimation accepts either a **qubit Hamiltonian**
    (:class:`~qdk_chemistry.data.QubitOperator`) or an unmapped **lattice Hamiltonian**
    (:class:`~qdk_chemistry.data.Hamiltonian`). Whichever form is given is handed
    straight to the nested unitary builder, so the two must agree: a lattice Hamiltonian
    requires a lattice-aware builder such as ``"plaquette"``.

    """

    def __init__(self):
        """Initialize the PhaseEstimation with default settings."""
        super().__init__()
        self._settings = PhaseEstimationSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    def _build_unitary(self, unitary_builder, hamiltonian: HamiltonianInput) -> UnitaryRepresentation:
        """Build the unitary representation, reporting a lattice/qubit input mismatch clearly.

        Args:
            unitary_builder: The nested Hamiltonian unitary builder.
            hamiltonian: The lattice or qubit Hamiltonian to represent.

        Returns:
            UnitaryRepresentation: The representation the builder produced.

        Raises:
            TypeError: If the builder cannot consume the input form it was given.

        """
        try:
            return unitary_builder.run(hamiltonian)
        except TypeError as error:
            if is_lattice_hamiltonian(hamiltonian):
                remedy = (
                    "Select a lattice-aware unitary builder such as 'plaquette', or map the "
                    "Hamiltonian to qubits first with a 'qubit_mapper'."
                )
            else:
                remedy = "Pass a Hamiltonian in the form the builder expects."
            raise TypeError(
                f"{self.name()!r} phase estimation was given a {describe_hamiltonian_input(hamiltonian)}, "
                f"which the nested {unitary_builder.name()!r} unitary builder cannot consume. {remedy}"
            ) from error

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: HamiltonianInput,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        r"""Run the phase estimation algorithm with the given state preparation circuit and Hamiltonian.

        This method implements the quantum phase estimation procedure:
        1. The state preparation circuit initializes the system in the desired quantum state.
        2. The unitary_builder constructs a unitary from the Hamiltonian it is given.
        3. The circuit_mapper transforms the unitary into controlled-U operations,
           where the control qubits are ancilla qubits used for phase readout.
        4. The circuit_executor runs the resulting quantum circuits on the target backend.
        5. Measurement results are processed to extract the eigenvalue phase estimates.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The lattice or qubit Hamiltonian for which to estimate eigenvalues.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the estimated phases and associated metadata.

        """


class PhaseEstimationFactory(AlgorithmFactory):
    """Factory class for creating PhaseEstimation instances."""

    def __init__(self):
        """Initialize the PhaseEstimationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    def default_algorithm_name(self) -> str:
        """Return the qdk_iterative as default algorithm name."""
        return "qdk_iterative"
