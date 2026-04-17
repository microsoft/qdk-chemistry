"""QDK/Chemistry unitary builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian, UnitaryRepresentation

__all__: list[str] = [
    "UnitaryBuilder",
    "UnitaryBuilderFactory",
    "TimeEvolutionBuilder",
    "TimeEvolutionBuilderFactory",
]


class UnitaryBuilder(Algorithm):
    """Base class for unitary builders in QDK/Chemistry algorithms.

    A unitary builder constructs a UnitaryRepresentation from a QubitHamiltonian.
    Subclasses implement specific models (e.g. time evolution, block encoding).
    """

    def __init__(self):
        """Initialize the UnitaryBuilder."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, *args, **kwargs) -> UnitaryRepresentation:
        """Construct a UnitaryRepresentation for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.
            *args: Additional positional arguments for concrete implementations.
            **kwargs: Additional keyword arguments for concrete implementations.

        Returns:
            UnitaryRepresentation: A UnitaryRepresentation for the given QubitHamiltonian.

        """

    @abstractmethod
    def phase_to_energy(self, phase_fraction: float, **model_params) -> float:
        """Convert a measured phase fraction to energy.

        The conversion is model-dependent: time-evolution uses ``E = angle / t``,
        while block encoding would use ``E = cos(angle) * lambda_norm``, etc.

        Args:
            phase_fraction: Fractional phase obtained from the phase register.
            **model_params: Model-specific parameters (e.g. ``time`` for time evolution).

        Returns:
            Energy estimate corresponding to ``phase_fraction``.

        """

    @abstractmethod
    def energy_alias_candidates(self, raw_energy: float, **model_params) -> list[float]:
        """Enumerate alias energies compatible with ``raw_energy``.

        Args:
            raw_energy: Energy derived from the measured phase.
            **model_params: Model-specific parameters.

        Returns:
            Sorted list of alias energy values.

        """

    def resolve_energy(self, raw_energy: float, reference_energy: float, **model_params) -> float:
        """Select the alias energy closest to a known reference value.

        Args:
            raw_energy: Energy derived from the measured phase.
            reference_energy: External reference guiding alias selection.
            **model_params: Model-specific parameters.

        Returns:
            Alias energy closest to ``reference_energy``.

        """
        candidates = self.energy_alias_candidates(raw_energy, **model_params)
        return min(candidates, key=lambda energy: abs(energy - reference_energy))

    # ------------------------------------------------------------------
    # Shared helpers used by Trotter, qDRIFT, and partially-randomized
    # builders.
    # ------------------------------------------------------------------

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


class TimeEvolutionBuilder(UnitaryBuilder):
    """Base class for time evolution builders in QDK/Chemistry algorithms.

    Time evolution builders construct unitaries of the form ``U = exp(-i H t)``.
    """

    def __init__(self):
        """Initialize the TimeEvolutionBuilder."""
        super().__init__()

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> UnitaryRepresentation:
        """Construct a UnitaryRepresentation representing the time evolution unitary for the given QubitHamiltonian.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian.
            time: The evolution time.

        Returns:
            UnitaryRepresentation: A UnitaryRepresentation representing the evolution of the given QubitHamiltonian.

        """

    def phase_to_energy(self, phase_fraction: float, **model_params) -> float:
        """Convert a measured phase fraction to energy using ``E = angle / t``.

        Args:
            phase_fraction: Fractional phase obtained from the phase register.
            **model_params: Must include ``time`` (evolution time).

        Returns:
            Energy estimate corresponding to ``phase_fraction``.

        """
        from qdk_chemistry.utils.phase import energy_from_phase  # noqa: PLC0415

        time = model_params.get("time", model_params.get("evolution_time"))
        if time is None:
            raise ValueError("TimeEvolutionBuilder.phase_to_energy requires 'time' or 'evolution_time' parameter.")
        return energy_from_phase(phase_fraction, evolution_time=time)

    def energy_alias_candidates(self, raw_energy: float, **model_params) -> list[float]:
        """Enumerate alias energies compatible with ``raw_energy``.

        Args:
            raw_energy: Energy derived from the measured phase.
            **model_params: Must include ``time`` (evolution time).

        Returns:
            Sorted list of alias energy values.

        """
        from qdk_chemistry.utils.phase import energy_alias_candidates as _energy_alias_candidates  # noqa: PLC0415

        time = model_params.get("time", model_params.get("evolution_time"))
        if time is None:
            raise ValueError(
                "TimeEvolutionBuilder.energy_alias_candidates requires 'time' or 'evolution_time' parameter."
            )
        shift_range = model_params.get("shift_range", range(-2, 3))
        return _energy_alias_candidates(raw_energy, evolution_time=time, shift_range=shift_range)


class UnitaryBuilderFactory(AlgorithmFactory):
    """Factory class for creating UnitaryBuilder instances."""

    def algorithm_type_name(self) -> str:
        """Return unitary_builder as the algorithm type name."""
        return "unitary_builder"

    def default_algorithm_name(self) -> str:
        """Return Trotter as the default algorithm name."""
        return "trotter"


# Backward-compatible alias
TimeEvolutionBuilderFactory = UnitaryBuilderFactory
