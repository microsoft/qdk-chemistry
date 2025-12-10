"""QDK/Chemistry time evolution pauli product formula container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitaryContainer


@dataclass(frozen=True)
class ExponentiatedPauliTerm:
    """Dataclass for an exponentiated Pauli term.

    Attributes:
        pauli_term: A dictionary mapping qubit indices to Pauli operators ('X', 'Y', 'Z').
        angle: The rotation angle for the exponentiation.

    Example:
        >>> term = ExponentiatedPauliTerm(pauli_term={0: 'X', 2: 'Z'}, angle=1.5708)

    """

    pauli_term: dict[int, str]
    angle: float


@dataclass(frozen=True)
class EvolutionOrdering:
    """Data class for evolution ordering.

    Attributes:
        indices: List of indices representing the order of evolution.

    Example:
        >>> # Apply the third term first, then the first, and the second lastly
        >>> ordering = EvolutionOrdering(indices=[2, 0, 1])

    """

    indices: list[int]

    def validate_ordering(self, num_terms: int) -> None:
        """Validate the evolution ordering.

        Args:
            num_terms: The total number of terms to be ordered.

        Raises:
            ValueError: If the ordering is invalid.

        """
        if len(self.indices) != num_terms:
            raise ValueError("Evolution ordering length must match the number of terms.")

        if sorted(self.indices) != list(range(num_terms)):
            raise ValueError("Evolution ordering must be a permutation of term indices.")


class PauliProductFormulaContainer(TimeEvolutionUnitaryContainer):
    """Data class for a Pauli product formula container."""

    # Class attribute for filename validation
    _data_type_name = "pauli_product_formula_container"

    def __init__(
        self, step_terms: list[ExponentiatedPauliTerm], evolution_ordering: EvolutionOrdering, step_reps: int
    ) -> None:
        """Initialize a PauliProductFormulaContainer.

        Args:
            step_terms: The list of exponentiated Pauli terms in a single step.
            evolution_ordering: A list of indices representing the order of evolution terms.
            step_reps: The number of repetitions of the single step.

        """
        super().__init__()
        self.step_terms = step_terms
        evolution_ordering.validate_ordering(len(step_terms))
        self.evolution_ordering = evolution_ordering
        self.step_reps = step_reps

    @property
    def type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary container.

        """
        return "pauli_product_formula"

    def set_ordering(self, evolution_ordering: EvolutionOrdering) -> None:
        """Set the evolution ordering.

        Args:
            evolution_ordering: A list of indices representing the order of evolution terms.

        """
        evolution_ordering.validate_ordering(len(self.step_terms))
        self.evolution_ordering = evolution_ordering

    def apply(self, state: Wavefunction) -> Wavefunction:
        """Apply the time evolution unitary to a given state.

        Args:
            state: The state to which the unitary is applied.

        Returns:
            The new state after applying the unitary.

        """
        # Placeholder implementation
        new_state = state
        for index in self.evolution_ordering.indices:
            term = self.terms[index]
            new_state = _apply_exponentiated_pauli_to_wavefunction(new_state, term)
        return new_state


def _apply_exponentiated_pauli_to_wavefunction(
    wavefunction: Wavefunction, term: ExponentiatedPauliTerm
) -> Wavefunction:
    """Apply an exponentiated Pauli term to a wavefunction.

    Args:
        wavefunction: The input wavefunction.
        term: The exponentiated Pauli term to apply.

    Returns:
        The new wavefunction after applying the term.

    """
    # TODO: Implement when bistring and statevector refacotring is done
