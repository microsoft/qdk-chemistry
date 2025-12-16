"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest

from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    EvolutionOrdering,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


@pytest.fixture
def step_terms():
    """Create a list of ExponentiatedPauliTerm instances for testing."""
    return [
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
        ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=1.2),
        ExponentiatedPauliTerm(pauli_term={0: "Y", 1: "X"}, angle=0.3),
    ]


@pytest.fixture
def valid_ordering():
    """Create a valid EvolutionOrdering instance for testing."""
    return EvolutionOrdering(indices=[2, 0, 1])


@pytest.fixture
def container(step_terms, valid_ordering):
    """Create a PauliProductFormulaContainer instance for testing."""
    return PauliProductFormulaContainer(
        step_terms=step_terms,
        evolution_ordering=valid_ordering,
        step_reps=4,
        num_qubits=2,
    )


class TestExponentiatedPauliTerm:
    """Tests for the ExponentiatedPauliTerm dataclass."""

    def test_attributes(self):
        """Test the attributes of ExponentiatedPauliTerm."""
        term = ExponentiatedPauliTerm(pauli_term={0: "X", 2: "Z"}, angle=1.57)

        assert term.pauli_term == {0: "X", 2: "Z"}
        assert np.isclose(
            term.angle, 1.57, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_frozen(self):
        """Test that ExponentiatedPauliTerm is immutable."""
        term = ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1)
        with pytest.raises(Exception, match="cannot be assigned to"):
            term.angle = 0.2


class TestEvolutionOrdering:
    """Tests for the EvolutionOrdering dataclass."""

    def test_valid_ordering(self):
        """Test a valid evolution ordering."""
        ordering = EvolutionOrdering(indices=[1, 0, 2])
        ordering.validate_ordering(num_terms=3)

    def test_invalid_length(self):
        """Test an evolution ordering with invalid length."""
        ordering = EvolutionOrdering(indices=[0, 1])
        with pytest.raises(ValueError, match="length must match"):
            ordering.validate_ordering(num_terms=3)

    def test_invalid_permutation(self):
        """Test an evolution ordering that is not a valid permutation."""
        ordering = EvolutionOrdering(indices=[0, 0, 2])
        with pytest.raises(ValueError, match="permutation"):
            ordering.validate_ordering(num_terms=3)


class TestPauliProductFormulaContainer:
    """Tests for the PauliProductFormulaContainer class."""

    def test_basic_properties(self, container):
        """Test basic properties of the container."""
        assert container.type == "pauli_product_formula"
        assert container.num_qubits == 2
        assert container.step_reps == 4
        assert len(container.step_terms) == 3
        assert container.evolution_ordering.indices == [2, 0, 1]

    def test_set_ordering(self, container):
        """Test setting a new valid evolution ordering."""
        new_ordering = EvolutionOrdering(indices=[1, 2, 0])
        container.set_ordering(new_ordering)

        assert container.evolution_ordering.indices == [1, 2, 0]

    def test_set_ordering_invalid(self, container):
        """Test setting an invalid evolution ordering."""
        bad_ordering = EvolutionOrdering(indices=[0, 1])
        with pytest.raises(ValueError, match="length must match"):
            container.set_ordering(bad_ordering)

    def test_to_json_roundtrip(self, container):
        """Test JSON serialization and deserialization roundtrip."""
        json_data = container.to_json()
        restored = PauliProductFormulaContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
        assert restored.evolution_ordering.indices == container.evolution_ordering.indices

        for t1, t2 in zip(restored.step_terms, container.step_terms, strict=False):
            assert t1.pauli_term == t2.pauli_term
            assert np.isclose(
                t1.angle, t2.angle, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )

    def test_to_hdf5_roundtrip(self, container, tmp_path):
        """Test HDF5 serialization and deserialization roundtrip."""
        file_path = tmp_path / "ppf_container.h5"

        with h5py.File(file_path, "w") as f:
            grp = f.create_group("container")
            container.to_hdf5(grp)

        with h5py.File(file_path, "r") as f:
            restored = PauliProductFormulaContainer.from_hdf5(f["container"])

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
        assert restored.evolution_ordering.indices == container.evolution_ordering.indices
        assert len(restored.step_terms) == len(container.step_terms)

    def test_summary(self, container):
        """Test the summary generation of the container."""
        summary = container.get_summary()

        assert "Pauli Product Formula Container" in summary
        assert "Number of qubits: 2" in summary
        assert "Number of step terms: 3" in summary
        assert "Step repetitions: 4" in summary
