"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Sequence

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import SparsePauliProductFormulaContainer, SparsePauliTerms
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
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


@pytest.fixture(params=[False, True], ids=["legacy", "sparse-factory"])
def container(step_terms, request):
    """Create a PauliProductFormulaContainer instance for testing."""
    if request.param:
        return SparsePauliProductFormulaContainer.from_sparse_terms(
            SparsePauliTerms(2, [{0: "X"}, {}, {0: "Y", 1: "X"}]), [0.5, 0.7, 0.3], step_reps=4, scale=1.7
        )
    return PauliProductFormulaContainer(
        step_terms=step_terms,
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
        with pytest.raises(Exception, match="cannot assign to field 'angle'"):
            term.angle = 0.2


class TestPauliProductFormulaContainer:
    """Tests for the PauliProductFormulaContainer class."""

    def test_basic_properties(self, container):
        """Test basic properties of the container."""
        assert container.type == "pauli_product_formula"
        assert container.num_qubits == 2
        assert container.step_reps == 4
        assert len(container.step_terms) == 3
        assert isinstance(container.step_terms, Sequence)
        assert container.step_terms.index(container.step_terms[-1]) == 2

    @pytest.mark.parametrize("step_reps", [0, -1])
    def test_non_positive_step_reps_raises(self, step_terms, step_reps):
        """A step repeated zero or fewer times has no defined unitary."""
        with pytest.raises(ValueError, match="step_reps must be a positive integer"):
            PauliProductFormulaContainer(step_terms=step_terms, step_reps=step_reps, num_qubits=2)

    @pytest.mark.parametrize("step_reps", [1.5, 2.0, True, "2", None])
    def test_non_integer_step_reps_raises(self, step_terms, step_reps):
        """A repetition count that is not an integer would only fail later, inside Q#."""
        with pytest.raises(TypeError, match="step_reps must be an integer"):
            PauliProductFormulaContainer(step_terms=step_terms, step_reps=step_reps, num_qubits=2)

    def test_numpy_integer_step_reps_is_normalised(self, step_terms):
        """Integers read back from HDF5 attributes are numpy scalars."""
        container = PauliProductFormulaContainer(step_terms=step_terms, step_reps=np.int64(3), num_qubits=2)

        assert container.step_reps == 3
        assert isinstance(container.step_reps, int)

    def test_update_ordering(self, container):
        """Test setting a new valid evolution ordering."""
        updated_container = container.reorder_terms([1, 2, 0])

        assert updated_container.step_terms[0] == container.step_terms[1]
        assert updated_container.step_terms[1] == container.step_terms[2]
        assert updated_container.step_terms[2] == container.step_terms[0]

    def test_update_ordering_invalid(self, container):
        """Test setting an invalid evolution ordering."""
        with pytest.raises(ValueError, match="must match the number of terms"):
            container.reorder_terms([0, 1])

        with pytest.raises(ValueError, match="Invalid permutation"):
            container.reorder_terms([0, 1, 3])

    @pytest.mark.parametrize("format_name", ["json", "hdf5"])
    def test_serialization_roundtrip(self, container, format_name, tmp_path):
        """Restore either representation through the parent loader, retaining numeric Pauli keys."""
        filename = tmp_path / f"formula.pauli_product_formula_container.{format_name}"
        container.to_file(filename, format_name)
        restored = PauliProductFormulaContainer.from_file(filename, format_name)
        assert isinstance(restored, PauliProductFormulaContainer)
        assert restored.to_json() == container.to_json()
        assert restored.content_hash() == container.content_hash()
        assert all(isinstance(key, int) for term in restored.step_terms for key in term.pauli_term)

    def test_combine_no_adjacent_identical(self):
        """Test combine when no adjacent terms share the same Pauli string."""
        a = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
                ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.2),
            ],
            step_reps=2,
            num_qubits=2,
        )
        b = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.3),
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.4),
            ],
            step_reps=2,
            num_qubits=2,
        )
        result = a.combine(b)

        # a expanded: [X, Z, X, Z], b expanded: [Y, X, Y, X]
        # No adjacent duplicates anywhere, so all 8 terms survive.
        assert result.step_reps == 1
        assert len(result.step_terms) == 8
        expected_angles = [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4]
        for term, expected in zip(result.step_terms, expected_angles, strict=True):
            assert np.isclose(term.angle, expected, atol=1e-14)

    def test_combine_with_adjacent_identical(self):
        """Merge adjacent equal factors despite differing dictionary insertion order."""
        a = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=1.5),
                ExponentiatedPauliTerm(pauli_term={1: "Z", 0: "X"}, angle=0.5),
            ],
            step_reps=2,
            num_qubits=2,
        )
        b = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "X", 1: "Z"}, angle=0.7),
                ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.5),
            ],
            step_reps=1,
            num_qubits=2,
        )
        result = a.combine(b)

        # a expanded: [Y, XZ, Y, XZ], b: [XZ, Z]; only the boundary XZ terms fuse.
        assert result.step_reps == 1
        assert len(result.step_terms) == 5

        assert result.step_terms[0].pauli_term == {0: "Y"}
        assert np.isclose(result.step_terms[0].angle, 1.5, atol=1e-14)
        assert result.step_terms[1].pauli_term == {0: "X", 1: "Z"}
        assert np.isclose(result.step_terms[1].angle, 0.5, atol=1e-14)
        assert result.step_terms[2].pauli_term == {0: "Y"}
        assert np.isclose(result.step_terms[2].angle, 1.5, atol=1e-14)
        assert list(result.step_terms[3].pauli_term.items()) == [(0, "X"), (1, "Z")]
        assert np.isclose(result.step_terms[3].angle, 1.2, atol=1e-14)
        assert result.step_terms[4].pauli_term == {0: "Z"}
        assert np.isclose(result.step_terms[4].angle, 1.5, atol=1e-14)

    @pytest.mark.parametrize(
        ("angles", "atol", "expected"),
        [
            ([1e16, 1.0, -1e16, 0.5], 0.0, [0.5]),
            ([1.0, -0.875], 0.125, []),
            ([1.0, np.nextafter(-0.875, 0.0)], 0.125, [1.0 + np.nextafter(-0.875, 0.0)]),
        ],
    )
    def test_fusion_preserves_rounding_and_threshold(self, angles, atol, expected):
        """Add angles sequentially and drop merged values at, but not above, the tolerance."""
        legacy = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, angle) for angle in angles], 1, 1)
        empty = PauliProductFormulaContainer([], 1, 1)
        assert [term.angle for term in legacy.combine(empty, atol).step_terms] == expected

    @pytest.mark.parametrize(("angle", "atol"), [(np.inf, 1e-12), (0.5, np.nan)])
    def test_fusion_preserves_nonfinite_behavior(self, angle, atol):
        """Preserve the existing cancellation rule for nonfinite angles or tolerance."""
        legacy = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, angle)], 1, 1)
        inverse = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, -angle)], 1, 1)
        assert not legacy.combine(inverse, atol).step_terms

    def test_summary(self, container):
        """Test the summary generation of the container."""
        summary = container.get_summary()

        assert "Pauli Product Formula Container" in summary
        assert "Number of qubits: 2" in summary
        assert "Number of step terms: 3" in summary
        assert "Step repetitions: 4" in summary

    def test_legacy_hash_and_reordering_scale_are_unchanged(self):
        """Legacy containers keep their baseline hash and scale when reordered."""
        original = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, 0.5)], 4, 2, scale=1.7)
        assert original.content_hash() == "c2b1c5b0979d3d48"  # da61805e2 baseline
        assert original.reorder_terms([0]).content_hash() == original.content_hash()

    def test_hdf5_preserves_numeric_term_order(self, tmp_path):
        """Double-digit term indices must not be restored in lexicographic order."""
        original = PauliProductFormulaContainer(
            [ExponentiatedPauliTerm({i % 2: "XYZ"[i % 3]}, 0.1 * i) for i in range(13)],
            step_reps=2,
            num_qubits=2,
            scale=1.7,
        )
        with h5py.File(tmp_path / "ordered.h5", "w") as group:
            original.to_hdf5(group)
            restored = PauliProductFormulaContainer.from_hdf5(group)
        assert list(restored.step_terms) == list(original.step_terms)
        assert restored.content_hash() == original.content_hash()

    @pytest.mark.parametrize("inverse_reps", [1, 4])
    def test_sparse_factory_inherits_fusion(self, container, inverse_reps):
        """The sparse factory uses the existing fusion rule, including complete cancellation and identities."""
        inverse = PauliProductFormulaContainer(
            [ExponentiatedPauliTerm(term.pauli_term, -term.angle) for term in reversed(container.step_terms)],
            inverse_reps,
            2,
            scale=container.scale,
        )
        result = container.combine(inverse)
        assert result.step_reps == 1
        assert result.scale == container.scale
        assert result.step_terms == list(container.step_terms) * (4 - inverse_reps)
