"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import h5py
import numpy as np
import pytest

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


@pytest.fixture(params=[False, True], ids=["legacy", "packed"])
def container(step_terms, request):
    """Create a PauliProductFormulaContainer instance for testing."""
    if request.param:
        return PauliProductFormulaContainer.from_sparse_arrays(
            np.array([0, 1, 1, 3]),
            np.array([0, 0, 1]),
            np.array([1, 2, 1]),
            np.array([0.5, 0.7, 0.3]),
            step_reps=4,
            num_qubits=2,
            scale=1.7,
        )
    return PauliProductFormulaContainer(
        step_terms=step_terms,
        step_reps=4,
        num_qubits=2,
    )


def _no_term_objects(*_args):
    """Fail when packed schedule operations unpack compatibility term dictionaries."""
    pytest.fail("Packed schedule operations must not unpack term objects")


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

    def test_large_register_reordering(self):
        """Sparse row selection retains empty rows, wide indices, and packed dtypes."""
        num_qubits = 2**32
        original = PauliProductFormulaContainer.from_sparse_arrays(
            [0, 2, 2, 3],
            [0, num_qubits - 1, 1],
            [1, 3, 2],
            [0.1, 0.2, 0.3],
            step_reps=7,
            num_qubits=num_qubits,
            scale=0.5,
        )
        reordered = original.reorder_terms([1, 2, 0])
        expected = ([0, 0, 1, 3], [1, 0, num_qubits - 1], [2, 1, 3], [0.2, 0.3, 0.1])
        for actual, values in zip(reordered.sparse_term_arrays(), expected, strict=True):
            np.testing.assert_array_equal(actual, values)
        assert reordered.reorder_terms([2, 0, 1]).content_hash() == original.content_hash()
        combined = original.combine(original)
        assert list(combined.step_terms) == list(original.step_terms) * 14

    def test_to_json_roundtrip(self, container):
        """Test JSON serialization and deserialization roundtrip."""
        json_data = container.to_json()
        restored = PauliProductFormulaContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
        assert restored.scale == container.scale
        assert restored.content_hash() == container.content_hash()

        for t1, t2 in zip(restored.step_terms, container.step_terms, strict=True):
            assert t1.pauli_term == t2.pauli_term
            assert np.isclose(
                t1.angle, t2.angle, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )

    def test_from_json_pauli_term_keys_are_int(self, container):
        """Regression: JSON keys are strings, but pauli_term keys must be int after deserialization."""
        json_data = container.to_json()
        # Simulate a real JSON roundtrip where all dict keys become strings
        json_string = json.dumps(json_data)
        parsed = json.loads(json_string)

        restored = PauliProductFormulaContainer.from_json(parsed)

        for term in restored.step_terms:
            for key in term.pauli_term:
                assert isinstance(key, int), f"pauli_term key {key!r} should be int, got {type(key).__name__}"

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
        assert len(restored.step_terms) == len(container.step_terms)
        assert list(restored.step_terms) == list(container.step_terms)
        assert restored.scale == container.scale
        assert restored.content_hash() == container.content_hash()

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
        packed = PauliProductFormulaContainer.from_sparse_arrays(
            np.arange(len(angles) + 1), [0] * len(angles), [1] * len(angles), angles, step_reps=1, num_qubits=1
        )
        empty = PauliProductFormulaContainer([], 1, 1)
        for source in (legacy, packed):
            result = source.combine(empty, atol)
            assert [term.angle for term in result.step_terms] == expected

    @pytest.mark.parametrize(("angle", "atol"), [(np.inf, 1e-12), (0.5, np.nan)])
    def test_fusion_preserves_nonfinite_behavior(self, angle, atol):
        """Legacy cancellation and packed finite-output validation must not silently change."""
        legacy = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, angle)], 1, 1)
        inverse = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, -angle)], 1, 1)
        assert not legacy.combine(inverse, atol).step_terms
        packed = PauliProductFormulaContainer.from_sparse_arrays([0, 1], [0], [1], [0.5], step_reps=1, num_qubits=1)
        if np.isfinite(angle):
            assert packed.combine(inverse, atol).step_terms[0].angle == 0.0
        else:
            with pytest.raises(ValueError, match="angles must be finite"):
                packed.combine(legacy, atol)

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


@pytest.mark.parametrize("container", [True], indirect=True)
class TestPackedPauliProductFormulaContainer:
    """Packed schedules preserve ownership and legacy combination semantics."""

    def test_owns_read_only_arrays(self, container):
        """Caller arrays and compatibility dictionaries cannot mutate stored factors or angles."""
        inputs = [values.copy() for values in container.sparse_term_arrays()]
        original = PauliProductFormulaContainer.from_sparse_arrays(*inputs, step_reps=2, num_qubits=2)
        original_hash = original.content_hash()
        for source, stored in zip(inputs, original.sparse_term_arrays(), strict=True):
            assert not np.shares_memory(source, stored)
            source[:] = 0
            assert not stored.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                stored[0] = 0
        assert original.content_hash() == original_hash
        assert original.step_terms[0] == ExponentiatedPauliTerm({0: "X"}, 0.5)
        original.step_terms[0].pauli_term[0] = "Z"
        assert original.step_terms[0].pauli_term == {0: "X"}

    @pytest.mark.parametrize("inverse_reps", [1, 4])
    def test_schedule_stays_packed_and_matches_legacy(self, container, inverse_reps, monkeypatch):
        """Reordering and mixed-storage combination preserve identity factors and complete cancellation."""
        legacy_left = PauliProductFormulaContainer(list(container.step_terms), 4, 2, scale=1.7)
        legacy_right = PauliProductFormulaContainer(
            [ExponentiatedPauliTerm(term.pauli_term, -term.angle) for term in reversed(legacy_left.step_terms)],
            inverse_reps,
            2,
            scale=1.7,
        )
        expected_terms = list(legacy_left.step_terms) * (4 - inverse_reps)
        assert list(legacy_left.combine(legacy_right).step_terms) == expected_terms
        with monkeypatch.context() as patch:
            patch.setattr(type(container.step_terms), "__getitem__", _no_term_objects)
            reordered = container.reorder_terms([2, 1, 0])
            offsets, indices, codes, angles = reordered.sparse_term_arrays()
            inverse = PauliProductFormulaContainer.from_sparse_arrays(
                offsets, indices, codes, -angles, step_reps=inverse_reps, num_qubits=2, scale=reordered.scale
            )
            results = [
                left.combine(right)
                for left, right in ((container, inverse), (container, legacy_right), (legacy_left, inverse))
            ]
            assert reordered.step_reps == container.step_reps
            restored = PauliProductFormulaContainer.from_json(reordered.to_json())
            assert reordered.content_hash() == restored.content_hash()
        for combined in results:
            assert combined.has_sparse_terms
            assert combined.step_reps == 1
            assert combined.scale == legacy_left.scale
            assert list(combined.step_terms) == expected_terms
        if inverse_reps == 4:
            assert not results[0].step_terms
            assert results[0].reorder_terms([]).content_hash() == results[0].content_hash()
            assert list(results[0].combine(container).step_terms) == list(legacy_left.step_terms) * 4

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("angles", [[0.5, 0.7, 0.3]]),
            ("angles", []),
            ("angles", [0.5, 0.7 + 1j, 0.3]),
            ("angles", [0.5, np.nan, 0.3]),
            ("angles", [0.5, np.inf, 0.3]),
            ("step_reps", 0),
            ("step_reps", True),
            ("step_reps", 1.5),
        ],
    )
    def test_invalid_angles_and_repetition_counts(self, container, field, value):
        """A packed formula requires one finite real angle per term and a positive integer repetition count."""
        fields = ("term_offsets", "qubit_indices", "pauli_codes", "angles")
        arguments = dict(zip(fields, container.sparse_term_arrays(), strict=True), step_reps=1, num_qubits=2)
        arguments[field] = value
        with pytest.raises((ValueError, TypeError)):
            PauliProductFormulaContainer.from_sparse_arrays(**arguments)
