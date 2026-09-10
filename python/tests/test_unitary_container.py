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
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
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
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.5),
    ]


@pytest.fixture
def container(step_terms):
    """Create a PauliProductFormulaContainer instance for testing."""
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


class TestStructuredExponentiatedPauliTerms:
    """Tests for structural batching and conjugation metadata."""

    def test_batch_owns_one_angle_and_disjoint_pauli_strings(self):
        """A batch represents equal-angle factors without neighbour-dependent IDs."""
        batch = BatchedExponentiatedPauliTerm(pauli_terms=[{0: "Z"}, {2: "X", 3: "Y"}], angle=0.25)

        assert batch.pauli_terms == [{0: "Z"}, {2: "X", 3: "Y"}]
        assert batch.angle == 0.25

    @pytest.mark.parametrize(
        ("pauli_terms", "match"),
        [
            ([{0: "Z"}], "at least two"),
            ([{}, {1: "Z"}], "identity"),
            ([{0: "Z"}, {0: "X"}], "disjoint support"),
        ],
    )
    def test_batch_rejects_invalid_hamming_weight_groups(self, pauli_terms, match):
        """Singleton, identity, and overlapping groups cannot use one weight register."""
        with pytest.raises(ValueError, match=match):
            BatchedExponentiatedPauliTerm(pauli_terms=pauli_terms, angle=0.25)

    def test_conjugation_stores_within_and_apply_blocks(self):
        """The representation directly records V D V-dagger."""
        within = [ExponentiatedPauliTerm({0: "X"}, 0.2)]
        apply = [ExponentiatedPauliTerm({0: "Z"}, 0.3)]

        conjugated = ConjugatedExponentiatedPauliTerm(within_terms=within, apply_terms=apply)

        assert conjugated.within_terms == within
        assert conjugated.apply_terms == apply


class TestPauliProductFormulaContainer:
    """Tests for the PauliProductFormulaContainer class."""

    def test_basic_properties(self, container):
        """Test basic properties of the container."""
        assert container.type == "pauli_product_formula"
        assert container.num_qubits == 2
        assert container.step_reps == 4
        assert len(container.step_terms) == 4
        assert container.conjugating_terms == []

    def test_outer_conjugation_is_distinct_from_the_repeated_step(self):
        """Moving a factor into the outer conjugation changes the represented circuit."""
        term = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.25)
        conjugated = PauliProductFormulaContainer(step_terms=[], step_reps=3, num_qubits=1, conjugating_terms=[term])
        repeated = PauliProductFormulaContainer(step_terms=[term], step_reps=3, num_qubits=1)

        assert conjugated.conjugating_terms == [term]
        assert conjugated.content_hash() != repeated.content_hash()

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
        conjugating = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.1)
        container = PauliProductFormulaContainer(
            step_terms=container.step_terms,
            step_reps=container.step_reps,
            num_qubits=container.num_qubits,
            conjugating_terms=[conjugating],
        )
        updated_container = container.reorder_terms([1, 2, 3, 0])

        assert updated_container.step_terms[0] == container.step_terms[1]
        assert updated_container.step_terms[1] == container.step_terms[2]
        assert updated_container.step_terms[2] == container.step_terms[3]
        assert updated_container.step_terms[3] == container.step_terms[0]
        assert updated_container.conjugating_terms == [conjugating]

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_structured_formula_persistence_fails_explicitly(self, file_format, tmp_path):
        """Persistence stays limited to the established flat wire format."""
        batch = BatchedExponentiatedPauliTerm([{0: "Z"}, {1: "Z"}], 0.4)
        container = PauliProductFormulaContainer(step_terms=[batch], step_reps=5, num_qubits=2)

        if file_format == "json":
            with pytest.raises(ValueError, match="Structured Pauli product formulas cannot be serialized"):
                container.to_json()
        else:
            path = tmp_path / "structured.h5"
            with (
                h5py.File(path, "w") as handle,
                pytest.raises(ValueError, match="Structured Pauli product formulas cannot be serialized"),
            ):
                container.to_hdf5(handle)

    def test_update_ordering_invalid(self, container):
        """Test setting an invalid evolution ordering."""
        with pytest.raises(ValueError, match="must match the number of terms"):
            container.reorder_terms([0, 1])

        with pytest.raises(ValueError, match="Invalid permutation"):
            container.reorder_terms([0, 1, 2, 4])

    def test_to_json_roundtrip(self, container):
        """Test JSON serialization and deserialization roundtrip."""
        json_data = container.to_json()
        restored = PauliProductFormulaContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
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
        """Test combine where adjacent identical Pauli terms get merged."""
        a = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=1.5),
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ],
            step_reps=2,
            num_qubits=1,
        )
        b = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.7),
                ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.5),
            ],
            step_reps=1,
            num_qubits=1,
        )
        result = a.combine(b)

        # a expanded: [Y(1.5), X(0.5), Y(1.5), X(0.5)], b expanded: [X(0.7), Z(1.5)]
        # Only the two adjacent X terms at the boundary are merged into X(1.2)
        assert result.step_reps == 1
        assert len(result.step_terms) == 5

        assert result.step_terms[0].pauli_term == {0: "Y"}
        assert np.isclose(result.step_terms[0].angle, 1.5, atol=1e-14)
        assert result.step_terms[1].pauli_term == {0: "X"}
        assert np.isclose(result.step_terms[1].angle, 0.5, atol=1e-14)
        assert result.step_terms[2].pauli_term == {0: "Y"}
        assert np.isclose(result.step_terms[2].angle, 1.5, atol=1e-14)
        assert result.step_terms[3].pauli_term == {0: "X"}
        assert np.isclose(result.step_terms[3].angle, 1.2, atol=1e-14)
        assert result.step_terms[4].pauli_term == {0: "Z"}
        assert np.isclose(result.step_terms[4].angle, 1.5, atol=1e-14)

    @pytest.mark.parametrize("structured", ["batch", "conjugated", "outer"])
    def test_combine_rejects_structured_formulas(self, structured):
        """Combining refuses to flatten explicit batches or conjugations."""
        x = ExponentiatedPauliTerm({0: "X"}, 0.3)
        z = ExponentiatedPauliTerm({0: "Z"}, 0.2)
        if structured == "batch":
            special = PauliProductFormulaContainer(
                step_terms=[BatchedExponentiatedPauliTerm([{0: "Z"}, {1: "Z"}], 0.3)],
                step_reps=1,
                num_qubits=2,
            )
        elif structured == "conjugated":
            special = PauliProductFormulaContainer(
                step_terms=[ConjugatedExponentiatedPauliTerm([x], [z])], step_reps=1, num_qubits=1
            )
        else:
            special = PauliProductFormulaContainer(step_terms=[z], step_reps=1, num_qubits=1, conjugating_terms=[x])
        plain = PauliProductFormulaContainer(
            step_terms=[ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.2)],
            step_reps=1,
            num_qubits=special.num_qubits,
        )
        with pytest.raises(ValueError, match="batched or conjugated"):
            plain.combine(special)
        with pytest.raises(ValueError, match="batched or conjugated"):
            special.combine(plain)

    def test_summary(self, container):
        """Test the summary generation of the container."""
        summary = container.get_summary()

        assert "Pauli Product Formula Container" in summary
        assert "Number of qubits: 2" in summary
        assert "Number of step terms: 4" in summary
        assert "Step repetitions: 4" in summary


class TestEigenvalueFromPhaseZeroScale:
    """A zero evolution time makes the phase-to-energy inversion undefined."""

    def test_zero_scale_raises_a_descriptive_error_not_zero_division(self):
        """``scale`` defaults to the evolution time, which is ``0.0`` for a default builder.

        Reaching ``E = -angle / t`` with ``t = 0`` previously raised a bare ``ZeroDivisionError``.
        It now raises a ``ValueError`` explaining that the evolution time is zero.
        """
        container = PauliProductFormulaContainer(
            step_terms=[ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3)],
            step_reps=1,
            num_qubits=1,
            scale=0.0,
        )
        with pytest.raises(ValueError, match=r"evolution time.*is zero"):
            container.eigenvalue_from_phase(0.25)

    def test_nonzero_scale_still_inverts(self):
        """A non-zero scale is unaffected by the guard."""
        container = PauliProductFormulaContainer(
            step_terms=[ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3)],
            step_reps=1,
            num_qubits=1,
            scale=2.0,
        )
        assert np.isfinite(container.eigenvalue_from_phase(0.25))


class TestBatchHashing:
    """Batching changes the circuit, so it must change the content hash."""

    @staticmethod
    def _container(terms):
        return PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=4)

    def test_batch_changes_the_hash(self):
        """Two containers differing only in batching are different circuits to cost."""
        plain = self._container([ExponentiatedPauliTerm({0: "Z"}, 0.3), ExponentiatedPauliTerm({1: "Z"}, 0.3)])
        batched = self._container([BatchedExponentiatedPauliTerm([{0: "Z"}, {1: "Z"}], 0.3)])
        assert plain.content_hash() != batched.content_hash()


class TestHdf5TermOrdering:
    """A product formula's factor order is part of what it means."""

    def test_hdf5_round_trip_preserves_order_past_ten_terms(self, tmp_path):
        """HDF5 lists members alphabetically, which puts ``term_10`` before ``term_2``.

        Reading the group back in listing order would silently change the unitary once
        there are eleven terms, which no round trip with fewer would catch.
        """
        terms = [ExponentiatedPauliTerm({0: "X"}, float(i) / 10.0) for i in range(12)]
        original = PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=1)
        path = tmp_path / "many.h5"
        with h5py.File(path, "w") as handle:
            original.to_hdf5(handle)
        with h5py.File(path, "r") as handle:
            restored = PauliProductFormulaContainer.from_hdf5(handle)
        assert [t.angle for t in restored.step_terms] == [t.angle for t in terms]
