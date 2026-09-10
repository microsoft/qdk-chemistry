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
    """Create a list of ExponentiatedPauliTerm instances for testing.

    Carries a control-exempt conjugating pair and a two-member batch as well as plain
    terms, so that anything round-tripping this fixture also covers both flags.
    """
    return [
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5, needs_control=False),
        ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=1.2, batch=1),
        ExponentiatedPauliTerm(pauli_term={0: "Y", 1: "X"}, angle=0.3, batch=1),
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.5, needs_control=False),
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


class TestPauliProductFormulaContainer:
    """Tests for the PauliProductFormulaContainer class."""

    def test_basic_properties(self, container):
        """Test basic properties of the container."""
        assert container.type == "pauli_product_formula"
        assert container.num_qubits == 2
        assert container.step_reps == 4
        assert len(container.step_terms) == 4
        assert container.before_repeated_terms == []
        assert container.after_repeated_terms == []

    def test_one_time_boundaries_are_distinct_from_the_repeated_step(self):
        """Moving a factor across a repetition boundary changes the represented circuit."""
        term = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.25)
        before = PauliProductFormulaContainer(step_terms=[], step_reps=3, num_qubits=1, before_repeated_terms=[term])
        repeated = PauliProductFormulaContainer(step_terms=[term], step_reps=3, num_qubits=1)
        after = PauliProductFormulaContainer(step_terms=[], step_reps=3, num_qubits=1, after_repeated_terms=[term])

        assert before.before_repeated_terms == [term]
        assert after.after_repeated_terms == [term]
        assert len({before.content_hash(), repeated.content_hash(), after.content_hash()}) == 3

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
        before = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.1)
        after = ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=-0.1)
        container = PauliProductFormulaContainer(
            step_terms=container.step_terms,
            step_reps=container.step_reps,
            num_qubits=container.num_qubits,
            before_repeated_terms=[before],
            after_repeated_terms=[after],
        )
        updated_container = container.reorder_terms([1, 2, 3, 0])

        assert updated_container.step_terms[0] == container.step_terms[1]
        assert updated_container.step_terms[1] == container.step_terms[2]
        assert updated_container.step_terms[2] == container.step_terms[3]
        assert updated_container.step_terms[3] == container.step_terms[0]
        assert updated_container.before_repeated_terms == [before]
        assert updated_container.after_repeated_terms == [after]

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_one_time_boundaries_roundtrip(self, file_format, tmp_path):
        """Both persistence formats preserve the one-time prefix and suffix."""
        before = ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.2, needs_control=False)
        repeated = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.4, batch=3)
        after = ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.2, needs_control=False)
        original = PauliProductFormulaContainer(
            step_terms=[repeated],
            step_reps=5,
            num_qubits=1,
            before_repeated_terms=[before],
            after_repeated_terms=[after],
        )

        if file_format == "json":
            restored = PauliProductFormulaContainer.from_json(json.loads(json.dumps(original.to_json())))
        else:
            path = tmp_path / "bounded.h5"
            with h5py.File(path, "w") as handle:
                original.to_hdf5(handle)
            with h5py.File(path, "r") as handle:
                restored = PauliProductFormulaContainer.from_hdf5(handle)

        assert restored.content_hash() == original.content_hash()
        assert restored.before_repeated_terms == [before]
        assert restored.step_terms == [repeated]
        assert restored.after_repeated_terms == [after]

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_loads_the_previous_flat_wire_format_with_empty_boundaries(self, file_format, tmp_path):
        """The new reader upgrades 0.2 files, while 0.3 makes old readers reject new fields."""
        term = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.4)
        old = PauliProductFormulaContainer(step_terms=[term], step_reps=2, num_qubits=1)

        if file_format == "json":
            payload = old.to_json()
            payload["version"] = "0.2.1"
            del payload["before_repeated_terms"]
            del payload["after_repeated_terms"]
            restored = PauliProductFormulaContainer.from_json(payload)
        else:
            path = tmp_path / "flat.h5"
            with h5py.File(path, "w") as handle:
                old.to_hdf5(handle)
                handle.attrs["version"] = "0.2.1"
                del handle["before_repeated_terms"]
                del handle["after_repeated_terms"]
            with h5py.File(path, "r") as handle:
                restored = PauliProductFormulaContainer.from_hdf5(handle)

        assert restored.step_terms == [term]
        assert restored.before_repeated_terms == []
        assert restored.after_repeated_terms == []

    def test_update_ordering_invalid(self, container):
        """Test setting an invalid evolution ordering."""
        with pytest.raises(ValueError, match="must match the number of terms"):
            container.reorder_terms([0, 1])

        with pytest.raises(ValueError, match="Invalid permutation"):
            container.reorder_terms([0, 1, 2, 4])

    def test_to_json_roundtrip(self, container):
        """Test JSON serialization and deserialization roundtrip.

        The fixture carries both flags, so this also covers that ``needs_control`` and
        ``batch`` survive: a reloaded circuit that lost either would cost differently.
        """
        json_data = container.to_json()
        restored = PauliProductFormulaContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
        assert restored.content_hash() == container.content_hash()

        for t1, t2 in zip(restored.step_terms, container.step_terms, strict=True):
            assert t1.pauli_term == t2.pauli_term
            assert t1.needs_control == t2.needs_control
            assert t1.batch == t2.batch
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
        """Test HDF5 serialization and deserialization roundtrip.

        The fixture carries both flags, so this also covers that ``needs_control`` and
        ``batch`` survive HDF5.
        """
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
        assert [t.batch for t in restored.step_terms] == [t.batch for t in container.step_terms]
        assert [t.needs_control for t in restored.step_terms] == [t.needs_control for t in container.step_terms]

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

    @pytest.mark.parametrize(
        ("terms", "num_qubits", "match"),
        [
            pytest.param(
                [
                    ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3, batch=1),
                    ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.3, batch=1),
                ],
                2,
                "batch",
                id="batched",
            ),
            pytest.param(
                [
                    ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.3, needs_control=False),
                    ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.3, needs_control=False),
                ],
                1,
                "needs_control",
                id="control-exempt",
            ),
        ],
    )
    def test_combine_rejects_neighbour_dependent_terms(self, terms, num_qubits, match):
        """Both flags are claims about a term's neighbours, which combining invalidates.

        A batch must stay consecutive and an exemption must keep its adjoint partner
        adjacent, but combining flattens and repeats the term lists. Rejected from
        either side, since the order of the operands must not decide correctness.
        """
        special = PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=num_qubits)
        plain = PauliProductFormulaContainer(
            step_terms=[ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.2)],
            step_reps=1,
            num_qubits=num_qubits,
        )
        with pytest.raises(ValueError, match=match):
            plain.combine(special)
        with pytest.raises(ValueError, match=match):
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
        batched = self._container(
            [
                ExponentiatedPauliTerm({0: "Z"}, 0.3, batch=1),
                ExponentiatedPauliTerm({1: "Z"}, 0.3, batch=1),
            ]
        )
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
