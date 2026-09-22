"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import SparsePauliProductFormulaContainer, SparsePauliTerms
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
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
            SparsePauliTerms(2, [{0: "X"}, {}, {0: "Y", 1: "X"}]),
            [0.5, 0.7, 0.3],
            step_reps=4,
            scale=1.7,
            layer_offsets=(0, 2, 3),
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
    @pytest.mark.parametrize("with_endpoints", [False, True])
    def test_serialization_roundtrip(self, container, format_name, tmp_path, with_endpoints):
        """Restore numeric Pauli keys and term order, including double-digit HDF5 term indices."""
        container = type(container)(
            [ExponentiatedPauliTerm(container.step_terms[i % 3].pauli_term, i * 0.1) for i in range(13)],
            container.step_reps,
            container.num_qubits,
            container.scale,
            beginning=container.step_terms[:1] if with_endpoints else (),
            end=container.step_terms[-1:] if with_endpoints else (),
            group_offsets=tuple(range(14)) if with_endpoints else None,
            layer_offsets=tuple(range(16 if with_endpoints else 14)),
        )
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
            layer_offsets=(0, 2),
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
        assert result.layer_offsets == (0, 2, 4, 5, 6, 7, 8)
        expected_angles = [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4]
        np.testing.assert_allclose([term.angle for term in result.step_terms], expected_angles, rtol=1e-5, atol=1e-14)

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
        assert [term.pauli_term for term in result.step_terms] == [
            {0: "Y"},
            {0: "X", 1: "Z"},
            {0: "Y"},
            {0: "X", 1: "Z"},
            {0: "Z"},
        ]
        np.testing.assert_allclose(
            [term.angle for term in result.step_terms], [1.5, 0.5, 1.5, 1.2, 1.5], rtol=1e-5, atol=1e-14
        )
        assert list(result.step_terms[3].pauli_term.items()) == [(0, "X"), (1, "Z")]

    @pytest.mark.parametrize(
        ("angles", "atol", "expected"),
        [
            ([1e16, 1.0, -1e16, 0.5], 0.0, [0.5]),
            ([1.0, -0.875], 0.125, []),
            ([1.0, np.nextafter(-0.875, 0.0)], 0.125, [1.0 + np.nextafter(-0.875, 0.0)]),
        ],
    )
    def test_fusion_preserves_rounding_and_threshold(self, angles, atol, expected):
        """Preserve sequential addition and threshold boundaries for finite angles."""
        legacy = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, angle) for angle in angles], 1, 1)
        empty = PauliProductFormulaContainer([], 1, 1)
        assert [term.angle for term in legacy.combine(empty, atol).step_terms] == expected

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
        if inverse_reps == 4 and container.layer_offsets is not None:
            assert result.layer_offsets == (0,)

    @pytest.mark.parametrize("cancel", [False, True])
    def test_group_boundary_fusion_stays_compact(self, cancel: bool) -> None:
        """Fuse reordered commuting boundaries without storing each repetition."""
        left = [ExponentiatedPauliTerm({0: "X"}, 0.125), ExponentiatedPauliTerm({1: "X"}, 0.25)]
        middle = [ExponentiatedPauliTerm({0: "Z"}, 0.3)]
        right = [ExponentiatedPauliTerm(t.pauli_term, -t.angle if cancel else t.angle) for t in reversed(left)]
        formula = PauliProductFormulaContainer(
            left + middle + right, 10**9, 2, group_offsets=(0, 2, 3, 5), layer_offsets=(0, 2, 3, 5)
        )
        fused = formula.combine(atol=0.0)
        assert fused.num_pauli_exponentials == 5 * 10**9 - (4 if cancel else 2) * (10**9 - 1)
        assert fused.num_stored_terms <= 8
        assert fused.beginning == left
        assert fused.end == middle + right
        assert fused.layer_offsets == ((0, 2, 3, 4, 6) if cancel else (0, 2, 3, 5, 6, 8))
        assert fused.combine(fused, atol=0.0).num_stored_terms <= 8

    def test_combine_different_bodies_includes_endpoints(self) -> None:
        """The general flatten-and-merge fallback preserves both formulas' endpoints."""
        x, y, z = [ExponentiatedPauliTerm({0: axis}, 0.125) for axis in "XYZ"]
        phase = ExponentiatedPauliTerm({}, 0.125)
        inverse_y = ExponentiatedPauliTerm(y.pauli_term, -y.angle)
        first = PauliProductFormulaContainer([x, z], 2, 1, beginning=[phase], end=[y], layer_offsets=(0, 1, 2, 3, 4))
        second = PauliProductFormulaContainer([x], 3, 1, beginning=[inverse_y], end=[phase], layer_offsets=(0, 1, 2, 3))
        combined = first.combine(second)
        assert combined.step_reps == 1
        assert combined.beginning == combined.end == []
        assert combined.step_terms == [phase, x, z, x, z, ExponentiatedPauliTerm(x.pauli_term, 0.375), phase]
        assert combined.layer_offsets == tuple(range(8))

    @pytest.mark.parametrize("format_name", ["json", "hdf5"])
    def test_legacy_02_read(self, container, tmp_path, format_name):
        """Flat 0.2 payloads without endpoint fields remain readable."""
        if format_name == "json":
            data = container.to_json()
            data["version"] = "0.2.0"
            del data["beginning"], data["end"]
            restored = PauliProductFormulaContainer.from_json(data)
        else:
            with h5py.File(tmp_path / "legacy.h5", "w") as group:
                container.to_hdf5(group)
                group.attrs["version"] = "0.2.0"
                del group["beginning"], group["end"]
                restored = PauliProductFormulaContainer.from_hdf5(group)
        assert restored.to_json() == container.to_json()

    @pytest.mark.parametrize(
        ("metadata", "match"),
        [
            ({"group_offsets": (0, True, 2)}, "group_offsets"),
            ({"group_offsets": (0, 2)}, "must commute"),
            ({"layer_offsets": (0, 1, 4)}, "beginning/body/end boundaries"),
            ({"layer_offsets": (0, 2, 3, 4)}, "disjoint"),
        ],
    )
    def test_invalid_fusion_metadata(self, metadata, match):
        """Check commuting certificates and endpoint-layer boundaries/support independently."""
        x, z = [ExponentiatedPauliTerm({0: axis}, 0.125) for axis in "XZ"]
        with pytest.raises(ValueError, match=match):
            PauliProductFormulaContainer([x, z], 2, 1, beginning=[x, x], **metadata)

    def test_reorder_preserves_endpoints_and_endpoint_layers(self):
        """A body permutation invalidates only body certificates, never endpoint data."""
        terms = [ExponentiatedPauliTerm({i: "X"}, 0.125) for i in range(2)]
        formula = PauliProductFormulaContainer(
            terms, 3, 2, beginning=terms, end=terms, group_offsets=(0, 2), layer_offsets=(0, 2, 4, 6)
        )
        reordered = formula.reorder_terms([1, 0])
        assert reordered.beginning == reordered.end == terms
        assert reordered.step_terms == list(reversed(terms))
        assert reordered.group_offsets is None
        assert reordered.layer_offsets == (0, 2, 3, 4, 6)

    @pytest.mark.parametrize("atol", [-1.0, np.inf, np.nan])
    def test_invalid_fusion_tolerance(self, atol):
        """Even a no-op fusion rejects invalid tolerances."""
        formula = PauliProductFormulaContainer([], 1, 1)
        with pytest.raises(ValueError, match="atol"):
            formula.combine(atol=atol)

    @pytest.mark.parametrize("angle", [np.inf, np.nan, 1e308])
    def test_fusion_rejects_nonfinite_angles_and_overflow(self, angle):
        """Reject invalid inputs and overflow on compact and fallback paths."""
        formula = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, angle)], 2, 1)
        with pytest.raises(ValueError, match="finite"):
            formula.combine()
        with pytest.raises(ValueError, match="finite"):
            formula.combine(PauliProductFormulaContainer([], 1, 1))
        formula = PauliProductFormulaContainer([], 1, 1, scale=np.inf)
        with pytest.raises(ValueError, match="scale"):
            formula.combine(formula)

    def test_partial_group_cancellation_retains_layers(self):
        """Drop cancelled boundary words without merging the surviving declared layers."""
        x = ExponentiatedPauliTerm({0: "X"}, 0.125)
        y = ExponentiatedPauliTerm({1: "X"}, 0.25)
        z = ExponentiatedPauliTerm({0: "Z"}, 0.5)
        terms = [x, y, z, ExponentiatedPauliTerm(y.pauli_term, -y.angle), x]
        formula = PauliProductFormulaContainer(terms, 4, 2, group_offsets=(0, 2, 3, 5), layer_offsets=(0, 2, 3, 4, 5))
        fused = formula.combine(atol=0)
        assert fused.step_terms == [z, ExponentiatedPauliTerm(x.pauli_term, 0.25)]
        assert fused.group_offsets == (0, 1, 2)
        assert fused.layer_offsets == (0, 2, 3, 4, 5, 6, 7)

    def test_structured_positional_api_and_flat_metadata_rejection(self):
        """The fifth positional argument remains conjugation, not flat endpoint metadata."""
        x = ExponentiatedPauliTerm({0: "X"}, 0.125)
        batch = BatchedExponentiatedPauliTerm([{0: "X"}, {1: "X"}], 0.25)
        assert PauliProductFormulaContainer([batch], 3, 2, 1.7, [x]).conjugating_terms == [x]
        with pytest.raises(ValueError, match="flat"):
            PauliProductFormulaContainer([batch], 2, 2, group_offsets=(0, 1))
        with pytest.raises(ValueError, match="flat"):
            PauliProductFormulaContainer([], 2, 2, beginning=[batch])

    def test_canonical_boundary_words_and_empty_body(self):
        """Canonical identities fuse, and full cancellation permits an empty compact body."""
        x = ExponentiatedPauliTerm({0: "X"}, 0.125)
        inverse = ExponentiatedPauliTerm({1: "I", 0: "X"}, -0.125)
        formula = PauliProductFormulaContainer([x, inverse], 10**9, 2, group_offsets=(0, 1, 2), layer_offsets=(0, 1, 2))
        fused = formula.combine(atol=0)
        assert fused.step_terms == []
        assert fused.beginning == [x]
        assert fused.end == [inverse]
        assert fused.group_offsets == (0,)
        assert fused.layer_offsets == (0, 1, 2)

    def test_different_layer_schedules_use_boundary_preserving_fallback(self):
        """Equal bodies cannot erase distinct producer-declared layer schedules."""
        terms = [ExponentiatedPauliTerm({0: "X"}, 0.125), ExponentiatedPauliTerm({1: "X"}, 0.25)]
        first = PauliProductFormulaContainer(terms, 2, 2, layer_offsets=(0, 2))
        second = PauliProductFormulaContainer(terms, 2, 2, layer_offsets=(0, 1, 2))
        combined = first.combine(second)
        assert combined.step_reps == 1
        assert combined.step_terms == terms * 4
        assert combined.layer_offsets == (0, 2, 4, 5, 6, 7, 8)
