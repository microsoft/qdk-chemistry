"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

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


@pytest.fixture(params=[False, True], ids=["ordinary", "sparse-factory"])
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
        assert updated_container.scale == container.scale

    def test_update_ordering_invalid(self, container):
        """Test setting an invalid evolution ordering."""
        with pytest.raises(ValueError, match="must match the number of terms"):
            container.reorder_terms([0, 1])

        with pytest.raises(ValueError, match="Invalid permutation"):
            container.reorder_terms([0, 1, 3])

    def test_to_json_roundtrip(self, container):
        """Test JSON serialization and deserialization roundtrip."""
        json_data = container.to_json()
        restored = PauliProductFormulaContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps

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

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    def test_old_packed_load_writes_canonical(self, tmp_path, file_format):
        """Convert old 0.3.0 arrays, retaining identity, last-axis-wins factors, angles and scale."""
        payload = {
            "version": "0.3.0",
            "container_type": "pauli_product_formula",
            "term_offsets": [0, 3, 3, 4],
            "qubit_indices": [1, 0, 1, 0],
            "pauli_codes": [1, 2, 3, 1],
            "angles": [0.5, -0.25, 0.0],
            "step_reps": 3,
            "num_qubits": 2,
            "scale": 1.7,
        }
        if file_format == "json":
            restored = PauliProductFormulaContainer.from_json(json.loads(json.dumps(payload)))
        else:
            with h5py.File(tmp_path / "old.h5", "w") as group:
                for key in ("version", "container_type", "step_reps", "num_qubits", "scale"):
                    group.attrs[key] = payload[key]
                for key in ("term_offsets", "qubit_indices", "pauli_codes", "angles"):
                    group.create_dataset(key, data=payload[key])
                restored = PauliProductFormulaContainer.from_hdf5(group)
        expected = PauliProductFormulaContainer(
            [
                ExponentiatedPauliTerm({1: "Z", 0: "Y"}, 0.5),
                ExponentiatedPauliTerm({}, -0.25),
                ExponentiatedPauliTerm({0: "X"}, 0.0),
            ],
            3,
            2,
            scale=1.7,
        )
        assert restored.to_json() == expected.to_json()
        assert list(restored.step_terms[0].pauli_term.items()) == [(1, "Z"), (0, "Y")]
        assert restored.content_hash() == expected.content_hash()
        path = tmp_path / f"canonical.pauli_product_formula_container.{file_format}"
        restored.to_file(path, file_format)
        if file_format == "json":
            written = json.loads(path.read_text())
            assert written == expected.to_json()
            assert not {"term_offsets", "qubit_indices", "pauli_codes", "angles"} & written.keys()
        else:
            with h5py.File(path, "r") as group:
                assert group.attrs["version"] == PauliProductFormulaContainer._serialization_version
                assert "step_terms" in group
                assert not {"term_offsets", "qubit_indices", "pauli_codes", "angles"} & group.keys()
        assert PauliProductFormulaContainer.from_file(path, file_format).content_hash() == expected.content_hash()

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("term_offsets", [0, 2, 1]),
            ("qubit_indices", [-1]),
            ("pauli_codes", [1.5]),
            ("angles", [[0.5]]),
        ],
    )
    def test_old_packed_rejects_invalid_arrays(self, key, value):
        """Malformed packed payloads must fail before constructing ordinary terms."""
        payload = {
            "version": "0.3.0",
            "num_qubits": 2,
            "step_reps": 3,
            "term_offsets": [0, 1],
            "qubit_indices": [0],
            "pauli_codes": [1],
            "angles": [0.5],
        }
        payload[key] = value
        with pytest.raises(ValueError, match="packed product-formula"):
            PauliProductFormulaContainer.from_json(payload)

    @pytest.mark.parametrize("file_format", ["json", "hdf5"])
    @pytest.mark.parametrize("with_endpoints", [False, True])
    def test_serialization_preserves_term_order_and_hash(self, container, file_format, tmp_path, with_endpoints):
        """Restore numeric Pauli keys and order, including double-digit HDF5 term indices."""
        container = type(container)(
            [ExponentiatedPauliTerm(container.step_terms[i % 3].pauli_term, i * 0.1) for i in range(13)],
            container.step_reps,
            container.num_qubits,
            container.scale,
            beginning=container.step_terms[:1] if with_endpoints else (),
            end=container.step_terms[-1:] if with_endpoints else (),
            group_offsets=tuple(range(14)) if with_endpoints else None,
            layer_offsets=tuple(range(16)) if with_endpoints else None,
        )
        filename = tmp_path / f"formula.pauli_product_formula_container.{file_format}"
        container.to_file(filename, file_format)
        restored = PauliProductFormulaContainer.from_file(filename, file_format)
        assert type(restored) is PauliProductFormulaContainer
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

    def test_summary(self, container):
        """Test the summary generation of the container."""
        summary = container.get_summary()

        assert "Pauli Product Formula Container" in summary
        assert "Number of qubits: 2" in summary
        assert "Number of step terms: 3" in summary
        assert "Step repetitions: 4" in summary

    def test_legacy_hash_and_reordering_scale_are_unchanged(self):
        """Ordinary formulas keep their baseline hash and scale when reordered."""
        original = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, 0.5)], 4, 2, scale=1.7)
        assert original.content_hash() == "c2b1c5b0979d3d48"  # da61805e2 baseline
        assert original.reorder_terms([0]).content_hash() == original.content_hash()

    @pytest.mark.parametrize("inverse_reps", [1, 4])
    def test_sparse_factory_inherits_fusion(self, container, inverse_reps):
        """Use the ordinary fusion rule, including complete cancellation and identities."""
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
