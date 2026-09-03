"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import h5py
import numpy as np
import pytest

from qdk_chemistry.data._type_name import class_data_type_name, declares_data_type_name
from qdk_chemistry.data.unitary_representation.containers.foqcs import FoqcsContainer, FoqcsFamily
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


@pytest.fixture
def container(step_terms):
    """Create a PauliProductFormulaContainer instance for testing."""
    return PauliProductFormulaContainer(
        step_terms=step_terms,
        step_reps=4,
        num_qubits=2,
    )


@pytest.fixture
def foqcs_families():
    """Create FOQCS families for an Ising-like chain: an X field and a ZZ coupling."""
    return [
        FoqcsFamily(paulis=("X",), offset=0, abs_coeff=0.6, phase=0.0),
        FoqcsFamily(paulis=("Z", "Z"), offset=1, abs_coeff=0.8, phase=-0.7853981633974483),
    ]


@pytest.fixture
def foqcs_container(foqcs_families):
    """Create a FoqcsContainer instance for testing."""
    return FoqcsContainer(num_sites=3, families=foqcs_families, scale=1.9, power=2)


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
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
        assert len(restored.step_terms) == len(container.step_terms)

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


class TestFoqcsFamily:
    """Tests for the FoqcsFamily dataclass."""

    def test_attributes(self):
        """Test the attributes of FoqcsFamily."""
        family = FoqcsFamily(paulis=("Z", "Z"), offset=2, abs_coeff=0.5, phase=1.25)

        assert family.paulis == ("Z", "Z")
        assert family.offset == 2
        assert np.isclose(
            family.abs_coeff, 0.5, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            family.phase, 1.25, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_frozen(self):
        """Test that FoqcsFamily is immutable."""
        family = FoqcsFamily(paulis=("X",), offset=0, abs_coeff=1.0, phase=0.0)
        with pytest.raises(Exception, match="cannot assign to field 'phase'"):
            family.phase = 0.2

    def test_identity_family_has_empty_pattern(self):
        """A constant shift is carried as a degenerate family that applies no Pauli."""
        family = FoqcsFamily(paulis=(), offset=0, abs_coeff=0.3, phase=np.pi / 2)

        assert family.paulis == ()


class TestFoqcsContainer:
    """Tests for the FoqcsContainer class."""

    def test_basic_properties(self, foqcs_container):
        """Test basic properties of the container."""
        assert foqcs_container.type == "foqcs"
        assert foqcs_container.num_sites == 3
        assert foqcs_container.num_families == 2
        assert foqcs_container.power == 2
        assert np.isclose(
            foqcs_container.scale,
            1.9,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_qubit_layout(self, foqcs_container):
        """The ancilla layout is [subPrepReg | xReg | zReg]: one qubit per family plus two site registers."""
        assert foqcs_container.num_target_qubits == 3
        assert foqcs_container.num_prepare_ancillas == 2 + 2 * 3
        assert foqcs_container.num_qubits == 3 + (2 + 2 * 3)

    def test_declares_its_own_data_type_name(self):
        """Regression: the container must declare its own wire-format identifier.

        A stale ``_data_type_name`` class attribute silently inherited
        ``"block_encoding_container"`` from the base, which mis-validates
        filename suffixes on save and load.
        """
        assert declares_data_type_name(FoqcsContainer)
        assert class_data_type_name(FoqcsContainer) == "foqcs_container"

    def test_attributes_are_immutable_after_construction(self, foqcs_container):
        """The public attributes are plain fields, but DataClass freezes them post-construction."""
        with pytest.raises(AttributeError):
            foqcs_container.num_sites = 5
        with pytest.raises(AttributeError):
            foqcs_container.scale = 1.0

    def test_families_are_copied_from_the_caller(self, foqcs_families):
        """Mutating the caller's list must not reach into the constructed container."""
        constructed = FoqcsContainer(num_sites=3, families=foqcs_families, scale=1.9)
        foqcs_families.append(FoqcsFamily(paulis=("Y",), offset=0, abs_coeff=0.1, phase=0.0))

        assert constructed.num_families == 2

    def test_to_json_roundtrip(self, foqcs_container):
        """Test JSON serialization and deserialization roundtrip."""
        json_data = foqcs_container.to_json()
        restored = FoqcsContainer.from_json(json.loads(json.dumps(json_data)))

        assert restored.type == foqcs_container.type
        assert restored.num_sites == foqcs_container.num_sites
        assert restored.power == foqcs_container.power
        assert restored.content_hash() == foqcs_container.content_hash()

        for restored_family, original in zip(restored.families, foqcs_container.families, strict=True):
            assert restored_family.paulis == original.paulis
            assert restored_family.offset == original.offset

    def test_to_hdf5_roundtrip(self, foqcs_container, tmp_path):
        """Test HDF5 serialization and deserialization roundtrip."""
        file_path = tmp_path / "foqcs_container.h5"

        with h5py.File(file_path, "w") as f:
            foqcs_container.to_hdf5(f.create_group("container"))

        with h5py.File(file_path, "r") as f:
            restored = FoqcsContainer.from_hdf5(f["container"])

        assert restored.content_hash() == foqcs_container.content_hash()

    def test_identity_family_survives_roundtrip(self):
        """An empty Pauli pattern must not be corrupted by serialization."""
        original = FoqcsContainer(
            num_sites=2,
            families=[
                FoqcsFamily(paulis=(), offset=0, abs_coeff=0.4, phase=np.pi / 2),
                FoqcsFamily(paulis=("X",), offset=0, abs_coeff=0.9, phase=0.0),
            ],
            scale=1.1,
        )

        restored = FoqcsContainer.from_json(json.loads(json.dumps(original.to_json())))

        assert restored.families[0].paulis == ()
        assert restored.content_hash() == original.content_hash()

    def test_content_hash_distinguishes_containers(self, foqcs_families):
        """Containers differing in any identifying field must hash differently."""
        base = FoqcsContainer(num_sites=3, families=foqcs_families, scale=1.9, power=2)

        assert (
            base.content_hash()
            != FoqcsContainer(num_sites=3, families=foqcs_families, scale=1.9, power=3).content_hash()
        )
        assert (
            base.content_hash()
            != FoqcsContainer(num_sites=4, families=foqcs_families, scale=1.9, power=2).content_hash()
        )
        assert (
            base.content_hash()
            != FoqcsContainer(num_sites=3, families=foqcs_families[:1], scale=1.9, power=2).content_hash()
        )

    def test_summary(self, foqcs_container):
        """Test the summary generation of the container."""
        summary = foqcs_container.get_summary()

        assert "FOQCS Container" in summary
        assert "Power: 2" in summary
        assert "Families (2)" in summary
        assert "ZZ (offset 1)" in summary

    def test_eigenvalue_from_phase_is_not_supported(self, foqcs_container):
        """A raw block encoding has no eigenvalue-phase relation; the walk container does."""
        with pytest.raises(NotImplementedError, match="LCUWalkContainer"):
            foqcs_container.eigenvalue_from_phase(0.25)

    def test_combine_is_not_supported(self, foqcs_container):
        """Combining FOQCS block encodings is not defined."""
        with pytest.raises(NotImplementedError, match="does not support combination"):
            foqcs_container.combine(foqcs_container)
