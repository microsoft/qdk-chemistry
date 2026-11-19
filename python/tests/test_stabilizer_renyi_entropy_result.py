"""Serialization, hashing, and immutability tests for entropy results."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import copysign

import h5py
import pytest

from qdk_chemistry.data import StabilizerRenyiEntropyResult, available_dataclasses, get_dataclass_type


def _result(truncated=False):
    """Create a valid result with optional truncation metadata."""
    return StabilizerRenyiEntropyResult(
        entropy=0.25,
        num_qubits=4,
        num_configurations=4,
        num_retained_configurations=3 if truncated else 4,
        truncation_enabled=truncated,
        coefficient_cutoff=1e-12 if truncated else None,
        discarded_weight=1e-26 if truncated else 0.0,
    )


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("format_type", ["json", "h5", "hdf5"])
def test_file_roundtrip(tmp_path, truncated, format_type):
    """Preserve result contents and hash through JSON and HDF5 files."""
    result = _result(truncated)
    path = tmp_path / f"state.stabilizer_renyi_entropy_result.{format_type}"
    result.to_file(path, format_type)
    loaded = StabilizerRenyiEntropyResult.from_file(path, format_type)
    assert type(loaded) is StabilizerRenyiEntropyResult
    assert loaded.to_json() == result.to_json()
    assert loaded.content_hash() == result.content_hash()


def test_json_dictionary_roundtrip():
    """Preserve tiny discarded probability through dictionary serialization."""
    result = _result(True)
    loaded = StabilizerRenyiEntropyResult.from_json(result.to_json())
    assert loaded.to_json() == result.to_json()
    assert loaded.discarded_weight == 1e-26


@pytest.mark.parametrize("entropy", [-0.0, 0.0])
def test_zero_entropy_is_canonical(entropy):
    """Give either sign of zero the same stored value, display and content hash."""
    data = _result().to_json()
    data["entropy"] = entropy
    result = StabilizerRenyiEntropyResult.from_json(data)
    data["entropy"] = 0.0
    expected = StabilizerRenyiEntropyResult.from_json(data)
    assert copysign(1.0, result.entropy) == 1.0
    assert copysign(1.0, result.to_json()["entropy"]) == 1.0
    assert result.get_summary().splitlines()[0] == "Stabilizer Renyi entropy (alpha=2): 0 bits"
    assert result.content_hash() == expected.content_hash()


@pytest.mark.parametrize("cutoff", [0, 0.0])
def test_coefficient_cutoff_is_canonical_float(cutoff):
    """Give equal integer and floating-point cutoffs the same serialized type and hash."""
    data = _result(True).to_json()
    data["coefficient_cutoff"] = cutoff
    result = StabilizerRenyiEntropyResult.from_json(data)
    data["coefficient_cutoff"] = 0.0
    expected = StabilizerRenyiEntropyResult.from_json(data)
    assert type(result.coefficient_cutoff) is float
    assert type(result.to_json()["coefficient_cutoff"]) is float
    assert result.to_json() == expected.to_json()
    assert result.content_hash() == expected.content_hash()


@pytest.mark.parametrize("truncated", [False, True])
def test_hdf5_group_can_be_reused(tmp_path, truncated):
    """Replace result attributes without leaving an obsolete cutoff or removing user metadata."""
    replacement = _result(truncated)
    with h5py.File(tmp_path / "reused.h5", "w") as file:
        group = file.create_group("result")
        group.attrs["user_note"] = "preserved"
        _result(not truncated).to_hdf5(group)
        replacement.to_hdf5(group)
        assert ("coefficient_cutoff" in group.attrs) is truncated
        assert group.attrs["user_note"] == "preserved"
        restored = StabilizerRenyiEntropyResult.from_hdf5(group)
        assert restored.to_json() == replacement.to_json()
        assert restored.content_hash() == replacement.content_hash()


def test_immutability_and_independent_serialization():
    """Keep the result immutable and independent of exported dictionaries."""
    result = _result()
    with pytest.raises(AttributeError):
        result.entropy = 0.0
    with pytest.raises(AttributeError):
        result.num_qubits = 2
    serialized = result.to_json()
    serialized["entropy"] = 0.0
    assert result.entropy == 0.25


def test_hashes_include_provenance():
    """Include individually varied entropy and provenance fields in the content hash."""
    result = _result()
    assert result.content_hash() == _result().content_hash()
    assert result.content_hash() != _result(True).content_hash()
    for field, value in (
        ("entropy", 0.5),
        ("num_qubits", 5),
        ("num_configurations", 5),
        ("num_retained_configurations", 2),
        ("coefficient_cutoff", 1e-10),
        ("discarded_weight", 1e-24),
    ):
        data = _result(True).to_json()
        data[field] = value
        assert StabilizerRenyiEntropyResult.from_json(data).content_hash() != _result(True).content_hash()


def test_registry_discovery():
    """Discover the result through the standard public data-class registry."""
    name = "stabilizer_renyi_entropy_result"
    assert name in available_dataclasses()
    assert get_dataclass_type(name) is StabilizerRenyiEntropyResult


def test_summary():
    """Display the entropy, representation and truncation metadata clearly."""
    summary = _result(True).get_summary()
    for text in ("0.25 bits", "alpha=2", "Active qubits: 4", "3/4", "jordan-wigner", "blocked", "1e-12", "1e-26"):
        assert text in summary
    assert "N_conf (retained/original): 3/4" in summary
    assert "Support" not in summary


@pytest.mark.parametrize("version", [None, "9.0.0"])
def test_json_version_required_and_checked(version):
    """Reject missing or unsupported JSON schema versions."""
    data = _result().to_json()
    if version is None:
        del data["version"]
    else:
        data["version"] = version
    with pytest.raises(RuntimeError):
        StabilizerRenyiEntropyResult.from_json(data)


@pytest.mark.parametrize("version", [None, "9.0.0"])
def test_hdf5_version_required_and_checked(tmp_path, version):
    """Reject missing or unsupported HDF5 schema versions."""
    with h5py.File(tmp_path / "version.h5", "w") as file:
        _result().to_hdf5(file)
        if version is None:
            del file.attrs["version"]
        else:
            file.attrs["version"] = version
        with pytest.raises(RuntimeError):
            StabilizerRenyiEntropyResult.from_hdf5(file)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("entropy", float("nan"), "entropy must be finite"),
        ("entropy", -0.1, "entropy must be finite"),
        ("entropy", 4.1, "entropy must be finite"),
        ("num_qubits", 0, "num_qubits must be a positive integer"),
        ("num_qubits", 2.5, "num_qubits must be a positive integer"),
        ("num_configurations", 17, "Configuration counts"),
        ("num_retained_configurations", 0, "num_retained_configurations must be a positive integer"),
        ("num_retained_configurations", 5, "Configuration counts"),
        ("truncation_enabled", "false", "truncation_enabled must be a boolean"),
        ("coefficient_cutoff", 1e-12, "Disabled truncation"),
        ("discarded_weight", -0.1, "discarded_weight must be finite"),
        ("discarded_weight", float("inf"), "discarded_weight must be finite"),
        ("discarded_weight", 1.0, "discarded_weight must be finite"),
        ("discarded_weight", 0.01, "Disabled truncation"),
        ("alpha", 3, "Only alpha=2"),
        ("log_base", 10, "Only alpha=2"),
        ("encoding", "bravyi-kitaev", "Only alpha=2"),
        ("fermion_mode_order", "interleaved", "Only alpha=2"),
    ],
)
def test_invalid_result_metadata(field, value, message):
    """Reject invalid or inconsistent result metadata with explicit errors."""
    data = _result().to_json()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        StabilizerRenyiEntropyResult.from_json(data)


@pytest.mark.parametrize("cutoff", [None, -1.0, float("nan"), float("inf")])
def test_enabled_truncation_requires_valid_cutoff(cutoff):
    """Require a finite, nonnegative cutoff when truncation is enabled."""
    data = _result(True).to_json()
    data["coefficient_cutoff"] = cutoff
    with pytest.raises(ValueError, match="Enabled truncation"):
        StabilizerRenyiEntropyResult.from_json(data)


@pytest.mark.parametrize("discarded_weight", [1e-26, 0.05])
def test_unchanged_configuration_counts_require_zero_discarded_weight(discarded_weight):
    """Reject discarded probability when every nonzero configuration was retained."""
    data = _result(True).to_json()
    data["num_retained_configurations"] = data["num_configurations"]
    data["discarded_weight"] = discarded_weight
    with pytest.raises(ValueError, match="Unchanged configuration counts require zero discarded_weight"):
        StabilizerRenyiEntropyResult.from_json(data)


@pytest.mark.parametrize("cutoff", [0.0, 1e-12])
def test_enabled_truncation_can_retain_every_configuration(cutoff):
    """Allow enabled truncation to retain every configuration without discarding probability."""
    data = _result(True).to_json()
    data["num_retained_configurations"] = data["num_configurations"]
    data["coefficient_cutoff"] = cutoff
    data["discarded_weight"] = 0.0
    result = StabilizerRenyiEntropyResult.from_json(data)
    assert result.truncation_enabled
    assert result.num_retained_configurations == result.num_configurations
    assert result.coefficient_cutoff == cutoff
    assert result.discarded_weight == 0.0
