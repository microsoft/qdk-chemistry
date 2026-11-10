"""Serialization, hashing, and immutability tests for entropy results."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import h5py
import pytest

from qdk_chemistry.data import StabilizerRenyiEntropyResult, available_dataclasses, get_dataclass_type


def _result(truncated=False):
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
    result = _result(truncated)
    path = tmp_path / f"state.stabilizer_renyi_entropy_result.{format_type}"
    result.to_file(path, format_type)
    loaded = StabilizerRenyiEntropyResult.from_file(path, format_type)
    assert type(loaded) is StabilizerRenyiEntropyResult
    assert loaded.to_json() == result.to_json()
    assert loaded.content_hash() == result.content_hash()


def test_json_dictionary_roundtrip():
    result = _result(True)
    loaded = StabilizerRenyiEntropyResult.from_json(result.to_json())
    assert loaded.to_json() == result.to_json()
    assert loaded.discarded_weight == 1e-26


def test_immutability_and_independent_serialization():
    result = _result()
    with pytest.raises(AttributeError):
        result.entropy = 0.0
    with pytest.raises(AttributeError):
        result.num_qubits = 2
    serialized = result.to_json()
    serialized["entropy"] = 0.0
    assert result.entropy == 0.25


def test_hashes_include_provenance():
    result = _result()
    assert result.content_hash() == _result().content_hash()
    assert result.content_hash() != _result(True).content_hash()
    for field, value in (("entropy", 0.5), ("num_qubits", 5), ("num_configurations", 5)):
        data = _result(True).to_json()
        data[field] = value
        assert StabilizerRenyiEntropyResult.from_json(data).content_hash() != _result(True).content_hash()


def test_registry_discovery():
    name = "stabilizer_renyi_entropy_result"
    assert name in available_dataclasses()
    assert get_dataclass_type(name) is StabilizerRenyiEntropyResult


def test_summary():
    summary = _result(True).get_summary()
    for text in ("0.25 bits", "alpha=2", "Active qubits: 4", "3/4", "jordan-wigner", "blocked", "1e-12", "1e-26"):
        assert text in summary
    assert "N_conf (retained/original): 3/4" in summary
    assert "Support" not in summary


@pytest.mark.parametrize("version", [None, "9.0.0"])
def test_json_version_required_and_checked(version):
    data = _result().to_json()
    if version is None:
        del data["version"]
    else:
        data["version"] = version
    with pytest.raises(RuntimeError):
        StabilizerRenyiEntropyResult.from_json(data)


@pytest.mark.parametrize("version", [None, "9.0.0"])
def test_hdf5_version_required_and_checked(tmp_path, version):
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
    data = _result().to_json()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        StabilizerRenyiEntropyResult.from_json(data)


@pytest.mark.parametrize("cutoff", [None, -1.0, float("nan"), float("inf")])
def test_enabled_truncation_requires_valid_cutoff(cutoff):
    data = _result(True).to_json()
    data["coefficient_cutoff"] = cutoff
    with pytest.raises(ValueError, match="Enabled truncation"):
        StabilizerRenyiEntropyResult.from_json(data)
