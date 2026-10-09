"""Tests for the model Hamiltonian description data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import numpy as np
import pytest

from qdk_chemistry.data import (
    FermiHubbardModelHamiltonianDescription,
    Hamiltonian,
    HamiltonianDescription,
    LatticeGeometry,
    LatticeGraph,
    ModelHamiltonianDescription,
)
from qdk_chemistry.data.hamiltonian_description import model_hamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian


class _ShellModelDescription(ModelHamiltonianDescription):
    """A model whose parameters can be arrays or per-shell values."""

    @property
    def kind(self) -> str:
        return "shell_test"


@pytest.fixture
def model() -> FermiHubbardModelHamiltonianDescription:
    """A periodic four-site Fermi-Hubbard chain."""
    return FermiHubbardModelHamiltonianDescription(LatticeGeometry.chain(4, periodic=True), t=1.0, u=4, epsilon=-0.5)


def test_holds_lattice_and_float_parameters(model: FermiHubbardModelHamiltonianDescription) -> None:
    """The description keeps its lattice and stores each parameter as a float."""
    assert isinstance(model, ModelHamiltonianDescription)
    assert isinstance(model, HamiltonianDescription)
    assert model.lattice.num_sites == 4
    assert model.parameters == {"t": 1.0, "u": 4.0, "epsilon": -0.5}
    assert isinstance(model.parameters["u"], float)
    assert "FermiHubbardModelHamiltonianDescription" in model.get_summary()
    assert "Lattice sites: 4" in model.get_summary()
    with pytest.raises(AttributeError):
        model.parameters = {}
    with pytest.raises(TypeError):
        model.parameters["u"] = 8.0  # type: ignore[index]


@pytest.mark.parametrize("format_type", ["json", "hdf5"])
def test_file_round_trip(model: FermiHubbardModelHamiltonianDescription, tmp_path, format_type: str) -> None:
    """The lattice and parameters survive a JSON or HDF5 round trip."""
    suffix = "json" if format_type == "json" else "h5"
    path = tmp_path / f"model.model_hamiltonian_description.{suffix}"
    model.to_file(path, format_type)

    restored = FermiHubbardModelHamiltonianDescription.from_file(path, format_type)

    assert restored.parameters == model.parameters
    assert restored.lattice.content_hash() == model.lattice.content_hash()
    assert restored.content_hash() == model.content_hash()


def test_content_hash_tracks_parameters(model: FermiHubbardModelHamiltonianDescription) -> None:
    """Changing a parameter changes the hash."""
    same = FermiHubbardModelHamiltonianDescription(model.lattice, t=1.0, u=4.0, epsilon=-0.5)
    changed = FermiHubbardModelHamiltonianDescription(model.lattice, t=1.0, u=8.0, epsilon=-0.5)

    assert same.content_hash() == model.content_hash()
    assert changed.content_hash() != model.content_hash()


def test_materialize_calls_create_hubbard_hamiltonian(model: FermiHubbardModelHamiltonianDescription) -> None:
    """The Fermi-Hubbard description materializes to the Hubbard Hamiltonian on nearest-neighbor bonds."""
    hamiltonian = model.materialize()
    expected = create_hubbard_hamiltonian(LatticeGraph.from_geometry(model.lattice), epsilon=-0.5, t=1.0, U=4.0)

    assert isinstance(hamiltonian, Hamiltonian)
    np.testing.assert_allclose(hamiltonian.get_one_body_integrals()[0], expected.get_one_body_integrals()[0])
    np.testing.assert_allclose(hamiltonian.get_two_body_integrals()[0], expected.get_two_body_integrals()[0])


def test_materialize_doubles_hopping_through_two_periodic_images() -> None:
    """A periodic direction of length 2 joins each neighbor pair through two images."""
    model = FermiHubbardModelHamiltonianDescription(
        LatticeGeometry.square(2, 2, periodic_x=True, periodic_y=True), t=1.0, u=4.0
    )
    expected = create_hubbard_hamiltonian(
        LatticeGraph.square(2, 2, periodic_x=True, periodic_y=True), epsilon=0.0, t=1.0, U=4.0
    )

    np.testing.assert_allclose(model.materialize().get_one_body_integrals()[0], expected.get_one_body_integrals()[0])


def test_base_raises_for_what_a_model_must_provide() -> None:
    """The base has no kind and cannot materialize."""
    base = ModelHamiltonianDescription(LatticeGeometry.chain(2), {"t": 1.0})

    with pytest.raises(NotImplementedError, match="must implement materialize"):
        base.materialize()
    with pytest.raises(NotImplementedError, match="must implement kind"):
        _ = base.kind


def test_from_json_dispatches_on_kind(model: FermiHubbardModelHamiltonianDescription) -> None:
    """The base loader rebuilds the recorded model and rejects an unknown kind."""
    data = model.to_json()
    restored = ModelHamiltonianDescription.from_json(data)

    assert data["kind"] == "fermi_hubbard"
    assert type(restored) is FermiHubbardModelHamiltonianDescription
    assert restored.content_hash() == model.content_hash()
    with pytest.raises(ValueError, match="Unknown ModelHamiltonianDescription kind: 'shell_test'"):
        ModelHamiltonianDescription.from_json(_ShellModelDescription(LatticeGeometry.chain(2), {"t": 1.0}).to_json())


def test_array_and_shell_parameters_are_frozen() -> None:
    """Arrays become read-only float copies and shell mappings become read-only and ordered by shell."""
    epsilon = np.array([0.1, 0.2, 0.3])
    model = _ShellModelDescription(LatticeGeometry.chain(3), {"epsilon": epsilon, "j": {2: 0.5, 1: np.array([1, 2])}})
    epsilon[0] = 9.0

    stored = model.parameters["epsilon"]
    assert isinstance(stored, np.ndarray)
    np.testing.assert_array_equal(stored, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="read-only"):
        stored[0] = 1.0
    shells = model.parameters["j"]
    assert list(shells) == [1, 2]
    assert shells[1].dtype == np.float64
    with pytest.raises(TypeError):
        shells[3] = 1.0  # type: ignore[index]
    assert "epsilon=array(3,), j={1: array(2,), 2: 0.5}" in model.get_summary()


@pytest.mark.parametrize("shell", [0, -1, 1.5, True, "1"])
def test_rejects_shells_that_are_not_positive_integers(shell: object) -> None:
    """Shell keys are kept as given, so anything but a positive integer is rejected."""
    with pytest.raises(ValueError, match="j shell indices must be positive integers"):
        _ShellModelDescription(LatticeGeometry.chain(3), {"j": {shell: 1.0}})


def test_hash_distinguishes_parameter_forms() -> None:
    """A float, an array and a shell mapping of the same value hash differently."""
    lattice = LatticeGeometry.chain(3)

    def content_hash(value: model_hamiltonian.ModelParameter) -> str:
        return _ShellModelDescription(lattice, {"j": value}).content_hash()

    forms = [1.0, np.array([1.0]), np.array([1.0, 1.0]), {1: 1.0}, {2: 1.0}, {1: np.array([1.0])}]
    assert len({content_hash(value) for value in forms}) == len(forms)
    assert content_hash({2: 0.5, 1: 1.0}) == content_hash({1: 1.0, 2: 0.5})
    assert content_hash(np.array([1, 2])) == content_hash(np.array([1.0, 2.0]))


def test_array_and_shell_parameters_round_trip_through_json() -> None:
    """Arrays serialize as lists and shell mappings as objects keyed by shell."""
    model = _ShellModelDescription(
        LatticeGeometry.chain(3), {"epsilon": np.array([0.1, 0.2, 0.3]), "j": {2: 0.5, 1: np.array([1.0, 2.0])}}
    )
    data = json.loads(json.dumps(model.to_json()))

    assert data["parameters"] == {"epsilon": [0.1, 0.2, 0.3], "j": {"1": [1.0, 2.0], "2": 0.5}}
    parameters = {name: model_hamiltonian._decode_parameter(value) for name, value in data["parameters"].items()}
    restored = _ShellModelDescription(LatticeGeometry.from_json(json.dumps(data["lattice"])), parameters)
    assert restored.content_hash() == model.content_hash()
