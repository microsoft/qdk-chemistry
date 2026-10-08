"""Tests for the ModelHamiltonianDescription data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import Hamiltonian, LatticeGeometry, LatticeGraph, ModelHamiltonianDescription, QubitOperator
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian, create_ising_hamiltonian


@pytest.fixture
def model() -> ModelHamiltonianDescription:
    """A periodic four-site Hubbard chain."""
    return ModelHamiltonianDescription(
        "hubbard", LatticeGeometry.chain(4, periodic=True), {"epsilon": 0.0, "t": 1.0, "U": 4}
    )


def test_holds_model_lattice_and_float_parameters(model: ModelHamiltonianDescription) -> None:
    """The description keeps its model and lattice and stores each parameter as a float."""
    assert model.model == "hubbard"
    assert model.lattice.num_sites == 4
    assert model.parameters == {"epsilon": 0.0, "t": 1.0, "U": 4.0}
    assert isinstance(model.parameters["U"], float)
    assert "Model: hubbard" in model.get_summary()
    assert "Lattice sites: 4" in model.get_summary()
    with pytest.raises(AttributeError):
        model.parameters = {}


def test_rejects_a_lattice_that_is_not_a_geometry() -> None:
    """Only a LatticeGeometry is accepted as the lattice."""
    with pytest.raises(TypeError, match="LatticeGeometry"):
        ModelHamiltonianDescription("hubbard", "chain", {"t": 1.0})


@pytest.mark.parametrize("format_type", ["json", "hdf5"])
def test_file_round_trip(model: ModelHamiltonianDescription, tmp_path, format_type: str) -> None:
    """The model, lattice and parameters survive a JSON or HDF5 round trip."""
    suffix = "json" if format_type == "json" else "h5"
    path = tmp_path / f"model.model_hamiltonian_description.{suffix}"
    model.to_file(path, format_type)

    restored = ModelHamiltonianDescription.from_file(path, format_type)

    assert restored.model == model.model
    assert restored.parameters == model.parameters
    assert restored.lattice.content_hash() == model.lattice.content_hash()
    assert restored.content_hash() == model.content_hash()


def test_content_hash_tracks_model_and_parameters(model: ModelHamiltonianDescription) -> None:
    """Changing the model or a parameter changes the hash; parameter order does not."""
    reordered = ModelHamiltonianDescription("hubbard", model.lattice, {"U": 4.0, "t": 1.0, "epsilon": 0.0})
    changed = ModelHamiltonianDescription("hubbard", model.lattice, {"epsilon": 0.0, "t": 1.0, "U": 8.0})
    renamed = ModelHamiltonianDescription("ppp", model.lattice, model.parameters)

    assert reordered.content_hash() == model.content_hash()
    assert changed.content_hash() != model.content_hash()
    assert renamed.content_hash() != model.content_hash()


def test_materialize_builds_a_fermionic_hamiltonian(model: ModelHamiltonianDescription) -> None:
    """A fermionic model materializes to the Hamiltonian its create function builds on nearest neighbors."""
    hamiltonian = model.materialize()
    expected = create_hubbard_hamiltonian(LatticeGraph.from_geometry(model.lattice), epsilon=0.0, t=1.0, U=4.0)

    assert isinstance(hamiltonian, Hamiltonian)
    np.testing.assert_allclose(hamiltonian.get_one_body_integrals()[0], expected.get_one_body_integrals()[0])
    np.testing.assert_allclose(hamiltonian.get_two_body_integrals()[0], expected.get_two_body_integrals()[0])


def test_materialize_builds_a_spin_qubit_operator() -> None:
    """A spin model materializes to a QubitOperator."""
    lattice = LatticeGeometry.chain(3)
    qubit_operator = ModelHamiltonianDescription("Ising", lattice, {"j": 1.0, "h": 0.5}).materialize()
    expected = create_ising_hamiltonian(LatticeGraph.from_geometry(lattice), j=1.0, h=0.5)

    assert isinstance(qubit_operator, QubitOperator)
    assert dict(qubit_operator.get_real_coefficients()) == dict(expected.get_real_coefficients())


def test_materialize_rejects_a_model_without_a_create_function() -> None:
    """A model the create functions do not cover cannot be materialized."""
    description = ModelHamiltonianDescription("plaquette", LatticeGeometry.chain(2), {"t": 1.0})
    with pytest.raises(ValueError, match="No Hamiltonian factory for model 'plaquette'"):
        description.materialize()
