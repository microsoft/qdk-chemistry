"""Tests for the ModelHamiltonianDescription data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.data import LatticeGeometry, ModelHamiltonianDescription


@pytest.fixture
def model() -> ModelHamiltonianDescription:
    """A periodic four-site Hubbard chain."""
    return ModelHamiltonianDescription(LatticeGeometry.chain(4, periodic=True), {"t": 1.0, "u": 4})


def test_holds_lattice_and_float_parameters(model: ModelHamiltonianDescription) -> None:
    """The description keeps its lattice and stores each parameter as a float."""
    assert model.lattice.num_sites == 4
    assert model.parameters == {"t": 1.0, "u": 4.0}
    assert isinstance(model.parameters["u"], float)
    assert "Lattice sites: 4" in model.get_summary()
    with pytest.raises(AttributeError):
        model.parameters = {}


def test_rejects_a_lattice_that_is_not_a_geometry() -> None:
    """Only a LatticeGeometry is accepted as the lattice."""
    with pytest.raises(TypeError, match="LatticeGeometry"):
        ModelHamiltonianDescription("chain", {"t": 1.0})


@pytest.mark.parametrize("format_type", ["json", "hdf5"])
def test_file_round_trip(model: ModelHamiltonianDescription, tmp_path, format_type: str) -> None:
    """The lattice and parameters survive a JSON or HDF5 round trip."""
    suffix = "json" if format_type == "json" else "h5"
    path = tmp_path / f"model.model_hamiltonian_description.{suffix}"
    model.to_file(path, format_type)

    restored = ModelHamiltonianDescription.from_file(path, format_type)

    assert restored.parameters == model.parameters
    assert restored.lattice.content_hash() == model.lattice.content_hash()
    assert restored.content_hash() == model.content_hash()


def test_content_hash_tracks_parameters(model: ModelHamiltonianDescription) -> None:
    """Changing a parameter changes the hash; parameter order does not."""
    reordered = ModelHamiltonianDescription(model.lattice, {"u": 4.0, "t": 1.0})
    changed = ModelHamiltonianDescription(model.lattice, {"t": 1.0, "u": 8.0})

    assert reordered.content_hash() == model.content_hash()
    assert changed.content_hash() != model.content_hash()
