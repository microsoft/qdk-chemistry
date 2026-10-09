"""Tests for the model Hamiltonian description data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian


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
    path = tmp_path / f"model.fermi_hubbard_model_hamiltonian_description.{suffix}"
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
