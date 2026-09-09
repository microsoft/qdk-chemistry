"""Tests for the standalone AuxiliaryBasis class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pickle
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.data import (
    AOType,
    AuxiliaryBasis,
    AuxiliaryBasisCollection,
    AuxiliaryBasisRole,
    BasisSet,
    OrbitalType,
    Shell,
    Structure,
    with_auxiliary_basis,
)


def _make_structure(symbols=("H", "H", "H")):
    positions = np.zeros((len(symbols), 3))
    positions[:, 0] = np.arange(len(symbols))
    return Structure(list(symbols), positions)


def _make_shells():
    return [
        Shell(2, OrbitalType.P, [1.0], [0.7]),
        Shell(0, OrbitalType.D, [3.0], [0.8]),
        Shell(2, OrbitalType.S, [2.0], [0.9]),
        Shell(0, OrbitalType.S, [0.5, 4.0], [0.4, 0.6]),
    ]


def test_construction_structure_ownership_and_canonical_ordering():
    structure = _make_structure()
    structure_hash = structure.content_hash()
    custom = AuxiliaryBasis(_make_shells(), structure)
    named = AuxiliaryBasis("density-fit", _make_shells(), structure)

    assert custom.get_name() == AuxiliaryBasis.custom_name == "custom_aux"
    assert named.get_name() == "density-fit"
    assert named.get_atomic_orbital_type() == AOType.Spherical
    assert named.get_num_atoms() == 3
    assert named.get_num_shells() == 4

    del structure
    assert named.get_structure().content_hash() == structure_hash

    shells = named.get_shells()
    assert [(shell.atom_index, shell.orbital_type) for shell in shells] == [
        (0, OrbitalType.S),
        (0, OrbitalType.D),
        (2, OrbitalType.S),
        (2, OrbitalType.P),
    ]
    assert shells[0].exponents[0] == 4.0
    assert named.get_shells_for_atom(1) == []


def test_orbital_counts_and_shell_access():
    structure = _make_structure()
    spherical = AuxiliaryBasis("spherical", _make_shells(), structure)
    cartesian = AuxiliaryBasis("cartesian", _make_shells(), structure, AOType.Cartesian)

    assert spherical.get_num_auxiliary_orbitals() == 10
    assert cartesian.get_num_auxiliary_orbitals() == 11
    assert cartesian.get_atomic_orbital_type() == AOType.Cartesian
    assert spherical.get_shell(0).orbital_type == OrbitalType.S
    assert len(spherical.get_shells_for_atom(0)) == 2

    with pytest.raises(IndexError, match="shell index"):
        spherical.get_shell(spherical.get_num_shells())
    with pytest.raises(IndexError, match="Atom index"):
        spherical.get_shells_for_atom(spherical.get_num_atoms())


def test_shell_access_returns_copy_of_immutable_basis_data():
    structure = _make_structure()
    auxiliary = AuxiliaryBasis("aux", _make_shells(), structure)
    original_type = auxiliary.get_shell(0).orbital_type

    detached_shell = auxiliary.get_shell(0)
    detached_shell.orbital_type = OrbitalType.I

    assert auxiliary.get_shell(0).orbital_type == original_type


def test_factories_load_by_name_element_and_index():
    structure = _make_structure(("O", "H", "H"))
    basis_name = "def2-universal-jfit"

    by_name = AuxiliaryBasis.from_basis_name(basis_name.upper(), structure, AOType.Cartesian)
    by_element = AuxiliaryBasis.from_element_map({"H": basis_name, "O": basis_name}, structure)
    by_index = AuxiliaryBasis.from_index_map({0: basis_name, 1: basis_name, 2: basis_name}, structure)

    assert by_name.get_name() == basis_name
    assert by_name.get_atomic_orbital_type() == AOType.Cartesian
    assert by_element.get_name() == AuxiliaryBasis.custom_name
    assert by_index.get_name() == AuxiliaryBasis.custom_name
    assert all(basis.get_num_shells() > 0 for basis in (by_name, by_element, by_index))
    assert all(basis.get_num_atoms() == 3 for basis in (by_name, by_element, by_index))


def test_validation_rejects_invalid_construction_and_maps():
    structure = _make_structure(("H",))
    shell = Shell(0, OrbitalType.S, [2.0], [1.0])

    with pytest.raises(TypeError, match="incompatible constructor arguments"):
        AuxiliaryBasis([shell])
    with pytest.raises(ValueError, match="nullptr"):
        AuxiliaryBasis([shell], None)
    with pytest.raises(ValueError, match="name cannot be empty"):
        AuxiliaryBasis("", [shell], structure)
    with pytest.raises(ValueError, match="shells cannot be empty"):
        AuxiliaryBasis([], structure)
    with pytest.raises(ValueError, match="radial powers"):
        AuxiliaryBasis([Shell(0, OrbitalType.S, [2.0], [1.0], [0])], structure)
    with pytest.raises(ValueError, match="local-potential"):
        AuxiliaryBasis([Shell(0, OrbitalType.UL, [2.0], [1.0])], structure)
    with pytest.raises(ValueError, match="atom_index"):
        AuxiliaryBasis([Shell(1, OrbitalType.S, [2.0], [1.0])], structure)

    water = _make_structure(("O", "H", "H"))
    basis_name = "def2-universal-jfit"
    with pytest.raises(ValueError, match="element"):
        AuxiliaryBasis.from_element_map({"H": basis_name}, water)
    with pytest.raises(ValueError, match="atom index"):
        AuxiliaryBasis.from_index_map({0: basis_name, 1: basis_name}, water)
    with pytest.raises(ValueError, match="Basis set file does not exist"):
        AuxiliaryBasis.from_element_map({"H": basis_name, "O": "invalid-basis-set"}, water)


def test_content_hash_tracks_all_owned_data():
    structure = _make_structure()
    shells = _make_shells()
    baseline = AuxiliaryBasis("aux", shells, structure)

    assert baseline.content_hash() == AuxiliaryBasis("aux", shells, structure).content_hash()
    assert baseline.content_hash() != AuxiliaryBasis("other", shells, structure).content_hash()
    assert baseline.content_hash() != AuxiliaryBasis("aux", shells, structure, AOType.Cartesian).content_hash()

    changed_shells = _make_shells()
    changed_shells[0].coefficients[0] = 0.5
    assert baseline.content_hash() != AuxiliaryBasis("aux", changed_shells, structure).content_hash()

    shifted_structure = Structure(
        ["H", "H", "H"],
        np.array([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert baseline.content_hash() != AuxiliaryBasis("aux", shells, shifted_structure).content_hash()


def test_summary_and_data_type():
    basis = AuxiliaryBasis("summary-aux", _make_shells(), _make_structure())

    summary = basis.get_summary()
    assert "AuxiliaryBasis: summary-aux" in summary
    assert "Number of auxiliary orbitals: 10" in summary
    assert str(basis) == summary
    assert repr(basis) == summary
    assert AuxiliaryBasis._data_type_name == "auxiliary_basis"


def test_json_string_and_file_round_trips(tmp_path: Path):
    basis = AuxiliaryBasis("json-aux", _make_shells(), _make_structure(), AOType.Cartesian)

    json_data = basis.to_json()
    assert isinstance(json_data, str)
    assert AuxiliaryBasis.from_json(json_data).content_hash() == basis.content_hash()

    json_file = tmp_path / "roundtrip.auxiliary_basis.json"
    basis.to_json_file(json_file)
    assert AuxiliaryBasis.from_json_file(json_file).content_hash() == basis.content_hash()

    generic_file = tmp_path / "generic.auxiliary_basis.json"
    basis.to_file(generic_file, "json")
    assert AuxiliaryBasis.from_file(generic_file, "json").content_hash() == basis.content_hash()


def test_hdf5_file_round_trip_and_pickle(tmp_path: Path):
    basis = AuxiliaryBasis("hdf5-aux", _make_shells(), _make_structure())

    hdf5_file = tmp_path / "roundtrip.auxiliary_basis.h5"
    basis.to_hdf5_file(hdf5_file)
    assert AuxiliaryBasis.from_hdf5_file(hdf5_file).content_hash() == basis.content_hash()

    restored = pickle.loads(pickle.dumps(basis))
    assert restored.content_hash() == basis.content_hash()
    assert restored.get_structure().content_hash() == basis.get_structure().content_hash()


def test_role_keyed_collection_and_persistence(tmp_path: Path):
    structure = _make_structure()
    jfit = AuxiliaryBasis("jfit-basis", _make_shells(), structure)
    jkfit = AuxiliaryBasis("jkfit-basis", _make_shells(), structure)
    empty = AuxiliaryBasisCollection()

    jk_collection = with_auxiliary_basis(empty, AuxiliaryBasisRole.JKFIT, jkfit)
    assert not empty.has_auxiliary_basis(AuxiliaryBasisRole.JKFIT)
    assert jk_collection.resolve_auxiliary_basis(AuxiliaryBasisRole.JFIT).get_name() == "jkfit-basis"

    j_collection = with_auxiliary_basis(empty, AuxiliaryBasisRole.JFIT, jfit)
    with pytest.raises(IndexError, match=r"JKFIT|jkfit"):
        j_collection.resolve_auxiliary_basis(AuxiliaryBasisRole.JKFIT)

    collection = with_auxiliary_basis(jk_collection, AuxiliaryBasisRole.JFIT, jfit)
    assert set(collection.get_auxiliary_bases()) == {
        AuxiliaryBasisRole.JFIT,
        AuxiliaryBasisRole.JKFIT,
    }
    assert collection.get_auxiliary_basis(AuxiliaryBasisRole.JFIT).get_name() == "jfit-basis"
    assert collection.content_hash() != empty.content_hash()

    restored = AuxiliaryBasisCollection.from_json(collection.to_json())
    assert restored.content_hash() == collection.content_hash()

    json_path = tmp_path / "fits.auxiliary_basis_collection.json"
    collection.to_json_file(json_path)
    restored = AuxiliaryBasisCollection.from_json_file(json_path)
    assert restored.content_hash() == collection.content_hash()

    hdf5_path = tmp_path / "fits.auxiliary_basis_collection.h5"
    collection.to_hdf5_file(hdf5_path)
    restored = AuxiliaryBasisCollection.from_hdf5_file(hdf5_path)
    assert restored.content_hash() == collection.content_hash()

    assert pickle.loads(pickle.dumps(collection)).content_hash() == collection.content_hash()


def test_collection_rejects_invalid_entries():
    structure = _make_structure()
    matching = AuxiliaryBasis("matching", _make_shells(), structure)
    shifted = Structure(
        ["H", "H", "H"],
        np.array([[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    mismatched = AuxiliaryBasis("shifted", _make_shells(), shifted)

    with pytest.raises(ValueError, match="same molecular structure"):
        AuxiliaryBasisCollection(
            {
                AuxiliaryBasisRole.JFIT: matching,
                AuxiliaryBasisRole.RIFIT: mismatched,
            }
        )

    with pytest.raises(ValueError, match="cannot be null"):
        AuxiliaryBasisCollection({AuxiliaryBasisRole.JFIT: None})

    collection = AuxiliaryBasisCollection({AuxiliaryBasisRole.JFIT: matching})
    with pytest.raises(ValueError, match="same molecular structure"):
        with_auxiliary_basis(collection, AuxiliaryBasisRole.RIFIT, mismatched)


def test_collection_is_independent_of_primary_basis():
    structure = _make_structure()
    primary = BasisSet("primary", _make_shells(), structure)
    primary_hash = primary.content_hash()
    auxiliary = AuxiliaryBasis("jfit", _make_shells(), structure)

    collection = AuxiliaryBasisCollection({AuxiliaryBasisRole.JFIT: auxiliary})

    assert primary.content_hash() == primary_hash
    assert "auxiliary_bases" not in primary.to_json()
    assert collection.get_auxiliary_basis(AuxiliaryBasisRole.JFIT).content_hash() == auxiliary.content_hash()
