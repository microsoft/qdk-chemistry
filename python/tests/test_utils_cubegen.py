"""Tests for cube file generation utilities."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from pyscf import gto

from qdk_chemistry.data import Orbitals, Structure
from qdk_chemistry.plugins.pyscf.conversion import pyscf_mol_to_qdk_basis
from qdk_chemistry.utils.cubegen import generate_cubefiles_from_orbitals


def _diatomic_orbitals(symbols: tuple[str, str], bond_length: float, multiplicity: int) -> Orbitals:
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
    structure = Structure(list(symbols), coordinates)
    mol = gto.M(
        atom=list(zip(symbols, coordinates, strict=True)),
        basis="sto-3g",
        unit="Bohr",
        spin=multiplicity - 1,
    )
    basis_set = pyscf_mol_to_qdk_basis(mol, structure, "sto-3g")
    return Orbitals(np.eye(mol.nao_nr()), None, None, basis_set)


def _o2_orbitals() -> Orbitals:
    return _diatomic_orbitals(("O", "O"), bond_length=2.282, multiplicity=3)


def _no_orbitals() -> Orbitals:
    return _diatomic_orbitals(("N", "O"), bond_length=2.175, multiplicity=2)


def test_generate_cubefiles_uses_singlet_fallback_for_even_electron_o2(capfd):
    # The fallback only checks electron-spin parity, so it selects a singlet even
    # though O2 has a triplet ground state.
    orbitals = _o2_orbitals()

    assert generate_cubefiles_from_orbitals(orbitals, indices=[]) == {}
    captured = capfd.readouterr()
    assert "using neutral singlet for cube-file generation" in captured.out


def test_generate_cubefiles_tries_singlet_then_doublet(capfd):
    orbitals = _no_orbitals()

    assert generate_cubefiles_from_orbitals(orbitals, indices=[]) == {}
    captured = capfd.readouterr()
    assert "using neutral doublet for cube-file generation" in captured.out


def test_generate_cubefiles_uses_explicit_charge_and_multiplicity(capfd):
    orbitals = _no_orbitals()

    assert generate_cubefiles_from_orbitals(orbitals, indices=[], charge=1, multiplicity=1) == {}
    captured = capfd.readouterr()
    assert "Charge and spin multiplicity were not provided" not in captured.out


@pytest.mark.parametrize(("charge", "multiplicity"), [(0, None), (None, 1)])
def test_generate_cubefiles_rejects_incomplete_charge_and_multiplicity(charge, multiplicity):
    with pytest.raises(ValueError, match="charge and multiplicity must either both be provided or both be None"):
        generate_cubefiles_from_orbitals(_no_orbitals(), indices=[], charge=charge, multiplicity=multiplicity)
