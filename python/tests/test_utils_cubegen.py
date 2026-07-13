"""Tests for cube file generation utilities."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

try:
    import pyscf  # noqa: F401
    from pyscf import gto

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")

if PYSCF_AVAILABLE:
    import qdk_chemistry.utils.cubegen as cubegen_utils
    from qdk_chemistry.data import Orbitals, Structure
    from qdk_chemistry.plugins.pyscf.conversion import basis_to_pyscf_mol, pyscf_mol_to_qdk_basis
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


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestCubegen:
    """Tests for cube file generation utilities."""

    def test_generate_cubefiles_singlet(self):
        orbitals = _o2_orbitals()
        assert generate_cubefiles_from_orbitals(orbitals, indices=[]) == {}

    def test_generate_cubefiles_doublet(self):
        orbitals = _no_orbitals()
        assert generate_cubefiles_from_orbitals(orbitals, indices=[]) == {}

    def test_generate_cubefiles_are_identical_for_singlet_and_triplet(self, monkeypatch):
        orbitals = _o2_orbitals()
        basis_set = orbitals.get_basis_set()
        molecules = iter(
            [
                basis_to_pyscf_mol(basis_set, charge=0, multiplicity=1),
                basis_to_pyscf_mol(basis_set, charge=0, multiplicity=3),
            ]
        )
        monkeypatch.setattr(cubegen_utils, "basis_to_pyscf_mol", lambda *_args, **_kwargs: next(molecules))

        singlet_cubes = generate_cubefiles_from_orbitals(orbitals, indices=[0], grid_size=(4, 4, 4))
        triplet_cubes = generate_cubefiles_from_orbitals(orbitals, indices=[0], grid_size=(4, 4, 4))

        assert singlet_cubes == triplet_cubes
