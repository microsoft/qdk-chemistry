"""Tests for RDKit plugin in building structures."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import data
from qdk_chemistry.data import Structure


try:
    import rdkit
    import rdkit.Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

if RDKIT_AVAILABLE:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Mol
    
    from qdk_chemistry.constants import ANGSTROM_TO_BOHR


pytestmark = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
print("Running RDKit structure tests...")


def create_structure_from_rdkit(molecule: Mol) -> Structure:
    """Create a Structure object from an RDKit molecule."""
    symbols = []
    coords =  molecule.GetConformer().GetPositions() * ANGSTROM_TO_BOHR

    for atom in molecule.GetAtoms():
        symbols.append(f"{atom.GetSymbol()}")
    
    return Structure(symbols, coords)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestRDKitPlugin:
    """Test class for getting structure from RDKit."""


    def test_rdkit_water_structure(self):
        """Test construction water from RDKit."""

        water_rdkit = Chem.MolFromSmiles("O")
        water_rdkit = Chem.AddHs(water_rdkit)
        AllChem.EmbedMolecule(water_rdkit)
        AllChem.UFFOptimizeMolecule(water_rdkit)
        water = create_structure_from_rdkit(water_rdkit)
        assert water.get_num_atoms() == 3
        assert water.get_total_nuclear_charge() == 10
        assert water.get_atom_symbol(0) == "O"
        assert water.get_atom_symbol(1) == "H"
        assert water.get_atom_symbol(2) == "H"
        assert np.allclose(water.get_coordinates(), water_rdkit.GetConformer().GetPositions() * ANGSTROM_TO_BOHR) 

    def test_rdkit_ethanol_structure(self):
        """Test construction ethanol from RDKit."""

        ethanol_rdkit = Chem.MolFromSmiles("CCO")
        ethanol_rdkit = Chem.AddHs(ethanol_rdkit)
        AllChem.EmbedMolecule(ethanol_rdkit)
        AllChem.UFFOptimizeMolecule(ethanol_rdkit)
        ethanol = create_structure_from_rdkit(ethanol_rdkit)
        assert ethanol.get_num_atoms() == 9
        assert ethanol.get_total_nuclear_charge() == 26
        assert ethanol.get_atom_symbol(0) == "C"
        assert ethanol.get_atom_symbol(1) == "C"
        assert ethanol.get_atom_symbol(2) == "O"
        assert ethanol.get_atom_symbol(3) == "H"
        assert ethanol.get_atom_symbol(4) == "H"
        assert ethanol.get_atom_symbol(5) == "H"
        assert ethanol.get_atom_symbol(6) == "H"
        assert ethanol.get_atom_symbol(7) == "H"
        assert ethanol.get_atom_symbol(8) == "H"
        assert np.allclose(ethanol.get_coordinates(), ethanol_rdkit.GetConformer().GetPositions() * ANGSTROM_TO_BOHR) 