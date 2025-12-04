"""Example for using RDKit to build structures."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import logging
from collections.abc import Sequence
import argparse

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from qdk_chemistry.constants import ANGSTROM_TO_BOHR

LOGGER = logging.getLogger(__file__)


def create_structure_from_rdkit(molecule: Mol) -> Structure:
    """Create a Structure object from an RDKit molecule."""
    symbols = []
    coords = molecule.GetConformer().GetPositions() * ANGSTROM_TO_BOHR

    for atom in molecule.GetAtoms():
        symbols.append(f"{atom.GetSymbol()}")

    return Structure(symbols, coords)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options in the same order as the workflow steps."""
    parser = argparse.ArgumentParser(description="simple water demo")

    parser.add_argument(
        "--basis",
        default="cc-pvdz",
        help="Basis set applied to the SCF solver (default: cc-pvdz).",
    )
    parser.add_argument(
        "--charge", type=int, default=0, help="Total molecular charge (default: 0)."
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=1,
        help="Spin multiplicity (2S+1). Default assumes a singlet (1).",
    )
    return parser.parse_args(argv)


    
def main(argv: Sequence[str] | None = None) -> None:
    """Example of constructing water from RDKit."""

    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)

    ########################################################################################
    # 1. Load the target structure (use rdkit).
    ########################################################################################
    

    water_rdkit = Chem.MolFromSmiles("O")
    water_rdkit = Chem.AddHs(water_rdkit)
    AllChem.EmbedMolecule(water_rdkit)
    AllChem.UFFOptimizeMolecule(water_rdkit)
    water = create_structure_from_rdkit(water_rdkit)
    
    LOGGER.info(water.get_summary())
    LOGGER.info("XYZ Geometry in Angstrom")
    LOGGER.info(water.to_xyz())    

    ########################################################################################
    # 2. Run the SCF stage to obtain the reference wavefunction.
    ########################################################################################
    nuclear_repulsion = water.calculate_nuclear_repulsion_energy()
    scf_solver = create("scf_solver")
    scf_solver.settings().set("basis_set", args.basis)
    e_scf, scf_wavefunction = scf_solver.run(water, args.charge, args.spin)
    total_scf_energy = e_scf + nuclear_repulsion
    LOGGER.info("SCF Energy: %.8f Hartree", total_scf_energy)

if __name__ == "__main__":
    main()
