"""Example for using RDKit to build structures."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
from collections.abc import Sequence

from qdk_chemistry.algorithms import create
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import Logger
from rdkit import Chem
from rdkit.Chem import AllChem, Mol


def create_structure_from_rdkit(molecule: Mol) -> Structure:
    """Create a Structure object from an RDKit molecule."""
    symbols = []
    coords = (
        molecule.GetConformer().GetPositions() * ANGSTROM_TO_BOHR
    )  # qdk_chemistry uses Bohr units

    for atom in molecule.GetAtoms():
        symbols.append(f"{atom.GetSymbol()}")

    return Structure(symbols, coords)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options in the same order as the workflow steps."""
    parser = argparse.ArgumentParser(
        description="Build QDK Chemistry structure from RDKit SMILES and run SCF"
    )

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

    Logger.set_global_level("info")
    args = parse_args(argv)

    ########################################################################################
    # 1. Load the target structure (use RDKit).
    ########################################################################################

    # Create water from SMILES string
    water_rdkit = Chem.MolFromSmiles("O")
    # Add hydrogens to the molecule
    water_rdkit = Chem.AddHs(water_rdkit)
    # Generate 3D coordinates
    AllChem.EmbedMolecule(water_rdkit)
    # Optimize geometry using UFF force field provided by RDKit.
    AllChem.UFFOptimizeMolecule(water_rdkit)
    # Convert to QDK Chemistry Structure
    water = create_structure_from_rdkit(water_rdkit)

    Logger.info(water.get_summary())
    Logger.info("XYZ Geometry in Angstrom")
    Logger.info(water.to_xyz())

    ########################################################################################
    # 2. Run the SCF stage to obtain the reference wavefunction.
    ########################################################################################
    nuclear_repulsion = water.calculate_nuclear_repulsion_energy()
    scf_solver = create("scf_solver")
    e_scf, scf_wavefunction = scf_solver.run(water, args.charge, args.spin, args.basis)
    total_scf_energy = e_scf + nuclear_repulsion
    Logger.info(f"SCF Energy: {total_scf_energy:.8f} Hartree")


if __name__ == "__main__":
    main()
