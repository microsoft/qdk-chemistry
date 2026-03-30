"""Tests for PySCF plugin functionality and basis set conversion utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.data import Ansatz, Settings, Structure

from ..reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
    mcscf_energy_tolerance,
    orthonormality_error_tolerance,
    plain_text_tolerance,
    scf_energy_tolerance,
    scf_orbital_tolerance,
    unitarity_error_tolerance,
)

try:
    import pyscf
    import pyscf.gto
    import pyscf.lo
    import pyscf.mcscf
    import pyscf.scf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

if PYSCF_AVAILABLE:
    from qdk_chemistry.constants import ANGSTROM_TO_BOHR
    from qdk_chemistry.data import AOType, BasisSet, OrbitalType, Shell
    from qdk_chemistry.plugins.pyscf.conversion import (
        basis_to_pyscf_mol,
        hamiltonian_to_scf,
        hamiltonian_to_scf_from_n_electrons_and_multiplicity,
        orbitals_to_scf,
        orbitals_to_scf_from_n_electrons_and_multiplicity,
        pyscf_mol_to_qdk_basis,
        structure_to_pyscf_atom_labels,
    )
    from qdk_chemistry.plugins.pyscf.mcscf import _mcsolver_to_fcisolver


pytestmark = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")


def create_n2_structure():
    """Create a nitrogen molecule structure."""
    symbols = ["N", "N"]
    coords = np.array(
        [
            [0.000000000, 0.0000000000, 2.000000000000 * ANGSTROM_TO_BOHR],
            [0.000000000, 0.0000000000, 0.000000000000],
        ]
    )
    return Structure(symbols, coords)


def create_water_structure():
    """Create a water molecule structure.

    Crawford geometry - same as used in C++ tests.
    """
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.000000000 * ANGSTROM_TO_BOHR, -0.0757918436 * ANGSTROM_TO_BOHR, 0.000000000000],
            [0.866811829 * ANGSTROM_TO_BOHR, 0.6014357793 * ANGSTROM_TO_BOHR, -0.000000000000],
            [-0.866811829 * ANGSTROM_TO_BOHR, 0.6014357793 * ANGSTROM_TO_BOHR, -0.000000000000],
        ]
    )
    return Structure(symbols, coords)


def create_helium_structure():
    """Create a helium atom structure."""
    return Structure(["He"], np.array([[0.0, 0.0, 0.0]]))


def create_li_structure():
    """Create a lithium atom structure."""
    return Structure(["Li"], np.array([[0.0, 0.0, 0.0]]))


def create_o2_structure():
    """Create an oxygen molecule (O2) structure."""
    symbols = ["O", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.208 * ANGSTROM_TO_BOHR]])
    return Structure(symbols, coords)


def create_uo2_structure():
    """Create a uranyl ion (UO2) structure."""
    symbols = ["U", "O", "O"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.85213 * ANGSTROM_TO_BOHR],
            [0.0, 0.0, -1.85213 * ANGSTROM_TO_BOHR],
        ]
    )
    return Structure(symbols, coords)


def pipek_objective_function(orbitals, mos):
    """Calculate the Pipek-Mezey objective function."""
    mf = orbitals_to_scf(orbitals, 0, 0)
    mol = mf.mol
    pm = pyscf.lo.PM(mol, mos, mf, pop_method="mulliken")
    return pm.cost_function(None)


def boys_objective_function(orbitals, mos):
    """Calculate the Foster-Boys objective function."""
    mf = orbitals_to_scf(orbitals, 0, 0)
    mol = mf.mol
    fb = pyscf.lo.Boys(mol, mos)
    return fb.cost_function(None)


def er_objective_function(orbitals, mos):
    """Calculate the Edmiston-Ruedenberg objective function."""
    mf = orbitals_to_scf(orbitals, 0, 0)  # true electron count not needed
    mol = mf.mol
    er = pyscf.lo.ER(mol, mos)
    return er.cost_function(np.eye(mos.shape[1]))
