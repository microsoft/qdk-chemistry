"""Complete SCF workflow example with settings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from pathlib import Path
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, BasisSet

# Create the default ScfSolver instance
scf_solver = create("scf_solver")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the SCF solver using the settings interface
# Note that the following line is optional, since hf is the default method
scf_solver.settings().set("method", "hf")

# end-cell-configure
################################################################################

################################################################################
# start-cell-run
# Load structure from XYZ file
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# Run scf
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="def2-tzvpp"
)
scf_orbitals = wfn.get_orbitals()

print(f"SCF Energy: {E_scf:.10f} Hartree")
# end-cell-run
################################################################################

################################################################################
# start-cell-alternative-run
# Run scf with an initial guess from previous calculation
E_scf2, wfn2 = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess=scf_orbitals
)

# Run scf with a custom basis set
basis_set = BasisSet.from_basis_name("def2-tzvpp", structure)
E_scf3, wfn3 = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess=basis_set
)
# end-cell-alternative-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("scf_solver"))  # ['pyscf', 'qdk']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-pyscf-example
from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

# Create and configure the PySCF solver
solver = create("scf_solver", "pyscf")
solver.settings().set("method", "b3lyp")
solver.settings().set("scf_type", "restricted")

# Run with basis set specified as input parameter
water_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.76, 0.59], [0.0, -0.76, 0.59]])
water = Structure(water_coords, symbols=["O", "H", "H"])
energy, wfn = solver.run(water, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz")
# end-cell-pyscf-example
################################################################################
