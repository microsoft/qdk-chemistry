"""J/K builder examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# docs:xyz ../data/water.structure.xyz
# start-cell-scf
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import AlgorithmRef, Structure

structure = Structure.from_xyz("""\
3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
""")

# J/K builders available to the SCF solver ("qdk" is the default)
print(available("jk_builder"))

# The qdk builder with its integrals stored in memory
scf_solver = create("scf_solver")
scf_solver.settings().set(
    "jk_builder", AlgorithmRef("jk_builder", "qdk", eri_method="incore")
)
energy, wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)
# end-cell-scf
################################################################################

################################################################################
# start-cell-build
import numpy as np

from qdk_chemistry.data import CoulombKernel, ErfCoulombKernel
from qdk_chemistry.data.symmetry import (
    SymmetryLabel,
    axes,
    make_spin_diagonal_rank2_sbt,
)

orbitals = wavefunction.get_orbitals()
basis_set = orbitals.get_basis_set()
occupations = np.zeros(orbitals.get_num_molecular_orbitals())
occupations[:5] = 1.0
density_alpha = orbitals.calculate_ao_density_matrix(occupations)

# J and K of the closed-shell density matrices (restricted: one alpha block)
densities = make_spin_diagonal_rank2_sbt(density_alpha)
coulomb, exchange = create("jk_builder").run(
    basis_set, CoulombKernel(), CoulombKernel(), densities
)
alpha = SymmetryLabel([axes.alpha()])
coulomb_alpha = coulomb.block([alpha, alpha])
print(f"Coulomb energy: {2.0 * np.sum(density_alpha * coulomb_alpha):.6f} Hartree")

# Long-range exchange only: a None kernel skips J
_, exchange_lr = create("jk_builder").run(
    basis_set, None, ErfCoulombKernel(omega=0.4), densities
)
exchange_lr_alpha = exchange_lr.block([alpha, alpha])
print(
    f"Long-range exchange energy: {-np.sum(density_alpha * exchange_lr_alpha):.6f} Hartree"
)
# end-cell-build
################################################################################

################################################################################
# start-cell-custom
import time

from qdk_chemistry.algorithms import JKBuilder
from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin


class TimedJKBuilder(JKBuilder):
    """J/K builder timing the builds of the qdk builder."""

    seconds = 0.0

    def __init__(self):
        super().__init__()
        self._qdk = None

    def name(self):
        return "timed"

    def _run_impl(self, basis_set, j_kernel, k_kernel, densities):
        if self._qdk is None:
            self._qdk = create(
                "jk_builder",
                "qdk",
                eri_threshold=self.settings().get("eri_threshold"),
            )
        start = time.perf_counter()
        result = self._qdk.run(basis_set, j_kernel, k_kernel, densities)
        TimedJKBuilder.seconds += time.perf_counter() - start
        return result


class TimedJKPlugin(QdkChemistryPlugin):
    """Register the timed J/K builder."""

    def register(self, registrar: PluginRegistrar) -> None:
        registrar.register_algorithm(lambda: TimedJKBuilder())


# Installed plugins are registered automatically through their entry point.
# This call only makes the standalone documentation example executable.
TimedJKPlugin().register(PluginRegistrar())

scf_solver = create("scf_solver")
scf_solver.settings().set("jk_builder", AlgorithmRef("jk_builder", "timed"))
energy, _ = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)
print(f"SCF energy: {energy:.10f} Eh, J/K builds: {TimedJKBuilder.seconds:.3f} s")
# end-cell-custom
################################################################################
