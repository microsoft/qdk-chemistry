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

# J and K of the closed-shell spin densities (restricted: one alpha block)
coulomb, exchange = create("jk_builder").run(
    basis_set, make_spin_diagonal_rank2_sbt(density_alpha)
)
alpha = SymmetryLabel([axes.alpha()])
coulomb_alpha = coulomb.block([alpha, alpha])
print(f"Coulomb energy: {2.0 * np.sum(density_alpha * coulomb_alpha):.6f} Hartree")
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

    def _run_impl(self, basis_set, densities):
        if self._qdk is None:
            keys = ("eri_threshold", "alpha", "beta", "omega")
            self._qdk = create(
                "jk_builder", "qdk", **{key: self.settings().get(key) for key in keys}
            )
        start = time.perf_counter()
        result = self._qdk.run(basis_set, densities)
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
energy, wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)
print(f"J/K builds: {TimedJKBuilder.seconds:.3f} s")
# end-cell-custom
################################################################################
