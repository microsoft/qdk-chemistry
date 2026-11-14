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
# start-cell-engine
import numpy as np

orbitals = wavefunction.get_orbitals()
basis_set = orbitals.get_basis_set()
n = basis_set.get_num_atomic_orbitals()
occupations = np.zeros(orbitals.get_num_molecular_orbitals())
occupations[:5] = 2.0
density = orbitals.calculate_ao_density_matrix(occupations)

# An engine for the basis set; J and K are filled in place
engine = create("jk_builder").run(basis_set, 1e-12)
coulomb = np.zeros((n, n))
exchange = np.zeros((n, n))
engine.build(density, coulomb, exchange)
print(f"Coulomb energy: {0.5 * np.sum(density * coulomb):.6f} Hartree")
# end-cell-engine
################################################################################

################################################################################
# start-cell-custom
import time

from qdk_chemistry.algorithms import JKBuilder, JKEngine
from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin


class TimedJKEngine(JKEngine):
    """J/K engine timing the builds of another engine."""

    seconds = 0.0

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def build(self, densities, coulomb, exchange, alpha, beta, omega):
        start = time.perf_counter()
        self.engine.build(densities, coulomb, exchange, alpha, beta, omega)
        TimedJKEngine.seconds += time.perf_counter() - start


class TimedJKBuilder(JKBuilder):
    """J/K builder timing the builds of the qdk engine."""

    def name(self):
        return "timed"

    def _run_impl(self, basis_set, eri_threshold):
        return TimedJKEngine(create("jk_builder", "qdk").run(basis_set, eri_threshold))


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
print(f"J/K builds: {TimedJKEngine.seconds:.3f} s")
# end-cell-custom
################################################################################
