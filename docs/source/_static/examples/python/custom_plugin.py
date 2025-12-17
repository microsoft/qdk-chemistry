"""Custom plugin examples for QDK/Chemistry.

This module demonstrates how to extend QDK/Chemistry with custom plugins:
1. Adding a new backend for an existing algorithm type
2. Defining an entirely new algorithm type
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-custom-settings
from qdk_chemistry.data import ElectronicStructureSettings


class CustomScfSettings(ElectronicStructureSettings):
    """Settings for the custom SCF solver."""

    def __init__(self):
        super().__init__()
        # Define additional settings beyond the inherited defaults
        self._set_default(
            "custom_option",
            "string",
            "default_value",
            "Description of the custom option",
        )


# end-cell-custom-settings
################################################################################

################################################################################
# start-cell-custom-scf-solver
from qdk_chemistry.algorithms import ScfSolver  # noqa: E402
from qdk_chemistry.data import Orbitals, Structure, Wavefunction  # noqa: E402


class CustomScfSolver(ScfSolver):
    """Custom SCF solver wrapping an external chemistry package."""

    def __init__(self):
        super().__init__()
        self._settings = CustomScfSettings()

    def name(self) -> str:
        return "custom"

    def _run_impl(
        self,
        structure: Structure,
        charge: int,
        spin_multiplicity: int,
        initial_guess: Orbitals | None = None,
    ) -> tuple[float, Wavefunction]:
        # Convert to external format
        # external_mol = external_package.Molecule(structure.positions, structure.elements)

        # Execute external calculation
        _basis_set = self.settings().get("basis_set")  # noqa: F841
        # energy, external_orbs = external_package.run_scf(external_mol, basis=_basis_set)

        # Convert results to QDK format
        # wavefunction = self._convert_to_wavefunction(external_orbs)

        return energy, wavefunction


# end-cell-custom-scf-solver
################################################################################

################################################################################
# start-cell-registration
from qdk_chemistry.algorithms.registry import register  # noqa: E402

# Registration during module import
register(lambda: CustomScfSolver())
# end-cell-registration
################################################################################

################################################################################
# start-cell-usage-after-registration
from qdk_chemistry.algorithms import ScfSolver  # noqa: E402

# Instantiate the custom solver
solver = ScfSolver.create("custom")
solver.settings()["basis_set"] = "cc-pvdz"
molecule = ...  # Assume molecule is defined elsewhere
energy, wavefunction = solver.run(molecule, charge=0, spin_multiplicity=1)  # noqa: F821

# Verify registration
print(ScfSolver.available())  # [..., 'custom']
# end-cell-usage-after-registration
################################################################################

################################################################################
# start-cell-geometry-settings
from qdk_chemistry.data import Settings  # noqa: E402


class GeometryOptimizerSettings(Settings):
    """Settings for geometry optimization algorithms."""

    def __init__(self):
        super().__init__()
        self._set_default(
            "max_steps", "int", 100, "Maximum optimization steps", (1, 10000)
        )
        self._set_default(
            "convergence_threshold", "double", 1e-5, "Gradient convergence threshold"
        )
        self._set_default("step_size", "double", 0.1, "Initial optimization step size")


# end-cell-geometry-settings
################################################################################

################################################################################
# start-cell-geometry-base-class
from qdk_chemistry.algorithms.base import Algorithm  # noqa: E402


class GeometryOptimizer(Algorithm):
    """Abstract base class for geometry optimization algorithms."""

    def type_name(self) -> str:
        return "geometry_optimizer"


# end-cell-geometry-base-class
################################################################################

################################################################################
# start-cell-geometry-factory
from qdk_chemistry.algorithms.base import AlgorithmFactory  # noqa: E402


class GeometryOptimizerFactory(AlgorithmFactory):
    """Factory for creating geometry optimizer instances."""

    def algorithm_type_name(self) -> str:
        return "geometry_optimizer"

    def default_algorithm_name(self) -> str:
        return "bfgs"


# end-cell-geometry-factory
################################################################################

################################################################################
# start-cell-geometry-implementations
from qdk_chemistry.data import Structure  # noqa: E402


class BfgsOptimizer(GeometryOptimizer):
    """BFGS quasi-Newton geometry optimizer."""

    def __init__(self):
        super().__init__()
        self._settings = GeometryOptimizerSettings()

    def name(self) -> str:
        return "bfgs"

    def _run_impl(self, structure: Structure) -> Structure:
        _max_steps = self.settings().get("max_steps")  # noqa: F841
        _threshold = self.settings().get("convergence_threshold")  # noqa: F841

        # BFGS optimization implementation
        optimized_structure = structure  # Placeholder
        return optimized_structure


# end-cell-geometry-implementations
################################################################################


################################################################################
# start-cell-steepest-descent
class SteepestDescentOptimizer(GeometryOptimizer):
    """Steepest descent geometry optimizer."""

    def __init__(self):
        super().__init__()
        self._settings = GeometryOptimizerSettings()

    def name(self) -> str:
        return "steepest_descent"

    def _run_impl(self, structure: Structure) -> Structure:
        # Steepest descent implementation
        optimized_structure = structure  # Placeholder
        return optimized_structure


# end-cell-steepest-descent
################################################################################

################################################################################
# start-cell-geometry-registration
from qdk_chemistry.algorithms import registry  # noqa: E402

# Register the factory
registry.register_factory(GeometryOptimizerFactory())

# Register implementations
registry.register(lambda: BfgsOptimizer())
registry.register(lambda: SteepestDescentOptimizer())
# end-cell-geometry-registration
################################################################################

################################################################################
# start-cell-geometry-usage
from qdk_chemistry.algorithms import available, create  # noqa: E402

# List available implementations
print(available("geometry_optimizer"))  # ['bfgs', 'steepest_descent']

# Instantiate and configure
optimizer = create("geometry_optimizer", "bfgs")
optimizer.settings().set("max_steps", 200)
optimizer.settings().set("convergence_threshold", 1e-6)

# Execute
initial_structure = ...  # Assume initial_structure is defined elsewhere
optimized_structure = optimizer.run(initial_structure)  # noqa: F821
# end-cell-geometry-usage
################################################################################
