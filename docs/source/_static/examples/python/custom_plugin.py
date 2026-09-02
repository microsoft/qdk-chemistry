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
from qdk_chemistry.algorithms import ScfSolver
from qdk_chemistry.data import (
    BasisSet,
    Orbitals,
    Structure,
    Wavefunction,
)


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
        basis_or_guess: Orbitals | BasisSet | str | None = None,
    ) -> tuple[float, Wavefunction]:
        """Perform a self-consistent field (SCF) calculation using a custom backend.

        This method should convert the input structure to the external format, run the SCF calculation
        using the specified method and basis set, and return the electronic energy and wavefunction
        in QDK/Chemistry format.

        Args:
            structure: The molecular structure to be calculated.
            charge: The total charge of the molecular system.
            spin_multiplicity: The spin multiplicity (2S+1) of the system.
            basis_or_guess: Basis set information or initial guess, which can be:
                - An Orbitals object (used as initial guess)
                - A BasisSet object
                - A string specifying the basis set name
                - None (use default from settings)

        Returns:
            Tuple of (energy, wavefunction)
        """
        # Convert to external format

        # Execute external calculation

        # Convert results to QDK format

        # energy = 0.0
        # wavefunction = Wavefunction(...)
        # return energy, wavefunction
        return 0.0, None


# end-cell-custom-scf-solver
################################################################################

################################################################################
# start-cell-registration
from qdk_chemistry.algorithms.registry import register

# Registration during module import
register(lambda: CustomScfSolver())
# end-cell-registration
################################################################################

################################################################################
# start-cell-usage-after-registration
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

# Define a molecular structure (e.g., H2 molecule)
coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
molecule = Structure(coords, symbols=["H", "H"])

# Instantiate the custom solver
solver = create("scf_solver", "custom")
# energy, wavefunction = solver.run(
#     molecule, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
# )

# Verify registration
print(available("scf_solver"))  # [..., 'custom']
# end-cell-usage-after-registration
################################################################################

################################################################################
# start-cell-descriptor-settings
from qdk_chemistry.data import Settings


class MolecularDescriptorSettings(Settings):
    """Settings for molecular descriptor algorithms."""

    def __init__(self):
        super().__init__()
        self._set_default("normalize", "bool", False, "Normalize the descriptor")


# end-cell-descriptor-settings
################################################################################

################################################################################
# start-cell-descriptor-base-class
from qdk_chemistry.algorithms.base import Algorithm


class MolecularDescriptorCalculator(Algorithm):
    """Abstract base class for molecular descriptor algorithms."""

    def type_name(self) -> str:
        return "molecular_descriptor_calculator"


# end-cell-descriptor-base-class
################################################################################

################################################################################
# start-cell-descriptor-factory
from qdk_chemistry.algorithms.base import AlgorithmFactory


class MolecularDescriptorCalculatorFactory(AlgorithmFactory):
    """Factory for creating molecular descriptor calculators."""

    def algorithm_type_name(self) -> str:
        return "molecular_descriptor_calculator"

    def default_algorithm_name(self) -> str:
        return "nuclear_charge"


# end-cell-descriptor-factory
################################################################################

################################################################################
# start-cell-descriptor-implementations
from qdk_chemistry.data import Structure


class NuclearChargeDescriptor(MolecularDescriptorCalculator):
    """Calculator for a nuclear-charge molecular descriptor."""

    def __init__(self):
        super().__init__()
        self._settings = MolecularDescriptorSettings()

    def name(self) -> str:
        return "nuclear_charge"

    def _run_impl(self, structure: Structure) -> float:
        descriptor = float(sum(structure.get_nuclear_charges()))
        if self.settings().get("normalize") and structure.get_num_atoms() > 0:
            descriptor /= structure.get_num_atoms()
        return descriptor


# end-cell-descriptor-implementations
################################################################################


################################################################################
# start-cell-mass-descriptor
class MassDescriptor(MolecularDescriptorCalculator):
    """Calculator for a molecular-mass descriptor."""

    def __init__(self):
        super().__init__()
        self._settings = MolecularDescriptorSettings()

    def name(self) -> str:
        return "mass"

    def _run_impl(self, structure: Structure) -> float:
        descriptor = float(sum(structure.get_masses()))
        if self.settings().get("normalize") and structure.get_num_atoms() > 0:
            descriptor /= structure.get_num_atoms()
        return descriptor


# end-cell-mass-descriptor
################################################################################

################################################################################
# start-cell-descriptor-registration
from qdk_chemistry import algorithms

# Register the factory
algorithms.registry.register_factory(MolecularDescriptorCalculatorFactory())

# Register implementations
algorithms.register(lambda: NuclearChargeDescriptor())
algorithms.register(lambda: MassDescriptor())
# end-cell-descriptor-registration
################################################################################

################################################################################
# start-cell-descriptor-usage
from qdk_chemistry.algorithms import available, create

# List available implementations
print(available("molecular_descriptor_calculator"))  # ['mass', 'nuclear_charge']

# Instantiate and configure
calculator = create("molecular_descriptor_calculator", "nuclear_charge")
calculator.settings().set("normalize", True)

# Execute
# descriptor = calculator.run(molecule)
# end-cell-descriptor-usage
################################################################################
