"""QDK/Chemistry plugins for external quantum chemistry software integration.

Plugins are bridge modules that connect QDK/Chemistry with external quantum chemistry
software packages. They implement QDK/Chemistry's algorithm interfaces (such as
:class:`~qdk_chemistry.algorithms.ScfSolver`, etc.) while internally using the external library
to perform the actual calculations. This allows users to leverage specialized capabilities
from established quantum chemistry packages while maintaining a consistent QDK/Chemistry API.

Purpose and Benefits
--------------------
Plugins serve several key purposes:

1. **Extended Functionality**: Leverage specialized capabilities from established quantum
   chemistry packages (e.g., PySCF's diverse DFT functionals, orbital localization methods).

2. **Automatic Format Conversion**: Handle conversion between QDK/Chemistry data structures
   (e.g., :class:`~qdk_chemistry.data.Structure`, :class:`~qdk_chemistry.data.Orbitals`,
   :class:`~qdk_chemistry.data.Hamiltonian`) and external library formats automatically.

3. **Registry Integration**: Plugin implementations are automatically registered with
   QDK/Chemistry's algorithm registry system, making them available through the
   :func:`~qdk_chemistry.algorithms.create` factory function.

4. **Consistent Interface**: Provide a uniform API for different backend implementations,
   allowing users to easily switch between different computational engines without
   changing their workflow code.

Available Plugins
-----------------
Currently available plugin packages:

* :mod:`qdk_chemistry.plugins.pyscf`: PySCF integration providing SCF solvers, coupled
  cluster calculators, orbital localization, active space selection (AVAS), and
  stability analysis.

Using Plugins
-------------
Plugins are used by importing the plugin module (which auto-registers implementations)
and then creating instances through the registry or directly:

Examples:
    >>> # Import the plugin to register its implementations
    >>> import qdk_chemistry.plugins.pyscf
    >>> # Create a PySCF SCF solver through the registry
    >>> from qdk_chemistry.algorithms import create
    >>> scf_solver = create("scf_solver", "pyscf")
    >>> # Configure and use like any other algorithm
    >>> scf_solver.settings()["max_iterations"] = 50
    >>> energy, orbitals = scf_solver.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

Notes:
    Plugin modules may have additional dependencies beyond core QDK/Chemistry.
    For example, the PySCF plugin requires the `pyscf` package to be installed
    separately.

See Also:
    :mod:`qdk_chemistry.algorithms`: Algorithm base classes and registry system
    :mod:`qdk_chemistry.plugins.pyscf`: PySCF integration plugin

"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
