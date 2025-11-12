QDK/Chemistry Comprehensive Documentation
================================

Welcome to the comprehensive documentation for the Quantum Development Kit (QDK) Chemistry library.
This documentation provides detailed information about QDK/Chemistry's structure, components, and usage patterns.

Overview
--------

The QDK Chemistry library is designed to facilitate quantum chemistry calculations and simulations. It
provides a set of tools and libraries for working with molecular structures, performing electronic structure
calculations, and analyzing quantum many-body systems.

QDK/Chemistry features a unified interface for communitry quantum chemistry software packages,
allowing seamless interoperability with established software while maintaining a consistent API. This enables users to
leverage specialized capabilities across different packages without changing their workflow.

Documentation Structure
-----------------------

This comprehensive documentation is organized into the following sections:

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Data Classes

   data/structure
   data/orbitals
   data/hamiltonian
   data/wavefunction
   data/basis_set
   data/basis_sets
   data/functionals

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Algorithms

   algorithms/scf_solver
   algorithms/localizer
   algorithms/active_space
   algorithms/hamiltonian_constructor
   algorithms/mc_calculator
   algorithms/dynamical_correlation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Advanced Topics

   advanced/design_principles
   advanced/factory_pattern
   advanced/serialization
   advanced/interfaces
   advanced/settings

- **Data Classes**: Documentation of the core data structures used in QDK/Chemistry

  - :doc:`Structure <data/structure>`: Molecular structure representation
  - :doc:`Orbitals <data/orbitals>`: Molecular orbital information
  - :doc:`Hamiltonian <data/hamiltonian>`: Electronic Hamiltonian representation
  - :doc:`Wavefunction <data/wavefunction>`: Wavefunction representation

- **Algorithms**: Documentation of the computational algorithms in QDK/Chemistry

  - :doc:`ScfSolver <algorithms/scf_solver>`: Self-Consistent Field calculations
  - :doc:`Localizer <algorithms/localizer>`: Orbital localization methods
  - :doc:`ActiveSpaceSelector <algorithms/active_space>`: Active space selection methods
  - :doc:`HamiltonianConstructor <algorithms/hamiltonian_constructor>`: Hamiltonian construction
  - :doc:`MCCalculator <algorithms/mc_calculator>`: Multi-Configurational calculations
  - :doc:`DynamicalCorrelation <algorithms/dynamical_correlation>`: Dynamical correlation methods

- **Advanced Topics**: Deeper insights into QDK/Chemistry's design and usage

  - :doc:`Design Principles <advanced/design_principles>`: Core architectural principles of QDK/Chemistry
  - :doc:`Factory Pattern <advanced/factory_pattern>`: Understanding the factory pattern and extending QDK/Chemistry
  - :doc:`Serialization <advanced/serialization>`: Data serialization and deserialization
  - :doc:`Interfaces <advanced/interfaces>`: QDK/Chemistry's interface system to external packages
  - :doc:`Settings <advanced/settings>`: Configuration system for algorithms

Getting Started
---------------

If you're new to QDK/Chemistry, we recommend starting with the :doc:`Quickstart Guide <../Quickstart>` to familiarize yourself with the basic concepts and workflow.

API Reference
-------------

For detailed API reference, please refer to the language-specific API documentation:

- :doc:`C++ API Reference <../../api/cpp_api>`
- :doc:`Python API Reference <../../api/python_api>`
