Algorithm classes
=================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Creation, Configuration and Extension of Algorithm Classes

   factory_pattern
   settings

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Available Algorithms

   active_space
   energy_estimator
   hamiltonian_constructor
   localizer
   mc_calculator
   mcscf
   pmc
   qubit_mapper
   scf_solver
   stability_checker
   state_preparation

QDK/Chemistry provides a comprehensive set of algorithm classes which express core methodological primitives for quantum and classical chemistry calculations.
All algorithms follow a :doc:`factory pattern <factory_pattern>` design, allowing you to create instances by name and configured through a unified :doc:`settings <settings>` interface.

Quick reference
---------------

The following table summarizes the available algorithm classes in QDK/Chemistry and their purposes. For detailed documentation, refer to the linked pages.

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Algorithm Class
     - Purpose
     - Input → Output
   * - :doc:`ScfSolver <scf_solver>`
     - Mean-field (:term:`HF`/:term:`DFT`) calculations
     - Structure → Orbitals
   * - :doc:`OrbitalLocalizer <localizer>`
     - Orbital transformations
     - Orbitals → Orbitals
   * - :doc:`ActiveSpaceSelector <active_space>`
     - Active space identification
     - Wavefunction → Wavefunction
   * - :doc:`HamiltonianConstructor <hamiltonian_constructor>`
     - Molecular Hamiltonian construction
     - Orbitals → Hamiltonian
   * - :doc:`MultiConfigurationCalculator <mc_calculator>`
     - Many-body wavefunction calculations
     - Hamiltonian → Wavefunction
   * - :doc:`ProjectedMultiConfigurationCalculator <pmc>`
     - Projected many-body wavefunction calculations
     - Hamiltonian → Wavefunction
   * - :doc:`MultiConfigurationScf <mcscf>`
     - Coupled `Orbital`-`Wavefunction` calculations.
     - Orbitals → Wavefunction
   * - :doc:`QubitMapper <qubit_mapper>`
     - Fermion-to-qubit mapping
     - Hamiltonian → QubitHamiltonian
   * - :doc:`StatePreparation <state_preparation>`
     - Quantum state preparation
     - Wavefunction → Circuit
   * - :doc:`EnergyEstimator <energy_estimator>`
     - Quantum energy expectation values
     - Circuit + QubitHamiltonian → Energy
   * - :doc:`StabilityChecker <stability_checker>`
     - :term:`SCF` stability analysis
     - Orbitals → Stability

Discovering implementations
---------------------------

Each algorithm class exposes multiple implementations that can be discovered at runtime.
Use ``available()`` to list registered implementations:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-discover-implementations
      :end-before: // end-cell-discover-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-discover-implementations
      :end-before: # end-cell-discover-implementations

For details on creating, loading, and using custom algorithm implementations, see the :doc:`plugin system <../plugins>` and :doc:`factory pattern <factory_pattern>` documentation.
