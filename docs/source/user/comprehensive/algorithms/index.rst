Algorithm classes
=================

QDK/Chemistry provides a comprehensive set of algorithm classes which express core methodological primitives for quantum and classical chemistry calculations.
All algorithms follow a :doc:`factory pattern <factory_pattern>` design, allowing you to create instances by name and configured through a unified :doc:`settings <settings>` interface.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   factory_pattern
   settings
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
   phase_estimation
   time_evolution_builder
   circuit_mapper
   circuit_executor


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
   * - :doc:`PhaseEstimation <phase_estimation>`
     - Quantum phase estimation
     - Circuit + QubitHamiltonian → QpeResult
   * - :doc:`TimeEvolutionBuilder <time_evolution_builder>`
     - Hamiltonian simulation unitaries
     - QubitHamiltonian → TimeEvolutionUnitary
   * - :doc:`ControlledEvolutionCircuitMapper <circuit_mapper>`
     - Controlled-unitary circuit synthesis
     - TimeEvolutionUnitary → Circuit
   * - :doc:`CircuitExecutor <circuit_executor>`
     - Quantum circuit execution
     - Circuit → CircuitExecutorData


.. _algorithms-term-grouper:

Term grouper
------------

The ``term_grouper`` algorithm type partitions the Pauli terms of a :class:`~qdk_chemistry.data.QubitHamiltonian` into algorithm-relevant subsets and stores the result on :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition`.
A grouper consumes a ``QubitHamiltonian`` and returns a *new* ``QubitHamiltonian`` whose ``term_partition`` field is populated; the input is not mutated.

The following strategies are registered out of the box:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Strategy
     - Description
     - Output kind
   * - ``"commuting"`` (default)
     - Greedy first-fit grouping by full Pauli commutation (``[P_i, P_j] = 0``).
       Used by Trotter-style decompositions to exponentiate a commuting block as a single ordered product without splitting error.
     - :class:`~qdk_chemistry.data.FlatPartition`
   * - ``"qubit_wise_commuting"``
     - Greedy first-fit grouping by qubit-wise commutation.
       All members of a group can be measured in a single basis, so this strategy is the basis-selection primitive used by :class:`~qdk_chemistry.algorithms.QdkEnergyEstimator`.
     - :class:`~qdk_chemistry.data.FlatPartition`
   * - ``"identity"``
     - Trivial grouping — every term is its own single-element group.
       Useful to clear or override an existing partition while still passing through the standard ``term_grouper`` interface.
     - :class:`~qdk_chemistry.data.FlatPartition`

Example::

    from qdk_chemistry.algorithms import registry

    grouper = registry.create("term_grouper", "qubit_wise_commuting")
    grouped = grouper.run(qubit_hamiltonian)
    grouped.term_partition  # FlatPartition(strategy="qubit_wise_commuting", ...)


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
