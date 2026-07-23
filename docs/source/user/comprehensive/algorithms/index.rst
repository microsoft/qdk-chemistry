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
   expectation_estimator
   hamiltonian_constructor
   localizer
   mc_calculator
   mcscf
   pmc
   qubit_mapper
   scf_solver
   stability_checker
   state_preparation
   hadamard_test
   phase_estimation
   qpe_circuit_builder
   evolution_circuit_builder
   hamiltonian_unitary_builder
   propagator
   hamiltonian_simulation
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
     - Hamiltonian → QubitOperator
   * - :doc:`StatePreparation <state_preparation>`
     - Quantum state preparation
     - Wavefunction → Circuit
   * - :doc:`HadamardTest <hadamard_test>`
     - Controlled-unitary overlap estimation
     - Circuit + UnitaryRepresentation → CircuitExecutorData
   * - :doc:`ExpectationEstimator <expectation_estimator>`
     - Quantum energy expectation values
     - Circuit + QubitOperator → Energy
   * - :doc:`StabilityChecker <stability_checker>`
     - :term:`SCF` stability analysis
     - Orbitals → Stability
   * - :doc:`PhaseEstimation <phase_estimation>`
     - Quantum phase estimation
     - Circuit + QubitOperator → QpeResult
   * - :doc:`QpeCircuitBuilder <qpe_circuit_builder>`
     - Phase estimation circuit composition
     - Circuit + QubitOperator → Circuit list
   * - :doc:`EvolutionCircuitBuilder <evolution_circuit_builder>`
     - Time-evolution circuit composition
     - TimeDependentQubitHamiltonian + Circuit → Circuit
   * - :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`
     - Hamiltonian simulation unitaries
     - QubitOperator → UnitaryRepresentation
<<<<<<< HEAD
   * - :doc:`Propagator <propagator>`
     - Effective Hamiltonians for time-dependent evolution
     - TimeDependentQubitHamiltonian → QubitOperator
   * - :doc:`HamiltonianSimulation <hamiltonian_simulation>`
     - Time-dependent Hamiltonian simulation
     - TimeDependentQubitHamiltonian + Circuit + Observables → (EnergyExpectationResult, MeasurementData) list
=======
>>>>>>> origin
   * - :doc:`ControlledCircuitMapper <circuit_mapper>`
     - Controlled-unitary circuit synthesis
     - UnitaryRepresentation → Circuit
   * - :doc:`CircuitExecutor <circuit_executor>`
     - Quantum circuit execution
     - Circuit → CircuitExecutorData


.. _algorithms-term-grouper:

Term grouper
------------

The ``term_grouper`` algorithm type partitions the Pauli terms of a :class:`~qdk_chemistry.data.QubitOperator` into algorithm-relevant subsets and stores the result on :attr:`~qdk_chemistry.data.QubitOperator.term_partition`.
A grouper consumes a ``QubitOperator`` and returns a *new* ``QubitOperator`` whose ``term_partition`` field is populated; the input is not mutated.

Strategies include full commutation grouping, qubit-wise commutation grouping, and trivial (identity) grouping.
Use ``registry.available("term_grouper")`` to list implementations.

Example::

    from qdk_chemistry.algorithms import registry

    grouper = registry.create("term_grouper", "qubit_wise_commuting")
    grouped = grouper.run(qubit_hamiltonian)
    grouped.term_partition  # FlatPartition(strategy="qubit_wise_commuting", ...)


Discovering implementations
---------------------------

Each algorithm class exposes multiple implementations that can be discovered at runtime.
Use ``available()`` to list registered implementations:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-discover-implementations
      :end-before: # end-cell-discover-implementations

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-discover-implementations
      :end-before: // end-cell-discover-implementations

For details on creating, loading, and using custom algorithm implementations, see the :doc:`plugin system <../plugins>` and :doc:`factory pattern <factory_pattern>` documentation.
