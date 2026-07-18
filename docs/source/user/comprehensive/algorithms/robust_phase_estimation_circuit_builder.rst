Robust phase estimation circuit builder
=======================================

The :class:`~qdk_chemistry.algorithms.RobustPhaseEstimationCircuitBuilder` separates robust phase-estimation circuit generation from execution.
It resolves the geometric evolution-time schedule, measurement workload, and unitary-builder configuration, then returns a lazy, re-iterable circuit collection.
The high-level :doc:`PhaseEstimation <phase_estimation>` algorithm consumes the same collection through a separately configured :doc:`CircuitExecutor <circuit_executor>`.

Overview
--------

Create the builder through the ``"robust_phase_estimation_circuit_builder"`` factory type with implementation name ``"qdk"``.
Its ``run`` method accepts a state-preparation :class:`~qdk_chemistry.data.Circuit` and a :class:`~qdk_chemistry.data.QubitOperator`, and returns a :class:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationCircuitSet`.

Unlike the eager ``list[Circuit]`` returned by :doc:`QpeCircuitBuilder <qpe_circuit_builder>`, this collection generates one X/Y Hadamard-test circuit pair at a time.
This keeps memory bounded when an experiment requires independent randomized circuit draws.
Re-iterating one circuit set recreates the same schedule and concrete draw seeds, so resource estimation and execution can consume the same public object without schedule drift.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Setting
     - Type
     - Description
   * - ``unitary_builder``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Time-evolution builder used to realize each scheduled unitary.
   * - ``hadamard_test_circuit_builder``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Builder used to generate the X- and Y-basis Hadamard-test circuits.
   * - ``target_accuracy``
     - float
     - Requested absolute accuracy of the final energy estimate.
   * - ``base_time``
     - float
     - Round-zero evolution time. ``0.0`` selects it from the Hamiltonian coefficient norm.
   * - ``unitary_accuracy_fraction``
     - float
     - Fraction of ``target_accuracy`` assigned to unitary synthesis when the selected builder supports accuracy-based sizing.
   * - ``epsilon_rpe`` / ``epsilon_unitary``
     - float
     - Optional explicit RPE and unitary error budgets. Both must be set together.
   * - ``energy_correction``
     - str
     - Phase-to-energy map: ``"auto"``, ``"linear"``, or ``"qdrift_tangent"``.
   * - ``seed``
     - int
     - Root random seed. ``-1`` chooses one entropy-backed seed when the circuit set is created.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/phase_estimation.py
      :language: python
      :start-after: # start-cell-configure-robust
      :end-before: # end-cell-configure-robust

Rounds and experiments
----------------------

Each :class:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationRound` exposes its zero-based round index, evolution time, shots per basis, number of circuit draws, scheduled unitary sample count, circuit multiplicity, draw seeds, and exact unitary-builder configuration.

Iterating the circuit set yields :class:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationExperiment` objects.
Each experiment contains one X-basis circuit, one Y-basis circuit, its round and draw coordinates, its concrete random seed when applicable, and the number of executions represented by each circuit.

For deterministic evolution, one circuit pair represents all shots in the round, so ``circuit_multiplicity`` equals ``shots_per_basis``.
For randomized evolution, every independent draw produces one pair and ``circuit_multiplicity`` is one.
The X and Y circuits in a pair always share the same unitary draw.

Resource estimation
-------------------

Every generated :class:`~qdk_chemistry.data.Circuit` supports :meth:`~qdk_chemistry.data.Circuit.get_qre_application`.
The QRE application describes one circuit; ``circuit_multiplicity`` remains separate workload metadata that callers should include when aggregating an experiment-level estimate.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/phase_estimation.py
      :language: python
      :start-after: # start-cell-robust-circuit-set
      :end-before: # end-cell-robust-circuit-set

Related classes
---------------

- :doc:`PhaseEstimation <phase_estimation>`: Executes the lazy circuit set and reconstructs the energy
- :doc:`HadamardTest <hadamard_test>`: Constructs and executes individual controlled-unitary overlap tests
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Builds the scheduled evolution unitaries
- :doc:`CircuitExecutor <circuit_executor>`: Executes generated circuits
- :class:`~qdk_chemistry.data.Circuit`: Provides QRE application conversion
