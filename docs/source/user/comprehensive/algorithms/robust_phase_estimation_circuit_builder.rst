Robust phase estimation circuit builder
=======================================

The :class:`~qdk_chemistry.algorithms.RobustPhaseEstimationCircuitBuilder` separates robust phase-estimation circuit generation from execution.
It resolves the geometric evolution-time schedule, measurement workload, and unitary-builder configuration, then returns a re-iterable circuit collection that generates circuits on demand.
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
     - Legacy fraction used by non-Trotter builders. It is not supported when ``unitary_builder`` selects Trotter.
   * - ``epsilon_rpe``
     - float
     - Legacy explicit RPE energy tolerance for non-Trotter builders. Trotter always uses ``target_accuracy`` as its RPE tolerance.
   * - ``epsilon_unitary``
     - float
     - Independent positive Trotter sizing tolerance. When omitted, Trotter uses ``0.85``.
   * - ``energy_correction``
     - str
     - Phase-to-energy map: ``"auto"``, ``"linear"``, or ``"qdrift_tangent"``.
   * - ``seed``
     - int
     - Root random seed. ``-1`` chooses one entropy-backed seed when the circuit set is created.

Trotter accuracy routing
------------------------

When ``unitary_builder`` selects the registered Trotter implementation, the builder uses independent quantities with different units:

.. math::

  \epsilon_{\mathrm{RPE}} = \epsilon_{\mathrm{total}} = \mathtt{target\_accuracy},
  \qquad
  \epsilon_u = \mathtt{epsilon\_unitary}.

The default ``epsilon_unitary`` is ``0.85``. It is dimensionless and controls Trotter step sizing; it is not added to the energy-valued ``target_accuracy``.
``unitary_accuracy_fraction`` and explicit ``epsilon_rpe`` are rejected for Trotter.

For ``0 < epsilon_unitary < sin(pi/3)``, exact eigenstate input, and a valid per-round Trotter operator bound, the automatic ladder gives

.. math::

  |\widehat E-E|
  \leq
  \frac{2}{\pi}\,\mathtt{target\_accuracy}\,\arcsin(\mathtt{epsilon\_unitary})
  < \mathtt{target\_accuracy}.

Any positive ``epsilon_unitary`` is accepted as a Trotter step-sizing input, but values at or above ``sin(pi/3)`` do not carry this uniform branch guarantee.
Other unitary-builder categories retain their existing routing.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/phase_estimation.py
      :language: python
      :start-after: # start-cell-configure-robust
      :end-before: # end-cell-configure-robust

Rounds and experiments
----------------------

Each :class:`~qdk_chemistry.data.RobustPhaseEstimationRound` exposes its zero-based round index, evolution time, shots per basis, number of circuit draws, scheduled unitary sample count, circuit multiplicity, draw seeds, and exact unitary-builder configuration.

Iterating the circuit set yields :class:`~qdk_chemistry.data.RobustPhaseEstimationExperiment` objects.
Each experiment contains one X-basis circuit, one Y-basis circuit, its round and draw coordinates, its concrete random seed when applicable, and the number of executions represented by each circuit.

Use :meth:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationCircuitSet.get_experiment` to materialize one X/Y pair directly.
For randomized evolution, provide the desired ``draw_index``; for deterministic evolution, omit it.
The paired form ensures that X and Y circuits use the same unitary draw.
When only one basis is needed, :meth:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationCircuitSet.get_circuit` returns the requested concrete circuit.

For deterministic evolution, one circuit pair represents all shots in the round, so ``circuit_multiplicity`` equals ``shots_per_basis``.
For randomized evolution, every independent draw produces one pair and ``circuit_multiplicity`` is one.
The X and Y circuits in a pair always share the same unitary draw.

Resource estimation
-------------------

Every generated :class:`~qdk_chemistry.data.Circuit` supports :meth:`~qdk_chemistry.data.Circuit.get_qre_application`.
The QRE application describes one circuit; ``circuit_multiplicity`` remains separate workload metadata that callers should include when aggregating an experiment-level estimate.
The circuit set's :attr:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationCircuitSet.schedule` is a serializable :class:`~qdk_chemistry.data.RobustPhaseEstimationSchedule` containing rounds, seeds, multiplicities, accuracy parameters, and builder configurations without materialized circuits.
A selected :class:`~qdk_chemistry.data.RobustPhaseEstimationExperiment` is also serializable after its X/Y pair has been materialized.
After loading a schedule, use :meth:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder.RobustPhaseEstimationCircuitSet.from_schedule` with the live state-preparation circuit and qubit Hamiltonian to resume on-demand generation.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/phase_estimation.py
      :language: python
      :start-after: # start-cell-robust-circuit-set
      :end-before: # end-cell-robust-circuit-set

Related classes
---------------

- :doc:`PhaseEstimation <phase_estimation>`: Executes the on-demand circuit set and reconstructs the energy
- :doc:`HadamardTest <hadamard_test>`: Constructs and executes individual controlled-unitary overlap tests
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Builds the scheduled evolution unitaries
- :doc:`CircuitExecutor <circuit_executor>`: Executes generated circuits
- :class:`~qdk_chemistry.data.Circuit`: Provides QRE application conversion
