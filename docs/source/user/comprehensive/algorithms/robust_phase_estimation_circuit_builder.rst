Robust phase estimation circuit builder
=======================================

The :class:`~qdk_chemistry.algorithms.RobustPhaseEstimationCircuitBuilder` separates robust phase-estimation circuit generation from execution.
It resolves the geometric evolution-time schedule, measurement workload, and unitary-builder configuration, then returns a re-iterable circuit collection that generates circuits on demand.
The high-level :doc:`PhaseEstimation <phase_estimation>` algorithm consumes the same collection through a separately configured :doc:`CircuitExecutor <circuit_executor>`.

Overview
--------

Create the builder through the ``"robust_phase_estimation_circuit_builder"`` factory type with implementation name ``"qdk"``.
Its ``run`` method accepts a state-preparation :class:`~qdk_chemistry.data.Circuit` and a :class:`~qdk_chemistry.data.QubitOperator`, and returns a :class:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet`.

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
     - Time-evolution builder used to realize each scheduled unitary. Its ``power`` setting must be ``1`` because RPE controls powers through the round-time schedule.
   * - ``hadamard_test_circuit_builder``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Builder used to generate the X- and Y-basis Hadamard-test circuits.
   * - ``target_accuracy``
     - float
     - Requested absolute accuracy of the final energy estimate.
   * - ``base_time``
     - float
     - Round-zero evolution time. ``0.0`` selects :math:`\pi/(2\lambda)` from the Hamiltonian coefficient norm; an explicit positive value must satisfy :math:`\mathtt{base\_time}\,\lambda < \pi` to avoid energy aliasing.
   * - ``unitary_accuracy_fraction``
     - float
     - Legacy fraction used by non-Trotter builders. Partially randomized evolution uses it only when explicitly set; it is not supported for Trotter.
   * - ``epsilon_rpe``
     - float
     - Explicit RPE energy tolerance for non-Trotter builders. Set it together with ``epsilon_unitary``. Trotter and default partially randomized evolution use ``target_accuracy``.
   * - ``epsilon_unitary``
     - float
     - Positive full-unitary tolerance. Trotter and default partially randomized evolution use an independent value of ``0.85`` when omitted.
   * - ``energy_correction``
     - str
     - Phase-to-energy map: ``"auto"``, ``"linear"``, or ``"qdrift_tangent"``.
   * - ``seed``
     - int
     - Root random seed. ``-1`` chooses one entropy-backed seed when the circuit set is created.

Independent unitary-accuracy routing
------------------------------------

When ``unitary_builder`` selects Trotter or partially randomized evolution, the default route uses independent quantities with different units:

.. math::

  \epsilon_{\mathrm{RPE}} = \epsilon_{\mathrm{total}} = \mathtt{target\_accuracy},
  \qquad
  \epsilon_u = \mathtt{epsilon\_unitary}.

The default ``epsilon_unitary`` is ``0.85``. It is dimensionless and controls unitary sizing; it is not added to the energy-valued ``target_accuracy``.
``unitary_accuracy_fraction`` and explicit ``epsilon_rpe`` are rejected for Trotter.
For partially randomized evolution, setting ``unitary_accuracy_fraction`` explicitly retains the legacy fractional route, while setting both ``epsilon_rpe`` and ``epsilon_unitary`` selects an explicit paired budget.

The partially randomized builder divides its own ``target_accuracy`` :math:`\eta` in quadrature using ``accuracy_split`` :math:`s`.
RPE maps the full-unitary tolerance to

.. math::

  \eta = \frac{\epsilon_u}{\sqrt{s}+\sqrt{1-s}},
  \qquad
  \epsilon_D=\sqrt{s}\,\eta,
  \qquad
  \epsilon_R=\sqrt{1-s}\,\eta,

so the conservative additive channel bound satisfies :math:`\epsilon_D+\epsilon_R=\epsilon_u`.
Since the longest RPE evolution has :math:`t_{\max}=O(1/\epsilon_{\mathrm{total}})` and :math:`\epsilon_R` is independent of the requested energy accuracy, the Campbell random-sample budget scales as

.. math::

  N_R=O\!\left(\frac{\lambda_R^2t_{\max}^2}{\epsilon_R}\right)
     =O\!\left(\epsilon_{\mathrm{total}}^{-2}\right).

For ``0 < epsilon_unitary < sin(pi/3)``, exact eigenstate input, and a valid per-round full-evolution error bound, the automatic ladder gives

.. math::

  |\widehat E-E|
  \leq
  \frac{2}{\pi}\,\mathtt{target\_accuracy}\,\arcsin(\mathtt{epsilon\_unitary})
  < \mathtt{target\_accuracy}.

Any positive ``epsilon_unitary`` is accepted as a Trotter step-sizing input, but values at or above ``sin(pi/3)`` do not carry this uniform branch guarantee.
Partially randomized evolution requires ``epsilon_unitary < sin(pi/3)`` for its independent route.
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

Use :meth:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet.get_experiment` to materialize one X/Y pair directly.
For randomized evolution, provide the desired ``draw_index``; for deterministic evolution, omit it.
The paired form ensures that X and Y circuits use the same unitary draw.
When only one basis is needed, :meth:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet.get_circuit` returns the requested concrete circuit.

For deterministic evolution, one circuit pair represents all shots in the round, so ``circuit_multiplicity`` equals ``shots_per_basis``.
For randomized evolution, every independent draw produces one pair and ``circuit_multiplicity`` is one.
The X and Y circuits in a pair always share the same unitary draw.

Resource estimation
-------------------

Every generated :class:`~qdk_chemistry.data.Circuit` supports :meth:`~qdk_chemistry.data.Circuit.get_qre_application`.
The QRE application describes one circuit; ``circuit_multiplicity`` remains separate workload metadata that callers should include when aggregating an experiment-level estimate.
A :class:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet` is an immutable :class:`~qdk_chemistry.data.DataClass` that can be saved to JSON or HDF5.
It stores its :attr:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet.schedule`, state-preparation circuit, and qubit Hamiltonian without materializing any X/Y circuit pairs.
Loading preserves these inputs and the nested builder configurations; on-demand generation can resume directly when the configured builders support the restored circuit representation.
A selected :class:`~qdk_chemistry.data.RobustPhaseEstimationExperiment` is also serializable after its X/Y pair has been materialized.
For a Q# factory-backed state preparation, serialization preserves QIR for resource estimation but cannot reconstruct the live Q# callable needed to build new Hadamard-test circuits.
Use :meth:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet.from_schedule` with the loaded schedule, original live state-preparation circuit, and qubit Hamiltonian to resume on-demand generation.

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
