Robust phase estimation circuit builder
=======================================

The robust phase-estimation circuit workflow separates scheduling, circuit construction, and execution.
The ``"rpe_experiment_scheduler"`` resolves one reproducible workload as a :class:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet`.
The ``"qdk_robust"`` :doc:`QpeCircuitBuilder <qpe_circuit_builder>` converts that workload into X- and Y-basis Hadamard-test circuits.
The high-level :doc:`PhaseEstimation <phase_estimation>` algorithm streams those pairs through a separately configured :doc:`CircuitExecutor <circuit_executor>`.

Overview
--------

Create the circuit builder through the ``"qpe_circuit_builder"`` factory type with implementation name ``"qdk_robust"``.
Its standard ``run`` method accepts a state-preparation :class:`~qdk_chemistry.data.Circuit` and a :class:`~qdk_chemistry.data.QubitOperator`, then returns a flat ``list[Circuit]`` consistent with the other QPE builders.

The robust builder additionally exposes three workload-aware methods:

``schedule(state_preparation, qubit_hamiltonian)``
   Resolve a :class:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet` without constructing concrete X/Y circuits.

``build(circuit_set)``
   Materialize the canonical flat circuit list described by an existing workload.

``iter_build(circuit_set)``
   Yield ``(experiment_spec, x_circuit, y_circuit)`` tuples one at a time.
   The final RPE algorithm uses this path to keep memory bounded for independently randomized draws.

Call ``schedule`` only once when the configured seed is ``-1``.
Scheduling concretizes one entropy-backed root seed, and all later construction or execution should consume that same circuit set.

Configuration
-------------

The robust circuit builder has one setting:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Setting
     - Type
     - Description
   * - ``experiment_scheduler``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Scheduler that resolves the complete reproducible workload. The default is ``AlgorithmRef("rpe_experiment_scheduler", "qdk")``.

The nested experiment scheduler defines:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Setting
     - Type
     - Description
   * - ``unitary_builder``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Time-evolution builder used to realize each scheduled unitary. Its ``power`` must be ``1`` because RPE owns the evolution-time ladder.
   * - ``hadamard_test_circuit_builder``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Builder used to generate the X- and Y-basis Hadamard tests.
   * - ``target_accuracy``
     - float
     - Requested absolute accuracy of the final energy estimate.
   * - ``base_time``
     - float
     - Round-zero evolution time. ``0.0`` selects :math:`\pi/(2\lambda)`; an explicit positive value must satisfy :math:`\mathtt{base\_time}\,\lambda < \pi`.
   * - ``unitary_accuracy_fraction``
     - float
     - Legacy fraction used by non-Trotter builders. Partially randomized evolution uses it only when explicitly set; Trotter does not support it.
   * - ``epsilon_rpe``
     - float
     - Optional explicit RPE energy tolerance for non-Trotter builders. Set it together with ``epsilon_unitary``.
   * - ``epsilon_unitary``
     - float
     - Positive full-unitary tolerance. Trotter and default partially randomized evolution use an independent value of ``0.85`` when omitted.
   * - ``energy_correction``
     - str
     - Phase-to-energy map: ``"auto"``, ``"linear"``, or ``"qdrift_tangent"``.
   * - ``seed``
     - int
     - Root random seed. ``-1`` chooses one entropy-backed seed when the circuit set is scheduled.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/phase_estimation.py
      :language: python
      :start-after: # start-cell-configure-robust
      :end-before: # end-cell-configure-robust

Independent unitary-accuracy routing
------------------------------------

When ``unitary_builder`` selects Trotter or partially randomized evolution, the default route uses independent quantities with different units:

.. math::

  \epsilon_{\mathrm{RPE}} = \epsilon_{\mathrm{total}} = \mathtt{target\_accuracy},
  \qquad
  \epsilon_u = \mathtt{epsilon\_unitary}.

The default ``epsilon_unitary`` is ``0.85``.
It is dimensionless and controls unitary sizing; it is not added to the energy-valued ``target_accuracy``.
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

For ``0 < epsilon_unitary < sin(pi/3)``, exact eigenstate input, and a valid per-round full-evolution error bound, the automatic ladder gives

.. math::

  |\widehat E-E|
  \leq
  \frac{2}{\pi}\,\mathtt{target\_accuracy}\,\arcsin(\mathtt{epsilon\_unitary})
  < \mathtt{target\_accuracy}.

Workload manifest
-----------------

The :class:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet` is the only independently serializable RPE data type.
It stores the global schedule metadata, state-preparation circuit, qubit Hamiltonian, nested builder snapshots, read-only round values, and :class:`~qdk_chemistry.data.RobustPhaseEstimationExperimentSpec` entries.

Each experiment specification identifies one X/Y pair and carries its round, randomized draw coordinates, concrete draw seed, and execution count.
For deterministic evolution, one pair represents every shot in its round.
For randomized evolution, each specification represents one independently sampled unitary and one shot per basis.
Both circuits in a pair always share the same unitary draw.

The ``x_circuit_index`` and ``y_circuit_index`` properties identify positions in the canonical flat list returned by ``build``.
When circuits are streamed, filtered, or batched, carry the experiment specification with the pair rather than treating those positions as persistent identifiers.

Resource estimation and serialization
-------------------------------------

Every generated :class:`~qdk_chemistry.data.Circuit` supports :meth:`~qdk_chemistry.data.Circuit.get_qre_application`.
Use ``spec.shots`` when aggregating an experiment-level resource estimate.

Saving and loading a circuit set preserves the complete workload and concrete randomized seeds without materializing circuits.
A restored Q# factory-backed state preparation contains QIR for resource estimation but does not contain the live Q# callable needed to construct new Hadamard-test circuits.
Call :meth:`~qdk_chemistry.data.RobustPhaseEstimationCircuitSet.rebind` with the original live state-preparation circuit before rebuilding such a workload.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/phase_estimation.py
      :language: python
      :start-after: # start-cell-robust-circuit-set
      :end-before: # end-cell-robust-circuit-set

Related classes
---------------

- :doc:`PhaseEstimation <phase_estimation>`: Streams the scheduled circuit pairs and reconstructs the energy
- :doc:`HadamardTest <hadamard_test>`: Constructs and executes individual controlled-unitary overlap tests
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Builds the scheduled evolution unitaries
- :doc:`CircuitExecutor <circuit_executor>`: Executes generated circuits
- :class:`~qdk_chemistry.data.Circuit`: Provides QRE application conversion
