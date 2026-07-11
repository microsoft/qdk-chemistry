Hadamard test
=============

The :class:`~qdk_chemistry.algorithms.HadamardTest` algorithm in QDK/Chemistry estimates expectation values associated with a target unitary by measuring a single control qubit after a controlled evolution.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a state-preparation :class:`~qdk_chemistry.data.Circuit` (typically from :doc:`StatePreparation <state_preparation>`), a :class:`~qdk_chemistry.data.UnitaryRepresentation` (for example from :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`), and shot count as input, and returns :class:`~qdk_chemistry.data.CircuitExecutorData` containing measurement bitstring counts.

Overview
--------

The Hadamard test is a standard primitive for extracting overlap information from a controlled unitary application.
Given a prepared system state :math:`\vert \psi\rangle` and a unitary :math:`U`, the algorithm builds a circuit with one control qubit and the system register:

1. Prepare the system in :math:`\vert \psi\rangle`
2. Apply :math:`H` to the control qubit
3. Apply controlled-:math:`U` to the system register (controlled on the control qubit)
4. Measure the control qubit in a selected basis

QDK/Chemistry supports control-qubit measurement in ``X`` and ``Y`` bases through the ``test_basis`` setting.

For the standard Hadamard-test circuit, ``test_basis="X"`` estimates :math:`\mathrm{Re}\langle \psi \vert U \vert \psi \rangle` and ``test_basis="Y"`` estimates :math:`\mathrm{Im}\langle \psi \vert U \vert \psi \rangle`.

The algorithm returns the measurement counts for the control qubit, which can be converted to an expectation estimate:

.. math::

   \hat{m} = \frac{N_0 - N_1}{N_0 + N_1}

Typical workflow
~~~~~~~~~~~~~~~~

A Hadamard-test workflow usually needs two prepared inputs:

1. A state-preparation :class:`~qdk_chemistry.data.Circuit` (often produced by :doc:`StatePreparation <state_preparation>` from a reference wavefunction)
2. A target :class:`~qdk_chemistry.data.UnitaryRepresentation` (commonly a Hamiltonian time-evolution operator from :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`)
3. A configured :class:`~qdk_chemistry.algorithms.HadamardTest` instance to execute the circuit and collect counts

In molecular pipelines, structure setup, :term:`SCF`, and multi-configuration steps are typically used only to provide the wavefunction and Hamiltonian needed to build those two inputs.


Using the HadamardTest
----------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a Hadamard test calculation.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.HadamardTest` requires the following inputs:

State preparation circuit
   A :class:`~qdk_chemistry.data.Circuit` that prepares the target system state.

UnitaryRepresentation
   A :class:`~qdk_chemistry.data.UnitaryRepresentation` describing the unitary to apply under control.
   This is often generated from a qubit operator via :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`.

Shots
   A positive integer number of circuit executions used to estimate expectation values from counts.

Settings
   The :class:`~qdk_chemistry.algorithms.HadamardTest` is configured via its settings object, which includes:

   - ``test_basis`` - Measurement basis for the control qubit. Allowed values are ``"X"`` and ``"Y"``.
   - ``controlled_circuit_mapper`` - A :class:`~qdk_chemistry.data.AlgorithmRef` to a :doc:`ControlledCircuitMapper <circuit_mapper>` implementation used to synthesize controlled-:math:`U`.
   - ``circuit_executor`` - A :class:`~qdk_chemistry.data.AlgorithmRef` to a :doc:`CircuitExecutor <circuit_executor>` implementation used to run the final circuit.

.. note::

   The state preparation circuit and unitary should be qubit-compatible and represent the same target system.

.. rubric:: Creating a Hadamard test algorithm

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hadamard_test.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hadamard_test.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hadamard_test.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.HadamardTest` currently provides one built-in implementation.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hadamard_test.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. rubric:: Factory name: ``"qdk"``

.. rubric:: Settings

Direct settings on :class:`~qdk_chemistry.algorithms.hadamard_test.hadamard_test.HadamardTest`:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``test_basis``
     - string
     - Measurement basis for the control qubit. Allowed values are ``"X"`` and ``"Y"``. Default is ``"X"``.
   * - ``controlled_circuit_mapper``
     - AlgorithmRef
     - Nested :doc:`ControlledCircuitMapper <circuit_mapper>` used to synthesize controlled-:math:`U`.
   * - ``circuit_executor``
     - AlgorithmRef
     - Nested :doc:`CircuitExecutor <circuit_executor>` used to execute the generated circuit.


Interpreting results
--------------------

The algorithm returns :class:`~qdk_chemistry.data.CircuitExecutorData`, whose ``bitstring_counts`` can be converted to an expectation estimate:

.. math::

   \hat{m} = \frac{N_0 - N_1}{N_0 + N_1}

Here :math:`N_0 + N_1` is the number of clean measurement outcomes recorded in ``bitstring_counts``.
This can differ from requested ``shots`` when an executor tracks loss separately.
Finite sampling still introduces statistical uncertainty, so increasing shots typically improves estimator stability.


Related classes
---------------

- :class:`~qdk_chemistry.algorithms.HadamardTest`: Main Hadamard test algorithm interface
- :class:`~qdk_chemistry.algorithms.hadamard_test.circuit_builder.base.HadamardTestCircuitBuilder`: Circuit-builder abstraction used internally
- :class:`~qdk_chemistry.data.Circuit`: State-preparation and executable circuit data model
- :class:`~qdk_chemistry.data.UnitaryRepresentation`: Input unitary data model
- :class:`~qdk_chemistry.data.CircuitExecutorData`: Measurement output container
- :class:`~qdk_chemistry.data.AlgorithmRef`: Nested algorithm configuration reference type


Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/hadamard_test.py>`_ script.
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Build target unitaries such as time evolution
- :doc:`StatePreparation <state_preparation>`: Build state-preparation circuits from wavefunctions
- :doc:`ControlledCircuitMapper <circuit_mapper>`: Synthesize controlled-unitary circuits
- :doc:`CircuitExecutor <circuit_executor>`: Execute circuits and collect bitstring counts
- :doc:`Settings <settings>`: Configure algorithm settings
- :doc:`Factory Pattern <factory_pattern>`: Create algorithms by type and implementation name
