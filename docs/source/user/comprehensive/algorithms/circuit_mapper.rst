Controlled circuit mapper
=========================

The :class:`~qdk_chemistry.algorithms.ControlledCircuitMapper` algorithm in QDK/Chemistry converts a :class:`~qdk_chemistry.data.UnitaryRepresentation` into a *controlled* quantum circuit.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.ControlledUnitary` as input and produces a :class:`~qdk_chemistry.data.Circuit` as output.

Overview
--------

Controlled unitaries — operations of the form :math:`C\text{-}U` that apply :math:`U` to a target register conditioned on the state of a control qubit — are a building block in many quantum algorithms.
Mathematically, for a single control qubit the controlled unitary acts as:

.. math::

   C\text{-}U \;=\; |0\rangle\langle 0| \otimes I \;+\; |1\rangle\langle 1| \otimes U

That is, the target register is left unchanged when the control is :math:`|0\rangle` and :math:`U` is applied when the control is :math:`|1\rangle`.

The :class:`~qdk_chemistry.algorithms.ControlledCircuitMapper` synthesises these controlled operations from the abstract :class:`~qdk_chemistry.data.UnitaryRepresentation` representation produced by a :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`.
This is a core component of algorithms such as :doc:`PhaseEstimation <phase_estimation>`, which requires repeated controlled applications :math:`C\text{-}U^{2^k}`.

The mapper takes two inputs:

1. A :class:`~qdk_chemistry.data.ControlledUnitary` — which pairs a :class:`~qdk_chemistry.data.UnitaryRepresentation` with the control qubit indices
2. An optional power parameter that controls how many times the unitary is repeated (:math:`U^{\text{power}}`)

The resulting :class:`~qdk_chemistry.data.Circuit` implements the controlled unitary and can be executed by a :doc:`CircuitExecutor <circuit_executor>`.


Using the ControlledCircuitMapper
------------------------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run the circuit mapper.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.ControlledCircuitMapper` requires:

ControlledUnitary
   A :class:`~qdk_chemistry.data.ControlledUnitary` wrapping a :class:`~qdk_chemistry.data.UnitaryRepresentation` and specifying which qubits serve as controls.

.. rubric:: Creating a mapper

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the mapper

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run


Available implementations
-------------------------

You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations


Pauli sequence mapper
~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"pauli_sequence"`` (default)

Given a time-evolution unitary expressed as a :class:`~qdk_chemistry.data.PauliProductFormulaContainer` — a sequence of exponentiated Pauli terms :math:`e^{-i\theta_j P_j}` — this mapper constructs a controlled version by:

1. Rotating each Pauli operator :math:`P_j` into the Z basis
2. Entangling the target qubits with a CNOT ladder
3. Applying a controlled :math:`R_z(2\theta_j)` rotation from the control qubit
4. Uncomputing the basis rotations and entangling operations

.. note::
   The current implementation supports a single control qubit.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``power``
     - int
     - Number of times the unitary is repeated (:math:`U^{\text{power}}`). Default is 1.


Related classes
---------------

- :class:`~qdk_chemistry.data.ControlledUnitary`: Input — pairs a unitary with control qubit indices
- :class:`~qdk_chemistry.data.UnitaryRepresentation`: The underlying unitary representation
- :class:`~qdk_chemistry.data.Circuit`: Output circuit
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Produces the :class:`~qdk_chemistry.data.UnitaryRepresentation` that this mapper consumes

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/circuit_mapper.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Uses the circuit mapper to build controlled-:math:`U` operations
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Constructs the input unitaries
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
