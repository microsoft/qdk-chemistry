Circuit
=======

The :class:`~qdk_chemistry.data.Circuit` class in QDK/Chemistry represents a quantum circuit.
It stores the circuit in one or more formats and converts between them on demand.


Worked example
--------------

The following example defines a Q# operation from a string, wraps the compiled circuit in a :class:`~qdk_chemistry.data.Circuit`, inspects the gate-level diagram, and runs resource estimation:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit.py
      :language: python
      :start-after: # start-cell-qsharp-workflow
      :end-before: # end-cell-qsharp-workflow


Overview
--------

A :class:`~qdk_chemistry.data.Circuit` wraps a quantum circuit that may be represented in any combination of the following formats:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Description
   * - ``qasm``
     - An OpenQASM string.
   * - ``qir``
     - A :term:`QIR` (Quantum Intermediate Representation) object for cross-platform compilation.
   * - ``qsharp``
     - A compiled ``qsharp._native.Circuit`` object for inspection and visualization.
   * - ``qsharp_op``
     - A Q# callable — a native Q# operation that can be composed with other Q# operations.
   * - ``qsharp_factory``
     - A ``QsharpFactoryData`` — a deferred Q# program that compiles on demand (see `Q# factory harness`_ below).

At least one representation must be provided at construction.
When a format that is not stored is requested, the class converts automatically — for example, calling :meth:`~qdk_chemistry.data.Circuit.get_qsharp_circuit` on a circuit that only has ``qasm`` will compile it to Q# via the ``qsharp`` package.

The full constructor signature is:

.. code-block:: python

   Circuit(
       qasm: str | None = None,
       qir: QirInputData | str | None = None,
       qsharp: qsharp._native.Circuit | None = None,
       qsharp_op: Callable | None = None,
       qsharp_factory: QsharpFactoryData | None = None,
       encoding: str | None = None,
   )

The ``encoding`` parameter records the fermion-to-qubit mapping assumed by the circuit (e.g., ``"jordan-wigner"``).


.. _qsharp-factory-harness:

Q# factory harness
-------------------

The Q# factory is the primary mechanism by which QDK/Chemistry builds circuits internally.
Rather than constructing a circuit as a gate sequence, the library stores a reference to a **Q# callable** along with the **classical parameters** that configure it.
The circuit is compiled only when a concrete representation is actually needed.

A ``QsharpFactoryData`` is a frozen dataclass with two fields:

``program``
   A Q# callable — a reference to a Q# operation that constructs the circuit.
   In practice this is typically a Q# factory function exposed via the ``qsharp`` Python package (e.g., ``QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit``).

``parameter``
   A ``dict[str, Any]`` of classical parameters that are passed to the Q# callable at compilation time.
   These are Python-native values — lists, ints, floats, nested dicts, or even other Q# operations — that the Q# program uses to configure the circuit it produces.

The standard pattern for building a factory is:

.. code-block:: python

   from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData

   # 1. Build a parameter object (often a Q# dataclass)
   params = SomeQsharpModule.SomeParamsClass(
       numQubits=4,
       coefficients=[0.5, -0.3, 0.1],
       targetIndices=[0, 1, 2, 3],
   )

   # 2. Create the factory — vars() converts the dataclass to a dict
   factory = QsharpFactoryData(
       program=SomeQsharpModule.MakeCircuit,
       parameter=vars(params),
   )

   # 3. Wrap in a Circuit — nothing is compiled yet
   circuit = Circuit(qsharp_factory=factory, encoding="jordan-wigner")

   # 4. Compilation happens on demand:
   circuit.get_qsharp_circuit()  # → qsharp.circuit(program, **params)
   circuit.get_qir()             # → qsharp.compile(program, **params)
   circuit.estimate()            # → qsharp.estimate(program, est_params, **params)

The parameter dict can also be constructed directly when the parameters are simple:

.. code-block:: python

   factory = QsharpFactoryData(
       program=SomeQsharpModule.MakeMeasurementCircuit,
       parameter={
           "baseCircuit": another_circuit._qsharp_op,  # a Q# callable as a parameter
           "bases": ["Z", "X", "Z"],
           "numQubits": 3,
       },
   )

This pattern is what enables end-to-end Q# composition: because the factory stores Q# callables rather than serialized gate sequences, multi-stage circuits compose natively in the Q# runtime without intermediate format conversions.


Conversion methods
------------------

Each method returns the circuit in the requested format, converting from whatever representation is available:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - :meth:`~qdk_chemistry.data.Circuit.get_qsharp_circuit`
     - Returns a ``qsharp._native.Circuit`` for inspection. Accepts ``prune_classical_qubits`` to remove unused qubits.
   * - :meth:`~qdk_chemistry.data.Circuit.get_qir`
     - Returns the :term:`QIR` representation. Compiles from Q# factory or converts from QASM if needed.
   * - :meth:`~qdk_chemistry.data.Circuit.get_qasm`
     - Returns the OpenQASM string. Converts from :term:`QIR` via Qiskit if only :term:`QIR` is available.
   * - :meth:`~qdk_chemistry.data.Circuit.get_qiskit_circuit`
     - Returns a Qiskit ``QuantumCircuit``. Requires ``qiskit`` to be installed.
   * - :meth:`~qdk_chemistry.data.Circuit.estimate`
     - Runs Q#'s resource estimator on the circuit. Accepts optional ``params`` for estimation configuration.


Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``qasm``
     - str | None
     - OpenQASM circuit string, if available.
   * - ``qir``
     - QirInputData | str | None
     - :term:`QIR` representation, if available. Lazily compiled from the Q# factory on first access.
   * - ``qsharp``
     - qsharp._native.Circuit | None
     - Compiled Q# circuit object, if available.
   * - ``encoding``
     - str | None
     - Qubit encoding label (e.g., ``"jordan-wigner"``), if set.


Serialization
-------------

:class:`~qdk_chemistry.data.Circuit` supports the same :doc:`serialization <serialization>` formats as other QDK/Chemistry data classes:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization

.. note::

   Serialization persists the OpenQASM and :term:`QIR` representations.
   Q# callables and factories are not directly serializable — they are reconstructed when the producing algorithm re-creates the circuit.


Related classes
---------------

- :class:`~qdk_chemistry.data.CircuitExecutorData`: Measurement results from circuit execution
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Often paired with a circuit for energy estimation

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/circuit.py>`_ script.
- :doc:`CircuitExecutor <../algorithms/circuit_executor>`: Execute circuits on backends
- :doc:`Serialization <serialization>`: Data persistence formats
