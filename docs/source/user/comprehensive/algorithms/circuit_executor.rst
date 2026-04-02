Circuit execution
=================

The :class:`~qdk_chemistry.algorithms.CircuitExecutor` algorithm in QDK/Chemistry executes quantum circuits and returns measurement results.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Circuit`, a shot count, and an optional :class:`~qdk_chemistry.data.QuantumErrorProfile` as input and returns a :class:`~qdk_chemistry.data.CircuitExecutorData` object containing bitstring counts and execution metadata.

Overview
--------

The :class:`~qdk_chemistry.algorithms.CircuitExecutor` provides a unified interface for running quantum circuits across different backends.
It is used by algorithms such as :doc:`PhaseEstimation <phase_estimation>` and :doc:`EnergyEstimator <energy_estimator>` to execute measurement circuits, but it can also be used independently for general-purpose circuit execution.

The algorithm supports:

Multiple backends
   QDK/Chemistry currently ships simulator backends — QDK's native Q# simulators (sparse-state and full-state) and Qiskit's Aer simulator via the :doc:`plugin system <../plugins>`. The interface is designed to be extensible to additional backends.

Noise modelling
   Depolarizing noise on individual gates via :class:`~qdk_chemistry.data.QuantumErrorProfile`, enabling realistic simulation of noisy quantum hardware.

Flexible circuit input
   Accepts circuits in OpenQASM, :term:`QIR`, or native Q# representations — the :class:`~qdk_chemistry.data.Circuit` class handles format conversion automatically.


Using the CircuitExecutor
-------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a circuit executor.
The ``run`` method returns a :class:`~qdk_chemistry.data.CircuitExecutorData` object containing bitstring measurement counts.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.CircuitExecutor` requires the following inputs:

Circuit
   A :class:`~qdk_chemistry.data.Circuit` object containing the quantum circuit to execute.
   The circuit can be provided in any supported format (OpenQASM, :term:`QIR`, Q# callable, or Q# factory).

Shots
   The number of measurement repetitions to perform.

Noise model (optional)
   A :class:`~qdk_chemistry.data.QuantumErrorProfile` specifying gate-level error rates.
   When provided, the simulator applies the specified noise during circuit execution.
   Not all backends support noise — see `Available implementations`_ for details.

.. rubric:: Creating a circuit executor

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings vary by implementation.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API (QDK)

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-configure-qdk
      :end-before: # end-cell-configure-qdk

.. tab:: Python API (Qiskit)

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-configure-qiskit
      :end-before: # end-cell-configure-qiskit

.. rubric:: Running a circuit

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

.. rubric:: Running with noise

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-noise
      :end-before: # end-cell-noise

Available implementations
-------------------------

You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations


QDK sparse-state simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_sparse_state_simulator"`` (default)

A sparse-state simulator that efficiently represents quantum states with limited entanglement.
Supports both Q# factory circuits and OpenQASM circuits.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``seed``
     - int
     - Random seed for reproducibility. Default is 42.

.. note::
   Noise modelling is not supported by the sparse-state simulator. Use the full-state simulator for noisy simulations.


QDK full-state simulator
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_full_state_simulator"``

A full-state simulator that supports CPU, GPU, and Clifford simulation modes, with optional depolarizing noise via :class:`~qdk_chemistry.data.QuantumErrorProfile`.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``type``
     - str
     - Simulator type: ``"cpu"`` (default), ``"gpu"``, or ``"clifford"``.
   * - ``seed``
     - int
     - Random seed for reproducibility. Default is 42.


Qiskit Aer simulator
~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qiskit_aer_simulator"``

Integration with Qiskit's Aer simulator, supporting multiple simulation methods and custom noise models.
This implementation is available through the :doc:`Qiskit plugin <../plugins>`.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``method``
     - str
     - Simulation method (e.g., ``"statevector"``). Default is ``"statevector"``.
   * - ``seed``
     - int
     - Random seed for reproducibility. Default is 42.
   * - ``transpile_optimization_level``
     - int
     - Qiskit transpilation optimization level (0–3). Default is 0.


Related classes
---------------

- :class:`~qdk_chemistry.data.Circuit`: Input quantum circuit
- :class:`~qdk_chemistry.data.CircuitExecutorData`: Output measurement data (bitstring counts, metadata)
- :class:`~qdk_chemistry.data.QuantumErrorProfile`: Noise model definition

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/circuit_executor.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Uses CircuitExecutor to run :term:`QPE` circuits
- :doc:`EnergyEstimator <energy_estimator>`: Uses CircuitExecutor for observable sampling
- :doc:`Plugins <../plugins>`: Plugin system for external backends
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
