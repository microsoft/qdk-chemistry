Circuit execution
=================

The :class:`~qdk_chemistry.algorithms.CircuitExecutor` algorithm in QDK/Chemistry executes quantum circuits and returns measurement results.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Circuit`, a shot count, and an optional :class:`~qdk_chemistry.data.QuantumErrorProfile` as input and returns a :class:`~qdk_chemistry.data.CircuitExecutorData` object containing bitstring counts and execution metadata.

Overview
--------

The :class:`~qdk_chemistry.algorithms.CircuitExecutor` provides a unified interface for running quantum circuits across different backends.
It is used by algorithms such as :doc:`PhaseEstimation <phase_estimation>` and :doc:`ExpectationEstimator <expectation_estimator>` to execute measurement circuits, but it can also be used independently for general-purpose circuit execution.

The algorithm supports:

Multiple backends
   QDK/Chemistry ships QDK's native Q# simulators (sparse-state and full-state), Qiskit's Aer simulator, and an Azure Quantum backend for submitting jobs to a remote target, all via the :doc:`plugin system <../plugins>`. The interface is designed to be extensible to additional backends.

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

.. tab:: Python API (Azure Quantum)

   .. literalinclude:: ../../../_static/examples/python/circuit_executor.py
      :language: python
      :start-after: # start-cell-configure-azure-quantum
      :end-before: # end-cell-configure-azure-quantum

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


Azure Quantum Backend
~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"azure_quantum_backend"``

Compiles circuits to :term:`QIR` and submits them to an Azure Quantum target, returning the measurement results.
Available through the :doc:`Azure Quantum plugin <../plugins>`; requires ``azure-quantum`` and ``azure-identity``.
Compiled QIR objects are passed directly to the SDK, preserving its QIR content type and submission metadata.

The four workspace settings ``subscription_id``, ``resource_group``, ``workspace_name``, and ``target_name`` have no defaults and must be provided. ``location`` is optional.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``subscription_id``
     - str
     - Azure subscription ID.
   * - ``resource_group``
     - str
     - Azure resource group.
   * - ``workspace_name``
     - str
     - Azure Quantum workspace name.
   * - ``location``
     - str
     - Workspace region.
   * - ``target_name``
     - str
     - Target to submit to.
   * - ``auth_mode``
     - str
     - Credential mode: ``"azure-cli"`` (default) or ``"default"``.
   * - ``input_params``
     - str
     - Job input parameters as a JSON object string. Default is ``"{}"``.
   * - ``job_name``
     - str
     - Name for the submitted job.
   * - ``timeout_secs``
     - int
     - Maximum seconds to wait for job completion. Default is 3600.
   * - ``output_dir``
     - str
     - Local directory for saved job attachments. Empty disables saving.
   * - ``attachments``
     - list[str]
     - Attachment names to download from the job container. Empty disables saving.

Set ``output_dir`` and ``attachments`` together to keep backend artifacts such as traces or raw output.
The saved paths and the job ID appear in the executor metadata.

Failed shots with error or unexpected outcomes are excluded from ``bitstring_counts`` and logged as a warning.
The entire shot is excluded even if only one measurement contains an error; failed shots are not counted as qubit loss.
``total_shots`` remains the requested shot count. For a returned executor result, use
``result.get_executor_metadata()["failed_shots"]`` for the failed-shot count and
``result.get_executor_metadata()["failed_results"]`` for the original failed histogram
entries (outcomes and counts, keyed by the SDK's display labels). The full histogram remains
available under ``"results"``.

If no clean measurement shots remain, the backend raises
``azure.quantum.job.JobFailedWithResultsError``, even if the service reports the job as
succeeded. Its ``get_failure_results()`` method provides the same metadata, including the
job ID, failed outcomes, and paths of saved attachments. No automatic retry is performed.
Combined IQPE requests one whole-circuit execution (``shots=1``), with ``shots_per_bit``
internal samples per phase bit for in-circuit voting. If that execution yields no clean
measurement result, there is no phase or energy to decode.

.. note::
   Custom :class:`~qdk_chemistry.data.QuantumErrorProfile` noise models are not supported. Configure noise through the target's own ``input_params`` instead.


Local demo execution through MCP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MCP tools ``run_azure_quantum_demo`` and ``run_iqpe_azure_quantum_demo``
exercise :class:`~qdk_chemistry.plugins.azure_quantum.circuit_executor.AzureQuantumBackend`
without Azure credentials or a workspace connection. They submit QIR to the local
``demo.target``, backed by QDK CPU full-state simulation, rather than to Azure.
Both tools run in an isolated process, bypass caching, and return
``local_demo=True``, ``submitted_to_azure=False``, and local job histograms.
The ``backend_display_name`` field provides the standard report label,
``"Azure Quantum local demo target"``. The ``reporting_guidance`` field asks
assistants to describe simulator and execution provenance once in Methods
rather than repeating cloud-versus-local disclaimers. These presentation hints
do not change the execution provenance.
Full-state memory grows exponentially with circuit width; use small inputs.

``run_azure_quantum_demo`` requires the ``azure-quantum`` extra. It accepts a
saved :class:`~qdk_chemistry.data.Circuit`, ``shots``, and an output filename for
:class:`~qdk_chemistry.data.CircuitExecutorData`. Measurement counts are returned
inline as well as saved.

``run_iqpe_azure_quantum_demo`` additionally requires ``qiskit-extras`` to compose
saved circuits with the ``qiskit_iterative`` builder. Supply a state-preparation
circuit, a :class:`~qdk_chemistry.data.QubitOperator`, and explicit ``num_bits``
and ``evolution_time`` (atomic units). For example:

.. code-block:: python

   run_iqpe_azure_quantum_demo(
       project_name="h2",
       state_prep_circuit_filename="hf.circuit.json",
       qubit_hamiltonian_filename="h2.qubit_hamiltonian.json",
       out_qpe_result_filename="iqpe.qpe_result.json",
       num_bits=1,
       evolution_time=1.0,
       shots_per_bit=3,
       seed=42,
       unitary_builder_settings={"num_divisions": 4, "order": 2},
   )

This runs one local job per phase bit and saves a
:class:`~qdk_chemistry.data.QpeResult`. The Trotter unitary uses one step by
default; ``unitary_builder_settings`` controls its approximation. Returned
energies exclude the Hamiltonian's core energy. A one-bit example demonstrates
the workflow, not a precision molecular energy calculation. Use
``overwrite=True`` to replace an existing valid result.


Related classes
---------------

- :class:`~qdk_chemistry.data.Circuit`: Input quantum circuit
- :class:`~qdk_chemistry.data.CircuitExecutorData`: Output measurement data (bitstring counts, metadata)
- :class:`~qdk_chemistry.data.QuantumErrorProfile`: Noise model definition

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/circuit_executor.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Uses CircuitExecutor to run :term:`QPE` circuits
- :doc:`ExpectationEstimator <expectation_estimator>`: Uses CircuitExecutor for observable sampling
- :doc:`Plugins <../plugins>`: Plugin system for external backends
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
