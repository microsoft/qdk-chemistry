State preparation
=================

The :class:`~qdk_chemistry.algorithms.StatePreparation` algorithm in QDK/Chemistry constructs quantum circuits that load classical representations of target wavefunctions onto qubits.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Wavefunction` instance as input and produces an OpenQASM circuit as output.
The output circuit, when executed, prepares the qubit register in a state that encodes the input wavefunction.

Overview
--------

The :class:`~qdk_chemistry.algorithms.StatePreparation` module provides tools for constructing quantum circuits that load classical representations of wavefunctions(e.g., a Slater determinant or a linear combination thereof, represented by the `Wavefunction` class)  onto qubits. It supports multiple approaches for state preparation, allowing users to choose the method best suited to their problem. Each approach is designed to efficiently encode quantum states for chemistry applications.

For details on individual methods and their technical implementations, see the `Available implementations`_ section below.

Using the StatePreparation
--------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a state preparation.
The ``run`` method returns an OpenQASM circuit string that, when executed, loads the input wavefunction onto a qubit register.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.StatePreparation` requires the following input:

Wavefunction
   A :class:`~qdk_chemistry.data.Wavefunction` instance containing the quantum state to be loaded onto qubits. This is typically obtained from a multi-configuration calculation using the :doc:`MultiConfigurationCalculator <mc_calculator>`. The method with which this encoding is achieved is implementation dependent.


.. rubric:: Creating a state preparation algorithm

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

Once configured, the :class:`~qdk_chemistry.algorithms.StatePreparation` can be used to generate a quantum circuit from a :class:`~qdk_chemistry.data.Wavefunction`.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.StatePreparation` provides a unified interface for state preparation methods.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

Sparse Isometry GF2+X
~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"sparse_isometry_gf2x"``

This method is an optimized approach that leverages sparsity in the target wavefunction. The GF2+X method, a modification of the original sparse isometry work in :cite:`Malvetti2021`, applies GF(2) Gaussian elimination to the binary matrix representation of the state to determine a reduced space representation of the sparse state. This reduced state is then densely encoded via regular isometry :cite:`Christandl2016` on a smaller number of qubits, and finally scattered to the full qubit space using X and :term:`CNOT` gates. These reductions correspond to efficient gate sequences that simplify the preparation basis. By focusing only on non-zero amplitudes, this approach substantially reduces circuit depth and gate count compared with dense isometry methods. This method is native to QDK/Chemistry and is especially efficient for wavefunctions with sparse amplitude structure.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``basis_gates``
     - list[str]
     - Basis gates for transpilation. Default is ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"].
   * - ``transpile``
     - bool
     - Whether to transpile the circuit. Default is True.
   * - ``transpile_optimization_level``
     - int
     - Optimization level for transpilation (0-3). Default is 1.

Regular Isometry
~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qiskit_regular_isometry"``

This method uses regular isometry synthesis via `Qiskit <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.StatePreparation>`_, implementing the isometry-based approach proposed by Matthias Christandl :cite:`Christandl2016`. It provides a general solution for state preparation, and is suitable for cases where a dense representation is required or preferred.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``basis_gates``
     - list[str]
     - Basis gates for transpilation. Default is ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"].
   * - ``transpile``
     - bool
     - Whether to transpile the circuit. Default is True.
   * - ``transpile_optimization_level``
     - int
     - Optimization level for transpilation (0-3). Default is 1.

For more details on how QDK/Chemistry interfaces with external packages, see the :ref:`plugin system <plugin-system>` documentation.

Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`: Input wavefunction for circuit construction

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/state_preparation.py>`_ script.
- :doc:`EnergyEstimator <energy_estimator>`: Estimate the energy of prepared states
- :doc:`QubitMapper <qubit_mapper>`: Map Hamiltonians to qubit operators
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
