State Preparation
=================

The :class:`~qdk_chemistry.algorithms.StatePreparation` algorithm in QDK/Chemistry constructs quantum circuits that load target wavefunctions onto qubits.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Wavefunction` instance as input and produces an OpenQASM circuit as output.
The output circuit, when executed, prepares the qubit register in a state that encodes the input wavefunction.

Overview
--------

The :class:`~qdk_chemistry.algorithms.StatePreparation` module provides two complementary approaches for constructing circuits that load wavefunctions onto qubits:

- **Sparse Isometry (GF2+X)**: An optimized approach that leverages sparsity in the target wavefunction. This method applies GF(2) Gaussian elimination to the binary matrix representation of the state, while removing duplicate rows, all-ones rows, and reducing the rank of the diagonal matrix. These reductions correspond to implementing CNOT and X gates that simplify the preparation basis. By performing this GF(2)+X elimination, the algorithm constructs circuits that prepare only the non-zero amplitudes, substantially reducing circuit depth and gate count compared with dense isometry methods.
- **Regular Isometry**: Isometry-based state preparation proposed by Matthias Christandl :cite:`Christandl2016`, with circuit synthesis integrated through the Qiskit plugin.

Using the StatePreparation
--------------------------

This section demonstrates how to create, configure, and run a state preparation.
The ``run`` method returns an OpenQASM circuit string that, when executed, loads the input wavefunction onto a qubit register.

**Creating a state preparation algorithm:**

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

**Configuring settings:**

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/state_preparation.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

**Running the calculation:**

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

   .. code-block:: python

      from qdk_chemistry.algorithms import registry
      print(registry.available("state_prep"))  # ['sparse_isometry_gf2x', 'regular_isometry']

Sparse Isometry GF2+X
~~~~~~~~~~~~~~~~~~~~~

**Factory name:** ``"sparse_isometry_gf2x"``

Native QDK/Chemistry implementation of sparse isometry state preparation using GF(2)+X elimination. Optimized for wavefunctions with sparse amplitude structure, substantially reducing circuit depth compared with dense methods.

**Settings:**

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

**Factory name:** ``"regular_isometry"``

State preparation using regular isometry synthesis via Qiskit. Implements the isometry-based approach proposed by Matthias Christandl.

**Settings:**

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
