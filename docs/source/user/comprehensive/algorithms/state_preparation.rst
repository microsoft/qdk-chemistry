State Preparation
=================

The :class:`~qdk_chemistry.algorithms.StatePreparation` algorithm in QDK/Chemistry constructs quantum circuits that prepare target wavefunctions on quantum hardware.

Overview
--------

State preparation is a critical step in quantum algorithms, where a quantum state is initialized to represent a specific wavefunction.
The :class:`~qdk_chemistry.algorithms.StatePreparation` module provides two complementary approaches for constructing circuits:

- **Regular Isometry**: Isometry-based state preparation proposed by Matthias Christandl :cite:`Christandl2016`, with circuit synthesis integrated through the Qiskit plugin
- **Sparse Isometry**: An optimized approach that leverages sparsity in the target wavefunction. This method applies GF(2) Gaussian elimination to the binary matrix representation of the state, while removing duplicate rows, all-ones rows, and reducing the rank of the diagonal matrix. These reductions correspond to implementing CNOT and X gates that simplify the preparation basis. By performing this GF(2)+X elimination, the algorithm constructs circuits that prepare only the non-zero amplitudes, substantially reducing circuit depth and gate count compared with dense isometry methods.

Capabilities
------------

The :class:`~qdk_chemistry.algorithms.StatePreparation` in QDK/Chemistry provides:

- **Efficient Circuit Construction**: Leveraging GF(2) elimination to optimize quantum circuits for sparse wavefunctions.
- **Circuit Transpilation Options**: Configurable transpilation settings to optimize circuits for specific quantum hardware backends.

Creating a StatePreparation
----------------------------

The :class:`~qdk_chemistry.algorithms.StatePreparation` created using the :doc:`factory pattern <../advanced/factory_pattern>`.

.. tab:: Python API

   .. literalinclude:: ../../../../examples/state_preparation.py
      :language: python
      :lines: 3-10

Configuring the StatePreparation
--------------------------------

The :class:`~qdk_chemistry.algorithms.StatePreparation` can be configured using the ``Settings`` object with the following parameters.

.. tab:: Python API

   .. literalinclude:: ../../../../examples/state_preparation.py
      :language: python
      :lines: 3-7, 11-13


Preparing a Quantum State
--------------------------

Once configured, the :class:`~qdk_chemistry.algorithms.StatePreparation` can be used to generate a quantum circuit in OpenQASM format from a :doc:`Wavefunction <../data/wavefunction>`.

.. tab:: Python API

   .. code-block:: python

      # --------------------------------------------------------------------------------------------
      # Copyright (c) Microsoft Corporation. All rights reserved.
      # Licensed under the MIT License. See LICENSE.txt in the project root for license information.
      # --------------------------------------------------------------------------------------------

      from qdk_chemistry.data import Wavefunction

      # Obtain a valid Wavefunction instance
      wavefunction = Wavefunction.from_json_file("molecule.wavefunction.json")

      # Generate the quantum circuit
      circuit_qasm = state_prep.run(wavefunction)

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.StatePreparation` accepts a range of settings to control its behavior.

Base settings
~~~~~~~~~~~~~

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

Implemented interface
---------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.StatePreparation` provides a unified interface for state preparation methods.

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **sparse_isometry_gf2x**: Native implementation of sparse isometry state preparation using GF(2)+X elimination

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **regular_isometry**: State preparation using regular isometry synthesis via Qiskit

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

Related classes
---------------

- :doc:`EnergyEstimator <./energy_estimator>`: Estimate the energy of prepared states
