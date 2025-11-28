Energy estimation
=================

The :class:`~qdk_chemistry.algorithms.EnergyEstimator` algorithm in QDK/Chemistry provides a flexible and efficient framework for computing expectation values with quantum circuit and Hamiltonians.
The estimator evaluates energies by generating measurement circuits, executing them on a backend with a configurable number of shots, and statistically aggregating the resulting bitstring outcomes.
It is designed to support multiple backend simulators.


Overview
--------

The :class:`~qdk_chemistry.algorithms.EnergyEstimator` evaluates the expectation value of a qubit Hamiltonian with respect to a given quantum circuit.
It takes a circuit in OpenQASM format with target qubit Hamiltonians, automatically generates the corresponding measurement circuits.
These circuits are executed on a selected backend simulator with the user-specified number of shots, and the resulting bitstring statistics are used to calculate per-term expectation values and the total energy.


Capabilities
------------

The :class:`~qdk_chemistry.algorithms.EnergyEstimator` provides the following capabilities:

- **Expectation Value and Variance Calculation**: Computes the energy expectation value and variance for a given quantum circuit and Hamiltonians. It supports multiple Hamiltonians for simultaneous evaluation.
- **Backend Flexibility**: Allows users to choose between Qsharp and Qiskit backends, each with unique features and configurations, such as noise modeling.

Creating an Energy Estimator
----------------------------

The :class:`~qdk_chemistry.algorithms.EnergyEstimator` is created using the :doc:`factory pattern <../design/factory_pattern>`.

.. tab:: Python API

   .. literalinclude:: ../../../../examples/energy_estimator.py
      :language: python
      :lines: 3-7,31-32,60-62


Configuring the Energy Estimator
--------------------------------

Qsharp Backend
~~~~~~~~~~~~~~

The Qsharp implementation of the :class:`~qdk_chemistry.algorithms.EnergyEstimator` leverages the QDK simulator to execute quantum circuits. Key features include:

- Support for depolarizing noise, bit flip noise, pauli noise, and phase flip noise.
- Simulation with qubit loss.

.. tab:: Python API

   .. literalinclude:: ../../../../examples/energy_estimator.py
      :language: python
      :lines: 3-7,40-60

Qiskit Backend
~~~~~~~~~~~~~~~

The Qiskit implementation uses the Aer simulator to execute quantum circuits. Key features include:

- Support for custom noise models.
- Support for other configurations of Aer simulator backends.


.. tab:: Python API

   .. literalinclude:: ../../../../examples/energy_estimator.py
      :language: python
      :lines: 3-7,70-85

Implemented interface
---------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.EnergyEstimator` provides a unified interface for selecting simulator backend for energy estimation.

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **qdk_base_simulator**: Native implementation with support for Qsharp simulator backends

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **qiskit_aer_simulator**: Qiskit Aer simulator backend with customizable noise models and configurations

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

Related Topics
--------------

- :doc:`StatePreparation <state_preparation>`: Prepare molecule wavefunctions into quantum circuits.
- :doc:`QubitMapper <qubit_mapper>`: Prepare qubit Hamiltonians.
