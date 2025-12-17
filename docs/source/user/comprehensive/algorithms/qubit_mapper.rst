Qubit mapping
=============

The :class:`~qdk_chemistry.algorithms.QubitMapper` algorithm in QDK/Chemistry transforms electronic-structure Hamiltonians into qubit Hamiltonians suitable for quantum computation.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Hamiltonian <../data/hamiltonian>` instance as input and produces a :class:`~qdk_chemistry.data.QubitHamiltonian` instance as output.
This transformation is essential for executing quantum chemistry algorithms on quantum hardware.

Overview
--------

The :class:`~qdk_chemistry.algorithms.QubitMapper` algorithm converts fermionic Hamiltonians into qubit-operator representations composed of Pauli strings.
This transformation preserves the operator algebra, particle-number constraints, and antisymmetry required by fermionic statistics.
The resulting qubit Hamiltonian is mathematically equivalent to the original fermionic Hamiltonian but is now in a form that can be executed on quantum hardware or simulated by quantum algorithms.

The mapper supports multiple encoding strategies:

- **Jordan-Wigner mapping** :cite:`Jordan-Wigner1928`: Encodes each fermionic mode in a single qubit whose state directly represents the orbital occupation.
- **Parity mapping** :cite:`Love2012`: Encodes qubits with cumulative electron-number parities of the orbitals.
- **Bravyi-Kitaev mapping** :cite:`Bravyi-Kitaev2002`: Distributes both occupation and parity information across qubits using a binary-tree (Fenwick tree) structure, reducing the average Pauli-string length to logarithmic scaling.

Using the QubitMapper
---------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a qubit mapping.
The ``run`` method returns a :class:`~qdk_chemistry.data.QubitHamiltonian` object containing the Pauli-string representation.

**Creating a mapper:**

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

**Configuring settings:**

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

**Running the calculation:**

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.QubitMapper` provides a unified interface for qubit mapping methods.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

Qiskit
~~~~~~

**Factory name:** ``"qiskit"``

Qubit mapping implementation integrated through the Qiskit plugin. Supports multiple fermion-to-qubit encoding strategies including Jordan-Wigner, parity, and Bravyi-Kitaev mappings.

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``encoding``
     - string
     - Qubit mapping strategy (``jordan-wigner``, ``bravyi-kitaev``, ``parity``)

For more details on how QDK/Chemistry interfaces with external packages, see the :ref:`plugin system <plugin-system>` documentation.

Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for mapping
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Output qubit operator representation

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/qubit_mapper.py>`_ script.
- :doc:`StatePreparation <state_preparation>`: Prepare quantum circuits from wavefunctions
- :doc:`EnergyEstimator <energy_estimator>`: Estimate energies using the qubit Hamiltonian
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
