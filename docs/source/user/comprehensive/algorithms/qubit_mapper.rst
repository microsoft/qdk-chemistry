Qubit mapping
=============

The ``QubitMapper`` algorithm in QDK/Chemistry performs the essential task of transforming electronic-structure Hamiltonians into qubit Hamiltonians suitable for quantum computation.


Overview
--------

The ``QubitMapper`` algorithm converts fermionic Hamiltonians into qubit-operator representations composed of Pauli strings.
This transformation preserves the operator algebra, particle-number constraints, and antisymmetry required by fermionic statistics.
The resulting qubit Hamiltonian is mathematically equivalent to the original fermionic Hamiltonian but is now in a form that can be executed on quantum hardware or simulated by quantum algorithms.

Capabilities
------------

The ``QubitMapper`` in QDK/Chemistry provides:

- **Encoding Options**:
   Support for different encoding options integrated through Qiskit plugin:

   - **Jordan-Wigner mapping** (Zeitschrift für Physik, 47, 631-651 (1928)): Encodes each fermionic mode in a single qubit whose state directly represents the orbital occupation.
   - **Parity mapping** (The Journal of chemical physics, 137(22), 224109 (2012)): Encodes qubits with cumulative electron-number parities of the orbitals.
   - **Bravyi-Kitaev mapping** (Annals of Physics, 298(1), 210-226 (2002)): Distributes both occupation and parity information across qubits using a binary-tree (Fenwick tree) structure, reducing the average Pauli-string length to logarithmic scaling.

Creating a QubitMapper
----------------------

The ``QubitMapper`` is created using the :doc:`factory pattern <../advanced/factory_pattern>`.

.. tab:: Python API

   .. literalinclude:: ../../../../examples/qubit_mapper.py
      :language: python
      :lines: 3-10

Mapping a Hamiltonian
----------------------

This mapper is used to create a :doc:`QubitHamiltonian <../data/qubit_hamiltonian>` object from a :doc:`Hamiltonian <../data/hamiltonian>`.

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.data import Hamiltonian

      # Obtain a valid Hamiltonian instance
      hamiltonian = Hamiltonian(...)

      # Map the Hamiltonian to a QubitHamiltonian
      qubit_hamiltonian = mapper.run(hamiltonian)

Available settings
------------------

The ``QubitMapper`` accepts a range of settings to control its behavior.

Base settings
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``encoding``
     - string
     - Qubit mapping strategy (``jordan-wigner``, ``bravyi-kitaev``, ``parity``)

Implemented interface
---------------------

QDK/Chemistry's ``QubitMapper`` provides a unified interface for qubit mapping methods.

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **qiskit**: Qiskit QubitMapper implementation with multiple encoding strategies

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../advanced/interfaces>` documentation.

Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for mapping
