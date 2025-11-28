Qubit mapping
=============

The :class:`~qdk_chemistry.algorithms.QubitMapper` algorithm in QDK/Chemistry performs the essential task of transforming electronic-structure Hamiltonians into qubit Hamiltonians suitable for quantum computation.


Overview
--------

The :class:`~qdk_chemistry.algorithms.QubitMapper` algorithm converts fermionic Hamiltonians into qubit-operator representations composed of Pauli strings.
This transformation preserves the operator algebra, particle-number constraints, and antisymmetry required by fermionic statistics.
The resulting qubit Hamiltonian is mathematically equivalent to the original fermionic Hamiltonian but is now in a form that can be executed on quantum hardware or simulated by quantum algorithms.

The mapper supports multiple encoding strategies:

   - **Jordan-Wigner mapping** :cite:`Jordan-Wigner1928`: Encodes each fermionic mode in a single qubit whose state directly represents the orbital occupation.
   - **Parity mapping** :cite:`Love2012`: Encodes qubits with cumulative electron-number parities of the orbitals.
   - **Bravyi-Kitaev mapping** :cite:`Bravyi-Kitaev2002`: Distributes both occupation and parity information across qubits using a binary-tree (Fenwick tree) structure, reducing the average Pauli-string length to logarithmic scaling.

Capabilities
------------

The :class:`~qdk_chemistry.algorithms.QubitMapper` in QDK/Chemistry provides:

- **Encoding Options**: Support for different encoding options integrated through Qiskit plugin (Jordan-Wigner, Parity, Bravyi-Kitaev).

Creating a QubitMapper
----------------------

The :class:`~qdk_chemistry.algorithms.QubitMapper` is created using the :doc:`factory pattern <../design/factory_pattern>`.

.. tab:: Python API

   .. literalinclude:: ../../../../examples/qubit_mapper.py
      :language: python
      :lines: 3-10

Mapping a Hamiltonian
----------------------

This mapper is used to create a :class:`~qdk_chemistry.data.QubitHamiltonian` object from a :class:`~qdk_chemistry.data.Hamiltonian`.

.. tab:: Python API

   .. code-block:: python

      # --------------------------------------------------------------------------------------------
      # Copyright (c) Microsoft Corporation. All rights reserved.
      # Licensed under the MIT License. See LICENSE.txt in the project root for license information.
      # --------------------------------------------------------------------------------------------

      from qdk_chemistry.data import Hamiltonian

      # Obtain a valid Hamiltonian instance
      hamiltonian = Hamiltonian.from_json_file("molecule.hamiltonian.json")

      # Map the Hamiltonian to a QubitHamiltonian
      qubit_hamiltonian = mapper.run(hamiltonian)

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.QubitMapper` accepts a range of settings to control its behavior.

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

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.QubitMapper` provides a unified interface for qubit mapping methods.

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **qiskit**: Qiskit QubitMapper implementation with multiple encoding strategies

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for mapping
