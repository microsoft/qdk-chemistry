Ansatz
======

The :class:`~qdk_chemistry.data.Ansatz` class in QDK/Chemistry represents a quantum chemical ansatz that combines a Hamiltonian operator with a wavefunction.
This pairing enables energy expectation value calculations and forms the foundation for various quantum chemistry methods.

Overview
--------

An ansatz in quantum chemistry represents a trial wavefunction paired with the system's Hamiltonian operator.
The :class:`~qdk_chemistry.data.Ansatz` class encapsulates this fundamental concept by combining:

- A :class:`~qdk_chemistry.data.Hamiltonian` operator describing the system's energy
- A :class:`~qdk_chemistry.data.Wavefunction` describing the quantum state

The class ensures consistency between the Hamiltonian and wavefunction components and provides methods for energy calculations and data access.

Properties
~~~~~~~~~~

The :class:`~qdk_chemistry.data.Ansatz` class provides access to:

- **Hamiltonian**: The energy operator for the molecular system
- **Wavefunction**: The quantum state description
- **Orbitals**: Molecular orbital basis set (derived from Hamiltonian)
- **Energy calculation**: Expectation value ⟨ψ|H|ψ⟩
- **Validation**: Consistency checks between components

Creating an ansatz
------------------

An :class:`~qdk_chemistry.data.Ansatz` object can be created from existing Hamiltonian and Wavefunction objects:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/ansatz.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/ansatz.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

Accessing ansatz data
---------------------

The :class:`~qdk_chemistry.data.Ansatz` class provides methods to access its components and perform calculations:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/ansatz.cpp
      :language: cpp
      :start-after: // start-cell-access
      :end-before: // end-cell-access

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/ansatz.py
      :language: python
      :start-after: # start-cell-access
      :end-before: # end-cell-access


Further reading
---------------

- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Wavefunction <wavefunction>`: Quantum state component of the ansatz
- :doc:`Hamiltonian <hamiltonian>`: Energy operator component of the ansatz
- :doc:`Orbitals <orbitals>`: Molecular orbital basis set
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Algorithms that produce ansatz objects
