MajoranaMapping
===============

The :class:`~qdk_chemistry.data.MajoranaMapping` class in QDK/Chemistry is an immutable data class that defines a fermion-to-qubit encoding by mapping Majorana operators to Pauli strings.
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

Fermion-to-qubit mappings transform fermionic creation and annihilation operators into qubit (Pauli) operators.
Every such mapping can be expressed as a table of Majorana-to-Pauli-string correspondences.
The :class:`~qdk_chemistry.data.MajoranaMapping` class encapsulates this table, making the :doc:`QubitMapper <../algorithms/qubit_mapper>` algorithm encoding-agnostic: the mapper receives the encoding as data rather than selecting it internally.

Convention
~~~~~~~~~~

``num_modes``
   The number of fermionic modes (spin-orbitals) in the system.

Pauli strings use little-endian qubit ordering, consistent with the rest of QDK/Chemistry's :doc:`PauliOperator <pauli_operator>` layer.

Built-in encodings
------------------

Factory methods construct standard encodings for a given number of modes.
Each returns a :class:`~qdk_chemistry.data.MajoranaMapping` with the appropriate Pauli-string table and a descriptive ``name``.

Jordan-Wigner
~~~~~~~~~~~~~

.. code-block:: python

   from qdk_chemistry.data import MajoranaMapping

   mapping = MajoranaMapping.jordan_wigner(num_modes=12)

Encodes each fermionic mode in a single qubit.
See :ref:`encoding-jordan-wigner` for a description of the encoding.

Bravyi-Kitaev
~~~~~~~~~~~~~

.. code-block:: python

   mapping = MajoranaMapping.bravyi_kitaev(num_modes=12)

Uses a binary-tree structure to reduce average Pauli-string weight.
See :ref:`encoding-bravyi-kitaev` for a description of the encoding.

Parity
~~~~~~

.. code-block:: python

   mapping = MajoranaMapping.parity(num_modes=12)

Encodes cumulative electron-number parities.
See :ref:`encoding-parity` for a description of the encoding.

Custom encodings
----------------

A custom encoding can be defined by providing a Pauli-string table directly:

.. code-block:: python

   from qdk_chemistry.data import MajoranaMapping

   # Provide a list of Pauli strings, one per Majorana operator
   mapping = MajoranaMapping(table=[...], name="my-custom-encoding")

Alternatively, construct from mode pairs:

.. code-block:: python

   mapping = MajoranaMapping.from_mode_pairs(...)

Validation
----------

At construction, the :class:`~qdk_chemistry.data.MajoranaMapping` validates that the provided table satisfies the Clifford algebra anti-commutation relations required for a valid fermion-to-qubit mapping.
Invalid tables raise an error immediately, preventing silent correctness issues downstream.

Serialization
-------------

:class:`~qdk_chemistry.data.MajoranaMapping` supports the same :doc:`serialization <serialization>` formats as other QDK/Chemistry data classes:

.. code-block:: python

   from qdk_chemistry.data import MajoranaMapping

   mapping = MajoranaMapping.jordan_wigner(num_modes=12)

   # JSON round-trip
   json_str = mapping.to_json()
   restored = MajoranaMapping.from_json(json_str)

   # HDF5 round-trip
   mapping.to_hdf5("mapping.h5")
   restored = MajoranaMapping.from_hdf5("mapping.h5")


Related classes
---------------

- :doc:`Hamiltonian <hamiltonian>`: Fermionic Hamiltonian transformed by the mapping
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Output of qubit mapping using this encoding
- :doc:`PauliOperator <pauli_operator>`: Pauli operator expressions used in the mapping table

Further reading
---------------

- :doc:`QubitMapper <../algorithms/qubit_mapper>`: The algorithm that consumes a ``MajoranaMapping`` to perform fermion-to-qubit transformations
- :doc:`Design principles <../design/index>`: Data class design principles in QDK/Chemistry
