MajoranaMapping
===============

The :class:`~qdk_chemistry.data.MajoranaMapping` class in QDK/Chemistry is an immutable data class that defines a fermion-to-qubit encoding.
It exposes the **bilinear** :math:`i\,\gamma_j\,\gamma_k` as the unified primitive available across every encoding, and (for Majorana-atomic encodings) individual Majorana operators :math:`\gamma_k` as an additional capability.
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

Fermion-to-qubit mappings transform fermionic creation and annihilation operators into qubit (Pauli) operators.
The :class:`~qdk_chemistry.data.MajoranaMapping` class encapsulates such an encoding as data, making the :doc:`QubitMapper <../algorithms/qubit_mapper>` algorithm encoding-agnostic: the mapper receives the encoding as data rather than selecting it internally.

Bilinears as the unified primitive
----------------------------------

Across fermion-to-qubit encodings, the most general primitive that admits a Pauli-string image is the **bilinear** :math:`i\,\gamma_j\,\gamma_k`.
Bilinears generate the parity-even subalgebra of the Majorana Clifford algebra, so any parity-conserving operator decomposes into ordered bilinear products, and higher-degree even monomials are products of bilinears.

Individual Majorana operators :math:`\gamma_k` have a Pauli image only in **Majorana-atomic** encodings.
The :class:`~qdk_chemistry.data.MajoranaMapping` API therefore exposes:

- :py:meth:`~qdk_chemistry.data.MajoranaMapping.bilinear` — the unified primitive available on every encoding.
- :py:meth:`~qdk_chemistry.data.MajoranaMapping.majorana` — the additional capability provided by Majorana-atomic encodings; gated by :py:attr:`~qdk_chemistry.data.MajoranaMapping.is_majorana_atomic`.

Convention
~~~~~~~~~~

``num_modes``
   The number of fermionic modes (spin-orbitals) in the system.

Pauli strings use little-endian qubit ordering, consistent with the rest of QDK/Chemistry's :doc:`PauliOperator <pauli_operator>` layer.
:py:meth:`~qdk_chemistry.data.MajoranaMapping.majorana` and :py:meth:`~qdk_chemistry.data.MajoranaMapping.bilinear` return Pauli strings in the encoding's native (pre-taper) qubit basis, i.e. of length ``len(mapping.table[0])``; any tapering specification is applied downstream by the qubit mapper.

Built-in encodings
------------------

Factory methods construct standard encodings for a given number of modes.
Each returns a :class:`~qdk_chemistry.data.MajoranaMapping` with the appropriate Pauli-string table and a descriptive ``name``.

Jordan-Wigner
~~~~~~~~~~~~~

.. code-block:: python

   from qdk_chemistry.data import MajoranaMapping

   mapping = MajoranaMapping.jordan_wigner(num_modes=12)

   # Bilinear (works on every encoding):
   coeff, pauli_str = mapping.bilinear(0, 1)

   # Single Majorana (Majorana-atomic encodings only):
   if mapping.is_majorana_atomic:
       gamma_0 = mapping.majorana(0)

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

A custom Majorana-atomic encoding can be defined by providing a Pauli-string table directly:

.. code-block:: python

   from qdk_chemistry.data import MajoranaMapping

   # Provide a list of Pauli strings, one per Majorana operator
   mapping = MajoranaMapping(table=[...], name="my-custom-encoding")

Alternatively, construct from mode pairs:

.. code-block:: python

   mapping = MajoranaMapping.from_mode_pairs(...)

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
   mapping.to_hdf5_file("mapping.h5")
   restored = MajoranaMapping.from_hdf5_file("mapping.h5")


Related classes
---------------

- :doc:`Hamiltonian <hamiltonian>`: Fermionic Hamiltonian transformed by the mapping
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Output of qubit mapping using this encoding
- :doc:`PauliOperator <pauli_operator>`: Pauli operator expressions used in the mapping table

Further reading
---------------

- :doc:`QubitMapper <../algorithms/qubit_mapper>`: The algorithm that consumes a ``MajoranaMapping`` to perform fermion-to-qubit transformations
- :doc:`Design principles <../design/index>`: Data class design principles in QDK/Chemistry
