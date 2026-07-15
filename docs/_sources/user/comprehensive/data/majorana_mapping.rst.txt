MajoranaMapping
===============

The :class:`~qdk_chemistry.data.MajoranaMapping` class defines a fermion-to-qubit encoding.
It exposes the **bilinear** :math:`i\,\gamma_j\,\gamma_k` as the unified primitive available across every encoding, and (for Majorana-atomic encodings) individual Majorana operators :math:`\gamma_k` as an additional capability.
It follows QDK/Chemistry's data-container conventions for immutable fermion-to-qubit encoding data.

Overview
--------

Fermion-to-qubit mappings transform fermionic creation and annihilation operators into qubit (Pauli) operators.
The :class:`~qdk_chemistry.data.MajoranaMapping` class encapsulates such an encoding as data, making the :doc:`QubitMapper <../algorithms/qubit_mapper>` algorithm encoding-agnostic: the mapper receives the encoding as data rather than selecting it internally.

Bilinears as the unified primitive
----------------------------------

Every fermion-to-qubit encoding can express the **bilinear** product :math:`i\,\gamma_j\,\gamma_k` as a Pauli string.
This makes the bilinear the most general building block that all encodings share.

Why bilinears?  Physical (parity-conserving) fermionic operators can always be
written as products of bilinears, so any Hamiltonian can be mapped through
bilinears alone — even if the encoding does not assign a Pauli image to
individual Majorana operators :math:`\gamma_k`.

For the formal foundations, see `Bravyi and Kitaev (2002) <https://arxiv.org/abs/quant-ph/0003137>`_,
which develops the Majorana-operator perspective on fermion-to-qubit encodings.

Individual Majorana operators :math:`\gamma_k` have a Pauli image only in **Majorana-atomic** encodings (e.g. Jordan-Wigner, Bravyi-Kitaev, Parity).
In **bilinear-only** encodings — where the number of qubits exceeds the number of fermionic modes — single Majoranas have no representation in the physical subspace; only the bilinears are observable.

The :class:`~qdk_chemistry.data.MajoranaMapping` supports both forms:

- :py:meth:`~qdk_chemistry.data.MajoranaMapping.bilinear` — the unified primitive available on every encoding.
- :py:meth:`~qdk_chemistry.data.MajoranaMapping.majorana` — the additional capability provided by Majorana-atomic encodings; gated by :py:attr:`~qdk_chemistry.data.MajoranaMapping.is_majorana_atomic`.

Majorana-atomic mappings are constructed from a Pauli-string table (the constructor or factory methods).
Bilinear-only mappings are constructed via :py:meth:`~qdk_chemistry.data.MajoranaMapping.from_bilinears`.

Convention
~~~~~~~~~~

``num_modes``
   The number of fermionic modes (spin-orbitals) in the system.

Pauli strings use little-endian qubit ordering, consistent with the rest of QDK/Chemistry's :doc:`PauliOperator <pauli_operator>` layer.
:py:meth:`~qdk_chemistry.data.MajoranaMapping.majorana` and :py:meth:`~qdk_chemistry.data.MajoranaMapping.bilinear` return Pauli strings in the encoding's native (pre-taper) qubit basis.

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

A custom Majorana-atomic encoding can be defined by providing a sparse Pauli-word table directly:

.. code-block:: python

   from qdk_chemistry.data import MajoranaMapping

   # Provide one sparse Pauli word per Majorana operator.
   # Entries are (qubit_index, operator_code), with X=1, Y=2, Z=3.
   mapping = MajoranaMapping.from_table([...], name="my-custom-encoding")

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
- :class:`~qdk_chemistry.data.QubitOperator`: Output of qubit mapping using this encoding
- :doc:`PauliOperator <pauli_operator>`: Pauli operator expressions used in the mapping table

Further reading
---------------

- :doc:`QubitMapper <../algorithms/qubit_mapper>`: The algorithm that consumes a ``MajoranaMapping`` to perform fermion-to-qubit transformations
- :doc:`Design principles <../design/index>`: Data class design principles in QDK/Chemistry
