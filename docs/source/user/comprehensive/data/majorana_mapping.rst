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

Across every fermion-to-qubit encoding, the most general primitive that admits a Pauli-string image is the **bilinear** :math:`i\,\gamma_j\,\gamma_k`.
This is not a stylistic choice: bilinears generate the parity-even subalgebra of the Majorana Clifford algebra in *every* encoding, so any parity-conserving operator (Hamiltonian terms, BdG anomalous terms, density-density couplings) decomposes into ordered bilinear products.
Quartics and higher-degree even monomials are products of bilinears.

By contrast, individual Majorana operators :math:`\gamma_k` only have a Pauli image in **Majorana-atomic** encodings.
For *redundant* encodings — Bravyi-Kitaev superfast, Verstraete-Cirac, Derby-Klassen compact, Setia-Whitfield, and the broader Majorana loop stabilizer code family — :math:`m > n` qubits represent :math:`n` modes, and a single :math:`\gamma_k` anticommutes with the total parity stabilizer; it has no representation in the codespace at all.

The :class:`~qdk_chemistry.data.MajoranaMapping` API therefore exposes:

- :py:meth:`~qdk_chemistry.data.MajoranaMapping.bilinear` — the unified primitive available on every encoding.
- :py:meth:`~qdk_chemistry.data.MajoranaMapping.majorana` — the additional capability that Majorana-atomic encodings provide; gated by :py:attr:`~qdk_chemistry.data.MajoranaMapping.is_majorana_atomic`.

The current factory methods (Jordan-Wigner, Bravyi-Kitaev, parity, SCBK) are all Majorana-atomic, so both APIs are available and produce consistent results: :py:meth:`~qdk_chemistry.data.MajoranaMapping.bilinear` is computed on demand from the stored Majorana table.

A common point of confusion: BdG (Bogoliubov-de Gennes) Hamiltonians contain anomalous terms like :math:`a_i^\dagger a_j^\dagger`. Despite breaking :math:`U(1)` particle number, these are parity-even bilinears in :math:`\gamma`'s and are representable on **any** backend, redundant or not. The cases that *require* a Majorana-atomic backend are single-Majorana observables (e.g. MZM measurements), state preparation by acting with a single :math:`a_j^\dagger` on the vacuum, and Bogoliubov quasiparticle operators viewed as observables.

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

Validation
----------

At construction, the :class:`~qdk_chemistry.data.MajoranaMapping` validates that the provided table satisfies the Clifford algebra anti-commutation relations required for a valid Majorana-atomic fermion-to-qubit mapping.
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
- Chen, Xu, Boettcher, "Equivalence between fermion-to-qubit mappings in two spatial dimensions" — unified treatment of every fermion-to-qubit mapping as a homomorphism from the parity-even Majorana Clifford algebra into the Pauli group, modulo stabilizers.
- Jiang, Kalev, Mruczkiewicz, Neven, "Optimal fermion-to-qubit mapping via ternary trees" and the related Majorana loop stabilizer code literature — redundant encodings as stabilizer codes.
