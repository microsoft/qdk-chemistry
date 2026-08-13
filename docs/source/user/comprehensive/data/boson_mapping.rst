BosonMapping
============

The :class:`~qdk_chemistry.data.BosonMapping` class defines a boson-to-qubit encoding.
It follows QDK/Chemistry's data-container conventions for immutable encoding data, and is the bosonic counterpart of :doc:`MajoranaMapping <majorana_mapping>`.

Overview
--------

A bosonic mode has an unbounded occupation number, so it is first truncated to a local Fock-space dimension :math:`d`, then encoded on :math:`n_q = \log_2 d` qubits.
The encoding is an injective *codeword* map :math:`\mathrm{cw}: \{0, \ldots, d-1\} \to \{0,1\}^{n_q}`, and the associated isometry is

.. math::

   V = \sum_n |\mathrm{cw}(n)\rangle\langle n| .

:class:`~qdk_chemistry.data.BosonMapping` stores exactly that map, making the :doc:`BosonQubitMapper <../algorithms/boson_qubit_mapper>` algorithm encoding-agnostic: the mapper receives the encoding as data rather than selecting it internally.

Convention
~~~~~~~~~~

``num_modes``
   The number of bosonic modes in the system.

``mode_dimension(i)``
   The local Fock-space dimension :math:`d = n_\mathrm{max} + 1` of mode ``i``.
   The cutoff is owned by the :class:`~qdk_chemistry.data.BosonicModes` basis and is attributed per mode, so every accessor takes a mode index.

Mode ``i`` owns the contiguous qubit block starting at :math:`\sum_{j>i} n_q(j)`, so mode 0 occupies the most significant block and the encoded basis index of an occupation tuple :math:`(n_0, \ldots, n_{L-1})` is row-major in that tuple.
Pauli strings use the same little-endian qubit ordering as the rest of QDK/Chemistry's :doc:`PauliOperator <pauli_operator>` layer.

.. _boson-power-of-two-cutoff:

Power-of-two truncation
~~~~~~~~~~~~~~~~~~~~~~~

Only power-of-two :math:`d` is accepted.
The codeword map is then a bijection onto the whole register, so :math:`V` is unitary, the unphysical subspace is empty, and leakage is identically zero — no penalty term is needed.
Padding a requested cutoff up to a power of two is free in Pauli-term count (:math:`d = 3` and :math:`d = 4` both cost 32 hopping terms) and only lowers the truncation error.
Padding is never applied implicitly; opt in with :meth:`~qdk_chemistry.data.BosonicModes.with_padded_dimensions`.

Built-in encodings
------------------

Factory methods construct the named encodings, either with a uniform cutoff or from a :class:`~qdk_chemistry.data.BosonicModes` basis.
Both encodings produce the same number of Pauli terms and differ only in which computational basis state represents which occupation number.

.. _boson-encoding-standard-binary:

Standard binary
~~~~~~~~~~~~~~~

.. code-block:: python

   from qdk_chemistry.data import BosonMapping, BosonicModes

   mapping = BosonMapping.standard_binary(num_modes=2, mode_dimension=4)
   mapping.codeword_table(0)  # [0, 1, 2, 3]

   # Or read the cutoff from a basis, which may differ per mode:
   mapping = BosonMapping.standard_binary(BosonicModes(2, 4))

Occupation :math:`n` maps to the binary representation of :math:`n`.

.. _boson-encoding-gray-code:

Gray code
~~~~~~~~~

.. code-block:: python

   mapping = BosonMapping.gray_code(num_modes=2, mode_dimension=4)
   mapping.codeword_table(0)  # [0, 1, 3, 2]

Occupation :math:`n` maps to :math:`n \oplus (n \gg 1)`, so adjacent occupations differ in a single qubit.
The Pauli-term count of the hopping operator is identical to standard binary; only the circuit depth of the diagonal terms differs.

Custom encodings
----------------

The encoding set is open.
An encoding *is* a codeword table, so :meth:`~qdk_chemistry.data.BosonMapping.from_codeword_table` accepts any table directly and the named factories are conveniences on top of it — the same relationship :meth:`~qdk_chemistry.data.MajoranaMapping.from_table` has to the named fermionic transforms.

.. code-block:: python

   # Any per-mode permutation of range(d) is a valid encoding.
   mapping = BosonMapping.from_codeword_table([[2, 0, 3, 1]] * 2, name="my-encoding")
   mapping.name  # 'my-encoding'

A valid table is exactly a permutation of ``range(d)`` for each mode: injective, with a power-of-two length, and with every codeword fitting in that mode's qubits.
This is the same surjectivity condition the named encodings satisfy, and it is what makes the isometry unitary (see :ref:`boson-power-of-two-cutoff`).

No table recognition is attempted, so a table that happens to coincide with a named encoding is still labelled by whatever ``name`` says.
The table — not the label — is the identity of the mapping: it is what is hashed, what is written to disk, and what is restored on read.

Operator primitives
-------------------

Each accessor returns the exact Pauli expansion of one bosonic operator, with global qubit indices.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Operator
   * - :meth:`~qdk_chemistry.data.BosonMapping.number`
     - :math:`\hat n`
   * - :meth:`~qdk_chemistry.data.BosonMapping.number_squared`
     - :math:`\hat n^2`
   * - :meth:`~qdk_chemistry.data.BosonMapping.number_times_number_minus_one`
     - :math:`\hat n(\hat n - 1)`, the Bose-Hubbard on-site interaction
   * - :meth:`~qdk_chemistry.data.BosonMapping.annihilation`
     - :math:`b`
   * - :meth:`~qdk_chemistry.data.BosonMapping.creation`
     - :math:`b^\dagger`
   * - :meth:`~qdk_chemistry.data.BosonMapping.diagonal`
     - Any diagonal function :math:`f(\hat n)` of the occupation
   * - :meth:`~qdk_chemistry.data.BosonMapping.ladder_product`
     - An ordered product of ladder operators across modes

:meth:`~qdk_chemistry.data.BosonMapping.diagonal` covers :math:`\hat n`, :math:`\hat n^2`, :math:`\hat n(\hat n - 1)` and occupation penalties alike, via a fast Walsh-Hadamard transform over the mode's :math:`Z`-products.

Hard-core bosons
----------------

At :math:`d = 2` the mode is a two-level system: :math:`b` is exactly the spin lowering operator and :math:`\hat n(\hat n - 1)` vanishes identically, so any on-site interaction :math:`U` gives the same operator.
Use :meth:`~qdk_chemistry.data.BosonicModes.hard_core` to make that limit explicit.

See also
--------

- :doc:`BosonQubitMapper <../algorithms/boson_qubit_mapper>` — the algorithm that consumes this data class.
- :doc:`MajoranaMapping <majorana_mapping>` — the fermionic counterpart.
- :doc:`Orbitals <orbitals>` — including :class:`~qdk_chemistry.data.BosonicModes`, the basis that owns the cutoff.
