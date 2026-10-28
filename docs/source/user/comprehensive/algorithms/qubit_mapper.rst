Qubit mapping
=============

The :class:`~qdk_chemistry.algorithms.QubitMapper` algorithm in QDK/Chemistry transforms electronic-structure Hamiltonians into qubit operators suitable for quantum computation.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Hamiltonian <../data/hamiltonian>` instance as input and produces a :class:`~qdk_chemistry.data.QubitOperator` instance as output.
This transformation is essential for executing quantum chemistry algorithms on quantum hardware.

Overview
--------

The :class:`~qdk_chemistry.algorithms.QubitMapper` algorithm converts fermionic Hamiltonians into qubit-operator representations, including explicit Pauli decompositions and structured sums of squares.
This transformation preserves the operator algebra, particle-number constraints, and antisymmetry required by fermionic statistics.
The resulting qubit operator is mathematically equivalent to the original fermionic Hamiltonian but is now in a form that can be executed on quantum hardware or simulated by quantum algorithms.

.. note::

   **Core energy handling:** Pauli-decomposition mappers exclude the input
   Hamiltonian's core energy (nuclear repulsion and frozen-orbital contributions).
   Add ``hamiltonian.get_core_energy()`` to their electronic energies.
   The :ref:`sum-of-squares mapper <sum-of-squares-qubit-mapper>` instead includes
   core energy in its scalar shift; its SOSSA-decoded energies are already total
   energies and must not receive that constant a second time.

Supported encodings
~~~~~~~~~~~~~~~~~~~

Different encoding strategies produce mathematically equivalent qubit operators but with different Pauli-string structures.
The choice of encoding can affect circuit depth and measurement requirements on quantum hardware.
Not every implementation supports all encodings — see `Available implementations`_ for details.

.. _encoding-jordan-wigner:

Jordan-Wigner :cite:`Jordan-Wigner1928`
   Encodes each fermionic mode in a single qubit whose state directly represents the orbital occupation.
   Fermionic antisymmetry is enforced through a Z-string on all lower-indexed qubits.

.. _encoding-bravyi-kitaev:

Bravyi-Kitaev :cite:`Seeley2012`
   Distributes both occupation and parity information across qubits using a binary-tree (Fenwick tree) structure, reducing the average Pauli-string weight to O(log n).

.. _encoding-parity:

Parity :cite:`Seeley2012`
   Encodes qubits with cumulative electron-number parities of the orbitals.

.. _encoding-scbk:

Symmetry-conserving Bravyi-Kitaev :cite:`Bravyi2017tapering`
   Exploits particle-number and spin-parity symmetries to reduce the qubit count by 2. Use :meth:`~qdk_chemistry.data.MajoranaMapping.symmetry_conserving_bravyi_kitaev` with a :class:`~qdk_chemistry.data.Symmetries` object.

.. _encoding-bk-tree:

Bravyi-Kitaev tree :cite:`Havlicek2017`
   A tree-based variant of the Bravyi-Kitaev transformation that uses a different qubit indexing strategy.

.. _encoding-verstraete-cirac:

Verstraete-Cirac :cite:`Verstraete2005`
   An auxiliary qubit encoding that eliminates non-local Z-strings by locally tracking parity. It operates on any connected :class:`~qdk_chemistry.data.LatticeGraph`; an optional Hamiltonian-path reordering (``dfs_ordering=True`` on the lattice factories) minimizes the number of auxiliary qubits and stabilizers. Use :meth:`~qdk_chemistry.data.MajoranaMapping.verstraete_cirac` with a :class:`~qdk_chemistry.data.LatticeGraph`.

Using the QubitMapper
---------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run a qubit mapping.
The ``run`` method requires a :class:`~qdk_chemistry.data.MajoranaMapping` as its second argument, which specifies the fermion-to-qubit encoding to use.
It returns a :class:`~qdk_chemistry.data.QubitOperator` containing the representation produced by the selected implementation.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.QubitMapper` requires the following inputs:

Hamiltonian
   A :doc:`Hamiltonian <../data/hamiltonian>` instance containing the fermionic one- and
   two-electron integrals. This is typically constructed using the
   :doc:`HamiltonianConstructor <hamiltonian_constructor>` algorithm.

   The Hamiltonian defines the fermionic operators that will be transformed into
   qubit (Pauli) operators using the selected encoding strategy.

MajoranaMapping
   A :doc:`MajoranaMapping <../data/majorana_mapping>` instance specifying the
   fermion-to-qubit encoding. Built-in factory methods are available for standard
   encodings (e.g., ``MajoranaMapping.jordan_wigner(num_modes=n)``), or a custom
   encoding can be constructed from a Pauli-string table.

.. note::

   Different encoding strategies produce mathematically equivalent qubit operators
   but with different Pauli-string structures. The choice of encoding can affect circuit
   depth and measurement requirements on quantum hardware.
   See `Supported encodings`_ above for descriptions.

.. rubric:: Creating a mapper

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

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

.. _extending-implementations:

Details for extending implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pauli-decomposition implementations fall into two groups. They use the
:class:`~qdk_chemistry.data.MajoranaMapping` argument differently:

**Table-driven backends** (QDK native)
   Read the Pauli-string table from the ``MajoranaMapping`` directly and
   pass it to the C++ mapping engine.  Any valid table works, including
   custom encodings that have no standard name.

**Third-party backends** (OpenFermion, Qiskit)
   **Ignore the Pauli table.**  They read
   :attr:`~qdk_chemistry.data.MajoranaMapping.base_encoding` — a string
   like ``"jordan-wigner"`` or ``"bravyi-kitaev-tree"`` — and pass it to
   their own library to select the matching transform.  The qubit
   operator is then built from scratch using the third-party library's
   own fermion-to-qubit code.

This distinction has practical consequences:

- **Custom mappings** (user-defined Pauli tables) work with the QDK
  backend but **cannot** be used with third-party backends, which have
  no way to interpret an arbitrary table.

- **Consistency is assumed, not verified.**  Factory-produced mappings
  (e.g. ``MajoranaMapping.jordan_wigner()``) guarantee that the Pauli
  table and the ``base_encoding`` name describe the same encoding.
  Cross-backend eigenvalue tests in the test suite verify this for every
  supported factory × backend combination.  However, if a
  ``MajoranaMapping`` is manually built with a table that does not
  match its name, a third-party backend will silently use the wrong
  transform.

- **Tapering** is each backend's responsibility.  The base class provides
  a ``_taper_result()`` helper that applies tapering and qubit relabeling
  to an *already mapped* ``QubitOperator``.  Backends must first run
  the base transform (typically using ``mapping.without_tapering()``) and
  then call ``_taper_result()`` on the output. The shipped Pauli-decomposition backends use
  this helper, but third-party backends are free to handle tapering
  however they choose.

.. _qdk-qubit-mapper:

QDK
~~~

.. rubric:: Factory name: ``"qdk"``

Native QDK/Chemistry qubit mapping implementation built on the :doc:`PauliOperator <../data/pauli_operator>` expression layer.
This is a **table-driven** backend: it reads the Pauli-string table from the :class:`~qdk_chemistry.data.MajoranaMapping` and passes it directly to the C++ mapping engine.
Any valid ``MajoranaMapping`` works — factory-produced or custom user-defined tables.
The mapping's ``name`` and ``base_encoding`` are used only for metadata on the output, not to select a transform.

Supported encodings: :ref:`Jordan-Wigner <encoding-jordan-wigner>`, :ref:`Bravyi-Kitaev <encoding-bravyi-kitaev>`, :ref:`Bravyi-Kitaev tree <encoding-bk-tree>`, :ref:`Parity <encoding-parity>`, :ref:`SCBK <encoding-scbk>`, :ref:`Verstraete-Cirac <encoding-verstraete-cirac>`, and any custom encoding

The native mapper uses blocked spin-orbital ordering internally (alpha orbitals first, then beta orbitals).
Use ``QubitOperator.to_interleaved()`` for alternative qubit orderings if needed.

Both restricted (RHF) and unrestricted (UHF) Hamiltonians are supported.

Custom encodings can be defined by constructing a :class:`~qdk_chemistry.data.MajoranaMapping` from a Pauli-string table.

.. rubric:: Integral symmetry

Restricted integrals must have the 4-fold symmetry
:math:`(pq|rs) = (qp|sr) = (rs|pq)`, which ordinary 8-fold integrals and those
produced by downfolding both satisfy. They use spin-summed symmetric and
antisymmetric excitation pairs, weighted by
:math:`g^{S,A}_{pq,rs} = \tfrac12\left[(pq|rs) \pm (pq|sr)\right]`.
A restricted tensor that defines a non-Hermitian operator raises, as does a
Hermitian one not stored with :math:`(pq|rs) = (rs|pq)`; since that exchange
leaves the operator unchanged, map its bra-ket average instead.

Unrestricted integrals use the same excitation-pair split when the ``aa`` and
``bb`` blocks are 4-fold and ``ab`` satisfies
:math:`(pq|rs)_{ab} = (qp|sr)_{ab}`; ``ab`` need not satisfy
:math:`(pq|rs)_{ab} = (rs|pq)_{ab}`. Other unrestricted integrals are mapped
channel by channel.

The ``integral_threshold`` setting screens :math:`g^S` and :math:`g^A` in the
excitation-pair split, so symmetry partners are always screened together, and
individual integrals in the channel-by-channel loop. For 8-fold integrals,
:math:`g^A` vanishes up to round-off, so antisymmetric products are formed
only where that round-off exceeds the threshold. In ill-conditioned bases, such
as augmented ones, round-off can exceed the default threshold of ``1e-12`` and
add Pauli terms of that size (2–6% more terms for an H₄ chain in
aug-cc-pVDZ); an ``integral_threshold`` of about ``1e-10`` removes most of them,
at the cost of also screening genuine integrals below it.

.. rubric:: Container-aware fast paths

The ``"qdk"`` backend consumes two-body integrals directly from the underlying
:doc:`HamiltonianContainer <../data/hamiltonian>` without reconstructing a
dense :math:`N^4` two-body tensor when the container stores its integrals in a
compressed form:

- :class:`~qdk_chemistry.data.SparseHamiltonianContainer` — the mapping loop
  iterates over the stored ``(p, q, r, s)`` integrals rather than the full
  :math:`O(N^4)` index space, skipping the zeros that dominate lattice/model
  Hamiltonians (e.g. those produced by
  :func:`~qdk_chemistry.utils.model_hamiltonians.create_hubbard_hamiltonian`
  and :func:`~qdk_chemistry.utils.model_hamiltonians.create_ppp_hamiltonian`).
  Stored entries are read **literally**: positions that are not stored are
  zero, exactly as in the dense view of the container, so a complete 4-fold
  suborbit needs no explicit zeros for the rest of its 8-fold orbit. To build
  entries from symmetry representatives, use
  :meth:`~qdk_chemistry.data.SparseHamiltonianContainer.expand_symmetry_representatives`
  with :class:`~qdk_chemistry.data.TwoBodyPermutationGroup`. Entry order does
  not affect the result.
- :class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` — the three-center
  (Cholesky / density-fitted) factors are kept in their
  :math:`O(N^2 \cdot n_\text{aux})` form and the auxiliary index is contracted
  in integral space, one ``(pq|.)`` row at a time (a vectorized matrix-vector
  product per orbital pair), so the mapping needs one additional
  :math:`N^2`-length row and suits systems whose dense ERI tensor does not fit
  in memory. The reconstructed integrals are checked element by element for
  :math:`(pq|rs) = (qp|sr)`, with the same tolerance as dense integrals. For
  factors that are symmetric in their orbital pair up to round-off, as
  Cholesky factors of ordinary integrals are, one pass over the factors with
  :math:`O(N^2)` memory settles the check. Otherwise the parts of the factors
  symmetric and antisymmetric in their orbital pair are copied, which doubles
  the factor memory, and a Frobenius bound in
  :math:`O(N^2 n_\text{aux}^2)` or, failing that, the exact largest violation
  in :math:`O(N^4 n_\text{aux})` decides it.

In all cases the result is a :class:`~qdk_chemistry.data.QubitOperator` that
is numerically equivalent, term by term, to the dense
:class:`~qdk_chemistry.data.CanonicalFourCenterHamiltonianContainer` path for
the same integrals. The behaviour of
``run()`` and the shape of the returned operator are unchanged, and the
selection is fully automatic based on the container type. The
:class:`~qdk_chemistry.data.CanonicalFourCenterHamiltonianContainer` continues
to use the dense path.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``threshold``
     - double
     - Threshold for pruning small Pauli coefficients. Default: ``1e-12``
   * - ``integral_threshold``
     - double
     - Threshold for filtering small integrals before transformation; 4-fold two-body integrals are screened in the symmetric and antisymmetric combinations above. Default: ``1e-12``

.. rubric:: Example

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/qubit_mapper.py
      :language: python
      :start-after: # start-cell-qdk-mapper
      :end-before: # end-cell-qdk-mapper

.. _qiskit-qubit-mapper:

Qiskit
~~~~~~

.. rubric:: Factory name: ``"qiskit"``

Qubit mapping implementation integrated through the Qiskit plugin.
This is a **third-party** backend: it reads ``mapping.base_encoding`` to select a Qiskit Nature mapper class and **ignores the Pauli table** (see :ref:`extending-implementations`).

Supported base encodings: :ref:`Jordan-Wigner <encoding-jordan-wigner>`, :ref:`Bravyi-Kitaev <encoding-bravyi-kitaev>`, :ref:`Parity <encoding-parity>`

Both restricted (RHF) and unrestricted (UHF) Hamiltonians are supported.

The plugin permutes the two-body tensor axes from QDK's chemist ordering to
Qiskit Nature's operator ordering, so 4-fold integrals are mapped exactly. An
unrestricted ``ab`` block that violates :math:`(pq|rs) = (qp|sr)` defines a
non-Hermitian operator, which Qiskit Nature cannot represent, so ``run()``
raises ``ValueError``.

.. rubric:: Settings

This implementation has no configurable settings.

.. _openfermion-qubit-mapper:

OpenFermion
~~~~~~~~~~~

.. rubric:: Factory name: ``"openfermion"``

Qubit mapping implementation integrated through the OpenFermion plugin.
This is a **third-party** backend: it reads ``mapping.base_encoding`` to select an OpenFermion transform function and **ignores the Pauli table** (see :ref:`extending-implementations`).

Supported base encodings: :ref:`Jordan-Wigner <encoding-jordan-wigner>`, :ref:`Bravyi-Kitaev <encoding-bravyi-kitaev>`, :ref:`Bravyi-Kitaev tree <encoding-bk-tree>`

Both restricted (RHF) and unrestricted (UHF) Hamiltonians are supported.

The plugin permutes the two-body tensor axes from QDK's chemist ordering to
OpenFermion's operator ordering, so 4-fold integrals are mapped exactly.

.. rubric:: Settings

This implementation has no configurable settings. The encoding strategy is
determined entirely by the :class:`~qdk_chemistry.data.MajoranaMapping` provided
to ``run()``.


.. _sum-of-squares-qubit-mapper:

Sum of squares
~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"sum_of_squares"``

The sum-of-squares mapper :cite:`Low2025` consumes a
:class:`~qdk_chemistry.data.DFTHCHamiltonianContainer`, including ordinary
DF data produced by :doc:`HamiltonianFactorization <hamiltonian_factorization>`.
It returns a :class:`~qdk_chemistry.data.QubitOperator` containing rotated
particle, hole, and spin-free generators:

.. math::

   H_{\mathrm{total}} = E_{\mathrm{shift}} I + \sum_j O_j^\dagger O_j.

The scalar shift includes core energy.
The ``threshold`` setting screens adjusted one-body eigenvalues below its
non-negative magnitude cutoff (default ``1e-12``).

The supported circuit workflow uses
:meth:`~qdk_chemistry.data.MajoranaMapping.jordan_wigner` over twice the
number of active spatial orbitals, with blocked spin ordering and no tapering.
Custom tables and reordered
modes are not supported by the SOSSA circuit path.

The output is numerical generator data, not an executable circuit.
Use the :ref:`SOSSA unitary builder <sossa-builder>` and its circuit mapper
to construct the block encoding.
Pauli-specific consumers reject this representation explicitly.
Dense and sparse matrix conversion of the SOS representation is not
implemented; conventional Hamiltonian access remains available on the
input DFTHC container.


Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for mapping
- :doc:`MajoranaMapping <../data/majorana_mapping>`: Fermion-to-qubit encoding passed to ``run()``
- :class:`~qdk_chemistry.data.QubitOperator`: Output qubit operator representation
- :doc:`Symmetries <../data/symmetries>`: Physical symmetries (e.g., conserved quantum numbers) for :meth:`~qdk_chemistry.data.MajoranaMapping.symmetry_conserving_bravyi_kitaev`

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/qubit_mapper.py>`_ script.
- :doc:`StatePreparation <state_preparation>`: Prepare quantum circuits from wavefunctions
- :doc:`ExpectationEstimator <expectation_estimator>`: Estimate energies using the qubit operator
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
