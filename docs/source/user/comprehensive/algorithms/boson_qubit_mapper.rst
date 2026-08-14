Boson-to-qubit mapping
======================

The :class:`~qdk_chemistry.algorithms.BosonQubitMapper` algorithm transforms bosonic Hamiltonians into qubit operators suitable for quantum computation.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Hamiltonian <../data/hamiltonian>` and a :doc:`BosonMapping <../data/boson_mapping>` as input and produces a :class:`~qdk_chemistry.data.QubitOperator` as output.
It is the bosonic counterpart of the :doc:`QubitMapper <qubit_mapper>`.

Overview
--------

A bosonic mode has an unbounded occupation number, so it is first truncated to a local Fock-space dimension ``d`` and encoded on ``log2(d)`` qubits.
The truncation and the encoding both live on the :class:`~qdk_chemistry.data.BosonMapping` passed to ``run()``, which keeps the algorithm encoding-agnostic.

The Hamiltonian is read in the same chemist notation used for every other Hamiltonian in the library,

.. math::

   H = \sum_{pq} h_{pq}\, b_p^\dagger b_q
     + \tfrac{1}{2} \sum_{pqrs} (pq|rs)\, b_p^\dagger b_r^\dagger b_s b_q ,

so no bosonic-specific container is required.
With :math:`h_{ii} = -\mu`, :math:`h_{ij} = -t` on bonds and :math:`(ii|ii) = U` this reproduces the Bose-Hubbard model exactly, the two-body contraction collapsing to :math:`\tfrac{U}{2}\sum_i n_i(n_i-1)`.

.. note::

   **Core energy handling:** The constant energy shift of the input Hamiltonian is **not**
   included in the output QubitOperator, matching :doc:`QubitMapper <qubit_mapper>`. To
   compute total energies, add ``hamiltonian.get_core_energy()`` to expectation values
   computed from the QubitOperator.

Supported encodings
~~~~~~~~~~~~~~~~~~~

Because every cutoff is a power of two, the encoding is surjective: the encoded subspace is the whole qubit Hilbert space, so there is no leakage and no penalty term is needed.
Encodings differ only in which computational basis state represents which occupation number, and produce the same number of Pauli terms.

:ref:`Standard binary <boson-encoding-standard-binary>`
   Occupation :math:`n` maps to the binary representation of :math:`n`.
   Use :meth:`~qdk_chemistry.data.BosonMapping.standard_binary`.

:ref:`Gray code <boson-encoding-gray-code>`
   Occupation :math:`n` maps to :math:`n \oplus (n \gg 1)`, so adjacent occupations differ in a single qubit.
   Use :meth:`~qdk_chemistry.data.BosonMapping.gray_code`.

Custom
   Any per-mode permutation of ``range(d)``, supplied to :meth:`~qdk_chemistry.data.BosonMapping.from_codeword_table`.

Using the BosonQubitMapper
--------------------------

.. note::
   This algorithm is currently available only in the Python API.

The ``run`` method requires a :class:`~qdk_chemistry.data.BosonMapping` as its second argument, which supplies both the encoding and the occupation cutoff.

Input requirements
~~~~~~~~~~~~~~~~~~

Hamiltonian
   A :doc:`Hamiltonian <../data/hamiltonian>` instance holding the bosonic one- and two-body
   integrals in chemist notation. Model Hamiltonians can be built with
   :func:`~qdk_chemistry.utils.model_hamiltonians.create_bose_hubbard_hamiltonian`, which
   attaches a :class:`~qdk_chemistry.data.BosonicModes` basis carrying the cutoff.

BosonMapping
   A :doc:`BosonMapping <../data/boson_mapping>` instance specifying the boson-to-qubit
   encoding and the per-mode cutoff. Its mode count and cutoff must agree with the
   Hamiltonian; a mismatch is a hard error rather than silently-wrong physics.

.. rubric:: Running the calculation

.. code-block:: python

   from qdk_chemistry.algorithms import create
   from qdk_chemistry.data import BosonMapping, LatticeGraph
   from qdk_chemistry.utils.model_hamiltonians import create_bose_hubbard_hamiltonian

   lattice = LatticeGraph.chain(2)
   hamiltonian = create_bose_hubbard_hamiltonian(lattice, t=1.0, U=4.0, mu=0.0, mode_dimension=4)

   mapper = create("boson_qubit_mapper")
   mapping = BosonMapping.standard_binary(num_modes=2, mode_dimension=4)
   qubit_hamiltonian = mapper.run(hamiltonian, mapping)

The mapping can also be built directly from the Hamiltonian's basis, which is the recommended
route when the cutoff varies by mode:

.. code-block:: python

   mapping = BosonMapping.standard_binary(hamiltonian.get_orbitals())

Available implementations
-------------------------

.. _qdk-boson-qubit-mapper:

QDK
~~~

.. rubric:: Factory name: ``"qdk"``

Native QDK/Chemistry boson mapping implementation built on the C++ mapping engine.
It reads the codeword table from the :class:`~qdk_chemistry.data.BosonMapping` and passes it directly to the engine, so any valid mapping works — named or custom.
The mapping's ``name`` is recorded as the ``encoding`` of the output operator and is not used to select a transform.

Settings
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Setting
     - Default
     - Description
   * - ``threshold``
     - ``1e-12``
     - Pauli terms with a coefficient below this are dropped from the result.
   * - ``integral_threshold``
     - ``1e-12``
     - Integrals below this are skipped, which improves performance.

See also
--------

- :doc:`BosonMapping <../data/boson_mapping>` — the encoding data class.
- :doc:`QubitMapper <qubit_mapper>` — the fermionic counterpart.
