Effective Hamiltonian construction
==================================

The :class:`~qdk_chemistry.algorithms.EffectiveHamiltonianConstructor` algorithm in QDK/Chemistry defines a common interface for downfolding a Hamiltonian from a larger orbital window into a smaller target space.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a reference :class:`~qdk_chemistry.data.Wavefunction`, an input :class:`~qdk_chemistry.data.Hamiltonian`, and target-space orbital indices as input and produces an effective :class:`~qdk_chemistry.data.Hamiltonian` as output.
The interface leaves the downfolding approximation and method-specific diagnostics to the concrete implementation.

Overview
--------

Let the orbital window :math:`W` be partitioned into a target space :math:`P` and an external space :math:`Q`:

.. math::

   W = P \cup Q,
   \qquad
   P \cap Q = \varnothing.

Using :math:`\hat P` and :math:`\hat Q` for the corresponding projectors, the full-window Hamiltonian has the block structure

.. math::

   H_W =
   \begin{pmatrix}
      H_{PP} & H_{PQ} \\
      H_{QP} & H_{QQ}
   \end{pmatrix},
   \qquad
   H_{PP} = \hat P H_W \hat P.

An implementation accounts for the influence of :math:`Q` while constructing an effective Hamiltonian that acts only in :math:`P`:

.. math::

   \mathcal{D}\!\left(\lvert \Psi_{\mathrm{ref}} \rangle, H_W, P\right)
   = H_{\mathrm{eff}}^{P},
   \qquad
   H_{\mathrm{eff}}^{P} = \hat P H_{\mathrm{eff}}^{P} \hat P,

where :math:`\mathcal{D}` denotes the selected downfolding implementation.
The interface does not prescribe how the coupling blocks :math:`H_{PQ}` and :math:`H_{QP}` or the external block :math:`H_{QQ}` contribute to :math:`H_{\mathrm{eff}}^{P}`.

Using the EffectiveHamiltonianConstructor
-----------------------------------------

The ``run`` method takes a reference wavefunction, a full-window Hamiltonian, and the target-space indices and returns the effective Hamiltonian acting in :math:`P`.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.EffectiveHamiltonianConstructor` requires the following inputs:

Reference wavefunction
   A :class:`~qdk_chemistry.data.Wavefunction` that defines the reference state used by the downfolding method.

Input Hamiltonian
   A :class:`~qdk_chemistry.data.Hamiltonian` expressed over the complete orbital window :math:`W`.

Target-space indices
   A :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet` containing the indices of :math:`P`.
   These are absolute molecular-orbital indices, drawn from the same index universe as ``Orbitals.active_indices()``, and must lie within the active space of :math:`W`.

Output contract
~~~~~~~~~~~~~~~

The returned :class:`~qdk_chemistry.data.Hamiltonian` is expressed over :math:`P` and satisfies:

- its orbitals have ``active_indices()`` equal to the requested target-space indices;
- its orbitals classify fully occupied orbitals of :math:`Q = W \setminus P` as inactive and unoccupied orbitals of :math:`Q` as virtual, while preserving the input Hamiltonian's inactive orbitals;
- its inactive Fock matrix, when present, is consistent with the output inactive orbitals and may therefore differ from the input Hamiltonian's inactive Fock matrix;
- the scalar shift from folding in :math:`Q` is added to the constant (zero-body) energy term, and the remaining :math:`Q` contribution is folded into the integrals.

The occupied/virtual partition of :math:`Q` is method dependent and is determined by the concrete implementation from its reference state.
Classifying an orbital of :math:`Q` as virtual describes its occupation in the output orbital metadata; it does not place that orbital in the effective active space.
Consumers of an effective Hamiltonian should not attempt to re-correlate :math:`Q`, since its contribution is already folded in.

Input validation is opt-in.
The ``run`` method does not validate its arguments; each concrete implementation decides whether to check the space contract :math:`P \subseteq W_H` and :math:`W_{\mathrm{ref}} \subseteq W_H` before computing.

The base interface defines no common settings.
Concrete implementations can expose method-specific configuration through the ``settings()`` object.
See :doc:`Settings <settings>` for a general treatment of algorithm settings in QDK/Chemistry.

.. rubric:: Creating a constructor

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/effective_hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/effective_hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. rubric:: Running the calculation

The example below prepares the inputs of the `QDK DUCC`_ implementation.
It obtains :math:`T_1` and :math:`T_2` amplitudes over the full window from a :class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator`, selects :math:`P` with an :doc:`ActiveSpaceSelector <active_space>`, and diagonalizes the effective Hamiltonian in :math:`P` with a :doc:`MultiConfigurationCalculator <mc_calculator>`.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/effective_hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

Available implementations
-------------------------

You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/effective_hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

QDK DUCC
~~~~~~~~

.. rubric:: Factory name: ``"qdk_ducc"`` (default)

The :class:`~qdk_chemistry.algorithms.QdkDuccConstructor` implements double unitary coupled-cluster (DUCC) downfolding :cite:`Bauman2019` using real singles and doubles amplitudes, :math:`T_1` and :math:`T_2`, without triples amplitudes.
Its input :class:`~qdk_chemistry.data.Wavefunction` must carry amplitudes over the same orbital window :math:`W` as the input :class:`~qdk_chemistry.data.Hamiltonian`, together with a single-determinant reference.
Store both singles and doubles; supply zero singles when the amplitude source contains only doubles.
Restricted closed-shell and unrestricted open-shell references are supported, including windows with frozen-core orbitals outside :math:`W`.
The alpha and beta channels must have equal orbital counts in both :math:`W` and the nonempty target :math:`P`, but their occupations and unrestricted orbital selections may differ.
Restricted amplitudes must obey the closed-shell spin relations; spatial doubles and their explicit same-spin antisymmetrizations are both accepted.
Unrestricted same-spin doubles may be stored as the full antisymmetric tensor or only for :math:`i<j` and :math:`a<b`.
The output preserves the input restrictedness.
Restricted two-electron integrals and unrestricted same-spin blocks satisfy :math:`(pq|rs)=(rs|pq)=(qp|sr)`, while the alpha-beta block satisfies only :math:`(pq|rs)=(qp|sr)`.
Unrestricted same-spin blocks store the antisymmetrized integrals :math:`\tfrac{1}{2}\left[(pq|rs)-(ps|rq)\right]`, which represent the same operator as plain integrals.

External amplitudes are those whose orbital indices are not entirely contained in :math:`P`.
The anti-Hermitian generator is

.. math::

   \sigma = T_{\mathrm{ext},1} + T_{\mathrm{ext},2}
            - \left(T_{\mathrm{ext},1} + T_{\mathrm{ext},2}\right)^\dagger.

Define :math:`C_0(A)=A` and :math:`C_{n+1}(A)=[C_n(A),\sigma]`.
For level :math:`1\leq L\leq 2`, the transformed operator is

.. math::

   \overline{H}^{(L)} = \sum_{n=0}^{L}\frac{C_n(H)}{n!}
                        + \frac{C_{L+1}(F)}{(L+1)!},

where :math:`F` is the reference Fock operator :cite:`Bauman2022`.
Intermediate commutators are not truncated by particle rank; only the final reference-normal-ordered scalar, one-body, and two-body terms are retained and downfolded into :math:`P`.
Level 0 performs bare downfolding without commutator corrections.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Setting
     - Type
     - Description
   * - ``level``
     - int
     - Commutator level, from 0 to 2 inclusive. Default: 2.

Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`: Provides the reference state
- :class:`~qdk_chemistry.data.Hamiltonian`: Represents the full-window input and target-space output Hamiltonians
- :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet`: Identifies the target orbital space while preserving symmetry blocks
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Builds the input Hamiltonian over the orbital window

Further reading
---------------

- :doc:`ActiveSpaceSelector <active_space>`: Identifies orbital subspaces for correlated calculations
- :doc:`Settings <settings>`: Configures algorithm implementations
- :doc:`Factory Pattern <factory_pattern>`: Discovers, registers, and creates algorithm implementations
