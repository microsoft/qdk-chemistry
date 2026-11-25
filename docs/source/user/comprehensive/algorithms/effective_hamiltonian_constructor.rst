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

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.EffectiveHamiltonianConstructor` provides a unified interface to downfolding methods.
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

DSRG-PT2
~~~~~~~~

.. rubric:: Factory name: ``"qdk_dsrg_pt2"`` (default)

Constructs the fixed-reference, second-order effective Hamiltonian of the driven similarity renormalization group (DSRG-PT2) :cite:`Evangelista2014,Li2015,Li2017` for a single-determinant reference state.
The result is an ordinary Hamiltonian over :math:`P`.

DSRG applies a continuous unitary transformation :math:`\bar H(s) = e^{-\hat A(s)} H_W e^{\hat A(s)}` with anti-Hermitian generator :math:`\hat A = \hat T - \hat T^\dagger` to suppress couplings between :math:`P` and :math:`Q` as a function of the flow parameter :math:`s`.
Through second order, the effective Hamiltonian :cite:`Li2017` is

.. math::

   H_{\mathrm{eff}}^{P} = \hat P\left(H_W + \tfrac{1}{2}\big[\tilde H, \hat A\big]\right)\hat P,

where :math:`\tilde H` is :math:`H_W` with renormalized couplings :cite:`Li2015,Li2017`.

The target space :math:`P` must be non-empty and must contain every singly occupied orbital of the reference.
Every other orbital of :math:`W` must be doubly occupied or empty in the reference determinant; those outside :math:`P` are folded as core or virtual orbitals, and reference core and virtual orbitals inside :math:`P` are embedded as doubly occupied and empty orbitals.
Below, the active orbitals are those of :math:`P`, the core orbitals are the doubly occupied orbitals of :math:`Q`, and the virtual orbitals are the rest of :math:`Q`.
The orbital energies :math:`\varepsilon_{p\sigma}` are the diagonal elements of the generalized Fock matrices

.. math::

   F^\sigma_{pq} = h^\sigma_{pq}
          + \sum_{\tau}\sum_{rs} \gamma^\tau_{rs}(pq|rs)^{\sigma\tau}
          - \sum_{rs} \gamma^\sigma_{rs}(pr|sq)^{\sigma\sigma}

in the semicanonical basis, which diagonalizes each :math:`F^\sigma` separately within the core, active, and virtual blocks.
Here :math:`\gamma^\sigma` is the reference 1-RDM of spin :math:`\sigma` over :math:`W`.

The first-order cluster operator :math:`\hat T` excites from core and active orbitals into active and virtual orbitals, excluding purely active excitations.
With the antisymmetrized integrals :math:`v^{ij}_{ab} = \langle ab \Vert ij \rangle` and the gaps :math:`\Delta^{ij}_{ab} = \varepsilon_i + \varepsilon_j - \varepsilon_a - \varepsilon_b` in the semicanonical basis, its doubles amplitudes and the renormalized couplings are

.. math::

   t^{ij}_{ab} = v^{ij}_{ab}\,\frac{1 - e^{-s(\Delta^{ij}_{ab})^2}}{\Delta^{ij}_{ab}},
   \qquad
   \tilde v^{ij}_{ab} = v^{ij}_{ab}\left(1 + e^{-s(\Delta^{ij}_{ab})^2}\right),

where :math:`s` is the flow parameter.
For the singles, :math:`t^i_a` applies :math:`(1 - e^{-s(\Delta^i_a)^2}) / \Delta^i_a` to :math:`f^i_a + b^i_a`, where :math:`b^i_a` is the active-space correction, while :math:`\tilde f^i_a = f^i_a(1 + e^{-s(\Delta^i_a)^2}) + b^i_a e^{-s(\Delta^i_a)^2}` :cite:`Li2015`.

The commutator in :math:`H_{\mathrm{eff}}^{P}` keeps its scalar, one- and two-body parts in reference normal order :cite:`Kutzelnigg1997`, and discards the residual three-body operator (including for two-electron active spaces).
The result is returned in ordinary normal order over the unrotated orbitals of :math:`P`.

``flow_parameter`` must be finite and nonnegative.
Larger :math:`s` regularizes less, and :math:`s = 0` disables the transformation, returning only the bare core-folded operator over :math:`P`.
At finite :math:`s`, an amplitude tends continuously to zero as its gap closes.
For a Hartree-Fock reference with every occupied orbital kept in :math:`P`, each MP2 pair energy carries the factor :math:`1 - e^{-2s\Delta^2}`, tending to bare MP2 as :math:`s \to \infty`.

This implementation currently supports real, normalized, self-dual single-determinant references: closed-shell determinants on restricted orbitals, and closed- or open-shell determinants on unrestricted orbitals.
Multi-determinant (correlated) references and open-shell determinants on restricted orbitals are rejected.
The input Hamiltonian must be Hermitian, and both spins must use the same spatial partitions.
Its inactive orbitals must be exactly the reference core outside :math:`W`, which its scalar and integrals already include.

The output orbitals preserve the reference orbital basis, and ``ModelOrbitals`` references give ``ModelOrbitals`` output.
The output keeps the spin restriction of the input.
The restricted and same-spin (``aaaa``, ``bbbb``) two-body blocks are 4-fold rather than 8-fold symmetric, :math:`(pq|rs) = (qp|sr) = (rs|pq)`: the commutator is not an electron-repulsion operator, so the bra swap :math:`(pq|rs) = (qp|rs)` does not hold.
For an unrestricted output, the mixed-spin ``aabb`` block satisfies 2-fold Hermiticity, :math:`(pq|rs)^{\alpha\beta} = (qp|sr)^{\alpha\beta}`.

The constructor logs the partition, the smallest gap, and the largest undamped ratio :math:`\lvert v/\Delta \rvert` over the coupled channels, and warns on large undamped amplitudes.

.. rubric:: Creating a downfolder

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

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``flow_parameter``
     - float
     - Flow parameter :math:`s`, in :math:`E_h^{-2}`; ``0`` returns the bare operator over :math:`P`. Default: 0.5

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

.. rubric:: Example

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-downfold
      :end-before: # end-cell-downfold

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/effective_hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-downfold
      :end-before: // end-cell-downfold

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
