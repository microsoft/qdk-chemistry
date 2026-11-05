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

QDK SW-PT2
~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_swpt2"``

This is the default implementation returned by ``create("effective_hamiltonian_constructor")``.

Second-order Schrieffer-Wolff (Van Vleck) downfolding :cite:`Schrieffer1966,Bravyi2011` with
diagonal generalized-Fock energy denominators.
Splitting :math:`H` into a part
:math:`H_{\mathrm{BD}}` that is block diagonal in the occupation of :math:`Q` and an
occupation-changing remainder :math:`H_{\mathrm{OD}} = \sum_T c_T\,\hat{T}`, the effective
Hamiltonian is :math:`\hat P e^{S} H e^{-S} \hat P` through second order,

.. math::

   H_{\mathrm{eff}} = \hat P\left(H_{\mathrm{BD}} + \tfrac{1}{2}[S, V]\right)\hat P,
   \qquad
   S = \sum_{T} \frac{g_T\,c_T}{\Delta_T}\,\hat{T},
   \qquad
   V = \sum_{T} (2 - g_T)\,c_T\,\hat{T},

with the denominators

.. math::

   \Delta_T = \sum_{\mathrm{creations}} \varepsilon_{p\sigma}
            - \sum_{\mathrm{annihilations}} \varepsilon_{p\sigma}

and a damping factor :math:`g_T` described below. Without damping, :math:`g_T = 1`, so
:math:`V = H_{\mathrm{OD}}` and :math:`S` solves :math:`[F_0, S] = H_{\mathrm{OD}}`, where
:math:`F_0`, the diagonal part of the reference's generalized Fock operator, approximates
:math:`H_{\mathrm{BD}}`. The orbital energies
:math:`\varepsilon_{p\sigma}` of spin :math:`\sigma` are the diagonal elements of the
generalized Fock matrices:

.. math::

   \varepsilon_{p\sigma} = F^\sigma_{pp},\qquad F^\sigma_{pq} = h^\sigma_{pq}
          + \sum_{\tau}\sum_{rs} \gamma^\tau_{rs}(pq|rs)^{\sigma\tau}
          - \sum_{rs} \gamma^\sigma_{rs}(pr|sq)^{\sigma\sigma}.

:math:`\gamma^\sigma` is the reference's 1-RDM of spin :math:`\sigma` over :math:`W`.
For restricted orbitals, the spin-traced 1-RDM :math:`\gamma` is split evenly,
:math:`\gamma^\alpha = \gamma^\beta = \gamma/2`. This spin-averaged construction
preserves spin symmetry and uses common orbital energies,
:math:`\varepsilon_{p\alpha} = \varepsilon_{p\beta}`. Unrestricted orbitals use the
reference's per-spin 1-RDMs, so the orbital energies can differ between spins.

By default, ``semicanonicalize`` diagonalizes each :math:`F^\sigma` within the inactive,
active, and virtual blocks. Integrals and density are rotated consistently,
and the emitted integrals are rotated back. Blocks that are already diagonal within a
tolerance are left unrotated; without the regularizer that tolerance is so tight that even a
converged SCF reference is rotated. With this option disabled, the denominators use the diagonals of
:math:`F^\sigma` in the supplied basis.

Near-degenerate channels have a small :math:`\Delta_T` and a large amplitude, which can make the
perturbative expansion unreliable. ``regularizer_flow`` sets the damping

.. math::

   g_T = 1 - e^{-s \Delta_T^2},

where :math:`s` is the flow parameter of the driven similarity renormalization group
(DSRG) :cite:`Evangelista2014`, in :math:`E_h^{-2}`. Each channel then contributes
:math:`|c_T|^2 (1 - e^{-2 s \Delta_T^2})/\Delta_T` to the second-order energy instead of
:math:`|c_T|^2/\Delta_T`. Larger :math:`s` regularizes less. The damping is not confined to
intruders: at the default :math:`s = 0.5\,E_h^{-2}`, channels with denominators of 0.3, 0.5, 1.0
and 2.0 :math:`E_h` keep approximately 9%, 22%, 63% and 98% of their energy.
Setting ``regularizer_flow`` to ``0`` turns the damping off (:math:`g_T = 1`), unlike the
:math:`s \to 0` limit, which damps every channel away. Channels with
:math:`|\Delta_T| < 10^{-8}\,E_h` always get :math:`g_T = 0`; without the regularizer a warning
is logged.

For a closed-shell restricted or an unrestricted Hartree-Fock reference, :math:`H_{\mathrm{eff}}`
is the second-order Hamiltonian of DSRG-PT2 :cite:`Li2015,Li2017`, up to the three-body
treatment below.

The kept space :math:`P` comes from ``p_indices`` alone; the reference supplies only
:math:`\gamma^\sigma`. Any restricted or unrestricted orbital set qualifies, canonical or not,
such as RHF, ROHF, CAS, localized or natural orbitals. It must share the input Hamiltonian's
molecular-orbital basis and keep every singly occupied orbital in its active space, and the
input Hamiltonian's inactive orbitals must be exactly its core orbitals outside :math:`W`. For
unrestricted orbitals, the input Hamiltonian's active space, ``p_indices`` and the reference
active space must select the same orbitals in both spins. Every orbital in
:math:`Q = W \setminus P` is folded as doubly occupied or empty by rounding its spin-traced
reference occupation to 2 or 0.
The rounding perturbs the mean field the active space feels, so
``max_folded_occupation_deviation`` bounds it, and a singly occupied orbital must be kept in
:math:`P`. The active space keeps the remaining electrons; the downfold is rejected if either
spin is left with fewer than zero or more electrons than active orbitals.

The commutator generates three-body terms that the emitted two-body operator cannot carry.
Reference-1-RDM pair contractions reduce these to lower-rank terms :cite:`Kutzelnigg1997`.
This Gaussian approximation recovers ordinary Wick contractions for a determinant in
its own orbitals, closed-shell restricted or unrestricted. It neglects connected higher
density cumulants for correlated references and spin-averaged open-shell references.
The residual three-body operator is discarded.
When the retained space contains at most two electrons, the three-body terms have no matrix
elements there and are dropped without folding.

.. warning::

   The emitted two-body blocks are only **4-fold** symmetric. Hermiticity
   :math:`(pq|rs) = (qp|sr)` and electron exchange :math:`(pq|rs) = (rs|pq)` hold as for the
   input integrals, but the bra swap :math:`(pq|rs) = (qp|rs)` of a genuine Coulomb integral
   does not: the commutator is not an electron-repulsion operator. Consumers must be given
   the full dense :math:`n^4` blocks.

The constructor logs the partition and, over the terms coupling :math:`P` and :math:`Q`, the
smallest denominator and largest unregularized amplitude; it warns on large occupation
rounding or amplitudes above one.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``regularizer_flow``
     - float
     - Flow parameter :math:`s` of the DSRG regularizer, in
       :math:`E_h^{-2}`; ``0`` selects the bare inverse. Default: 0.5
   * - ``semicanonicalize``
     - bool
     - Diagonalize each spin's generalized Fock matrix blockwise before forming
       denominators. Default: true
   * - ``max_folded_occupation_deviation``
     - float
     - Largest deviation from an integer reference occupation allowed for a folded
       orbital, in :math:`[0,1)`. Default: 0.5

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
