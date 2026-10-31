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
occupation-changing remainder :math:`H_{\mathrm{OD}}`, the effective Hamiltonian is

.. math::

   H_{\mathrm{eff}} = \hat P\left(H_{\mathrm{BD}} +
   \tfrac{1}{2}[S,H_{\mathrm{OD}}]\right)\hat P.

The generator :math:`S` is calculated from

.. math::

   [F_0, S] = H_{\mathrm{OD}},

where :math:`F_0` -- the diagonal part of the input reference wavefunction's
generalized Fock operator -- is used to approximate :math:`H_{\mathrm{BD}}`, which
allows for efficient computation of :math:`S`. This results in

.. math::

   S = \sum_{T} \frac{c_T}{\Delta_T}\,\hat{T},
   \qquad
   \Delta_T = \sum_{\mathrm{creations}} \varepsilon_{p\sigma}
            - \sum_{\mathrm{annihilations}} \varepsilon_{p\sigma},

where :math:`c_T` are the coefficients of the operator terms :math:`\hat{T}` in
:math:`H_{\mathrm{OD}} = \sum_T c_T\,\hat{T}`, and the orbital energies
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
and the emitted integrals are rotated back. A block whose off-diagonal elements
are negligible is left unrotated: at most :math:`10^{-6}/\max(1,\sigma)\,E_h`
with the regularizer described below, the level a converged SCF leaves, and at
most :math:`10^{-10}\,E_h` without it, because bare denominators can amplify the
neglected elements without bound.
With this option disabled, the denominators use the diagonals of
:math:`F^\sigma` in the supplied basis.

Near-degenerate channels give a small :math:`\Delta_T` and a large amplitude, which can make the
perturbative expansion unreliable. ``regularizer_sigma2`` damps them by replacing the
bare inverse denominator with

.. math::

   \frac{1}{\Delta_T} \to \frac{1 - e^{-\sigma \Delta_T^2}}{\Delta_T},

the :math:`\sigma^2` regularizer :cite:`Shee2021`, whose :math:`\sigma` (in :math:`E_h^{-2}`)
plays the role of the DSRG flow parameter :math:`s` in the generator
:cite:`Evangelista2014`. Because the correction is the half-commutator above,
second-order energies match DSRG-PT2 at :math:`s = \sigma/2`; the default
:math:`\sigma = 1` corresponds to :math:`s = 0.5`. Larger :math:`\sigma`
regularizes less. Setting it to ``0`` selects a guarded bare pseudoinverse:
coupled channels with :math:`|\Delta_T| < 10^{-8}\,E_h` are omitted, with a warning.
This differs from the :math:`\sigma \to 0` limit, which damps all channels to zero.
Damping or omitting channels makes the generator equation approximate; the
correction remains the half-commutator above, not a full DSRG expansion.

The damping is not confined to intruders. A channel keeps the fraction
:math:`1 - e^{-\sigma \Delta_T^2}` of its bare amplitude, so at the default
:math:`\sigma = 1\,E_h^{-2}` denominators of 0.3, 0.5, 1.0 and 2.0 :math:`E_h` retain
approximately 9%, 22%, 63% and 98% of their bare generator amplitudes, respectively.
Even well-separated channels are damped; increase :math:`\sigma` to approach the
bare amplitudes.

The kept space :math:`P` is defined by the ``p_indices`` argument. The reference wavefunction
only supplies the density matrices :math:`\gamma^\sigma`, and is independent of :math:`P`.
Every orbital in :math:`Q = W \setminus P` is folded, as doubly
occupied if its spin-traced reference occupation rounds to two and as empty if it rounds to zero. Rounding
an occupation that is far from 2 or 0 perturbs the mean field the active space feels, so
``max_folded_occupation_deviation`` bounds how far it may stray before the downfold is
rejected. A singly occupied orbital therefore cannot be folded and must be kept in :math:`P`.
The active space keeps whatever electrons the folded orbitals do not take.

The commutator :math:`[S, H_{\mathrm{OD}}]` generates three-body terms that the emitted
two-body operator cannot carry. With ``fold_above_two_body``, reference-1-RDM pair
contractions reduce these to lower-rank terms :cite:`Kutzelnigg1997`.
This Gaussian approximation recovers ordinary Wick contractions for a determinant in
its own orbitals, closed-shell restricted or unrestricted. It neglects connected higher
density cumulants for correlated references and spin-averaged open-shell references.
The residual three-body operator is discarded.
Folding can improve accuracy but adds cost; it is skipped when the retained space
contains at most two electrons.

The reference enters only through :math:`\gamma^\sigma`, so what is accepted is a property of the
input, not of the method that produced it: any restricted or unrestricted orbital set
works, canonical or not. RHF, ROHF, CAS, localized and natural orbitals all qualify,
including UHF natural orbitals carried as restricted orbitals with their 1-RDM. Unrestricted
orbitals, such as UHF, give an unrestricted effective Hamiltonian; the input Hamiltonian's
active space, ``p_indices`` and the reference active space must then select the same orbital
indices in both spins, and both spins of a folded index are folded together. Rejected is any
singly occupied orbital outside the reference active space, which could only be folded on an
arbitrary rounding.
The reference and the input Hamiltonian must share the same molecular-orbital basis, and the
orbitals the Hamiltonian folded into its core energy must be exactly the reference core
orbitals outside :math:`W`, since otherwise that core energy and :math:`\gamma^\sigma` describe
different states.

.. warning::

   The emitted two-body blocks are only **4-fold** symmetric. Hermiticity
   :math:`(pq|rs) = (qp|sr)` and electron exchange :math:`(pq|rs) = (rs|pq)` hold as for the
   input integrals, but the bra swap :math:`(pq|rs) = (qp|rs)` of a genuine Coulomb integral
   does not: the commutator is not an electron-repulsion operator. Consumers must be given
   the full dense :math:`n^4` blocks.

The constructor logs regularization, active electron count, folded-occupation
deviation, minimum coupled denominator, and maximum raw amplitude.
The amplitude diagnostic covers the terms that connect :math:`P` and :math:`Q`
and uses undamped couplings -- the generalized Fock matrix for single
excitations -- with the denominator floored at
:math:`10^{-8}\,E_h`. Large occupation or charge-rounding errors and amplitudes
above one trigger warnings. These are diagnostics, not convergence guarantees.

SeQuant derives the contraction equations on the first downfold with a nonempty
external space and reuses them within the process. Contraction orders are chosen
for the inactive, active, and virtual dimensions and rebuilt when they change.
This adds a setup cost.
Numerical cost and memory depend on the window, kept space, and density fold.
Evaluation uses per-spin spatial blocks; memory includes the dense integrals,
contraction workspace, and cached plans.

.. rubric:: Settings

``regularizer_sigma2`` (float, default ``1.0``)
   Strength :math:`\sigma` of the :math:`\sigma^2` denominator regularizer, in
   :math:`E_h^{-2}`; ``0`` selects the guarded bare pseudoinverse.

``semicanonicalize`` (bool, default ``True``)
   Diagonalize each spin's generalized Fock matrix blockwise before forming denominators.

``fold_above_two_body`` (bool, default ``True``)
   Approximate the three-body terms using reference-1-RDM pair contractions
   instead of discarding them; higher density cumulants are neglected.

``max_folded_occupation_deviation`` (float, default ``0.5``)
   Largest deviation from an integer reference occupation allowed for a folded orbital.
   This tolerance must lie in :math:`[0,1)`.

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
