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

where :math:`F_0` -- the spin-free generalized Fock operator based on the input reference wavefunction -- is used to approximate :math:`H_{\mathrm{BD}}`, which allows for efficient computation of :math:`S`. This results in

.. math::

   S = \sum_{T} \frac{c_T}{\Delta_T}\,\hat{T},
   \qquad
   \Delta_T = \sum_{\mathrm{creations}} \varepsilon_p
            - \sum_{\mathrm{annihilations}} \varepsilon_p,

where :math:`c_T` are the coefficients of the operator terms :math:`\hat{T}` in
:math:`H_{\mathrm{OD}} = \sum_T c_T\,\hat{T}`, and the orbital energies
:math:`\varepsilon_p` are the diagonal elements of the generalized Fock matrix:

.. math::
   \varepsilon_p = F_{pp},\qquad F_{pq} = h_{pq}
          + \sum_{rs} \gamma_{rs}\left[(pq|rs) - \tfrac{1}{2}(pr|sq)\right].

:math:`\gamma` is the reference wavefunction's spin-traced one-particle density over
:math:`W`. Because :math:`\gamma` is spin traced, the two spin channels share the same orbital
energies, :math:`\varepsilon_{p\alpha} = \varepsilon_{p\beta}`, so the emitted operator
commutes with the total spin and the active-space solve selects the spin sector.

These denominators assume :math:`F` is diagonal, which holds for canonical orbitals but not in
general. With ``semicanonicalize``, :math:`F` is diagonalized within each orbital-role block
before the denominators are formed and the emitted operator is rotated back to the caller's
basis, making the setting a no-op for canonical orbitals.

Near-degenerate channels give a small :math:`\Delta_T` and a large amplitude, where the
perturbative expansion stops converging. ``regularizer_sigma2`` damps them by replacing the
bare inverse denominator with

.. math::

   \frac{1}{\Delta_T} \to \frac{1 - e^{-\sigma \Delta_T^2}}{\Delta_T},

the :math:`\sigma^2` regularizer :cite:`Shee2021`, whose :math:`\sigma` (in :math:`E_h^{-2}`)
is equivalently the DSRG flow parameter :cite:`Evangelista2014`. Larger :math:`\sigma`
regularizes less. Setting it to ``0`` selects a guarded bare pseudoinverse: coupled channels
with :math:`|\Delta_T| < 10^{-8}\,E_h` are mapped to zero, while the remaining channels are
still downfolded. The constructor logs a warning when this cutoff is used because the result
then depends on omitting near-degenerate channels. ``0`` is a mode switch, not the
:math:`\sigma \to 0` limit, which damps every channel away entirely.

The damping is not confined to intruders. A channel keeps the fraction
:math:`1 - e^{-\sigma \Delta_T^2}` of its bare amplitude, so at the default
:math:`\sigma = 1\,E_h^{-2}` denominators of 0.3, 0.5, 1.0 and 2.0 :math:`E_h` retain 9%,
22%, 63% and 98% of the second-order result. Ordinary, well-separated channels are damped
along with the near-degenerate ones; raise :math:`\sigma` to recover them.

The kept space :math:`P` is defined by the ``p_indices`` argument. The reference wavefunction
only supplies the density matrix :math:`\gamma`, and is independent of :math:`P`.
Every orbital in :math:`Q = W \setminus P` is folded, as doubly
occupied if its reference occupation rounds to two and as empty if it rounds to zero. Rounding
an occupation that is far from 2 or 0 perturbs the mean field the active space feels, so
``max_folded_occupation_deviation`` bounds how far it may stray before the downfold is
rejected. The active space keeps whatever electrons the folded orbitals do not take.

The commutator :math:`[S, H_{\mathrm{OD}}]` generates three-body terms that the emitted
two-body operator cannot carry. With ``fold_above_two_body`` they are folded onto the reference
density using pair contractions formed from the spin-traced 1-RDM. This is the Gaussian-reference
truncation of the generalized normal-ordering framework :cite:`Kutzelnigg1997`. For a determinant
reference these are the ordinary Wick contractions, but the residual three-body operator is still
discarded. For a correlated CAS reference the approximation additionally neglects the two-body and
higher density cumulants. It can improve on discarding the three-body terms, especially as the
active electron count grows, but is not a complete correlated-reference normal ordering and costs
more to evaluate.

The reference enters only through :math:`\gamma`, so what is accepted is a property of the
input, not of the method that produced it: any restricted orbital set works, canonical or
not. RHF, ROHF, CAS, localized and natural orbitals all qualify, including UHF natural
orbitals carried as restricted orbitals with their 1-RDM. Rejected are unrestricted
orbitals, and any singly occupied orbital outside the reference active space, which could
only be folded on an arbitrary rounding.
The reference and the input Hamiltonian must share the same molecular-orbital basis, and the
orbitals the Hamiltonian folded into its core energy must be exactly the reference core
orbitals outside :math:`W`, since otherwise that core energy and :math:`\gamma` describe
different states.

.. warning::

   The emitted two-body block is only **4-fold** symmetric. Hermiticity
   :math:`(pq|rs) = (qp|sr)` and electron exchange :math:`(pq|rs) = (rs|pq)` survive the
   transformation, but the bra swap :math:`(pq|rs) = (qp|rs)` of a genuine Coulomb integral
   does not: the commutator is not an electron-repulsion operator. Consumers must be given
   the full dense :math:`n^4` block; one that reads only the canonical 8-fold-unique
   elements silently reconstructs a different operator, with no error raised.

The constructor logs the active regularization, the derived active electron count, the largest
folded occupation deviation, the minimum denominator and the maximum raw amplitude
:math:`|c_T/\Delta_T|`. It warns when a folded deviation or the folded core's excess charge is
large, and when the amplitude exceeds one, where the perturbation series stops contracting.

Cost is dominated by the projected commutator, and it grows steeply with the kept space;
building the window's dense integral block is negligible beside it. Widening the window is
cheaper than enlarging the kept space, but not free, and memory is bounded by the window's
dense four-center integrals. ``fold_above_two_body`` makes the downfold more expensive,
increasingly so as the kept space grows.

.. rubric:: Settings

``regularizer_sigma2`` (float, default ``1.0``)
   Strength :math:`\sigma` of the :math:`\sigma^2` denominator regularizer, in
   :math:`E_h^{-2}`; ``0`` selects the guarded bare pseudoinverse.

``semicanonicalize`` (bool, default ``True``)
   Diagonalize the generalized Fock matrix blockwise before forming denominators.

``fold_above_two_body`` (bool, default ``True``)
   Approximate the three-body terms using reference-1-RDM pair contractions
   instead of discarding them; higher density cumulants are neglected.

``max_folded_occupation_deviation`` (float, default ``0.5``)
   Largest deviation from an integer reference occupation allowed for a folded orbital.

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
