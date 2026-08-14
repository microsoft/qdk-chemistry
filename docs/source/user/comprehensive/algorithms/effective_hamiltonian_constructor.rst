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

QDK SW-PT2
~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_swpt2"`` (aliases ``"swpt2"``, ``"schrieffer_wolff"``)

This is the default implementation returned by ``create("effective_hamiltonian_constructor")``.

Second-order Schrieffer-Wolff (Van Vleck) downfolding. Splitting :math:`H` into a part
:math:`H_{\mathrm{BD}}` that is block diagonal in the occupation of :math:`Q` and an
occupation-changing remainder :math:`H_{\mathrm{OD}}`, the effective Hamiltonian is

.. math::

   H_{\mathrm{eff}} = P\left(H_{\mathrm{BD}} +
   \tfrac{1}{2}[S,H_{\mathrm{OD}}]\right)P,
   \qquad [F_0, S] = H_{\mathrm{OD}},

truncated to at most two-body operators. :math:`F_0` is a diagonal, spin-free generalized
Fock operator. Enabling a denominator regularizer replaces the bare inverse denominators of
:math:`S`, which then satisfies the commutator equation only approximately.

The input Hamiltonian must be built with every orbital of the window marked active, since the
Hamiltonian constructor folds inactive orbitals into the constant energy term and drops
virtual orbitals. Building it over :math:`P` alone removes the :math:`P \leftrightarrow Q`
couplings that the method needs.

Restricted HF, restricted open-shell HF, and spin-adapted CAS references are supported;
for ROHF every singly occupied orbital must be active. Unrestricted orbitals are rejected.
The reference and the input Hamiltonian must use the same molecular-orbital basis, and the
Hamiltonian's inactive orbitals must be the reference core orbitals outside the window.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 30 12 16 42

   * - Setting
     - Type
     - Default
     - Description
   * - ``denom_flow``
     - float
     - ``1.0``
     - Flow-parameter regularizer, :math:`1/D \to (1-e^{-sD^2})/D`, in :math:`E_h^{-2}`. ``0`` disables it.
   * - ``denom_imaginary_shift``
     - float
     - ``0.0``
     - Imaginary level shift, :math:`1/D \to D/(D^2+s^2)`, in :math:`E_h`. ``0`` disables it.
   * - ``denom_floor``
     - float
     - ``1e-8``
     - Absolute cutoff in :math:`E_h` used by unregularized denominators.
   * - ``semicanonicalize``
     - bool
     - ``True``
     - Diagonalize the generalized Fock matrix within each orbital-role block.
   * - ``fold_above_two_body``
     - bool
     - ``True``
     - Fold the three-body terms the transformation generates onto the reference density instead of discarding them. Ignored when :math:`P` holds at most two electrons.
   * - ``max_folded_occupation_deviation``
     - float
     - ``0.5``
     - Largest deviation from an integer reference occupation allowed for a folded orbital. Must be below 1.

``denom_flow`` and ``denom_imaginary_shift`` are mutually exclusive: a positive value enables
that scheme, and setting both raises an error rather than silently applying one. With both at
zero the unregularized inverse is used, floored by ``denom_floor``. The flow option borrows the
DSRG damping form but does not make this a DSRG calculation.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Example

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/effective_hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-downfold
      :end-before: # end-cell-downfold

SW-PT2 algorithm
^^^^^^^^^^^^^^^^

Each folded orbital's reference occupation is rounded to doubly occupied or empty. Rounding
does not change the total electron count, because the active space receives whatever the
folded orbitals do not take, and the resulting integer active electron count is logged for the
active-space solver. Rounding does perturb the mean field the active space feels, at first
order and without being damped by the regularizer, so both the largest folded deviation and
the net charge the folded core carries in excess of the reference density are logged, with a
warning when either is large. Keeping a correlated pair together on the folded side makes its
roundings cancel.

Noncanonical orbitals are semicanonicalized independently within the inactive, active, and
virtual blocks before the denominators are formed, and the emitted operator is rotated back to
the caller's basis.

The constructor logs the active regularization, the minimum denominator, the maximum raw
amplitude :math:`|V/\Delta|`, and whether a semicanonical rotation was applied. Small
denominators and large amplitudes indicate sensitivity to intruder states; an amplitude above
one, where the perturbation series stops contracting, also produces a warning. The default flow
parameter is a policy default rather than an accuracy guarantee, so compare regularization
choices when the logged values indicate sensitivity.

This implementation emits at most two-body operators and uses diagonal
generalized-Fock denominators. :math:`\tfrac{1}{2}[S,H_{\mathrm{OD}}]` also generates three-body
terms, which a Hamiltonian cannot hold. Discarding them outright is not a small correction: a
three-body operator has no matrix elements below three electrons, so it would be harmless only
while :math:`P` holds at most two, and its cost grows with the electron count in :math:`P`.
Folding a single valence virtual of water in a minimal basis into a six-electron kept space
that way costs about 0.2 :math:`E_h`, more than the orbital is worth and of the opposite sign.

Instead of discarding them, the terms above two-body are folded onto the reference: each is
normal-ordered against the reference one-particle density :math:`\gamma` and whatever falls to
two-body is kept, so only the reference-normal-ordered residual is lost rather than the whole
term. Nothing here requires :math:`\gamma` to describe a single determinant, so open-shell,
natural-orbital and correlated active-space references are all handled; what is neglected is
the two-body density cumulant.

What folding buys is a *bounded* error rather than a uniformly smaller one. Discarding is
erratic -- sometimes accidentally near-exact, sometimes catastrophic, with nothing in the
inputs to say which -- while folding lands in a narrow band. Measured over 64 cases spanning
ten molecules, two basis sets, closed- and open-shell references, active spaces from 4 to 8
electrons, and one to three folded virtuals, against full CI in the same window:

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Error vs full CI
     - median
     - mean
     - worst
   * - Discarding
     - 0.244
     - 0.499
     - 2.687
   * - Folding
     - 0.005
     - 0.006
     - 0.016

The gap widens as more orbitals are folded, which is the direction of practical interest.
Folding more orbitals generates more of the discarded terms, so discarding degrades sharply
while folding stays flat:

.. list-table::
   :header-rows: 1
   :widths: 22 16 16 23 23

   * - Folded virtuals
     - Cases
     - Folding loses
     - Median, discarding
     - Median, folding
   * - 1
     - 34
     - 5
     - 0.087
     - 0.005
   * - 2
     - 19
     - 2
     - 0.359
     - 0.005
   * - 3
     - 11
     - 0
     - 1.264
     - 0.005

Open-shell references behave at least as well as closed-shell ones: of 11 ROHF cases folding
won 10 and lost none, by factors of 14 to 125. The spin-traced density gives each singly
occupied orbital half an electron per spin, which keeps the emitted operator spin-free.

Folding is nevertheless **not** a strict improvement. It lost in 7 of the 64 cases, all with a
single folded virtual, all in multiply bonded systems (N2, CO) where the discarded terms are
small or cancel so their reference contractions move a nearly exact answer away from full CI.
The largest such loss was 0.011 :math:`E_h`. Correlation strength does not predict the
direction, so there is no useful criterion to gate on, and no substitute for checking against
a larger calculation when the answer matters.

The benefit also grows with the electron count in :math:`P`. In a separate sweep over 86
cases, folding won 17 of 18 at eight active electrons and 8 of 9 at ten, while at four
electrons it was an even split -- three-body operators simply matter more as :math:`P` fills.

Folding also removes a spurious sensitivity to the regularizer. Discarded three-body terms
used to leave flow and bare denominators disagreeing by 0.4 :math:`E_h` on the equilibrium
case, which looked like an intruder problem and was not; folded, they agree to about 0.002
:math:`E_h`.

Two consequences follow. The emitted operator now depends on the reference density, so its
accuracy degrades as the active-space solution moves away from that reference. And the
occupations are read after semicanonicalization, since that rotation mixes occupied and empty
orbitals within the kept space.

Folding is not free, and its cost depends on the reference. Reaching two-body means
enumerating contractions that the two-body truncation would otherwise let the projection skip,
and the enumeration prunes a branch as soon as a reference propagator vanishes. A determinant
reference therefore costs about six to seven times the kernel time, while a correlated
reference, whose density is dense after semicanonicalization, costs roughly sixteen to
twenty-three times. Folding does not change the asymptotic scaling, which stays near
:math:`O(N_{\mathrm{active}}^5)` either way; it is a constant multiplier.

It is therefore controlled by ``fold_above_two_body``, on by default because discarding is the
larger error. A kept space holding at most two electrons skips the cost automatically, since
the discarded terms have no matrix elements to contribute there.

Dense four-center integrals require :math:`O(N^4)` storage, and semicanonicalizing a
noncanonical window costs :math:`O(N^5)`. The retained commutator also grows steeply with the
size of :math:`P`. Use it for modest dense windows rather than windows of hundreds of orbitals.

Dense four-center integrals require :math:`O(N^4)` storage, and semicanonicalizing a
noncanonical window costs :math:`O(N^5)`. The retained commutator also grows steeply with the
size of :math:`P`. Use it for modest dense windows rather than windows of hundreds of orbitals.

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
