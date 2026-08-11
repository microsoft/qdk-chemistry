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
The ``run`` method does not validate its arguments; each concrete implementation decides whether to check the nested-space contract :math:`P \subseteq W_H` and :math:`W_{\mathrm{ref}} \subseteq W_H` before computing.
:math:`P` is not required to lie inside the reference active space: a kept orbital takes its correlation from the subsequent active-space solve, so only the folded orbitals of :math:`Q` rely on the reference density.

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

This implementation truncates the transformed Hamiltonian to at most two-body operators and
uses diagonal generalized-Fock denominators. The truncation is not a uniformly small
correction: :math:`\tfrac{1}{2}[S,H_{\mathrm{OD}}]` contains three-body terms that the output
Hamiltonian cannot represent, so the emitted operator reproduces the exact second-order Van
Vleck operator only while :math:`P` holds at most two electrons -- a three-body operator has no
matrix elements below three electrons. From three electrons on the discarded term contributes,
and it grows with the electron count in :math:`P`. This is a property of the kept space, not of
how many orbitals are folded: folding a single valence virtual of water in a minimal basis into
a six-electron kept space costs about 0.2 :math:`E_h`, where that orbital is worth
:math:`-0.02\ E_h` exactly and :math:`-0.06\ E_h` at untruncated second order, but the
truncated operator returns :math:`+0.14\ E_h`. Nothing in the intruder diagnostics detects
this, since it is a truncation error rather than a small denominator. Treat downfolded results
for active spaces holding more than two electrons as qualitative unless you can check them
against a larger calculation.

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
