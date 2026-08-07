Effective Hamiltonian construction
==================================

The :class:`~qdk_chemistry.algorithms.EffectiveHamiltonianConstructor`
algorithm folds orbitals outside a selected active space into an effective
Hamiltonian that acts only on that active space. This differs from
:class:`~qdk_chemistry.algorithms.HamiltonianConstructor`, which builds a bare
Hamiltonian from molecular orbitals.

Second-order Schrieffer-Wolff downfolding
-----------------------------------------

The ``"qdk_swpt2"`` implementation evaluates

.. math::

   H_{\mathrm{eff}} = P\left(H_{\mathrm{BD}} +
   \frac{1}{2}[S,H_{\mathrm{OD}}]\right)P,
   \qquad [F_0,S] = H_{\mathrm{OD}},

and retains scalar, one-body, and two-body operators in the active space
:math:`P`. Here :math:`F_0` is a diagonal, spin-free generalized Fock
operator. Restricted HF, restricted open-shell HF, and spin-adapted CAS
references are supported. Unrestricted orbitals are not supported.

Preparing the inputs
--------------------

The three inputs describe different orbital spaces:

``reference``
   A :class:`~qdk_chemistry.data.Wavefunction` whose occupations and reduced
   density matrices define the reference density over the window :math:`W`. It
   does not select :math:`P`. A CAS reference should contain its active one-body
   reduced density matrix. A restricted mean-field reference can also be used
   directly.

``window_hamiltonian``
   A :class:`~qdk_chemistry.data.Hamiltonian` built after marking every orbital
   in the full downfolding window :math:`W=P\cup Q` as active. Its inactive
   orbitals must match the reference core. Building this Hamiltonian only over
   :math:`P` removes the :math:`P\leftrightarrow Q` couplings before
   downfolding and is therefore invalid.

``p_indices``
   A :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet` naming the
   kept space :math:`P` as window orbital indices. It need not equal the
   reference active space. Reusing
   ``reference.get_orbitals().active_indices()`` keeps :math:`P` equal to the
   reference active space.

Each folded orbital's reference occupation is rounded to doubly occupied or
empty. Rounding does not change the total electron count: the active space
receives whatever the folded orbitals do not take, and the resulting integer
active electron count is logged. That is the count to pass to the active-space
solver. ``max_folded_occupation_deviation`` bounds how far a folded occupation
may sit from an integer; a half-occupied orbital can never be folded, because
rounding it either way would change the electron count and break spin symmetry.

Rounding a fractional occupation perturbs the mean field the active space feels
at first order, and this error is not damped by denominator regularization. The
active space itself is unaffected: it receives an integer electron count and its
density is determined by the subsequent active-space solve. The error lives
entirely on the folded side, and is summarized by the net electron count the
folded core carries in excess of the reference density. Individual roundings of
opposite sign cancel in that sum, so keeping a correlated pair together on the
folded side is preferable: the leftover density error is then neutral and short
ranged. Both the largest folded deviation and the excess are logged, and a
warning is raised when either is large.

The reference and window Hamiltonian must use the same restricted molecular
orbital basis, and every reference-active orbital must occur in the window.
For a restricted open-shell reference, every singly occupied orbital must be
active.

Running the downfold
--------------------

After preparing the reference over :math:`P` and the Hamiltonian over
:math:`W`, create and run the constructor:

.. code-block:: python

   from qdk_chemistry import algorithms

   downfolder = algorithms.create(
       "effective_hamiltonian_constructor", "qdk_swpt2"
   )
   p_indices = reference.get_orbitals().active_indices()
   effective_hamiltonian = downfolder.run(
       reference, window_hamiltonian, p_indices
   )

The returned Hamiltonian uses the reference active-space indexing and can be
passed to an active-space solver. The original inputs are not modified.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 28 14 16 42

   * - Setting
     - Type
     - Default
     - Description
   * - ``regularizer``
     - string
     - ``"flow"``
     - Inverse-denominator scheme: ``"flow"``, ``"shift"``, or ``"bare"``.
   * - ``denom_flow``
     - float
     - ``1.0``
     - Flow parameter in :math:`E_h^{-2}` for flow regularization.
   * - ``denom_shift``
     - float
     - ``0.0``
     - Shift in :math:`E_h` for shifted inverse denominators.
   * - ``denom_floor``
     - float
     - ``1\times10^{-8}``
     - Absolute cutoff in :math:`E_h` used by bare denominators.
   * - ``intruder_warn_amplitude``
     - float
     - ``1.0``
     - Warn when the largest raw generator amplitude exceeds this value.
   * - ``semicanonicalize``
     - bool
     - ``True``
     - Diagonalize the generalized Fock matrix within each orbital-role block.
   * - ``semicanonical_tolerance``
     - float
     - ``1\times10^{-10}``
     - Skip a role-block rotation below this off-diagonal tolerance.
   * - ``max_folded_occupation_deviation``
     - float
     - ``0.5``
     - Largest deviation from an integer reference occupation allowed for a
       folded orbital. Must be below 1.

For example, select shifted denominators before the first run:

.. code-block:: python

   settings = downfolder.settings()
   settings.set("regularizer", "shift")
   settings.set("denom_shift", 0.5)

Settings lock when ``run`` begins.

Diagnostic logging
------------------

The constructor logs the selected regularizer, minimum denominator, maximum raw
amplitude, whether a semicanonical rotation was applied, and the derived active
electron count. Small denominators
and large raw amplitudes indicate sensitivity to intruder states; amplitudes
above ``intruder_warn_amplitude`` also produce a warning. The default flow
parameter is a policy default, not a universal accuracy guarantee. Compare
regularization choices for the system being studied when the logged values
indicate sensitivity.

Approximations and scaling
--------------------------

This implementation truncates the transformed Hamiltonian to at most two-body
operators and uses diagonal generalized-Fock denominators. Flow regularization
uses a DSRG-style damping function, but is not a full DSRG calculation.

Dense four-center integrals require :math:`O(N^4)` storage. When a
noncanonical window is semicanonicalized, the four-index transformation
requires :math:`O(N^5)` work. The retained commutator also grows steeply with
active-space size. Use this implementation for modest dense windows; it is not
intended for windows containing hundreds of orbitals.
