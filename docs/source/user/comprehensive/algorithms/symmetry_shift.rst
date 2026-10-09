Symmetry shift (BLISS)
======================

The :class:`~qdk_chemistry.algorithms.SymmetryShifter` algorithm in QDK/Chemistry subtracts a number-symmetry operator from a Hamiltonian to reduce its :term:`LCU` 1-norm without changing the energy of the target electron-number sector.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Hamiltonian` and a target alpha/beta electron count as input and produces a new, shifted :class:`~qdk_chemistry.data.Hamiltonian` as output.
For more information about this pattern, see the :doc:`Factory Pattern <factory_pattern>` documentation.

QDK/Chemistry currently provides one implementation, :class:`~qdk_chemistry.algorithms.FermionicLowRankShifter` (``fermionic_low_rank``), the fermionic low-rank variant of the Block-Invariant Symmetry Shift (:term:`BLISS`) method :cite:`Loaiza2023,Patel2024`.

Overview
--------

Qubitization-based algorithms, such as qubitized :term:`QPE`, encode a Hamiltonian as a linear combination of unitaries.
Their cost scales with the 1-norm :math:`\lambda` of that decomposition, so reducing :math:`\lambda` reduces the runtime of the quantum algorithm directly, without changing the chemistry being simulated.

:term:`BLISS` exploits the fact that a Hamiltonian is only ever used within a fixed electron-number sector.
Any operator that annihilates every :math:`N_e`-electron state may be subtracted from :math:`H` for free: the spectrum within that sector is unchanged, while :math:`\lambda` generally drops.
The operator used here is parametrized by :math:`(\mu_1, \mu_2, \xi)`:

.. math::

   \hat K = \mu_1 (\hat N - N_e)
          + \mu_2 (\hat N^2 - N_e^2)
          + (\hat N - N_e) \sum_{ij} \xi_{ij} \hat E_{ij},

where :math:`\hat N` is the number operator and :math:`\hat E_{ij}` the spin-summed excitation operator.
Each term carries a factor of :math:`(\hat N - N_e)`, so :math:`\hat K \lvert \Psi_{N_e} \rangle = 0` and :math:`H - \hat K` reproduces the energies of :math:`H` in the :math:`N_e`-electron sector exactly.
A shifter chooses :math:`(\mu_1, \mu_2, \xi)` to minimize :math:`\lambda` of :math:`H - \hat K`.

.. _fermionic-low-rank-bliss:

Fermionic low-rank BLISS
~~~~~~~~~~~~~~~~~~~~~~~~

The fermionic low-rank method :cite:`Patel2024` takes an already *double-factorized* Hamiltonian as its input,

.. math::

   H = \sum_{ij} h_{ij} \hat E_{ij}
       + \frac{1}{2} \sum_r \Big( \sum_{ij} M^{(r)}_{ij} \hat E_{ij} \Big)^2
       + \dots,
   \qquad
   M^{(r)}_{ij} = \sum_b W^{(r)}_b\, U^{(r)}_{bi} U^{(r)}_{bj},

where each fragment :math:`r` is fixed by an orbital rotation :math:`U^{(r)}` and its eigenvalues :math:`W^{(r)}`.
Producing that form is the job of the separate ``double_factorization`` algorithm, not of BLISS.

.. note::
   The square carries a factor of :math:`\tfrac{1}{2}`, since QDK/Chemistry factorizes the raw tensor :math:`g` while :cite:`Patel2024` works with :math:`V = \tfrac{1}{2} g`.
   The shift is therefore computed from the rescaled eigenvalues :math:`\varepsilon^{(r)}_b = W^{(r)}_b / \sqrt{2}` and reported on the raw-:math:`g` scale, which gives the :math:`\mu_2` term below an extra factor of two relative to Eqs. 6-7 of :cite:`Patel2024`; the remaining difference, :math:`-\xi - \mu_2 I` on the one-body tensor, is the normal-ordering correction.

The fermionic 1-norm splits into a one-electron and a fragment contribution, :math:`\lambda = \lambda_{1e} + \lambda_{\mathrm{DF}}`, and both are minimized in closed form:

#. **Per-fragment shift.** Shifting fragment :math:`r` by :math:`\phi_r` moves all its eigenvalues by a constant, so the minimizer of its contribution to :math:`\lambda_{\mathrm{DF}}` is the *median* of the fragment's eigenvalues.
   Summing the per-fragment shifts gives the global two-electron parameters :math:`(\mu_2, \xi)`.
#. **One-electron shift.** :math:`\mu_1` is then the median of the eigenvalues of the *effective* one-electron operator of :math:`H - \hat K`, that is, the one-body tensor with the mean-field contraction of the shifted two-electron integrals folded in.
   Optimizing against the effective operator, rather than the bare :math:`h`, is what makes the reduction of :math:`\lambda_{1e}` carry over to the true 1-norm.

Because the shift is built from per-fragment medians, it is absorbed exactly into the existing fragments: the rotations :math:`U^{(r)}` are untouched and only the fragment eigenvalues, the one-body tensor, and the constant energy change,

.. math::

   \tilde h_{ij} &= h_{ij} + (N_e - 1)\,\xi_{ij} - (\mu_1 + \mu_2)\,\delta_{ij}, \\
   \tilde g_{ijkl} &= g_{ijkl} - 2\mu_2\,\delta_{ij}\delta_{kl}
                      - \xi_{ij}\delta_{kl} - \delta_{ij}\xi_{kl}, \\
   E'_{\mathrm{core}} &= E_{\mathrm{core}} + \mu_1 N_e + \mu_2 N_e^2 .

No dense :math:`n_{\mathrm{orb}}^4` tensor is ever formed, and the output is again a factorized Hamiltonian that can be block-encoded without re-factorization.

.. note::
   The one- and two-electron norms are minimized sequentially, not jointly, so the total 1-norm is not guaranteed to decrease.
   If the computed shift would increase :math:`\lambda`, the implementation logs a warning and returns the Hamiltonian unchanged with a zero shift.

Running a symmetry shift
------------------------

The ``run`` method takes a :class:`~qdk_chemistry.data.Hamiltonian` and the target electron counts and returns a new, shifted :class:`~qdk_chemistry.data.Hamiltonian`.
Computing a shift without applying it is deliberately not exposed, since how a shift folds into a Hamiltonian depends on the representation the implementation consumes.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.SymmetryShifter` requires the following inputs:

Hamiltonian
   A :class:`~qdk_chemistry.data.Hamiltonian` to shift.
   :class:`~qdk_chemistry.algorithms.FermionicLowRankShifter` requires a spin-restricted Hamiltonian backed by a :class:`~qdk_chemistry.data.FactorizedHamiltonianContainer`, that is, the output of the ``double_factorization`` :class:`~qdk_chemistry.algorithms.HamiltonianFactorization` algorithm.
   Anything else raises ``ValueError`` (``std::invalid_argument`` in C++).

Alpha electron count (``n_alpha_electrons``)
   The target number of alpha electrons in the active space.

Beta electron count (``n_beta_electrons``)
   The target number of beta electrons in the active space.

.. note::
   Only the total electron count :math:`N_e = N_\alpha + N_\beta` enters the fermionic low-rank shift; it does not use :math:`S_z`, so ``(5, 5)`` and ``(6, 4)`` produce the same result.

.. rubric:: Creating a shifter

The fermionic low-rank shifter has no tunable settings: truncation and the choice of decomposition belong to the ``double_factorization`` algorithm whose output it consumes.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/symmetry_shift.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/symmetry_shift.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. rubric:: Preparing a factorized Hamiltonian

Double factorization is not part of the shift; this step only produces an input of the form the shifter accepts.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/symmetry_shift.py
      :language: python
      :start-after: # start-cell-factorize
      :end-before: # end-cell-factorize

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/symmetry_shift.cpp
      :language: cpp
      :start-after: // start-cell-factorize
      :end-before: // end-cell-factorize

.. rubric:: Applying the shift

The fermionic 1-norm is a property of the factorized representation, so both the before and after values come from ``get_lambda()`` on the factorized container.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/symmetry_shift.py
      :language: python
      :start-after: # start-cell-shift
      :end-before: # end-cell-shift

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/symmetry_shift.cpp
      :language: cpp
      :start-after: // start-cell-shift
      :end-before: // end-cell-shift

.. rubric:: Inspecting the applied shift

The shifted Hamiltonian is the whole result; the :math:`(\mu_1, \mu_2, \xi)` parameters are internal to the implementation and are not returned.
``fermionic_low_rank`` logs the applied :math:`\mu_1` and :math:`\mu_2`, together with a summary of the per-fragment medians, at ``info`` level.

.. rubric:: Persisting the shifted Hamiltonian

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/symmetry_shift.py
      :language: python
      :start-after: # start-cell-persist
      :end-before: # end-cell-persist

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/symmetry_shift.cpp
      :language: cpp
      :start-after: // start-cell-persist
      :end-before: // end-cell-persist

Available implementations
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Name
     - Description
     - Requirements
   * - ``fermionic_low_rank``
     - Fermionic low-rank BLISS :cite:`Patel2024`: closed-form per-fragment median shift plus the optimal one-electron shift, absorbed back into the fragments.
     - Restricted, double-factorized Hamiltonian

``fermionic_low_rank`` is also the default returned by the factory when no name is given.
Use ``available("symmetry_shifter")`` to list registered implementations at runtime, as shown in :doc:`Factory Pattern <factory_pattern>`.

Related classes
---------------

- :class:`~qdk_chemistry.data.Hamiltonian`: The input and output of a shift
- :class:`~qdk_chemistry.data.FactorizedHamiltonianContainer`: Holds the double factorization and exposes ``get_lambda()``
- :class:`~qdk_chemistry.algorithms.DoubleFactorization`: Produces the factorized Hamiltonian the shifter consumes

Further reading
---------------

- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Builds the molecular Hamiltonian to be factorized
- :doc:`QubitMapper <qubit_mapper>`: Maps a shifted Hamiltonian to qubits
- :doc:`PhaseEstimation <phase_estimation>`: A consumer of the reduced 1-norm
- :doc:`Settings <settings>`: Configures algorithm implementations
