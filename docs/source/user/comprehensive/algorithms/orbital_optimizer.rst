Orbital optimization
====================

The :class:`~qdk_chemistry.algorithms.OrbitalOptimizer` algorithm proposes orbital rotations from a correlated :class:`~qdk_chemistry.data.Wavefunction`.
Unlike an :doc:`orbital localizer <localizer>`, an orbital optimizer may mix inactive, active, and virtual orbitals, thereby changing the active-space projector.

Contract
--------

The input and output are intentionally asymmetric:

.. code-block:: text

   correlated Wavefunction
       |
       |  OrbitalOptimizer.run(...)
       v
   OrbitalOptimizationResult
       |
       +-- proposed Orbitals
       +-- initial and final objectives
       +-- convergence information

The result contains an orbital proposal, not a wavefunction in the new basis.
After orbitals cross subspace boundaries, the molecular integrals and optimal correlated state change.
A workflow that accepts the proposal must therefore rebuild the Hamiltonian and solve for a new correlated wavefunction.

QIO, active-space QIO localization, and QICAS
---------------------------------------------

QDK/Chemistry distinguishes three related operations:

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Operation
     - Allowed rotations
     - Objective
     - Result
   * - :ref:`Active-space QIO localizer <localizer-qdk-active-space-qio>`
     - Active-active only
     - Entropy within a fixed active space
     - Transformed wavefunction representation
   * - Full-window quantum-information orbitals (QIO)
     - All orbital pairs, including pairs across subspace boundaries
     - Total single-orbital entropy :cite:`Liao2024`
     - One orbital proposal
   * - Quantum information-assisted complete active space optimization (QICAS)
     - Rotations that update a target active-space projector
     - Entropy outside the target active space :cite:`Ding2023`
     - Self-consistently optimized active space and wavefunction

Full-window QIO and QICAS both require correlated reduced density matrices over
a space that includes the orbitals allowed to cross the target active-space
boundary. Completing active-space density matrices with doubly occupied
inactive orbitals and empty virtual orbitals does not provide that correlation
signal and cannot drive projector-changing rotations.

Available implementations
-------------------------

No built-in :class:`~qdk_chemistry.algorithms.OrbitalOptimizer`
implementation is currently registered. The contract and factory support
custom implementations while full-window correlated RDM support is developed.
