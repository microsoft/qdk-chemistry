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
   * - :class:`~qdk_chemistry.algorithms.QdkQIOOrbitalOptimizer`
     - All orbital pairs, including pairs across subspace boundaries
     - Total single-orbital entropy :cite:`Liao2024`
     - One orbital proposal
   * - Quantum information-assisted complete active space optimization (QICAS)
     - Rotations that update a target active-space projector
     - Entropy outside the target active space :cite:`Ding2023`
     - Self-consistently optimized active space and wavefunction

The QIO orbital optimizer is not a QICAS workflow.
It performs one optimization against the reduced density matrices of its input wavefunction and does not rebuild the Hamiltonian or recompute the correlated state.
QICAS requires an outer macro-iteration that repeatedly solves the correlated problem, updates the active-space projector, and tests self-consistent convergence.

Running orbital optimization
----------------------------

Create the default implementation and run it on a correlated wavefunction:

.. code-block:: python

   from qdk_chemistry import algorithms

   optimizer = algorithms.create("orbital_optimizer")
   result = optimizer.run(wavefunction)
   proposed_orbitals = result.orbitals

The input wavefunction must contain restricted orbitals with an overlap matrix, an active-space partition, and real spin-dependent active-space one- and two-particle reduced density matrices.

Available implementations
-------------------------

QDK full-window quantum-information orbitals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_qio"`` (default)

The :class:`~qdk_chemistry.algorithms.QdkQIOOrbitalOptimizer` reconstructs full-window density matrices by treating inactive orbitals as doubly occupied and virtual orbitals as empty, then embedding the correlated active-space density matrices.
It minimizes the total single-orbital entropy over all orbitals using Jacobi sweeps and permits rotations across inactive, active, and virtual boundaries.
The active and inactive index labels are retained on the returned orbital columns, so cross-boundary rotations change the corresponding subspace projectors.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``max_cycles``
     - int
     - ``200``
     - Maximum number of Jacobi sweeps
   * - ``convergence_tolerance``
     - float
     - ``1e-10``
     - Sweep-to-sweep objective change used for convergence
   * - ``coarse_angle_step``
     - float
     - ``0.02``
     - Coarse pair-rotation angle spacing in radians
   * - ``fine_samples``
     - int
     - ``201``
     - Number of samples used to refine the best coarse angle
   * - ``improvement_tolerance``
     - float
     - ``1e-12``
     - Minimum objective decrease required to accept a pair rotation
