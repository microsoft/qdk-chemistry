Hamiltonian basis transformation
================================

The :class:`~qdk_chemistry.algorithms.HamiltonianBasisTransformer` algorithm
re-expresses an existing :class:`~qdk_chemistry.data.Hamiltonian` in a supplied
orbital basis without modifying the source Hamiltonian.

The native ``"qdk"`` implementation supports real, restricted, spin-only
:class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` data. It reuses the
stored three-center factors, avoiding another AO integral evaluation and
Cholesky decomposition.

This is an explicit, opt-in operation. The default ``"qdk"`` Hamiltonian
constructor returns canonical four-center integrals, which this implementation
does not support. Start with ``"qdk_cholesky"`` when planning to reuse a
Hamiltonian after active-orbital rotations.

Complete workflow
-----------------

The complete Python example uses only native algorithms. It selects a
two-electron, three-orbital active space for LiH, freezes the core orbital,
solves CASCI, rotates only the active
orbitals into natural orbitals, and reuses the Cholesky Hamiltonian before
solving again. The CASCI energy is invariant under this change of basis.
No optional plugin is required.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_basis_transformer.py
      :language: python
      :start-after: # start-cell-transform
      :end-before: # end-cell-transform

.. tab:: C++ API

   The corresponding transformation step in C++ is:

   .. code-block:: cpp

      auto transformer = HamiltonianBasisTransformerFactory::create("qdk");
      auto transformed_hamiltonian =
          transformer->run(source_hamiltonian, target_orbitals);

Run the complete Python example from the repository root:

.. code-block:: bash

   python docs/source/_static/examples/python/hamiltonian_basis_transformer.py

Supported transformations
-------------------------

The source and target orbitals must have the same AO basis, overlap matrix,
active/inactive index sets, and molecular orbitals outside the active space.
The active columns may differ by an orthogonal transformation. Additional
spatial symmetry labels and unrestricted Hamiltonians are not supported.
Source one-body integrals, three-center factors, inactive Fock matrix values,
and the core energy must be finite. After validating overlap-matrix symmetry,
the implementation uses its explicitly symmetrized value as the orbital
metric. It also validates the target active orbitals against the projected
difference between the symmetrized source and target metrics, preventing small
AO-matrix differences from being amplified in numerical null modes.

For each Cholesky factor :math:`L_Q` and recovered active-space rotation
:math:`U`, the implementation evaluates

.. math::

   h' = U^T h U, \qquad L'_Q = U^T L_Q U.

The corresponding full-orbital rotation is applied to the inactive Fock
matrix, while the core energy is unchanged. Optional AO Cholesky vectors remain
on the unchanged source Hamiltonian and are omitted from the returned
Hamiltonian to avoid copying this potentially large cache.

Performance benchmark
---------------------

The reproducible benchmark compares transformation with a fresh
``"qdk_cholesky"`` rebuild for LiH, water, and benzene. It reports SCF and initial
source construction separately from subsequent rotations, including all timing
samples, basis and active-space sizes, Cholesky rank, numerical thresholds, and
agreement between the returned integral payloads.

.. code-block:: bash

   python examples/benchmarks/hamiltonian_basis_transformer.py --case all --threads 1 --warmups 1 --repeats 5

The script sets thread environment variables before importing the numerical
libraries. Each ``BENCHMARK_RESULT`` JSON record contains warmed transformation
and rebuild timings. These measure the case where the source Hamiltonian is
already available, not total workflow runtime. Speedup depends on the machine,
basis, and active-space size; CI checks the benchmark's output and numerical
agreement, not a wall-clock speedup threshold.

Settings
--------

``validation_tolerance``
   Absolute tolerance used to validate the orbital-basis relationship. Active
   orbital checks are evaluated after mapping the orbitals into the AO-overlap
   metric, including the projected difference between the source and target
   symmetrized AO metrics. Structural rank requirements are enforced
   independently of this setting. The tolerance does not threshold integral
   values. Supported range: ``0`` through ``1e-2``. Default: ``1e-10``.
