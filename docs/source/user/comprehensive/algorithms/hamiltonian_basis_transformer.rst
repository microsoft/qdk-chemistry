Hamiltonian basis transformation
================================

The :class:`~qdk_chemistry.algorithms.HamiltonianBasisTransformer` algorithm
re-expresses an existing :class:`~qdk_chemistry.data.Hamiltonian` in a supplied
orbital basis without modifying the source Hamiltonian.

The native ``"qdk"`` implementation supports real, restricted, spin-only
:class:`~qdk_chemistry.data.CholeskyHamiltonianContainer` data. It reuses the
stored three-center factors, avoiding another AO integral evaluation and
Cholesky decomposition.

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import create

      transformer = create("hamiltonian_basis_transformer")
      no_hamiltonian = transformer.run(source_hamiltonian, no_orbitals)

.. tab:: C++ API

   .. code-block:: cpp

      auto transformer = HamiltonianBasisTransformerFactory::create("qdk");
      auto no_hamiltonian =
          transformer->run(source_hamiltonian, no_orbitals);

The source and target orbitals must have the same AO basis, overlap matrix,
active/inactive index sets, and molecular orbitals outside the active space.
The active columns may differ by an orthogonal transformation. Additional
spatial symmetry labels and unrestricted Hamiltonians are not supported.

For each Cholesky factor :math:`L_Q` and recovered active-space rotation
:math:`U`, the implementation evaluates

.. math::

   h' = U^T h U, \qquad L'_Q = U^T L_Q U.

The corresponding full-orbital rotation is applied to the inactive Fock
matrix, while the core energy is unchanged. Optional AO Cholesky vectors remain
on the unchanged source Hamiltonian and are omitted from the returned
Hamiltonian to avoid copying this potentially large cache.

Settings
--------

``validation_tolerance``
   Absolute tolerance used to validate the orbital-basis relationship. It does
   not threshold integral values. Default: ``1e-10``.
