Effective Hamiltonian construction
==================================

The ``EffectiveHamiltonianConstructor`` algorithm defines a common interface
for downfolding a Hamiltonian from a larger orbital window into a smaller
target space.

Problem definition
------------------

Let the orbitals be partitioned as ``W = P union Q``, where ``P`` is the
target space retained in the effective problem and ``Q`` is the external space
that is integrated out.  A constructor receives:

Reference wavefunction
   A :doc:`Wavefunction <../data/wavefunction>` that defines the reference
   state used by the downfolding method.

Input Hamiltonian
   A :doc:`Hamiltonian <../data/hamiltonian>` expressed over the complete
   window ``W``.

Target-space indices
   A :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet` containing
   the indices of ``P`` within the active space of ``W``.

The result is a :doc:`Hamiltonian <../data/hamiltonian>` whose active space is
``P``. The interface does not prescribe the approximation used to account for
``Q``; that behavior and any method-specific diagnostics belong to the
concrete implementation.

.. important::

   QDK/Chemistry currently provides the interface but no built-in
   effective-Hamiltonian implementation or default constructor.  Pass the name
   of a registered plugin or custom implementation to ``registry.create``.


Related documentation
---------------------

- :doc:`Hamiltonian construction <hamiltonian_constructor>`: Builds the input
  Hamiltonian over the orbital window.
- :doc:`Active-space selection <active_space>`: Identifies orbital subspaces
  for correlated calculations.
- :doc:`Factory pattern <factory_pattern>`: Discovers, registers, and creates
  algorithm implementations.
