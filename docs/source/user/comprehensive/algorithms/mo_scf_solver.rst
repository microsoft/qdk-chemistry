MO-basis self-consistent field solver
====================================

The :class:`~qdk_chemistry.algorithms.MoScfSolver` algorithm optimizes a Hartree-Fock reference directly from a :doc:`Hamiltonian <../data/hamiltonian>` in an orthonormal molecular-orbital (MO) basis.
It does not recompute atomic-orbital (AO) integrals or require a molecular geometry.
This is useful when an imported or active-space Hamiltonian has a reference determinant that is not stationary with respect to orbital rotations.

The native implementation is registered as ``mo_scf_solver/qdk``.
It shares the molecular :doc:`SCF solver <scf_solver>` implementation: the iteration loop, convergence checks, incremental Fock updates, :term:`DIIS`, :term:`GDM` line searches and BFGS history, and in-core integral contractions.
There is no separate optimization loop and no PySCF runtime dependency.

Inputs and outputs
------------------

``run(hamiltonian, n_active_alpha_electrons, n_active_beta_electrons)`` accepts real, restricted input integrals in a common orthonormal spatial-orbital basis.
Two-electron integrals use chemists' notation :math:`g_{pqrs}=(pq|rs)`, with dense tensors flattened in row-major order.
The input tensor must have the eightfold permutation symmetry of real, spin-independent spatial-orbital integrals.
The one-body integrals and constant energy must already include any frozen-core contributions.
Electron counts refer only to the Hamiltonian's active space, not the full molecule.
The first requested number of active orbitals in each spin channel defines the initial determinant.

The result is ``(energy, ansatz)``:

* ``energy`` is the total energy in Hartree, including the input constant energy.
* ``ansatz.get_wavefunction()`` is the optimized single-determinant wavefunction.
* ``ansatz.get_hamiltonian()`` contains the one- and two-body integrals transformed into the same optimized basis.

Both inputs and outputs follow QDK/Chemistry's immutable data model.
The input Hamiltonian is not modified.
Use the returned Hamiltonian with the returned wavefunction; the original Hamiltonian remains in its original basis.

Only active orbitals are rotated.
Inactive and external orbitals remain fixed, their index sets are preserved, and an available full-space inactive Fock matrix is transformed consistently.
If the input supplies AO coefficients, the optimized coefficients are composed with them.
For :class:`~qdk_chemistry.data.ModelOrbitals` inputs, the returned coefficients describe rotations relative to the original orthonormal model basis.
When only part of the orbital space is optimized, output orbital energies are absent: active-only integrals cannot determine the updated spectator-orbital energies.

Usage
-----

.. code-block:: python

   from qdk_chemistry import algorithms
   from qdk_chemistry.data import Hamiltonian

   hamiltonian = Hamiltonian.from_file("input.hamiltonian.h5", "hdf5")
   solver = algorithms.create(
       "mo_scf_solver",
       "qdk",
       scf_type="unrestricted",
       scf_algorithm="diis_gdm",
       convergence_threshold=1e-8,
       max_iterations=300,
   )
   energy, ansatz = solver.run(hamiltonian, 5, 4)
   ansatz.to_file("optimized.ansatz.h5", "hdf5")
   ansatz.get_hamiltonian().to_fcidump_file("optimized.fcidump", 5, 4)

.. code-block:: cpp

   #include <qdk/chemistry/algorithms/mo_scf.hpp>

   auto solver = qdk::chemistry::algorithms::MoScfSolverFactory::create();
   solver->settings().set("scf_type", "unrestricted");
   solver->settings().set("scf_algorithm", "gdm");
   auto [energy, ansatz] = solver->run(hamiltonian, 5, 4);

FCIDUMP export uses the standard restricted layout for RHF/ROHF.
For UHF it uses ``IUHF=1`` and separate ``aaaa``, ``bbbb``, ``aabb``, alpha one-body, and beta one-body blocks.
Mixed-spin integrals retain all distinct bra/ket pairs: :math:`(pq|rs)_{\alpha\beta}` need not equal :math:`(rs|pq)_{\alpha\beta}`.

Spin treatment and stationarity
------------------------------

``scf_type="auto"`` chooses RHF for equal electron counts and UHF otherwise.
``"restricted"`` chooses RHF or ROHF; ROHF requires at least as many alpha electrons as beta electrons.
ROHF also supports empty closed-shell or virtual subspaces with DIIS, GDM, and the hybrid algorithm.
``"unrestricted"`` allows separate alpha and beta orbital rotations, including for equal electron counts.
An equal-spin initial density is not automatically perturbed to break spin symmetry.

RHF and UHF stationarity imply vanishing occupied-virtual Fock elements in their allowed spin channels.
ROHF has a constrained shared-orbital variational space instead.
For closed orbitals :math:`i`, singly occupied orbitals :math:`u`, and virtual orbitals :math:`a`, its conditions involve

.. math::

   F^\beta_{iu}=0,\qquad
   F^\alpha_{ua}=0,\qquad
   (F^\alpha+F^\beta)_{ia}=0.

Consequently, a converged ROHF reference does not generally have zero occupied-virtual elements in each spin Fock matrix separately.
The solver uses QDK's existing effective-ROHF convergence criterion and constrained GDM gradient, not an unrestricted Brillouin test.
SCF convergence alone does not establish stability or a global energy minimum.
After convergence, all three algorithms canonicalize within equal-occupation subspaces to preserve the optimized determinant, including a non-Aufbau stationary reference.
Output orbital energies come from the physical Fock matrix; any level shift used during DIIS is excluded.

Settings and limitations
------------------------

The native AO and MO solvers share the following iteration controls:

* ``scf_algorithm``: ``"diis"``, ``"gdm"``, ``"diis_gdm"``, or ``"auto"`` (default).
  ``"auto"`` honors ``enable_gdm``: true selects the hybrid, false selects DIIS.
* ``convergence_threshold`` and ``max_iterations``: the same normalized orbital-gradient and density convergence checks and iteration limit as molecular SCF.
* ``level_shift``, ``energy_thresh_diis_switch``, ``gdm_max_diis_iteration``, ``gdm_bfgs_history_size_limit``, and ``fock_reset_steps``.

See :ref:`scf-convergence-algorithms` for defaults and algorithm details.
Failure to converge raises an exception; no partially optimized result is returned.

The current implementation supports Hartree-Fock only (``method="hf"``).
It materializes full four-index integrals with :math:`O(n_\mathrm{active}^4)` storage and does not preserve a sparse or factorized input representation.
Spin-dependent input Hamiltonians, DFT, nuclear properties, and multi-rank MPI are not supported.
UHF output is supported even though the input basis must be common to both spins.

This algorithm is separate from ``OrbitalLocalizer`` because it needs Hamiltonian integrals and returns a consistently transformed Hamiltonian as well as orbitals.
The localizer's wavefunction-only input and output contract is unchanged.
