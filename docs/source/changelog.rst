=========
Changelog
=========

For detailed release notes with code examples and feature walkthroughs, see the :doc:`release-notes/index` section.

Version 2.0.0
=============

Work in progress

Breaking changes
-----------------

- ``QdkEnergyEstimator`` no longer auto-groups terms by qubit-wise commutativity.
  Pre-group with ``create("term_grouper", "qubit_wise_commuting")`` to restore
  the previous behavior.

- The on-disk serialization format for :class:`~qdk_chemistry.data.Orbitals`,
  :class:`~qdk_chemistry.data.ModelOrbitals`, :class:`~qdk_chemistry.data.Hamiltonian` containers, and
  the :class:`~qdk_chemistry.data.StateVectorContainer` wavefunction container was bumped from
  ``0.1.0`` to ``0.2.0`` to reflect the switch to ``SymmetryBlockedTensor``-backed storage
  (:class:`~qdk_chemistry.data.AmplitudeContainer` is unchanged at ``0.1.0``). Direct loading
  rejects files written by earlier versions; upgrade them in place with the shipped converter
  (``python -m qdk_chemistry.migrate old.h5 new.h5``; see :doc:`user/migrating-data-files`) or
  re-generate them. Backward-compatible direct loading of ``0.1.0`` files is planned for a future
  release.

- Wavefunction containers were consolidated. The single-determinant,
  complete-active-space, and selected-configuration-interaction containers are
  now a single :class:`~qdk_chemistry.data.StateVectorContainer`, and the
  Møller-Plesset and coupled-cluster containers are now a single
  :class:`~qdk_chemistry.data.AmplitudeContainer`. The former Python names
  (``SlaterDeterminantContainer``, ``CasWavefunctionContainer``,
  ``SciWavefunctionContainer``, ``MP2Container``, ``CoupledClusterContainer``)
  remain importable as deprecated aliases (see :ref:`Deprecations <v2-deprecations>`).
  Accordingly, :meth:`~qdk_chemistry.data.Wavefunction.get_container_type`
  now returns ``"state_vector"`` or ``"amplitude"`` (was ``"sd"`` / ``"cas"`` /
  ``"sci"`` / ``"mp2"`` / ``"cc"``).

- **Silent semantic change:** the spin-resolved two-particle RDM accessor
  ``get_active_two_rdm_spin_dependent()`` now returns the blocks in the order
  ``(aaaa, aabb, bbbb)`` (was ``(aabb, aaaa, bbbb)``), and the ``aabb`` block is
  now defined in *alpha-alpha-beta-beta* index order (was
  *alpha-beta-alpha-beta*). Code that unpacks this tuple positionally will read
  silently incorrect data until updated.

- :class:`~qdk_chemistry.data.Configuration` string and bitset constructors were
  replaced by explicit factories:
  :meth:`~qdk_chemistry.data.Configuration.from_spin_half_string`,
  :meth:`~qdk_chemistry.data.Configuration.from_bitstring`, and
  ``from_spin_half_bitset``.

- The qubit mapper no longer accepts an ``encoding`` string. Construct a
  :class:`~qdk_chemistry.data.MajoranaMapping` and pass it to ``run()``
  (``mapper.run(hamiltonian, mapping)``) instead of
  ``QdkQubitMapper(encoding="jordan-wigner")``.

- The ``TimeEvolutionUnitary`` family was renamed to the ``Unitary`` family
  (:class:`~qdk_chemistry.data.UnitaryRepresentation`,
  :class:`~qdk_chemistry.data.UnitaryContainer`); the old names remain importable
  as deprecated aliases.

- C++ only: the FCIDUMP writer (``Hamiltonian::to_fcidump_file``), the Cholesky
  container's base class and dense constructor, and the
  ``ModelOrbitals(size_t, bool)`` constructor changed. See
  :ref:`What's New in Version 2.0 <release-v2.0.0>` for details.

.. _v2-deprecations:

Deprecations
------------

The v1 :class:`~qdk_chemistry.data.Orbitals` data accessors are retained as
tested facades over the new ``SymmetryBlocked`` storage, but now emit a
``DeprecationWarning``. Migrate to the v2 accessors:

- ``get_coefficients()`` / ``get_coefficients_alpha()`` /
  ``get_coefficients_beta()`` → :meth:`~qdk_chemistry.data.Orbitals.coefficients`
  (a :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2`).
- ``get_energies()`` / ``get_energies_alpha()`` / ``get_energies_beta()`` →
  :meth:`~qdk_chemistry.data.Orbitals.energies`
  (a :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank1`).
- ``get_active_space_indices()`` / ``get_inactive_space_indices()`` →
  :meth:`~qdk_chemistry.data.Orbitals.active_indices` /
  :meth:`~qdk_chemistry.data.Orbitals.inactive_indices`
  (each a :class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet`), with
  :meth:`~qdk_chemistry.data.Orbitals.num_active_orbitals` /
  :meth:`~qdk_chemistry.data.Orbitals.num_inactive_orbitals` for sizes.

Importing a removed or renamed ``qdk_chemistry.data`` name (the wavefunction and
unitary aliases above, as well as ``EncodingMismatchError`` and
``validate_encoding_compatibility``) resolves to its v2 replacement and emits a
``DeprecationWarning``. See :ref:`What's New in Version 2.0 <release-v2.0.0>` for
a complete migration guide.

Version 1.1.0
=============

See :ref:`release-v1.1.0` for full details.

- Q#-native circuit architecture with lazy QIR compilation
- Model Hamiltonians (Hückel, Hubbard, PPP, Ising, Heisenberg)
- Arbitrary-order Trotter-Suzuki product formulas
- Native ROHF with DIIS acceleration
- Cholesky-based AO→MO integral transformation
- OpenFermion qubit-mapping plugin
- MACIS active-space expansion to 2048 orbitals
- One-shot VVHV orbital localization

Version 1.0.2
=============

- Make qiskit-aer and qiskit-nature optional dependencies
- Loosen matplotlib version requirement to >=3.10.0
- Fixed installation instructions for Ubuntu compatibility
- Improved iQPE demo notebook

Version 1.0.1
=============

- Added support for Python 3.10
- Enhanced INSTALL.md with clearer installation steps

Version 1.0.0
=============

- Initial release of QDK/Chemistry
