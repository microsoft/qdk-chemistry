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
  :class:`~qdk_chemistry.data.Wavefunction` containers was bumped from ``0.1.0`` to ``0.2.0`` to
  reflect the switch to ``SymmetryBlockedTensor``-backed storage. Files
  written by earlier versions are **not** loaded by this release; re-generate
  them with the current version. Backward-compatible loading of ``0.1.0`` files
  is planned for a future release.

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
