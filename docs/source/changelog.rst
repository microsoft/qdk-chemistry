=========
Changelog
=========

For detailed release notes with code examples and feature walkthroughs, see the :doc:`release-notes/index` section.

Version 2.0.0
=============

See :ref:`release-v2.0.0` for full details and migration guidance.

- Symmetry-blocked storage unifying restricted and unrestricted orbital, integral, and index data
- Consolidated wavefunction containers (``StateVectorContainer``, ``AmplitudeContainer``)
- Nuclear gradients and finite-difference Hessians
- Stabilized SCF and ROHF geometric direct minimization
- Block-encoding/LCU qubitization, Hadamard test, and Zassenhaus product formulas
- Real-time dynamics for driven, time-dependent Hamiltonians
- Fermion-to-qubit mapping carried as data (``MajoranaMapping``), including a Verstraete-Cirac encoding
- Explicit Pauli term grouping and generalized expectation estimation
- Composable standard and iterative phase-estimation circuit builders
- Algorithm result caching and data-file migration tooling
- Windows build support with CI

Breaking changes:

- Wavefunction containers consolidated; ``get_container_type()`` now returns ``"state_vector"`` / ``"amplitude"``
- Serialization version bumped for orbitals, Hamiltonian containers, wavefunction, and unitary schemas
- Expectation estimator no longer auto-groups terms
- Qubit mapper takes a ``MajoranaMapping`` instead of an ``encoding`` string
- ``get_active_two_rdm_spin_dependent()`` block order changed silently
- ``Configuration`` string and bitset constructors replaced by explicit factories
- Renames (deprecated aliases retained): ``QubitHamiltonian`` to ``QubitOperator``, ``EnergyEstimator`` to ``ExpectationEstimator``, the ``TimeEvolutionUnitary`` to ``Unitary`` family
- v1 ``Orbitals`` dense accessors deprecated in favor of the symmetry-blocked accessors
- C++ only: FCIDUMP writer, Cholesky container, and ``ModelOrbitals`` constructor changes

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
