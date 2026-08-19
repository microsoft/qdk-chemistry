=========
Changelog
=========

For detailed release notes with code examples and feature walkthroughs, see the :doc:`release-notes/index` section.

Version 2.1.0
=============

See :ref:`release-v2.1.0` for full details.

- Geometry optimization as a first-class algorithm type (``geometry_optimizer``), with a geomeTRIC-backed implementation driven by the existing nuclear derivative calculators
- Vendored Q# utilities load into a private ``qdk.Context`` instead of the global interpreter, so importing ``qdk_chemistry`` no longer changes the caller's target profile
- Qiskit extras supported on Python 3.14

Bug fixes:

- Corrected the iterative phase estimation phase convention: for :math:`U = e^{-iHt}` the energy is recovered as :math:`E = -\theta / t`, the feedback phase is negated, and ``QpeResult.bits_msb_first`` is now genuinely most-significant-bit first
- ``QpeResult`` files written by v1 are now upgraded by ``python -m qdk_chemistry.migrate``

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
- On-demand robust phase-estimation circuit builder with round, draw, seed, and multiplicity metadata for shared QRE and execution workflows
- Robust phase estimation derives independent reproducible executor seeds per round, draw, and measurement basis, preventing repeated one-shot simulator streams in randomized evolution and X/Y stream reuse in deterministic evolution
- QDK Quantum Resource Estimator integration for generated circuits (``Circuit.estimate`` / ``get_qre_application``)
- Algorithm result caching and data-file migration tooling
- Windows build support with CI

Breaking changes:

- Robust phase estimation with Trotter and partially randomized unitary builders now defaults each to ``epsilon_rpe = target_accuracy`` and an independent dimensionless ``epsilon_unitary`` of ``0.85``. Partially randomized evolution maps that full-unitary tolerance through its quadrature split, restoring inverse-square random-sample scaling; explicit fractional and paired budgets remain supported. Trotter callers must replace ``unitary_accuracy_fraction`` or explicit ``epsilon_rpe`` with ``target_accuracy`` plus an optional positive ``epsilon_unitary``.
- The automatic partially randomized split now minimizes the implemented pre-merge rotation count, including integer sample sizing, the configured per-step sample floor, and all-random and all-deterministic limits.
- The partially randomized ``num_random_samples`` default is now ``1``; accuracy-aware mode increases it automatically from the Campbell bound, while callers can still request a larger floor explicitly.
- Wavefunction containers consolidated from five classes into two: ``SlaterDeterminantContainer``, ``CasWavefunctionContainer``, and ``SciWavefunctionContainer`` (structurally identical apart from a type tag) merge into ``StateVectorContainer``, distinguished by a stored sector; ``MP2Container`` and ``CoupledClusterContainer`` merge into ``AmplitudeContainer``, distinguished by an ``AmplitudeType`` tag. ``Wavefunction.get_container_type()`` now returns ``"state_vector"`` / ``"amplitude"`` (old names remain as deprecated aliases)
- Serialization schema bumped from ``0.1.0`` to ``0.2.0`` for the ``Orbitals`` / ``ModelOrbitals``, ``CanonicalFourCenterHamiltonianContainer`` / ``SparseHamiltonianContainer`` / ``CholeskyHamiltonianContainer``, ``Wavefunction`` / ``StateVectorContainer``, ``QpeResult``, and ``UnitaryRepresentation`` data classes; ``AmplitudeContainer`` and the top-level ``Hamiltonian`` are unchanged. Upgrade supported files with ``python -m qdk_chemistry.migrate``; v1 ``TimeEvolutionUnitary`` files must be regenerated because they do not store the scale required by v2
- Expectation estimator no longer auto-groups terms
- Qubit mapper takes a ``MajoranaMapping`` instead of an ``encoding`` string
- Silent semantic change: the ``Wavefunction`` accessor ``get_active_two_rdm_spin_dependent()`` now returns blocks as ``(aaaa, aabb, bbbb)`` (was ``(aabb, aaaa, bbbb)``), with the ``aabb`` block in alpha-alpha-beta-beta index order (was alpha-beta-alpha-beta), to match the two-electron integral block order in ``Hamiltonian``; the ``Wavefunction`` constructor takes the same new order. Positional unpacking reads incorrect data until updated
- ``Configuration`` string and bitset constructors replaced by explicit factories
- Phase estimation now uses the ``"qdk_iterative"`` and ``"qdk_standard"`` variants; ``num_bits``, unitary-builder, evolution-time, and controlled-mapper settings belong to the nested ``qpe_circuit_builder`` reference, and the executor is configured through ``circuit_executor``. The v1 ``"qiskit_standard"`` variant is composed from ``"qdk_standard"`` with Qiskit builder and executor references
- Renames (deprecated aliases retained): ``QubitHamiltonian`` to ``QubitOperator``, ``EnergyEstimator`` to ``ExpectationEstimator``, ``TimeEvolutionUnitary`` to ``UnitaryRepresentation``, and ``TimeEvolutionUnitaryContainer`` to ``UnitaryContainer``
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
