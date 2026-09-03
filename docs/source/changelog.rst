=========
Changelog
=========

For detailed release notes with code examples and feature walkthroughs, see the :doc:`release-notes/index` section.

Version 2.2.0
=============

See :ref:`release-v2.2.0` for full details.

- Model Context Protocol server (``qcmcp``) and command-line interface (``qc``) exposing the chemistry pipeline as structured tools, plus installable Copilot skills and MCP-configuration assets
- Remote algorithm execution and result caching: ``run()`` accepts ``remote`` and ``cache`` for serializable arguments and results, backed by a file-based request format and pluggable remote and cache backends
- Remote backend for Microsoft Discovery, installed with the new ``discovery`` extra
- Unary-iteration phase estimation (``phase_estimation`` / ``qdk_unary``) over a qubitized walk; execution currently requires the sparse-state simulator
- Binary encoding for sparse-isometry state preparation, reducing the dense amplitude-loading width when :math:`\lceil \log_2 d \rceil` is smaller than the reduced support width, in exchange for lookup ancillas and CCZ operations
- Controlled-SWAP circuit mapper (``controlled_circuit_mapper`` / ``cswap_pauli_sequence``) with a vacuum-annihilating term grouper and an additional system-sized vacuum register
- Amplitude amplification (``amplitude_amplification`` / ``qdk_base``) with a registered QPE subspace-marking oracle
- Gauge-fixing orbital localizer (``qdk_gauge_fixing``) that searches occupation-degenerate blocks for a lower mapped coefficient norm, and an active-space quantum-information orbital localizer (``qdk_active_space_qio``) that optimizes total single-orbital entropy
- ``effective_hamiltonian_constructor`` interface for registering methods that downfold a Hamiltonian onto a target orbital space; no implementation ships in this release
- Mulliken population analysis (``population_analyzer``) and ``Wavefunction.compute_s_squared()``
- Native in-process cube generation for orbitals, with no third-party quantum chemistry package required
- Native Windows wheels for x86-64 and Arm64, subject to the optional-dependency caveats in the release notes
- New tutorial: ground-state molecular energies with quantum phase estimation

Behavior changes:

- The ``state_prep`` implementation ``"sparse_isometry_gf2x"`` is renamed ``"sparse_isometry"``; the old name remains as a deprecated alias
- Orbital cube generation defaults to the native backend instead of PySCF, and default labels are zero-based (orbital ``0`` now writes ``orbital_0000.cube`` instead of ``orbital_0001.cube``)
- :term:`ECP`-adjusted nuclear charges now determine SCF electron counts, nuclear repulsion energies, and Hamiltonian core energies, changing results for systems with effective core potentials; ECP assignments are keyed by atom rather than element, and pre-2.2 cached results for these systems must be cleared
- When a stored one-particle :term:`RDM` does not match its determinant, orbital occupations are now its natural occupation numbers rather than integer mean-field occupations
- Duplicate algorithm, data-class, remote-backend, and cache-backend registrations now raise ``DuplicateRegistrationError`` and preserve the existing registration
- Custom ``DataClass`` subclasses must declare a static ``data_type_name()`` method instead of the former ``_data_type_name`` class attribute
- Supported BLAS backends are temporarily restricted to one thread inside GauXC-backed SCF, gradient, and response operations because nested BLAS threading could cause oversubscription or wrong results

Packaging and build changes:

- The new ``mcp`` extra installs the MCP server dependencies; on Windows Arm64 it is omitted from the ``all`` and ``test`` extras because ``cryptography`` has no native wheel
- The ``jupyter`` extra no longer installs the ``plugins`` extra; plugin backends require explicit opt-in
- Source builds and CI now use Libint2 2.13.1 consistently
- Fixed a BLIS/libFLAME symbol collision in Linux wheel builds

Bug fixes:

- Importing ``qdk_chemistry`` no longer emits the ``MP2NaturalOrbitalLocalizer`` deprecation message, while explicit use of ``qdk_mp2_natural_orbitals`` still warns
- Derived orbitals survive JSON round trips: ``Orbitals.from_json`` returns the concrete type instead of a sliced base ``Orbitals``
- Cache plugin discovery failures are logged instead of being swallowed
- Corrected exponent rendering in the controlled-IQPE tutorial diagram

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
- QDK Quantum Resource Estimator integration for generated circuits (``Circuit.estimate`` / ``get_qre_application``)
- Algorithm result caching and data-file migration tooling
- Windows build support with CI

Breaking changes:

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
