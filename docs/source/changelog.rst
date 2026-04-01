=========
Changelog
=========

Version 1.1.0
=============

Highlights
----------

- **Q#-native circuit architecture** — circuits are now built and composed as native Q# operations, enabling a streamlined end-to-end QPE code path with lazy QIR compilation.
- **Model Hamiltonians** — native construction of fermionic and spin lattice Hamiltonians (Hückel, Fermi-Hubbard, PPP, Ising, Heisenberg) on arbitrary lattice geometries.
- **Arbitrary-order Trotter-Suzuki** — product formulas of any even order with accuracy-aware step sizing and commutator-based error bounds.
- **Native ROHF** — DIIS-accelerated restricted open-shell Hartree-Fock in the QDK SCF solver.
- **Cholesky AO→MO transformation** — pivoted Cholesky decomposition for two-electron integrals, enabling treatment of larger molecular systems.
- **OpenFermion plugin** — Jordan-Wigner, Bravyi-Kitaev, and symmetry-conserving Bravyi-Kitaev mappings via OpenFermion.


Model Hamiltonians
^^^^^^^^^^^^^^^^^^

Running quantum phase estimation on a full *ab initio* molecular Hamiltonian is general and accurate, but the number of qubits and gates grows with the orbital basis set, making even modest molecules expensive to simulate.
An alternative is to **map the essential physics onto a simpler model Hamiltonian** that captures the key interactions with far fewer degrees of freedom.
New model-Hamiltonian constructors make it possible to build these Hamiltonians directly, without quantum-chemistry input files or external electronic-structure codes.

**Fermionic models** — Hückel, Hubbard, and Pariser-Parr-Pople (PPP) — return a second-quantized ``Hamiltonian``:

.. code-block:: python

   from qdk_chemistry.data import LatticeGraph
   from qdk_chemistry.utils.model_hamiltonians import (
       create_huckel_hamiltonian,
       create_hubbard_hamiltonian,
       create_ppp_hamiltonian,
       ohno_potential,
   )

   # 8-site open chain with tight-binding (Hückel) hopping
   chain = LatticeGraph.chain(8)
   ham = create_huckel_hamiltonian(chain, epsilon=-0.5, t=1.0)

   # Add on-site Coulomb repulsion (Hubbard)
   ham = create_hubbard_hamiltonian(chain, epsilon=-0.5, t=1.0, U=4.0)

   # 4-site periodic ring with intersite Coulomb via Ohno potential (PPP)
   ring = LatticeGraph.chain(4, periodic=True)
   V = ohno_potential(ring, U=11.26, R=1.4, nearest_neighbor_only=True)
   ham = create_ppp_hamiltonian(ring, epsilon=0.0, t=2.4, U=11.26, V=V, z=1.0)

All parameters (``epsilon``, ``t``, ``U``, ``V``, ``z``) accept either a scalar or per-site/per-bond arrays for inhomogeneous lattices.

**Spin models** — Ising and Heisenberg — return a ``QubitHamiltonian`` directly:

.. code-block:: python

   from qdk_chemistry.utils.model_hamiltonians import (
       create_ising_hamiltonian,
       create_heisenberg_hamiltonian,
   )

   # Transverse-field Ising model (ZZ coupling + transverse X field)
   ising = create_ising_hamiltonian(chain, j=1.0, h=0.5)

   # Isotropic Heisenberg (XXX) model
   heisenberg = create_heisenberg_hamiltonian(chain, jx=1.0, jy=1.0, jz=1.0)

**Lattice geometries.** All constructors take a ``LatticeGraph`` that defines the site connectivity. Common topologies are built in:

.. code-block:: python

   LatticeGraph.chain(8, periodic=True)     # ring
   LatticeGraph.square(4, 4)                # 2D grid
   LatticeGraph.triangular(3, 3)
   LatticeGraph.honeycomb(3, 3)
   LatticeGraph.kagome(3, 3)

   # Or from an adjacency matrix / edge dict
   LatticeGraph({(0, 1): 1.0, (1, 2): 0.5}, num_sites=3)

See `examples/extended_hubbard.ipynb <https://github.com/microsoft/qdk-chemistry/blob/v1.1.0/examples/extended_hubbard.ipynb>`_ for a complete walkthrough that models cyclobutadiene with the extended Hubbard (PPP) model and runs QPE on it.
The interoperability examples under ``examples/interoperability/`` also demonstrate constructing model Hamiltonians directly without relying on the built-in infrastructure.


Q#-Native Circuit Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Circuit`` class now composes quantum operations as native Q# callables (exposed in `qdk>=1.27.0`). This enables:

- End-to-end QPE without intermediate format conversions
- Lazy compilation to QIR only at execution time
- Direct access to Q#'s resource estimator on any ``Circuit``
- A leaner default install with framework-specific dependencies as optional extras

``Circuit`` can hold any combination of:

- ``qsharp_op`` — a callable Q# operation
- ``qsharp_factory`` — a lazy ``QsharpFactoryData(program, parameter)`` that compiles on demand
- ``qasm`` / ``qir`` — for interoperability with external tools

The QPE pipeline composes Q# callables end-to-end — state preparation, controlled time evolution, and phase estimation are all native Q# operations:

.. code-block:: python

   from qdk_chemistry import create

   state_prep = create("state_prep", "sparse_isometry")
   circuit = state_prep.run(wavefunction, hamiltonian, qubit_mapper)

   # Inspect the Q# circuit
   circuit.get_qsharp_circuit(prune_classical_qubits=True)

   # Compile to QIR for external use
   qir = circuit.get_qir()

   # Use Q#'s resource estimator directly
   import qsharp
   qsharp.estimate(
       circuit._qsharp_factory.program, None,
       *circuit._qsharp_factory.parameter.values(),
   )

**Lightweight core dependencies.** Framework-specific dependencies are now isolated as optional extras, keeping the core install lean:

.. code-block:: bash

   pip install qdk-chemistry                        # core (Q#-native)
   pip install 'qdk-chemistry[qiskit-extras]'       # + Qiskit interop
   pip install 'qdk-chemistry[openfermion-extras]'   # + OpenFermion interop

See `examples/qpe_stretched_n2.ipynb <https://github.com/microsoft/qdk-chemistry/blob/v1.1.0/examples/qpe_stretched_n2.ipynb>`_ and `examples/state_prep_energy.ipynb <https://github.com/microsoft/qdk-chemistry/blob/v1.1.0/examples/state_prep_energy.ipynb>`_ for worked end-to-end QPE notebooks.


Arbitrary-Order Trotter-Suzuki
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Time-evolution builders now support product formulas of any even order via recursive Suzuki composition [Suzuki1992]_.

.. code-block:: python

   from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter

   builder = Trotter(order=4, target_accuracy=1e-3, error_bound="commutator")
   unitary = builder.run(qubit_hamiltonian, time=1.0)

When both ``num_divisions`` and ``target_accuracy`` are specified, the builder uses whichever requires more Trotter steps. The ``error_bound`` parameter selects between two bounds:

- ``"naive"`` — triangle-inequality bound: :math:`N = \lceil (\sum_j |\alpha_j|)^2 t^2 / \epsilon \rceil`
- ``"commutator"`` — tighter commutator-scaling bound from [Childs2021]_: :math:`N = \lceil \frac{t^2}{2\epsilon} \sum_{j<k} \lVert [\alpha_j P_j, \alpha_k P_k] \rVert \rceil`

**Error-bound utilities** are available for manual step-count analysis:

.. code-block:: python

   from qdk_chemistry.algorithms.time_evolution.builder.trotter_error import (
       trotter_steps_naive,
       trotter_steps_commutator,
       commutator_bound_first_order,
       commutator_bound_second_order,
   )

   n_steps = trotter_steps_commutator(hamiltonian, time=1.0, epsilon=1e-3, order=4)

.. [Childs2021] Childs, A. M., et al. "Theory of Trotter Error with Commutator Scaling." *Physical Review X* 11.1 (2021): 011020. `arXiv:1912.08854 <https://arxiv.org/abs/1912.08854>`_

.. [Suzuki1992] Suzuki, M. "General theory of higher-order decomposition of exponential operators and symplectic integrators." *Physics Letters A* 165.5-6 (1992): 387-395.

See `examples/extended_hubbard.ipynb <https://github.com/microsoft/qdk-chemistry/blob/v1.1.0/examples/extended_hubbard.ipynb>`_ for a worked example using higher-order Trotterization on a model Hamiltonian.


Native ROHF in QDK SCF
^^^^^^^^^^^^^^^^^^^^^^^

The QDK SCF solver now supports restricted open-shell Hartree-Fock (ROHF) with DIIS acceleration for systems with unpaired electrons. Previously, open-shell calculations required an unrestricted formulation through the QDK SCF code path.

ROHF is automatically selected when specifying ``scf_type="restricted"`` on a system with multiplicity > 1. The DIIS extrapolation works with the ROHF-effective Fock matrix and total density matrix.

.. code-block:: python

   scf = create("scf", "qdk_scf")
   scf.settings().set("method", "hf")
   scf.settings().set("scf_type", "restricted")
   energy, wavefunction = scf.run(structure, charge=0, multiplicity=2, basis="sto-3g")


Cholesky-Based AO-to-MO Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new Hamiltonian constructor uses pivoted Cholesky decomposition of the AO electron-repulsion integrals (ERIs), reducing memory and compute cost compared to full four-center ERI evaluation. This enables treatment of active space methods for larger molecular systems.

.. code-block:: python

   constructor = create("hamiltonian_constructor", "qdk_cholesky")
   constructor.settings().set("cholesky_tolerance", 1e-8)
   constructor.settings().set("eri_threshold", 1e-12)
   constructor.settings().set("store_cholesky_vectors", True)
   hamiltonian = constructor.run(orbitals)

The implementation includes Schwarz-based shell-quartet screening, stability guards for near-singular decompositions, and diagnostic logging for convergence monitoring.


OpenFermion Plugin
^^^^^^^^^^^^^^^^^^

A new plugin provides qubit mappings backed by `OpenFermion <https://quantumai.google/openfermion>`_:

.. code-block:: bash

   pip install 'qdk-chemistry[openfermion-extras]'

.. code-block:: python

   from qdk_chemistry import create
   from qdk_chemistry.data import Symmetries

   mapper = create("qubit_mapper", "openfermion", encoding="jordan-wigner")
   qh = mapper.run(hamiltonian)

   # Symmetry-conserving Bravyi-Kitaev (reduces qubit count by 2)
   sym = Symmetries(n_alpha=2, n_beta=2)
   mapper = create("qubit_mapper", "openfermion",
                    encoding="symmetry-conserving-bravyi-kitaev")
   qh = mapper.run(hamiltonian, sym)

Available encodings: ``jordan-wigner``, ``bravyi-kitaev``, ``bravyi-kitaev-tree``, ``symmetry-conserving-bravyi-kitaev``.
Please see the OpenFermion documentation for more information

See `examples/interoperability/openFermion/molecular_hamiltonian_jordan_wigner.py <https://github.com/microsoft/qdk-chemistry/blob/v1.1.0/examples/interoperability/openFermion/molecular_hamiltonian_jordan_wigner.py>`_ for a worked example.


MACIS Improvements
^^^^^^^^^^^^^^^^^^

- **Expanded orbital limits.** The selected-CI dispatch now supports active spaces up to **2048 orbitals** (up from 127 in v1.0). The internal sCI bit-string representation was widened to 255 orbitals.
- **Single-orbital entropies and mutual information.** MACIS calculators can now compute S1 entropies and two-orbital mutual information for active-space analysis:

  .. code-block:: python

     mc = create("multi_configuration", "qdk_asci")
     mc.settings().set("calculate_single_orbital_entropies", True)
     mc.settings().set("calculate_mutual_information", True)
     energy, wavefunction = mc.run(hamiltonian)

- **ASCI refinement stability.** The ASCI refinement loop now detects sign-flipping energy oscillations and terminates gracefully instead of looping indefinitely.
- **Unrestricted-orbital guard.** MACIS calculators (CAS, ASCI, PMC) now raise a clear error if passed an unrestricted Hamiltonian, which is not supported by the MACIS backend.


Orbital Localization
^^^^^^^^^^^^^^^^^^^^

- **One-shot VVHV localization.** The valence-virtual hard-virtual (VVHV) localizer was rewritten to use a proper one-shot localization for proto hard-virtuals, replacing the previous canonicalization-based approach. This improves numerical stability with better weighted orthogonalization and singular-value diagnostics.
- **Bug fix** in the proto-HV construction that could produce incorrect localized orbitals in certain basis sets.


Other Improvements
^^^^^^^^^^^^^^^^^^

- **Refactored energy estimator.** ``QdkEnergyEstimator`` now takes a single ``QubitHamiltonian``, groups commuting terms internally via ``group_commuting(qubit_wise=True)``, and delegates execution to a ``CircuitExecutor``:

  .. code-block:: python

     estimator = create("energy_estimator", algorithm_name="qdk")
     energy_result, measurement_data = estimator.run(
         circuit=circuit,
         qubit_hamiltonian=qubit_hamiltonian,
         circuit_executor=circuit_executor,
         total_shots=1_500_000,
     )

- **CI coefficients for CC and MP2.** Coupled-cluster and MP2 wavefunctions now expose a CI-like expansion with ``get_coefficients()``, ``get_coefficient(...)``, and ``get_active_determinants()`` methods, enabling determinant-level analysis of post-HF wavefunctions.
- **Configurable ERI threshold.** The ``eri_threshold`` setting (default ``1e-12``) controls shell-quartet screening in Hamiltonian construction. A separate ``shell_pair_threshold`` is available for SCF pre-screening.
- **Wavefunction file I/O with** ``pathlib.Path``. Python bindings for ``to_file`` / ``from_file``, ``to_json_file`` / ``from_json_file``, and ``to_hdf5_file`` / ``from_hdf5_file`` now accept ``pathlib.Path`` objects in addition to strings.
- **Dedicated** ``[jupyter]`` **install extra.** Notebook dependencies are now isolated from the core install: ``pip install 'qdk-chemistry[jupyter]'``.
- **``Symmetries`` data class.** A new immutable container for active-space quantum numbers, used by qubit mappers that exploit particle-number or spin symmetries (e.g., symmetry-conserving Bravyi-Kitaev):

  .. code-block:: python

     from qdk_chemistry.data import Symmetries

     sym = Symmetries(n_alpha=3, n_beta=2)
     sym.n_particles    # 5
     sym.sz             # 0.5
     sym.spin_multiplicity  # 2

     # Or construct from an existing wavefunction/ansatz
     sym = Symmetries.from_wavefunction(wfn)
     sym = Symmetries.from_ansatz(ansatz)


Bug Fixes
^^^^^^^^^

- Fixed inactive Fock matrix dimension validation — matrices are now checked for full NMO × NMO shape before Hamiltonian construction, preventing silent size mismatches.
- Fixed active-space MP2 orbital energies — denominators now use active-space orbital energies rather than full-system indices for both restricted and unrestricted MP2.
- Fixed JSON serialization of ``TimeEvolutionContainer`` — Pauli-term keys are now correctly serialized as strings and deserialized back to integer qubit indices.
- Fixed SCF logging suppression — the solver no longer silences the global logger during execution, which was hiding unrelated log messages.
- Aligned logger flush policy with the configured log level.


Infrastructure and Packaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **macOS ARM64 wheels.** Pre-built wheels are now published for macOS Apple Silicon alongside Linux x86_64 and Linux aarch64.
- **Modular install extras.** Framework-specific dependencies are now optional extras (``qiskit-extras``, ``openfermion-extras``, ``jupyter``), keeping the default install lightweight.
- **Telemetry sanitization.** Tests and CI builds now force telemetry off (``QSHARP_PYTHON_TELEMETRY=false``) to prevent accidental data collection during development.
- **Centralized** ``VERSION`` **file.** A single ``VERSION`` file at the repo root is the source of truth for CMake, Python packaging, and CI. Format: ``X.Y.Z`` or ``X.Y.Z.T``.
- Broadened CI/CD coverage and release pipeline hardening.
- Documentation restructured with clearer navigation and a layered quick-reference style.
- Added `qdk-chemistry-data <https://github.com/microsoft/qdk-chemistry-data>`_ companion repo references for curated datasets and benchmark materials.

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
