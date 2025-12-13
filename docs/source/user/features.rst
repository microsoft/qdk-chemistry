Features, methods and dependencies
==================================

This document provides an overview of QDK/Chemistry's features, supported methods, and software dependencies.

.. contents:: On This Page
   :local:
   :depth: 2

Features Overview
-----------------

.. todo:: Add a high-level summary of QDK/Chemistry's core features and design philosophy.


Supported Methods
-----------------

Classical Quantum Chemistry Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Self-Consistent Field (SCF)
"""""""""""""""""""""""""""

QDK/Chemistry provides access to a variety of robust, high-performance implementations mean-field electronic structure methods that produce optimized molecular orbitals and reference energies. In particular, the following SCF types are supported:

**Hartree-Fock (HF):**
  - Restricted (RHF), Unrestricted (UHF), Restricted Open-Shell (ROHF)

**Density Functional Theory (DFT):**
  - Kohn-Sham methods: RKS, UKS, ROKS

See :doc:`comprehensive/algorithms/scf_solver` for further details about available SCF methods and implementations.

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Geometric Direct Minimization (GDM)**: The QDK SCF Solver implements the GDM algorithm :cite:`VanVoorhis2002` for robust and efficient SCF convergence of challenging systems. This methods is particularly helpful for open-shell and small gap systems, which are of high interest in quantum computing applications, yet challenging for many SCF solvers.

- **Stability Analysis and Reoptimization**: Challenging SCF problems often converge to local minima that do not represent the true mean-field ground state. QDK/Chemistry includes automated :doc:`comprehensive/algorithms/stability_checker` :cite:`Schlegel1991` tools to identify, perturb and reoptimize unstable solutions, helping users obtain physically meaningful reference states for subsequent calculations.


Orbital Localization
""""""""""""""""""""

The canonical orbitals produced by SCF calculations are typically delocalized over the entire molecule, which can complicate chemical interpretation and slow the convergence of post-SCF correlation methods. QDK/Chemistry provides several classes of orbital localization techniques to yield specialized representations which accelerate the convergence of correlation methods and enhance chemical insight:

- **Optimized Based Localizations**: The vast majority of orbitals localization methods fall into this category, where a cost function is defined and minimized to yield localized orbitals. See :cite:`Lehtola2013` for a discussion of this class of methods. QDK/Chemistry supports, either through our :ref:`native implementations <localizer-qdk-pipek-mezey>` or via :ref:`integration with external libraries <localizer-pyscf-multi>`, several popular choices of cost functions, including: **Pipek-Mezey** :cite:`Pipek1989`, **Foster-Boys** :cite:`Foster1960`, and **Edmiston-Ruedenberg** :cite:`Edmiston1963`.

- **Natural Orbital Based Methods**: The notion of locality is not limited to spatial localtion, and can be extened to notions which minimize quantities such as entanglement, as well as methods which optimize for wavefunction sparsity. Given a :doc:`multi-configurational wavefunction <comprehensive/algorithms/mc_calculator>`, QDK/Chemistry offers methods to compute **Natural Orbitals** :cite:`Lowdin1956`, which can be particularly useful for :doc:`active space selection <comprehensive/algorithms/active_space>`.

See :doc:`comprehensive/algorithms/localizer` for further details about available orbital localization methods and implementations.


Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Valence Virtual--Hard Virtual (VVHV) Orbital Localization**: Localization of molecular orbitals expressed in near-complete :doc:`./comprehensive/data/basis_set` is numerically ill-posed and challenging for most localizers. This can lead to orbitals which do not vary smoothly with molecular geometry, numerically unstable results, and reproduction difficulties on various architectures and compute environments. Do address this problem. QDK/Chemistry includes an implementation of orbitals locations withing the VVHV separation :cite:`Subotnik2005` (and subsequent improvements to the numerical procedure described in :cite:`Wang2025`), which separates orbitals into valence, virtual, and hard virtual categories for more numerically stable treatments. This can be particularly useful for selecting consistent active spaces across molecular geometries.

Active Space Selection
""""""""""""""""""""""

Automatically identify strongly correlated orbitals for multi-reference calculations.

**Automated approaches:**
  - **AutoCAS** :cite:`Stein2016` — entropy-based selection using orbital entanglement analysis
  - **Occupation-based** — select orbitals with fractional occupation numbers
  - **AVAS** — Automated Valence Active Space for transition metal systems

**Manual approaches:**
  - Valence selection with user-specified active electrons and orbitals

→ :doc:`comprehensive/algorithms/active_space`

Multi-Configuration Methods
"""""""""""""""""""""""""""

Capture static correlation for bond-breaking, transition states, and open-shell systems using multi-determinant wavefunctions.

**Configuration Interaction:**
  - Full CI (FCI), Complete Active Space CI (CASCI)
  - Selected CI (SCI) — iteratively identify important configurations

**Orbital-optimized methods:**
  - Multi-Configuration SCF (MCSCF)
  - Complete Active Space SCF (CASSCF)

→ :doc:`comprehensive/algorithms/mc_calculator` · :doc:`comprehensive/algorithms/mcscf`

Dynamical Correlation Methods
"""""""""""""""""""""""""""""

Capture instantaneous electron-electron interactions for quantitative accuracy.

**Perturbation theory:**
  - Second-order Møller-Plesset (MP2)

**Coupled Cluster:**
  - CCSD, CCSD(T)

→ :doc:`comprehensive/algorithms/dynamical_correlation`

Quantum Methods
^^^^^^^^^^^^^^^

- **State Preparation and Ansätze**: Construct parameterized quantum circuits for variational algorithms, including chemistry-inspired ansätze like UCCSD.
  See :doc:`comprehensive/algorithms/state_preparation` and :doc:`comprehensive/data/ansatz`.

- **Hamiltonian Representations**: Transform fermionic Hamiltonians to qubit operators using mappings such as Jordan-Wigner, Bravyi-Kitaev, and parity.
  See :doc:`comprehensive/algorithms/hamiltonian_constructor` and :doc:`comprehensive/algorithms/qubit_mapper`.

- **Measurement and Energy Estimation**: Estimate expectation values from quantum measurements using various grouping and sampling strategies.
  See :doc:`comprehensive/algorithms/energy_estimator`.


Software Dependencies
---------------------

Required Dependencies
^^^^^^^^^^^^^^^^^^^^^

.. todo:: List the core dependencies required for QDK/Chemistry to function.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

.. todo:: Document optional dependencies and the features they enable.


Backend Integrations
^^^^^^^^^^^^^^^^^^^^

.. todo:: Describe integrations with quantum computing frameworks (Qiskit, PennyLane, Q#, etc.).


Extensibility and Plugins
-------------------------

.. todo:: Document the plugin architecture and how users can extend QDK/Chemistry.


Platform Support
----------------

.. todo:: Document supported operating systems, compilers, and Python versions.


See Also
--------

- :doc:`quickstart` - Get started with QDK/Chemistry
- :doc:`comprehensive/index` - Comprehensive documentation
