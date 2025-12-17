Features, methods and dependencies
==================================

This document provides an overview of QDK/Chemistry's features, supported methods, and software dependencies.

.. contents:: On This Page
   :local:
   :depth: 2


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

- **Geometric Direct Minimization (GDM)**: The :ref:`QDK SCF Solver <qdk-scf-native>` implements the GDM algorithm :cite:`VanVoorhis2002` for robust and efficient SCF convergence of challenging systems. This methods is particularly helpful for open-shell and small gap systems, which are of high interest in quantum computing applications, yet challenging for many SCF solvers. See the :ref:`SCF Convergence Algorithms <scf-convergence-algorithms>` section for further details.

- **Stability Analysis and Reoptimization**: Challenging SCF problems often converge to local minima that do not represent the true mean-field ground state. QDK/Chemistry includes automated :doc:`comprehensive/algorithms/stability_checker` :cite:`Schlegel1991` tools to identify, perturb and reoptimize unstable solutions, helping users obtain physically meaningful reference states for subsequent calculations.


Orbital Localization
""""""""""""""""""""

The canonical orbitals produced by SCF calculations are typically delocalized over the entire molecule, which can complicate chemical interpretation and slow the convergence of post-SCF correlation methods. QDK/Chemistry provides several classes of orbital transformation techniques to yield specialized representations which accelerate the convergence of correlation methods and enhance chemical insight:

- **Optimization-Based Methods**: The vast majority of orbital localization methods fall into this category, where a cost function is iteratively minimized to yield localized orbitals :cite:`Lehtola2013`. QDK/Chemistry supports, either through our :ref:`native implementations <localizer-qdk-pipek-mezey>` or via :ref:`integration with external libraries <localizer-pyscf-multi>`, several popular choices of cost functions, including: **Pipek-Mezey** :cite:`Pipek1989`, **Foster-Boys** :cite:`Foster1960`, and **Edmiston-Ruedenberg** :cite:`Edmiston1963`.

- **Analytical Methods**: These methods transform orbitals in a single step through analytical techniques rather than iterative optimization. **Natural orbitals** :cite:`Lowdin1956`, which diagonalize the one-particle reduced density matrix, and **Cholesky localization** :cite:`Aquilante2006`, which uses Cholesky decomposition of the one particle density matrix for efficient approximate localization, are prominent examples. The localization of the "hard-virtual" orbitals in the :ref:`VVHV localization <vvhv-algorithm>` also falls into this category.

See :doc:`comprehensive/algorithms/localizer` for further details about available orbital localization methods and implementations.


Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Valence Virtual--Hard Virtual (VVHV) Orbital Localization**: Localization of molecular orbitals expressed in a near-complete :doc:`./comprehensive/data/basis_set` is numerically ill-posed and challenging for most iterative methods. QDK/Chemistry includes an implementation of the VVHV separation :cite:`Subotnik2005`, which partitions the virtual orbital space into valence-virtual and hard-virtual subspaces for more numerically stable treatments. This produces orbitals that vary smoothly with molecular geometry, which is particularly useful for selecting consistent active spaces along reaction pathways. See the :ref:`VVHV Algorithm <vvhv-algorithm>` section for further details.

Active Space Selection
""""""""""""""""""""""

Accurate treatment of strongly correlated systems requires identifying which molecular orbitals exhibit significant multireference character. QDK/Chemistry provides a range of automated and manual approaches to select chemically relevant orbitals for :doc:`multi-configurational calculations <comprehensive/algorithms/mc_calculator>`.

The challenge lies in balancing accuracy and computational cost: an ideal active space should include all orbitals with significant multireference character while remaining as compact as possible. QDK/Chemistry supports several strategies for making this selection:

**Automated Approaches:**

- **Entanglement-Based Methods**: Utilizing concepts from quantum information theory, these methods identify strongly correlated orbitals based on their entanglement with the rest of the system. QDK/Chemistry includes a native implementation of the AutoCAS algorithm :cite:`Stein2019`, which computes single-orbital entropies from reduced density matrices to systematically select active spaces (see below for details).

- **Occupation-based Methods**: Automatic selection based on natural orbital occupation numbers. Orbitals with fractional occupations indicate strong correlation and are included in the active space.

- **AVAS (Automated Valence Active Space)** :cite:`Sayfutyarova2017` — Projects molecular orbitals onto a target atomic orbital basis (e.g., metal 3d orbitals) to systematically identify valence active spaces. This functionality is available through the :ref:`PySCF Plugin <pyscf-avas-plugin>`.

**Manual Approaches:**

- **Valence selection** — User-specified active electrons and orbitals, typically centered around the HOMO-LUMO gap.

See :doc:`comprehensive/algorithms/active_space` for further details about available methods and implementations.

.. _active-space-highlights:

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

- **AutoCAS**: QDK/Chemistry includes a native implementation of the AutoCAS algorithm :cite:`Stein2019`, which leverages quantum information concepts to identify strongly correlated orbitals. The method computes single-orbital entropies—measures of how entangled each orbital is with the rest of the system—from the one- and two-electron reduced density matrices of a multi-configuration wavefunction. Orbitals with high entropy are strongly entangled and should be treated explicitly in the active space. QDK/Chemistry's implementation includes both standard AutoCAS and an enhanced variant using entanglement of orbitals with entropy differences (AutoCAS-EOS) for improved robustness. See the :ref:`AutoCAS Algorithm <autocas-algorithm-details>` documentation for further details.


Multi-Configuration Methods
"""""""""""""""""""""""""""

Multi-configuration (:term:`MC`) methods represent the electronic wavefunction as a linear combination of many Slater determinants, enabling accurate description of static (strong) correlation effects.QDK/Chemistry provides access to a hierarchy of :term:`MC` methods:

**Configuration Interaction:**

- **Complete Active Space CI (CASCI)** — Exact solution within a defined active space, with core orbitals frozen and virtual orbitals excluded.

- **Selected CI (SCI)** — Iteratively identifies and includes only the most important configurations, enabling treatment of larger active spaces at the cost of approximation.

**Orbital-Optimized Methods:**

- **Multi-Configuration SCF (MCSCF)** — Simultaneously optimizes configuration coefficients and orbital shapes for improved accuracy.

See :doc:`comprehensive/algorithms/mc_calculator` and :doc:`comprehensive/algorithms/mcscf` for further details.

.. _mc-highlights:

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Adaptive Sampling Configuration Interaction** (:term:`ASCI`): QDK/Chemistry integrates :term:`MACIS` (Many-body Adaptive Configuration Interaction Solver) :cite:`Williams-Young2023`, a high-performance, parallel implementation of the Adaptive Sampling Configuration Interaction (:term:`ASCI`) algorithm :cite:`Tubman2016,Tubman2020`. ASCI iteratively grows the determinant space by identifying configurations with the largest contributions to the wavefunction, achieving near-CASCI accuracy at a fraction of the cost. This enables treatment of active spaces that would be intractable for conventional CASCI.


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


Community Open Source Software Dependencies
-------------------------------------------

QDK/Chemistry builds upon a foundation of well-established open source libraries developed by the quantum chemistry community. For a complete list of software dependencies, see the `Installation Guide <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_.

.. note::

   If you use QDK/Chemistry in published work, please cite the underlying libraries as described below to acknowledge the community's contributions.


Basis Sets and Effective Core Potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Basis Set Exchange (BSE)** :cite:`Pritchard2019, Feller1996, Schuchardt2007`
   A comprehensive repository of standardized basis sets for quantum chemistry calculations. All of the basis sets and effective core potentials distributed with QDK/Chemistry are sourced from the BSE. If you publish results obtained with detault basis sets provided with QDK/Chemistry, in addition to the reference for the basis set itself, please cite the BSE. For guirance on citing specific basis sets and effective core potentials, see the `Basis Set Exchange Website <https://www.basissetexchange.org/>`_.

**Libecpint** :cite:`Shaw2017,Shaw2021`
   Provides efficient evaluation of effective core potential (ECP) integrals over Gaussian-type orbitals. QDK/Chemistry's native SCF solver relies on Libecpint for ECP integral computation. If you publish results obtained with any of the native quantum chemistry modules within QDK/Chemistry that utilize ECPs, please cite Libecpint. The `Libecpint repository <https://github.com/robashaw/libecpint>`_ includes additional guidance on citing Libecpint.

Integral Evaluation
^^^^^^^^^^^^^^^^^^^

**Libint** :cite:`Libint2_290`
   Provides efficient evaluation of molecular integrals over Gaussian-type orbitals, including one- and two-electron repulsion integrals essential for all electronic structure methods. QDK/Chemistry's native SCF solver, orbital localization, and post-SCF modules rely on Libint for integral computation. If you publish results obtained with any of the native quantum chemistry modules within QDK/Chemistry, please cite Libint. The `Libint repository <https://github.com/evaleev/libint>`_ includes additional guidance on citing Libint.

**GauXC** :cite:`Petrone2018,williams20on,williams2021achieving,williams2023distributed,kovtun2024relativistic`
   Handles numerical integration on atom-centered grids, which is required for evaluating exchange-correlation contributions in density functional theory. GauXC supports both CPU and GPU acceleration, enabling scalable :term:`DFT` calculations on modern hardware. If you publish :term:`DFT` results obtained with QDK/Chemistry, please cite GauXC. See the `GauXC repository <https://github.com/wavefunction91/gauxc>`_ for guidance on citing GauXC.


Exchange-Correlation Functionals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Libxc** :cite:`Lehtola2013`
   A comprehensive library providing implementations of over 600 exchange-correlation functionals spanning LDA, GGA, meta-GGA, and hybrid rungs of Jacob's ladder. GauXC depends on Libxc for functional evaluation. If you publish :term:`DFT` results obtained with QDK/Chemistry, in addition to the reference for the functional used, please cite Libxc. For guidance on citing specific functionals, see the `Libxc Website <https://tddft.org/programs/libxc/>`_.


Multi-Configuration Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MACIS** :cite:`Williams-Young2023`
   The Many-body Adaptive Configuration Interaction Solver powers QDK/Chemistry's selected CI capabilities. MACIS implements the :term:`ASCI` algorithm with distributed-memory parallelism, enabling treatment of active spaces far beyond the reach of conventional exact diagonalization.


Plugins
^^^^^^^

QDK/Chemistry is distributed with the following officially supported plugins that extend its capabilities by integrating with external quantum chemistry and quantum algorithms frameworks. If implementations of QDK/Chemistry interfaces using these plugins are used in your published work, please cite both QDK/Chemistry and the underlying package as described in the respective plugin documentation.

**PySCF Plugin**
   Integrates QDK/Chemistry with the PySCF quantum chemistry package, providing access to its extensive suite of electronic structure methods and tools.
   See the `PySCF documentation <https://pyscf.org/about.html>`_ for guidance on citing PySCF.

**Qiskit Plugin**
   Enables interoperability between QDK/Chemistry and the Qiskit quantum computing framework
   See the `Qiskit documentation <https://qiskit.org/documentation/getting_started.html>`_ for guidance on citing Qiskit.

See Also
--------

- :doc:`quickstart` - Get started with QDK/Chemistry
- :doc:`comprehensive/index` - Comprehensive documentation
