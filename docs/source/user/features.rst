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

QDK/Chemistry provides access to a variety of robust, high-performance implementations mean-field electronic structure methods that produce optimized molecular orbitals and reference energies. In particular, the following :term:`SCF` types are supported:

Hartree-Fock (:term:`HF`)
   Restricted (:term:`RHF`), Unrestricted (:term:`UHF`), Restricted Open-Shell (:term:`ROHF`)

Density Functional Theory (:term:`DFT`)
   Kohn-Sham methods: :term:`RKS`, :term:`UKS`, :term:`ROKS`

See :doc:`comprehensive/algorithms/scf_solver` for further details about available :term:`SCF` methods and implementations.

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

Geometric Direct Minimization (:term:`GDM`)
   The :ref:`QDK SCF Solver <qdk-scf-native>` implements the :term:`GDM` algorithm :cite:`VanVoorhis2002` for robust and efficient convergence of challenging :term:`SCF` systems.
   This method is particularly helpful for open-shell and small gap systems, which are of high interest in quantum computing applications, yet challenging for many :term:`SCF` solvers.
   See the :ref:`SCF Convergence Algorithms <scf-convergence-algorithms>` section for further details.

Stability Analysis and Reoptimization
   Challenging :term:`SCF` problems often converge to local minima that do not represent the true mean-field ground state, which in turn can lead to incorrect and unintuitive results in post-:term:`SCF` calculations.
   QDK/Chemistry includes automated :doc:`comprehensive/algorithms/stability_checker` :cite:`Schlegel1991` tools to identify, perturb and reoptimize unstable solutions, helping users obtain physically meaningful reference states for subsequent calculations.


Orbital Localization
""""""""""""""""""""

The canonical orbitals produced by :term:`SCF` calculations are typically delocalized over the entire molecule, which can complicate chemical interpretation and slow the convergence of post-:term:`SCF` correlation methods. QDK/Chemistry provides several classes of orbital transformation techniques to yield specialized representations which accelerate the convergence of correlation methods and enhance chemical insight:

Optimization-Based Methods
   The vast majority of orbital localization methods fall into this category, where a cost function is iteratively minimized to yield localized orbitals :cite:`Lehtola2013`.
   QDK/Chemistry supports, either through our :ref:`native implementations <localizer-qdk-pipek-mezey>` or via :ref:`integration with external libraries <localizer-pyscf-multi>`, several popular choices of cost functions, including: **Pipek-Mezey** :cite:`Pipek1989`, **Foster-Boys** :cite:`Foster1960`, and **Edmiston-Ruedenberg** :cite:`Edmiston1963`.

Analytical Methods
   These methods transform orbitals in a single step through analytical techniques rather than iterative optimization.
   Prominent examples include **natural orbitals** :cite:`Lowdin1956`, which diagonalize the one-particle reduced density matrix, and **Cholesky localization** :cite:`Aquilante2006`, which uses Cholesky decomposition of the one particle density matrix for efficient approximate localization.
   The localization of the "hard-virtual" orbitals in the :ref:`VVHV localization <vvhv-algorithm>` also falls into this category.

See :doc:`comprehensive/algorithms/localizer` for further details about available orbital localization methods and implementations.


Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

Valence Virtual--Hard Virtual (:term:`VVHV`) Orbital Localization
   Localization of molecular orbitals expressed in a near-complete :doc:`./comprehensive/data/basis_set` is numerically ill-posed and challenging for most iterative methods.
   QDK/Chemistry includes an implementation of the :term:`VVHV` separation :cite:`Subotnik2005`, which partitions the virtual orbital space into valence-virtual and hard-virtual subspaces for more numerically stable treatments.
   This produces orbitals that vary smoothly with molecular geometry, which is particularly useful for selecting consistent active spaces along reaction pathways.
   See the :ref:`VVHV Algorithm <vvhv-algorithm>` section for further details.


Active Space Selection
""""""""""""""""""""""

Accurate treatment of strongly correlated systems requires identifying which molecular orbitals exhibit significant entanglement character. QDK/Chemistry provides a range of automated and manual approaches to select chemically relevant orbitals for :doc:`multi-configurational calculations <comprehensive/algorithms/mc_calculator>`.

The challenge lies in balancing accuracy and computational cost: an ideal active space should include all orbitals with significant entanglement character while remaining as compact as possible. QDK/Chemistry supports several strategies for making this selection:

.. rubric:: Automated Approaches

Entropy-Based Methods
   Using concepts from quantum information theory, these methods identify strongly correlated orbitals based on their entanglement with the rest of the system.
   QDK/Chemistry includes a native implementation of the AutoCAS algorithm :cite:`Stein2019`, which leverages single-orbital entropies computed from reduced density matrices to systematically select active spaces (see below for details).

Occupation-based Methods
   Automatic selection based on natural orbital occupation numbers obtained from correlated many-body methods.
   Orbitals with fractional occupations indicate entanglement and are included in the active space.

:term:`AVAS` (Automated Valence Active Space) :cite:`Sayfutyarova2017`
   Projects molecular orbitals onto a target atomic orbital basis (e.g., metal 3d orbitals) to systematically identify valence active spaces.
   This functionality is available through the :ref:`PySCF Plugin <pyscf-avas-plugin>`.

.. rubric:: Manual Approaches

Valence selection
   User-specified active electrons and orbitals, centered around the :term:`HOMO`-:term:`LUMO` gap (Fermi level).

See :doc:`comprehensive/algorithms/active_space` for further details about available methods and implementations.

.. _active-space-highlights:

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

AutoCAS
   QDK/Chemistry includes a native implementation of the AutoCAS algorithm :cite:`Stein2019`, which leverages quantum information concepts to identify strongly correlated orbitals.
   The method computes single-orbital entropies—measures of how entangled each orbital is with the rest of the system—from the one- and two-electron reduced density matrices of a multi-configurational wavefunction.
   Orbitals with high entropy are strongly entangled and should be treated explicitly within the active space. QDK/Chemistry's implementation includes both standard AutoCAS and an enhanced variant (AutoCAS-EOS) for improved robustness.
   See the :ref:`AutoCAS Algorithm <autocas-algorithm-details>` documentation for further details.


Multi-Configuration Methods
"""""""""""""""""""""""""""

Multi-configuration (:term:`MC`) methods represent the electronic wavefunction as a linear combination of many Slater determinants, enabling accurate description of static correlation effects.
QDK/Chemistry provides access to a hierarchy of :term:`MC` methods:

.. rubric:: Configuration Interaction

Complete Active Space :term:`CI` (:term:`CASCI`)
   Exact solution within a defined active space, with core orbitals frozen and virtual orbitals excluded.

Selected :term:`CI` (:term:`SCI`)
   Iteratively identifies and includes only the most important configurations, enabling treatment of larger active spaces at the cost of approximation.

.. rubric:: Orbital-Optimized Methods

Multi-Configuration :term:`SCF` (:term:`MCSCF`)
   Simultaneously optimizes configuration coefficients and orbital shapes for improved accuracy.

See :doc:`comprehensive/algorithms/mc_calculator` and :doc:`comprehensive/algorithms/mcscf` for further details.

.. _mc-highlights:

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive Sampling Configuration Interaction (:term:`ASCI`)
   QDK/Chemistry integrates :term:`MACIS` (Many-body Adaptive Configuration Interaction Solver) :cite:`Williams-Young2023`, a high-performance, parallel implementation of the Adaptive Sampling Configuration Interaction (:term:`ASCI`) algorithm :cite:`Tubman2016,Tubman2020`.
   :term:`ASCI` iteratively grows the determinant space by identifying configurations with the largest contributions to the wavefunction, achieving near-:term:`CASCI` accuracy at a fraction of the cost.
   This enables treatment of active spaces that would be intractable for conventional :term:`CASCI`. See the :ref:`ASCI Algorithm <asci-algorithm>` section for details.


Quantum Algorithms
^^^^^^^^^^^^^^^^^^

State Preparation
"""""""""""""""""

Core to quantum algorithms for chemistry is the ability to efficiently prepare quantum states that approximate the ground or excited states of molecular systems. In QDK/Chemistry, this process is generally viewed as a mapping between a classical representation of the molecular wavefunction (e.g., a Slater determinant or a linear combination thereof, represented by the `Wavefunction` class) and a circuit that prepares the corresponding quantum state on a quantum computer given a particular qubit encoding (e.g. binary, gray code, etc). QDK/Chemistry provides several state preparation techniques to facilitate this mapping, including:

Dense Isometry State Preparation
   Implementations of general-purpose state preparation algorithms that can prepare arbitrary quantum states, represented in the occupation number formalism, given their amplitudes.

Sparse Isometry State Preparation
   Optimized algorithms for preparing quantum states with a small number of non-zero amplitudes, such as those arising from selected :term:`CI` methods.

See :doc:`comprehensive/algorithms/state_preparation` for further details about available state preparation methods and implementations.

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

GF2+X Sparse Isometry State Preparation
   QDK/Chemistry implements an optimized state preparation algorithm for wavefunctions with sparse amplitude structure.
   The GF2+X method, a modification of the original sparse isometry work in :cite:`Malvetti2021`, applies GF(2) Gaussian elimination to the binary matrix representation of the quantum state to determine a reduced space representation of the sparse state.
   This reduced state is then densely encoded via regular isometry :cite:`Christandl2016` on a smaller number of qubits, and finally scattered to the full qubit space using X and :term:`CNOT` gates.
   By focusing only on non-zero amplitudes, this approach substantially reduces circuit depth and gate count compared to dense isometry methods, making it especially suitable for selected :term:`CI` and other sparse wavefunctions.


Hamiltonian Encoding
""""""""""""""""""""

Classical quantum chemistry methods are generally expressed in the language of second quantization, using fermionic creation and annihilation operators to describe interactions between electron configurations. However, quantum computers operate on qubits, necessitating a transformation from fermionic operators to qubit operators. QDK/Chemistry implements several standard mapping techniques to achieve this:

Jordan-Wigner Transformation :cite:`Jordan-Wigner1928`
   A straightforward mapping that encodes fermionic operators directly onto qubits, preserving the algebraic structure of the operators.

Bravyi-Kitaev Transformation :cite:`Seeley2012`
   A more efficient mapping that reduces the number of qubits required for certain operations by balancing locality and parity information.

Parity Transformation :cite:`Seeley2012`
   An alternative mapping that encodes fermionic operators based on the parity of occupation numbers, offering advantages in specific contexts.

QDK/Chemistry provides both a native qubit mapper implementation and integration with external libraries through plugins.
See :doc:`comprehensive/algorithms/qubit_mapper` for available implementations and usage details.
QDK/Chemistry also provides :doc:`Pauli operator arithmetic <comprehensive/data/pauli_operator>` for building and manipulating qubit Hamiltonians using natural mathematical notation.

.. _qubit-mapper-highlights:

Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~

Native QDK Qubit Mapper
   QDK/Chemistry includes a high-performance native implementation of fermion-to-qubit mappings built on the :doc:`PauliOperator <comprehensive/data/pauli_operator>` expression layer.
   This implementation supports both Jordan-Wigner and Bravyi-Kitaev encodings :cite:`Seeley2012`, using the Seeley-Richard-Love algorithm for the Bravyi-Kitaev mapping, with configurable coefficient thresholds for controlling numerical precision.
   The native mapper applies thresholds after the complete transformation, ensuring mathematically consistent results across different molecular systems.


Observable Sampling
"""""""""""""""""""

After preparing a quantum state representing the molecular system, a common next step is to compute physical observables, such as the ground state energy.
One canonical choice for this task is to estimate the expectation value through statistical sampling of measurements performed on the quantum state relative to the operator of interest.
This generally involves the following steps:

1. **Operator Decomposition**:
   The target operator (e.g., the electronic Hamiltonian) is decomposed into a sum of measurable components, often expressed in terms of Pauli operators.
   This decomposition facilitates efficient measurement on quantum hardware.
   Starting from a qubit-mapped Hamiltonian, this task generally involves grouping Pauli terms into sets of mutually commuting operators that can be measured simultaneously.
   QDK/Chemistry provides utilities to perform, for example, `Pauli grouping by qubit-wise commutativity <https://qiskit.org/documentation/stubs/qiskit.opflow.grouping.PauliGrouper.html>`_ through its Qiskit plugin.
2. **Circuit Execution and Measurement**:
   Given the state preparation circuit and the decomposed operator, quantum circuits are executed on quantum hardware or simulators to obtain measurement outcomes.
3. **Classical Post-Processing**:
   The measurement results are processed classically to estimate the expectation value of the operator.

See :doc:`comprehensive/algorithms/energy_estimator` for further details about available observable sampling methods and implementations.



Community Open Source Software Dependencies
-------------------------------------------

QDK/Chemistry builds upon a foundation of well-established open source libraries developed by the quantum chemistry community. For a complete list of software dependencies, see the `Installation Guide <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_.

.. note::

   If you use QDK/Chemistry in published work, please cite the underlying libraries as described below to acknowledge the community's contributions.


Basis Sets and Effective Core Potentials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basis Set Exchange (:term:`BSE`) :cite:`Pritchard2019, Feller1996, Schuchardt2007`
   A comprehensive repository of standardized basis sets for quantum chemistry calculations.
   All of the basis sets and effective core potentials distributed with QDK/Chemistry are sourced from the :term:`BSE`.
   If you publish results obtained with default basis sets provided with QDK/Chemistry, in addition to the reference for the basis set itself, please cite the :term:`BSE`.
   For guidance on citing specific basis sets and effective core potentials, see the `Basis Set Exchange Website <https://www.basissetexchange.org/>`_.

Libecpint :cite:`Shaw2017,Shaw2021`
   Provides efficient evaluation of effective core potential (:term:`ECP`) integrals over Gaussian-type orbitals. QDK/Chemistry's native :term:`SCF` solver relies on Libecpint for :term:`ECP` integral computation. If you publish results obtained with any of the native quantum chemistry modules within QDK/Chemistry that utilize ECPs, please cite Libecpint. The `Libecpint repository <https://github.com/robashaw/libecpint>`_ includes additional guidance on citing Libecpint.

Integral Evaluation
^^^^^^^^^^^^^^^^^^^

Libint :cite:`Libint2_290`
   Provides efficient evaluation of molecular integrals over Gaussian-type orbitals, including one- and two-electron repulsion integrals essential for all electronic structure methods. QDK/Chemistry's native :term:`SCF` solver, orbital localization, and post-:term:`SCF` modules rely on Libint for integral computation. If you publish results obtained with any of the native quantum chemistry modules within QDK/Chemistry, please cite Libint. The `Libint repository <https://github.com/evaleev/libint>`_ includes additional guidance on citing Libint.

GauXC :cite:`Petrone2018,williams20on,williams2021achieving,williams2023distributed,kovtun2024relativistic`
   Handles numerical integration on atom-centered grids, which is required for evaluating exchange-correlation contributions in density functional theory. GauXC supports both CPU and GPU acceleration, enabling scalable :term:`DFT` calculations on modern hardware. If you publish :term:`DFT` results obtained with QDK/Chemistry, please cite GauXC. See the `GauXC repository <https://github.com/wavefunction91/gauxc>`_ for guidance on citing GauXC.


Exchange-Correlation Functionals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Libxc :cite:`Lehtola2013`
   A comprehensive library providing implementations of over 600 exchange-correlation functionals spanning :term:`LDA`, :term:`GGA`, meta-:term:`GGA`, and hybrid rungs of Jacob's ladder. GauXC depends on Libxc for functional evaluation. If you publish :term:`DFT` results obtained with QDK/Chemistry, in addition to the reference for the functional used, please cite Libxc. For guidance on citing specific functionals, see the `Libxc Website <https://tddft.org/programs/libxc/>`_.


Multi-Configuration Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

MACIS :cite:`Williams-Young2023`
   The Many-body Adaptive Configuration Interaction Solver powers QDK/Chemistry's selected CI capabilities. :term:`MACIS` implements the :term:`ASCI` algorithm with distributed-memory parallelism, enabling treatment of active spaces far beyond the reach of conventional exact diagonalization.


Plugins
^^^^^^^

QDK/Chemistry is distributed with the following officially supported plugins that extend its capabilities by integrating with external quantum chemistry and quantum algorithms frameworks. If implementations of QDK/Chemistry interfaces using these plugins are used in your published work, please cite both QDK/Chemistry and the underlying package as described in the respective plugin documentation.

PySCF Plugin
   Integrates QDK/Chemistry with the PySCF quantum chemistry package, providing access to its extensive suite of electronic structure methods and tools.
   See the `PySCF documentation <https://pyscf.org/about.html>`_ for guidance on citing PySCF.

Qiskit Plugin
   Enables interoperability between QDK/Chemistry and the Qiskit quantum computing framework
   See the `Qiskit documentation <https://qiskit.org/documentation/getting_started.html>`_ for guidance on citing Qiskit.


Visual Studio Code Integration
------------------------------

QDK/Chemistry provides integration with Visual Studio Code to enhance the development experience when using the Python API.

Type-Aware Autocompletion for Factory Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QDK/Chemistry dynamically generates type stubs that enable intelligent autocompletion in VS Code (via Pylance) for the :doc:`factory pattern <comprehensive/algorithms/factory_pattern>` used throughout the library.
When calling ``registry.create()``, the editor provides:

- **Algorithm type suggestions**: Autocompletion for valid ``algorithm_type`` values (e.g., ``"scf_solver"``, ``"orbital_localizer"``, ``"active_space_selector"``)
- **Algorithm name suggestions**: Context-aware suggestions for ``algorithm_name`` based on the selected algorithm type
- **Settings parameter hints**: Typed parameter hints for algorithm-specific settings
- **Return type inference**: Accurate return type information for the created algorithm instance

This integration significantly improves discoverability of available algorithms and reduces errors from typos in string literals.
The autocompletion features are also going to work with other IDEs that support type stubs.

.. note::

   **Limitation**: Type stub generation only works for algorithms shipped with QDK/Chemistry and its official plugins.
   Custom plugins registered at runtime will not have autocompletion support for their algorithm types and names.

Dev Container Support
^^^^^^^^^^^^^^^^^^^^^

For developers who want to modify or extend QDK/Chemistry and compile from source, the repository includes a pre-configured Dev Container.
This provides a complete build environment with all system dependencies installed, once the container is built.
See the `Installation Guide <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_ for details.


See Also
--------

- :doc:`quickstart` - Get started with QDK/Chemistry
- :doc:`comprehensive/index` - Comprehensive documentation
