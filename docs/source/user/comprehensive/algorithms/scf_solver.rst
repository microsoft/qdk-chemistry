ScfSolver
=========

The ``ScfSolver`` algorithm in QDK/Chemistry performs Self-Consistent Field (SCF) calculations to optimize molecular orbitals for a
given molecular structure. Following QDK/Chemistry's :doc:`algorithm design principles <../advanced/design_principles>`, it takes a
:doc:`Structure <../data/structure>` instance as input and produces an :doc:`Orbitals <../data/orbitals>` instance as
output. Its primary purpose is to find the best single-particle orbitals within a mean-field approximation. For
Hartree-Fock (HF) theory, it yields the mean field energy, which misses electron correlation and typically requires
post-HF methods for accurate energetics. For Density Functional Theory (DFT), some correlation effects are included
through the exchange-correlation functional.

Overview
--------

Self-Consistent Field theory encompasses both Hartree-Fock (HF) and Density Functional Theory (DFT) methods in quantum
chemistry. Both methods rely on a single Slater determinant representation of the many-electron wavefunction, using
molecular orbitals that are optimized to minimize the electronic energy. This single-determinant approach is a key
simplification that makes these methods computationally efficient but limits their ability to capture certain
correlation effects. The SCF procedure iteratively refines these orbitals until self-consistency is achieved.

At its core, an SCF calculation:

1. **Initializes a starting guess** for the molecular orbitals, typically using a superposition of atomic orbitals
2. **Constructs the Fock matrix** which represents the effective one-electron Hamiltonian
3. **Diagonalizes the matrix** to obtain a new set of molecular orbitals and their energies
4. **Computes the electron density** from the occupied molecular orbitals
5. **Checks for convergence** by comparing the new density with that from the previous iteration
6. **Repeats steps 2-5** until the density and energy no longer change significantly between iterations

This iterative process is called "self-consistent" because the orbitals used to construct the Fock/Kohn-Sham operator
must be consistent with the orbitals obtained by solving the resulting eigenvalue equation. The final result provides:

- Optimized molecular orbitals and their energies
- Mean-field energy (for HF), which excludes electron correlation
- Approximated ground state energy (for DFT), with correlation treated through the functional
- Electron density distribution
- Various electronic properties derived from the wavefunction/density

SCF methods provide an excellent starting point, but they miss important electronic correlation effects:

- **Static correlation**: Essential for systems with near-degenerate states or bond-breaking processes. See
  :doc:`MCCalculator <mc_calculator>` documentation.
- **Dynamic correlation**: Required for all molecular systems to account for instantaneous electron-electron
  interactions. See :doc:`Dynamic Correlation <dynamical_correlation>` documentation.

The orbitals from SCF calculations typically serve as input for post-SCF methods that capture these correlation effects.

SCF methods thus serve as the foundation for more advanced electronic structure calculations and provide essential
insights into molecular properties, reactivity, and spectroscopic characteristics.

Capabilities
------------

The ``ScfSolver`` in QDK/Chemistry provides the following calculation types for both Hartree-Fock and Density Functional Theory
methods:

- **Restricted calculations**: For closed-shell systems with paired electrons

  - Restricted Hartree-Fock (RHF)
  - Restricted Kohn-Sham DFT (RKS)

- **Unrestricted calculations**: For open-shell systems with unpaired electrons (TODO: Implementation details)

  - Unrestricted Hartree-Fock (UHF)
  - Unrestricted Kohn-Sham DFT (UKS)

.. todo::
   TODO (NAB):  finish SCF Solver documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41382

- **Restricted Open-shell calculations**: For open-shell systems with restricted orbitals (TODO: Implementation details)

  - Restricted Open-shell Hartree-Fock (ROHF)
  - Restricted Open-shell Kohn-Sham DFT (ROKS)

.. todo::
   TODO (NAB):  finish SCF Solver documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41382

- **DFT-specific features**:

  - Support for various :doc:`exchange-correlation functionals <../data/functionals>` including LDA, GGA, meta-GGA, hybrid,
    and range-separated functionals

- **Basis set support**:

  - Extensive library of standard quantum chemistry :doc:`basis sets <../data/basis_sets>` including Pople (STO-nG, 3-21G,
    6-31G, etc.), Dunning (cc-pVDZ, cc-pVTZ, etc.), and Karlsruhe (def2-SVP, def2-TZVP, etc.) families
  - Support for custom basis sets and effective core potentials (ECPs) (TODO)

.. todo::
   TODO (NAB):  finish SCF Solver documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41382

Creating an SCF Solver
----------------------

As an algorithm class in QDK/Chemistry, the ``ScfSolver`` follows the
:doc:`factory pattern design principle <../advanced/design_principles>` and is created using its corresponding factory:

Available Solvers
~~~~~~~~~~~~~~~~~

QDK/Chemistry currently provides the following registered solvers:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Solver
     - Type
     - Description
   * - **default**
     - Default solver implemented directly in QDK/Chemistry, optimized for performance and versatility
   * - **pyscf**
     - Third-party
     - Integration with the PySCF quantum chemistry package

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::algorithms;

      // Create the default ScfSolver instance
      auto scf_solver = ScfSolverFactory::create();

      // Or specify a particular solver implementation
      auto pyscf_solver = ScfSolverFactory::create("pyscf");

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.algorithms import create_scf_solver

      # Create the default ScfSolver instance
      scf_solver = create_scf_solver()

      # Or specify a particular solver implementation
      pyscf_solver = create_scf_solver(implementation="pyscf")

Configuring the SCF Calculation
-------------------------------

The ``ScfSolver`` can be configured using the ``Settings`` object.
QDK/Chemistry provides standard SCF settings that apply to all solver implementations, as well as specialized settings for specific solvers or algorithms.

QDK/Chemistry provides both standard settings that work across all SCF solver implementations and specialized settings for specific algorithms or implementations.
See the `Available Settings`_ section below for a complete list of configuration options.

.. note::
   For a complete list of available basis sets and their specifications, see the
   :doc:`Supported Basis Sets <../data/basis_sets>` documentation. This reference provides detailed information about all
   pre-defined basis sets you can use with the ``basis_set`` setting.

.. tab:: C++ API

   .. code-block:: cpp

      // Standard settings that work with all solvers
      // Set the method
      scf_solver.settings().set("method", "dft")
      // Set the basis set
      scf_solver->settings().set("basis_set", "def2-tzvpp");

      // For DFT calculations, set the exchange-correlation functional
      scf_solver->settings().set("functional", "B3LYP");

.. tab:: Python API

   .. code-block:: python

      # Standard settings that work with all solvers
      # Set the method
      scf_solver.settings().set("method", "dft")
      # Set the basis set
      scf_solver.settings().set("basis_set", "def2-tzvpp")

      # For DFT calculations, set the exchange-correlation functional
      scf_solver.settings().set("functional", "B3LYP")

Running an SCF Calculation
--------------------------

Once configured, the SCF calculation can be executed on a molecular structure. The ``solve`` method returns two values:

1. A scalar ``double`` value representing the converged SCF energy
2. An :doc:`Orbitals <../data/orbitals>` object containing the optimized molecular orbitals

.. tab:: C++ API

   .. code-block:: cpp

      // Create a structure (or load from a file)
      Structure structure;
      // configuring structure ...

      // Run the SCF calculation
      // Return types are: std::tuple<double, Orbitals>
      auto [E_scf, scf_orbitals] = scf_solver->solve(structure);
      std::cout << "SCF Energy: " << E_scf << " Hartree" << std::endl;

.. tab:: Python API

   .. code-block:: python

      # Create a structure (or load from a file)
      structure = Structure()
      # configuring structure ...

      # Run the SCF calculation
      # Returns: (float, Orbitals)
      E_scf, scf_orbitals = scf_solver.solve(structure)
      print(f"SCF Energy: {E_scf} Hartree")

Available Settings
------------------

The ``ScfSolver`` accepts a range of settings to control its behavior. These settings are divided into base settings
(common to all SCF calculations) and specialized settings (specific to certain SCF variants).

Base Settings
~~~~~~~~~~~~~

.. todo::
   🔧 **TODO:** Add comprehensive list of all actual base settings used by ScfSolver implementations

   TODO (NAB):  finish SCF Solver documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41382

These settings apply to all SCF calculations:

.. todo::
   TODO (NAB):  finish SCF Solver documentation below
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41382

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``method``
     - string
     - TODO
     - The method to use for the calculation
   * - ``basis_set``
     - string
     - TODO
     - The basis set to use for the calculation
   * - ``convergence_threshold``
     - float
     - TODO
     - Energy convergence criterion for SCF iterations
   * - ``max_iterations``
     - int
     - TODO
     - Maximum number of SCF iterations
   * - ``multiplicity``
     - int
     - 1
     - Spin multiplicity of the system (TODO: move this to structure)
   * - ``charge``
     - int
     - 0
     - Total charge of the system (TODO: move this to structure)

Specialized Settings
~~~~~~~~~~~~~~~~~~~~

.. todo::
   🔧 **TODO:** Add comprehensive list of all actual specialized settings used by ScfSolver implementations

   TODO (NAB):  finish SCF Solver documentation above and below
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41382

These settings apply only to specific variants of SCF calculations:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 30 20

   * - Setting
     - Type
     - Default
     - Description
     - Applicable To
   * - ``functional``
     - string
     - TODO
     - Exchange-correlation functional for DFT (empty for HF); see :doc:`functionals documentation <../data/functionals>`
     - DFT only
   * - ``level_shift``
     - float
     - 0.0
     - Energy level shifting for virtual orbitals to aid convergence
     - All SCF types

Implemented Interface
---------------------

QDK/Chemistry's ``ScfSolver`` provides a unified interface to SCF calculations across various quantum chemistry packages:

QDK/Chemistry Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **qdk**: Native implementation with support for RHF, UHF, RKS and UKS

Third-Party Interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **pyscf**: Comprehensive Python-based quantum chemistry package with extensive DFT capabilities

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../advanced/interfaces>` documentation.

Related Classes
---------------

- :doc:`Structure <../data/structure>`: Input molecular structure
- :doc:`Orbitals <../data/orbitals>`: Output optimized molecular orbitals

Related Topics
--------------

- :doc:`Settings <../advanced/settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <../advanced/factory_pattern>`: Understanding algorithm creation
