Self-consistent field solver
============================

The :class:`~qdk_chemistry.algorithms.ScfSolver` algorithm in QDK/Chemistry performs Self-Consistent Field (SCF) calculations to optimize molecular orbitals for a given molecular structure.
Following QDK/Chemistry's :doc:`algorithm design principles <../advanced/design_principles>`, it takes a :doc:`Structure <../data/structure>` instance as input and produces an :doc:`Orbitals <../data/orbitals>` instance as output.
Its primary purpose is to find the best single-particle orbitals within a mean-field approximation.
For Hartree-Fock (HF) theory, it yields the mean field energy, which misses electron correlation and typically requires post-HF methods for accurate energetics.
For Density Functional Theory (DFT), some correlation effects are included through the exchange-correlation functional.

Overview
--------

:term:`SCF` theory encompasses both :term:`HF` and :term:`DFT` methods in quantum chemistry.
Both methods rely on a single Slater determinant representation of the many-electron wavefunction, using molecular orbitals that are optimized to minimize the electronic energy.
This single-determinant approach is a key simplification that makes these methods computationally efficient but limits their ability to capture certain correlation effects.
The :term:`SCF` procedure iteratively refines these orbitals until self-consistency is achieved.

At its core, an :term:`SCF` calculation:

1. **Initializes a starting guess** for the molecular orbitals, typically using a superposition of atomic orbitals
2. **Constructs the Fock matrix** which represents the effective one-electron Hamiltonian
3. **Diagonalizes the matrix** to obtain a new set of molecular orbitals and their energies
4. **Computes the electron density** from the occupied molecular orbitals
5. **Checks for convergence** by comparing the new density with that from the previous iteration
6. **Repeats steps 2-5** until the density and energy no longer change significantly between iterations

This iterative process is called "self-consistent" because the orbitals used to construct the Fock/Kohn-Sham operator must be consistent with the orbitals obtained by solving the resulting eigenvalue equation.
The final result provides:

- Optimized molecular orbitals and their energies
- Mean-field energy (for :term:`HF`), which excludes electron correlation
- Approximated ground state energy (for :term:`DFT`), with correlation treated through the functional
- Electron density distribution
- Various electronic properties derived from the wavefunction/density

:term:`SCF` methods provide an excellent starting point, but they miss important electronic correlation effects:

- **Static correlation**: Essential for systems with near-degenerate states or bond-breaking processes.
  See :doc:`MCCalculator <mc_calculator>` documentation.
- **Dynamic correlation**: Required for all molecular systems to account for instantaneous electron-electron interactions.
  .. See :doc:`Dynamic Correlation <dynamical_correlation>` documentation.

The orbitals from :term:`SCF` calculations typically serve as input for post-:term:`SCF` methods that capture these correlation effects.
:term:`SCF` methods thus serve as the foundation for more advanced electronic structure calculations and provide essential insights into molecular properties, reactivity, and spectroscopic characteristics.

Capabilities
------------

The :class:`~qdk_chemistry.algorithms.ScfSolver` in QDK/Chemistry provides the following calculation types for both :term:`HF` and :term:`DFT` methods:

- **Restricted calculations**: For closed-shell systems with paired electrons

  - Restricted Hartree-Fock (RHF)
  - Restricted Kohn-Sham :term:`DFT` (RKS)

- **Unrestricted calculations**: For open-shell systems with unpaired electrons

  - Unrestricted Hartree-Fock (UHF)
  - Unrestricted Kohn-Sham :term:`DFT` (UKS)

- **Restricted open-shell calculations**: For open-shell systems with restricted orbitals

  - Restricted Open-shell Hartree-Fock (ROHF)
  - Restricted Open-shell Kohn-Sham :term:`DFT` (ROKS)

- **DFT-specific features**:

  - Support for various exchange-correlation functionals including :term:`LDA`, :term:`GGA`, meta-:term:`GGA`, hybrid, and range-separated functionals

- **Basis set support**:

  - Extensive library of standard quantum chemistry basis sets including Pople (STO-nG, 3-21G,
    6-31G, etc.), Dunning (cc-pVDZ, cc-pVTZ, etc.), and Karlsruhe (def2-SVP, def2-TZVP, etc.) families
  - Support for custom basis sets and effective core potentials (ECPs)

Creating an :term:`SCF` solver
------------------------------

As an algorithm class in QDK/Chemistry, the ``ScfSolver`` follows the :doc:`factory pattern design principle <../advanced/design_principles>` and is created using its corresponding factory.

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

   .. literalinclude:: ../../../../examples/factory_pattern.py
      :language: python

Configuring the :term:`SCF` calculation
---------------------------------------

The ``ScfSolver`` can be configured using the ``Settings`` object.
QDK/Chemistry provides standard :term:`SCF` settings that apply to all solver implementations, as well as specialized settings for specific solvers or algorithms.

QDK/Chemistry provides both standard settings that work across all :term:`SCF` solver implementations and specialized settings for specific algorithms or implementations.
See the `Available Settings`_ section below for a complete list of configuration options.

.. note::
   For a complete list of available basis sets and their specifications, see the :doc:`Supported Basis Sets <../data/basis_sets>` documentation.
   This reference provides detailed information about all pre-defined basis sets you can use with the ``basis_set`` setting.

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

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 4-12

Running an :term:`SCF` calculation
--------------------------

Once configured, the :term:`SCF` calculation can be executed on a molecular structure.
The ``solve`` method returns two values:

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

   .. literalinclude:: ../../../../examples/scf_solver.py
      :language: python
      :lines: 1-17

Available settings
------------------

The ``ScfSolver`` accepts a range of settings to control its behavior.
These settings are divided into base settings (common to all :term:`SCF` calculations) and specialized settings (specific to certain :term:`SCF` variants).

Base settings
~~~~~~~~~~~~~

.. note::
   This table is under construction.

These settings apply to all :term:`SCF` calculations:

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

Specialized settings
~~~~~~~~~~~~~~~~~~~~

.. note::
   This table is under construction.

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
     - Exchange-correlation functional for :term:`DFT` (empty for :term:`HF`); see :doc:`functionals documentation <../data/functionals>`
     - :term:`DFT`` only
   * - ``level_shift``
     - float
     - 0.0
     - Energy level shifting for virtual orbitals to aid convergence
     - All :term:`SCF` types

Implemented interface
---------------------

QDK/Chemistry's ``ScfSolver`` provides a unified interface to SCF calculations across various quantum chemistry packages:

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **qdk**: Native implementation with support for RHF, UHF, RKS and UKS

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **pyscf**: Comprehensive Python-based quantum chemistry package with extensive DFT capabilities

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../advanced/interfaces>` documentation.

Related classes
---------------

- :doc:`Structure <../data/structure>`: Input molecular structure
- :doc:`Orbitals <../data/orbitals>`: Output optimized molecular orbitals

Related topics
--------------

- :doc:`Settings <../advanced/settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <../advanced/factory_pattern>`: Understanding algorithm creation
