Self-consistent field solver
============================

The :class:`~qdk_chemistry.algorithms.ScfSolver` algorithm in QDK/Chemistry performs Self-Consistent Field (SCF) calculations to optimize molecular orbitals for a given molecular structure.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Structure <../data/structure>` instance as input and produces an :doc:`Orbitals <../data/orbitals>` instance as output.
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
  See :doc:`ReferenceDerivedCalculator <reference_derived>` documentation.

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

  - Support for :doc:`various exchange-correlation functionals <../basis_functionals>` including :term:`LDA`, :term:`GGA`, meta-:term:`GGA`, hybrid, and range-separated functionals

- **Basis set support**:

  - Extensive library of standard quantum chemistry basis sets including Pople (STO-nG, 3-21G,
    6-31G, etc.), Dunning (cc-pVDZ, cc-pVTZ, etc.), and Karlsruhe (def2-SVP, def2-TZVP, etc.) families
  - Support for custom basis sets and effective core potentials (ECPs)

Running an :term:`SCF` calculation
----------------------------------

Below is an example of how to run SCF using the default (Microsoft) QDK/Chemistry solver, with mostly default settings (except the basis set):

.. tab:: C++ API

   .. code-block:: cpp

      // Specify a structure
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
      std::vector<std::string> symbols = {"H", "H"};

      Structure structure(coords, symbols);

      // Create default ScfSolver instance (HF)
      auto scf_solver = ScfSolverFactory::create();

      // Set basis set
      scf_solver->settings().set("basis_set", "sto-3g");
      const auto& [E_scf, wfn_scf] = scf_solver->run(structure, 0, 1);


.. tab:: Python API

   .. literalinclude:: ../../../../examples/scf_solver.py
      :language: python
      :lines: 16-28

Likewise for DFT:

.. tab:: C++ API

   .. code-block:: cpp

      // Create default ScfSolver instance
      auto scf_solver_dft = ScfSolverFactory::create();

      // Set DFT method via functional name
      scf_solver_dft->settings().set("method", "b3lyp");
      const auto& [E_dft, wfn_dft] = scf_solver_dft->run(structure, 0, 1);


.. tab:: Python API

   .. literalinclude:: ../../../../examples/scf_solver.py
      :language: python
      :lines: 34-41

The PyScf solver class is available only in python (for more details regarding 3rd party implementations, refer to :doc:`Interfaces <../design/interfaces>`
.. tab:: Python API

   .. literalinclude:: ../../../../examples/scf_solver.py
      :language: python
      :lines: 48-53


Related classes
---------------

- :doc:`Structure <../data/structure>`: Input molecular structure
- :doc:`Orbitals <../data/orbitals>`: Output optimized molecular orbitals

Related topics
--------------
- :doc:`Interfaces <../design/interfaces>`: Interfaces of 3rd party codes into QDK/Chemistry
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <../design/factory_pattern>`: Understanding algorithm creation
- :doc:`../basis_functionals`: Exchange-correlation functionals for DFT calculations
