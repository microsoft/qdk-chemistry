Orbital localization
====================

The :class:`~qdk_chemistry.algorithms.Localizer` algorithm in QDK/Chemistry performs various orbital transformations to create localized or otherwise transformed molecular orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes an :doc:`Orbitals <../data/orbitals>` instance as input and produces a new :doc:`Orbitals <../data/orbitals>` instance as output.
These transformations preserve the overall electronic state but provide orbitals with different properties that are useful for chemical analysis or subsequent calculations.

Overview
--------

Canonical molecular orbitals from :term:`SCF` calculations are often delocalized over the entire molecule, which can make chemical interpretation difficult and lead to slow convergence in post-:term:`HF` methods.
The :class:`~qdk_chemistry.algorithms.Localizer` algorithm applies unitary transformations to these orbitals to obtain alternative representations that may be more physically intuitive or computationally advantageous.
Multiple localization methods are available through a unified interface, each optimizing different criteria to achieve localization.

Localization methods
--------------------

QDK/Chemistry provides several orbital transformation methods through the :class:`~qdk_chemistry.algorithms.Localizer` interface:

**Pipek-Mezey Localization**
   Maximizes the sum of squared Mulliken charges on each atom for each orbital, creating orbitals that are maximally localized on specific atoms or bonds.

**MP2 Natural Orbitals** 
   Transforms canonical orbitals into natural orbitals based on MP2 density matrices, providing orbitals that diagonalize the correlation effects.

**Valence Virtual Hard Virtual (VVHV) Orbitals**
   Separates orbitals into valence, virtual, and hard virtual categories for more efficient treatment in correlation methods.

Usage
-----

Before performing localization, you need an :doc:`Orbitals <../data/orbitals>` instance as input.
This is typically obtained from an :doc:`ScfSolver <scf_solver>` calculation, as localization is usually applied to converged :term:`SCF` orbitals.

The most common use case is localizing occupied orbitals after an SCF calculation:

.. tab:: C++ API

   .. code-block:: cpp

      // Create H2O molecule
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.757, 0.587}, {0.0, -0.757, 0.587}};
      std::vector<std::string> symbols = {"O", "H", "H"};
      Structure structure(coords, symbols);

      // Obtain orbitals from SCF calculation
      auto scf_solver = ScfSolverFactory::create();
      scf_solver->settings().set("basis_set", "sto-3g");
      auto [E_scf, wavefunction] = scf_solver->run(structure, 0, 1);
      auto orbitals = wavefunction.get_orbitals();

      // Create Pipek-Mezey localizer
      auto localizer = LocalizerFactory::create("qdk_pipek_mezey");

      // Localize occupied orbitals
      std::vector<size_t> occupied_indices = {0, 1, 2, 3, 4};  // 5 occupied orbitals for H2O
      auto localized_orbitals = localizer->run(orbitals, occupied_indices, occupied_indices);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/localizer.py
      :language: python
      :lines: 16-32


Implemented interfaces
---------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.Localizer` provides a unified interface for localization methods.

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **QDK/Chemistry**: Native implementation of Pipek-Mezey and :term:`MP2` natural orbital localization
- **QDK/Chemistry**: VVHV orbital separation method

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **PySCF**: Interface to PySCF's orbital localization methods (Pipek-Mezey, Boys, etc.)

The factory pattern allows seamless selection between these implementations.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

Related classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input and output orbitals for localization
- :doc:`ScfSolver <scf_solver>`: Produces initial orbitals for localization
- :doc:`ActiveSpaceSelector <active_space>`: Often used with localized orbitals for better active space selection
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Can build Hamiltonians using localized orbitals
- :doc:`Wavefunction <../data/wavefunction>`: Container for orbitals and electronic state information

Related topics
--------------

- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Factory pattern <../design/factory_pattern>`: Creating algorithm instances
