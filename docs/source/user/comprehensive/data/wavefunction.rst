Wavefunction
============

The :class:`~qdk_chemistry.data.Wavefunction` class in QDK/Chemistry represents quantum mechanical wavefunctions for molecular systems.
This class provides access to wavefunction coefficients, determinants, reduced density matrices (RDMs), and other quantum chemical properties essential for post-Hartree-Fock calculations.

Overview
--------

A wavefunction in quantum chemistry describes the quantum state of a molecular system.
In QDK/Chemistry, the :class:`~qdk_chemistry.data.Wavefunction` class encapsulates various wavefunction types, from simple single-determinant Hartree-Fock wavefunctions to complex multi-reference wavefunctions used in advanced correlation methods.

The class uses a container-based design where different wavefunction types (Slater determinants, configuration interaction, etc.) are implemented as specialized container classes, while the main :class:`~qdk_chemistry.data.Wavefunction` class provides a unified interface.

Wavefunction Types
~~~~~~~~~~~~~~~~~~

QDK/Chemistry supports multiple wavefunction representations through different container types:

- **Slater Determinant**: Single-determinant wavefunctions (e.g., from Hartree-Fock calculations)
- **CAS (Complete Active Space)**: Multi-determinant wavefunctions with full CI in the active space
- **SCI (Selected Configuration Interaction)**: Sparse multi-determinant wavefunctions

#TODO PVG once the unrestricted PR goes in, we need to add containers for CC and MP2.

Properties
~~~~~~~~~~

The :class:`~qdk_chemistry.data.Wavefunction` class provides access to:

- **Coefficients**: Expansion coefficients for each determinant
- **Determinants**: Electronic configurations
- **Electron counts**: Total and active space electron numbers
- **Orbital occupations**: Natural orbital occupations for all and active orbitals
- **Reduced Density Matrices (RDMs)**: One- and two-particle density matrices
- **Orbital entropies**: Single orbital entropies for correlation analysis
- **Overlap calculations**: Inner products between wavefunctions

Mathematical representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Container type**: Underlying implementation (CAS, SCI, Slater determinant, etc.)

Container types
---------------

QDK/Chemistry supports different wavefunction container types for various quantum chemistry methods:

#TODO PVG once unrestricted PR is merged, add mp2 and cc containers here as well.

Slater determinant container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wavefunctions are represented as linear combinations of determinants:

.. math::

   |\Psi\rangle = \sum_I c_I |\Phi_I\rangle

where :math:`c_I` are expansion coefficients and :math:`|\Phi_I\rangle` are Slater determinants.

Usage
-----

The :class:`~qdk_chemistry.data.Wavefunction` class is typically created as the output of electronic structure calculations.

Most commonly, wavefunctions are obtained from :doc:`SCF calculations <../algorithms/scf_solver>`:

.. tab:: C++ API

   .. code-block:: cpp

      // Create H2 molecule
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
      std::vector<std::string> symbols = {"H", "H"};
      Structure structure(coords, symbols);

      // Run SCF calculation
      auto scf_solver = ScfSolverFactory::create();
      scf_solver->settings().set("basis_set", "sto-3g");
      auto [E_scf, wavefunction] = scf_solver->run(structure, 0, 1);

      // Access wavefunction properties
      auto orbitals = wavefunction->get_orbitals();
      std::string container_type = wavefunction->get_container_type(); // "sd"
      WavefunctionType wf_type = wavefunction->get_type();

      // Get determinant information
      auto determinants = wavefunction->get_active_determinants();
      auto coefficients = wavefunction->get_coefficients();
      size_t num_dets = wavefunction->size();

      // Get electron counts
      auto [n_alpha_total, n_beta_total] = wavefunction->get_total_num_electrons();
      auto [n_alpha_active, n_beta_active] = wavefunction->get_active_num_electrons();

      // Get RDMs for active orbitals
      auto one_rdm_active = wavefunction->get_active_one_rdm_spin_traced();
      auto two_rdm_active = wavefunction->get_active_two_rdm_spin_traced();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/wavefunction.py
      :language: python
      :start-after: # start-cell-create-cas
      :end-before: # end-cell-create-cas

SCI wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

For Selected Configuration Interaction methods.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-sci
      :end-before: // end-cell-create-sci

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-sci
      :end-before: # end-cell-create-sci

Accessing wavefunction data
---------------------------

The :class:`~qdk_chemistry.data.Wavefunction` class provides methods to access coefficients, determinants, and derived properties:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-data
      :end-before: // end-cell-access-data

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-data
      :end-before: # end-cell-access-data


Related classes
---------------

- :doc:`Structure <structure>`: Molecular structure that defines the system
- :doc:`Orbitals <orbitals>`: Orbital basis set for the wavefunction
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from wavefunction
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces SCF wavefunctions
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Algorithm for multi-configuration wavefunctions

Related topics
--------------

- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Active space methods <../algorithms/active_space>`: Active space selection from wavefunctions
