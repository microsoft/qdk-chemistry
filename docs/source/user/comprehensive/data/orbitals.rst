Orbitals
========

The :class:`~qdk_chemistry.data.Orbitals` class in QDK/Chemistry represents a set of molecular orbitals.
This class stores orbital coefficients, energies, and other properties necessary for quantum chemical calculations.

Overview
--------

Molecular orbitals are a fundamental concept in quantum chemistry.
They are formed through linear combinations of atomic orbitals and provide a framework for understanding chemical bonding and electronic structure.
In QDK/Chemistry, the :class:`~qdk_chemistry.data.Orbitals` class encapsulates all relevant information about these orbitals, including their coefficients, energies, and occupation numbers.

Restricted vs. unrestricted calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.data.Orbitals` class supports both restricted and unrestricted calculations:

- **Restricted**: Alpha and beta electrons share the same spatial orbitals (:term:`RHF`, :term:`RKS`)
- **Unrestricted**: Alpha and beta electrons have separate spatial orbitals (:term:`UHF`, :term:`UKS`)

For restricted calculations, the alpha and beta components are identical. The class maintains separate alpha and beta
data internally, but they reference the same underlying data for restricted cases.

Model Orbitals
~~~~~~~~~~~~~~

Model orbitals are a simpler subclass of Orbitals in QDK/Chemistry, for model systems without any basis set information.
This class allows to fully specify model Hamiltonians and Wavefunctions.
Several properties present for the Orbitals subclass are missing for Model Orbitals: coefficients, energies, etc. These are summarized in the properties table below.

Properties
~~~~~~~~~~

The following table summarizes the properties available for the different orbital types:

.. list-table:: Orbital Properties Availability
   :widths: 25 40 15 20
   :header-rows: 1

   * - Property
     - Description
     - Orbitals
     - ModelOrbitals
   * - **Coefficients**
     - Matrix of orbital coefficients [AO x MO] for alpha and beta spin channels
     - ✓
     - ✗
   * - **Energies**
     - Vector of orbital energies for alpha and beta spin channels
     - ✓
     - ✗
   * - **Active space indices**
     - Active space indices for alpha and beta spin channels
     - ✓
     - ✓
   * - **Inactive space indices**
     - Inactive space indices for alpha and beta spin channels
     - ✓
     - ✓
   * - **Virtual space indices**
     - Virtual space indices for alpha and beta spin channels
     - ✓
     - ✓
   * - **MO overlap**
     - Overlap matrices between molecular orbitals, for both spin channels
     - ✓
     - ✓*
   * - **Basis Set**
     - Comprehensive basis set information
     - ✓
     - ✗
   * - **AO overlap**
     - Overlap matrices between atomic orbitals, for both spin channels
     - ✓
     - ✗

.. note::
   \* For ModelOrbitals, MO overlap matrices return identity matrices since model systems assume orthonormal orbitals.

For detailed information about basis sets in QDK/Chemistry, including available basis sets, creation, manipulation, and serialization, refer to the :doc:`Basis Set documentation <basis_set>`.


Usage
-----

The :class:`~qdk_chemistry.data.Orbitals` class is typically created as the output of an :doc:`SCF calculation <../algorithms/scf_solver>` or :doc:`orbital transformation <../algorithms/localizer>`.
It serves as input to various post-:term:`HF` methods such as :doc:`active space selection <../algorithms/active_space>` and :doc:`Hamiltonian construction <../algorithms/hamiltonian_constructor>`.

- **Orbital Localization**: Transform delocalized :term:`SCF` orbitals into localized representations for better chemical interpretation and more efficient correlation methods.
  See :doc:`Localizer <../algorithms/localizer>` for details.

- **Active Space Selection**: Automatically identify important orbitals for multi-reference calculations based on
  various criteria.
  See :doc:`ActiveSpaceSelector <../algorithms/active_space>` for details.

- **Hamiltonian Construction**: Build electronic Hamiltonians for post-HF methods using the orbital information.
  See :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` for details.

The below example illustrates the typical access to Orbitals (via an SCF):

.. tab:: C++ API

   .. code-block:: cpp

      // Create H2 molecule
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
      std::vector<std::string> symbols = {"H", "H"};
      Structure structure(coords, symbols);

      // Obtain orbitals from an SCF calculation
      auto scf_solver = ScfSolverFactory::create();
      scf_solver->settings().set("basis_set", "sto-3g");
      auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);
      std::shared_ptr<Orbitals> orbitals = wfn.get_orbitals();

      // Access orbital coefficients
      auto [coeffs, coeffs_beta] = orbitals->get_coefficients();

      // Access orbital energies
      auto [energies, energies_beta] = orbitals->get_energies();

      // Access atomic orbital overlap matrix
      const Eigen::MatrixXd& = orbitals->get_overlap_matrix();


.. tab:: Python API

   .. literalinclude:: ../../../../examples/orbitals.py
      :language: python
      :lines: 16-34

Below demonstrates the simple construction of ModelOrbitals, which can then be passed to the :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`

.. tab:: C++ API

   .. code-block:: cpp

   // Set basis set size
   size_t basis_size = 6;

   // Set active orbitals
   std::vector<size_t> alpha_active = {1,2};
   std::vector<size_t> beta_active = {2,3,4};
   std::vector<size_t> alpha_inactive = {0,3,4,5};
   std::vector<size_t> beta_inactive = {0,1,5};

   ModelOrbitals model_orbitals(basis_size, std::make_tuple(alpha_active, beta_active, alpha_inactive, beta_inactive));

   // We can then pass this object to a custom Hamiltonian constructor


.. tab:: Python API

   .. literalinclude:: ../../../../examples/orbitals.py
      :language: python
      :lines: 40-51


Related classes
---------------

- :doc:`Structure <structure>`: Molecular structure representation
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from orbitals
- :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`: Algorithm that builds Hamiltonians from orbitals
- :doc:`ActiveSpaceSelector <../algorithms/active_space>`: Algorithm for selecting active spaces from orbitals
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces orbitals
- :doc:`Localizer <../algorithms/localizer>`: Algorithms for orbital transformations

Related topics
--------------

- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
