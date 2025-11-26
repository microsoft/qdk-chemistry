Hamiltonian
===========

The ``Hamiltonian`` class in QDK/Chemistry represents the electronic Hamiltonian operator, which describes the physics of a quantum system.
It contains the one- and two-electron integrals that are essential for quantum chemistry calculations, particularly for active space methods.

Overview
--------

In quantum chemistry, the electronic Hamiltonian is the operator that gives the energy of a system of electrons.
The ``Hamiltonian`` class in QDK/Chemistry stores the matrix elements of this operator in the basis of molecular orbitals.
These matrix elements consist of one-electron integrals (representing kinetic energy and electron-nucleus interactions) and two-electron integrals (representing electron-electron repulsion).

Design principles
~~~~~~~~~~~~~~~~~

The ``Hamiltonian`` class follows an immutable data model design principle as described in the :doc:`QDK/Chemistry Design Principles <../advanced/design_principles>` document.
Once properly constructed, the Hamiltonian data is typically not modified during calculations.
This const-correctness approach ensures data integrity throughout computational workflows and prevents accidental modifications of the core quantum system representation.
While setter methods are available for construction and initialization purposes, in normal operation the Hamiltonian object should be treated as immutable after it has been fully populated.

Properties
----------

- **One-electron integrals**: Matrix of one-electron integrals (h₁)
- **Two-electron integrals**: Vector of two-electron integrals (h₂) in physicist notation :math:`\left\langle ij|kl \right\rangle`
- **Core energy**: Constant energy term combining nuclear repulsion and inactive orbital contributions
- **Inactive Fock matrix**: Matrix representing interactions between active and inactive orbitals
- **Orbitals**: Molecular orbital information for the system (see the :doc:`Orbitals <orbitals>` documentation for detailed information about orbital properties and representations)
- **Selected orbital indices**: Indices defining the active space orbitals
- **Number of electrons**: Count of electrons in the active space

Restricted vs. unrestricted Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Hamiltonian`` class supports both restricted and unrestricted representations:

- **Restricted**: Uses the same spatial orbitals for alpha and beta electrons. Suitable for closed-shell systems where alpha and beta electrons occupy the same spatial orbitals with opposite spins.
- **Unrestricted**: Allows different spatial orbitals for alpha and beta electrons.

For unrestricted Hamiltonians, the one-electron and two-electron integrals are stored separately for each spin channel:

- One-electron integrals: :math:`h_{\alpha\alpha}` and :math:`h_{\beta\beta}`
- Two-electron integrals: :math:`h_{\alpha\alpha\alpha\alpha}`, :math:`h_{\alpha\beta\alpha\beta}`, and :math:`h_{\beta\beta\beta\beta}`

Usage
-----

The ``Hamiltonian`` class is typically used as input to correlation methods such as Configuration Interaction (CI) and Multi-Configuration Self-Consistent Field (MCSCF) calculations.
The :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm is the primary tool for generating ``Hamiltonian`` objects from molecular data.

Creating a Hamiltonian object
-----------------------------

The ``Hamiltonian`` object is typically created using the :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm (recommended approach for most users), or it can be created directly with the appropriate integral data. Once properly constructed with all required data, the
Hamiltonian object should be considered constant and not modified:

.. tab:: C++ API

   .. code-block:: cpp
      // Create a simple structure
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}};
      std::vector<std::string> symbols = {"H", "H"};
      Structure structure(coords, symbols);

      // Run initial SCF
      auto scf_solver = ScfSolverFactory::create();
      auto [E_HF, wfn_HF] = scf_solver->run(structure, 0, 1);

      // Create a Hamiltonian constructor
      auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

      // Construct the Hamiltonian from orbitals
      auto hamiltonian = hamiltonian_constructor->run(wfn_HF->get_orbitals());

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 16-29

An unrestricted Hamiltonian is created by default, if an open-shell system is specified. The orbitals passed to the Hamiltonian need to be unrestricted as well, but this will also happen by default using the same pipeline as above. For example:

.. tab:: C++ API

   .. code-block:: cpp
      // Create O2 (spin and multiplicity are defined below)
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {2.3, 0.0, 0.0}};
      std::vector<std::string> symbols = {"O", "O"};
      Structure structure(coords, symbols);

      // Run initial SCF
      auto scf_solver = ScfSolverFactory::create();
      auto [E_UHF, wfn_UHF] = scf_solver->run(structure, 0, 3); // open shell - this will run UHF

      // Create a Hamiltonian constructor
      auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

      // Construct the Hamiltonian from orbitals
      auto hamiltonian = hamiltonian_constructor->run(wfn_UHF->get_orbitals());
      // Here, the Hamiltonian will be unrestricted by default and use the UHF orbitals

      // Can double check it is unrestricted like
      bool is_unrestricted = hamiltonian->is_unrestricted();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 35-52

Custom construction patterns
----------------------------
If desired, the Hamiltonian can also be constructed directly using one- and two-electron integrals and inactive Fock matrices. As above, there are related constructors for restricted and unrestricted Hamiltonians.

.. tab:: C++ API

   .. code-block:: cpp
      // Create restricted orbitals
      size_t num_orbitals = 2;

      // Create basis set
      std::vector<Shell> shells;
      for (size_t i = 0; i < num_orbitals; i++) {
         Eigen::VectorXd exponents(1);
         exponents << 1.0;
         Eigen::VectorXd coefficients(1);
         coefficients << 1.0;
         shells.emplace_back(Shell(i, OrbitalType::S, exponents, coefficients));
      }
      auto basis_set = std::make_shared<BasisSet>("test", shells);

      // Create restricted orbitals
      Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(num_orbitals, num_orbitals);
      auto restricted_orbitals = std::make_shared<Orbitals>(
         coeffs,
         std::nullopt,
         std::nullopt,  // No orbital energies
         basis_set
      );

      // Create one-body integrals
      Eigen::MatrixXd one_body(num_orbitals, num_orbitals);
      one_body << 1.0, 0.2,
                  0.2, 1.5;

      // Create two-body integrals
      std::random_device rd;
      std::mt19937 gen(42);  // Use seed 42 for reproducibility
      std::uniform_real_distribution<> dis(0.0, 1.0);
      Eigen::VectorXd two_body(num_orbitals * num_orbitals * num_orbitals * num_orbitals);
      for (size_t i = 0; i < two_body.size(); i++) {
         two_body(i) = dis(gen);
      }

      // Inactive Fock matrix
      Eigen::MatrixXd inactive_fock(num_orbitals, num_orbitals);
      inactive_fock << 0.5, 0.1,
                     0.1, 0.7;

      // Construct Hamiltonian directly
      Hamiltonian h_restricted(
         one_body,
         two_body,
         restricted_orbitals,
         2.0,  // core_energy
         inactive_fock
      );

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 59-90

Likewise for unrestricted:
.. tab:: C++ API

   .. code-block:: cpp
      // Create test unrestricted orbitals
      size_t num_orbitals = 2;

      // Create basis set
      std::vector<Shell> shells;
      for (size_t i = 0; i < num_orbitals; i++) {
         Eigen::VectorXd exponents(1);
         exponents << 1.0;
         Eigen::VectorXd coefficients(1);
         coefficients << 1.0;
         shells.emplace_back(Shell(0, OrbitalType::S, exponents, coefficients));
      }
      auto basis_set = std::make_shared<BasisSet>("test", shells);

      // Create unrestricted orbitals with different alpha and beta coefficients
      Eigen::MatrixXd coeffs_alpha = Eigen::MatrixXd::Identity(num_orbitals, num_orbitals);
      Eigen::MatrixXd coeffs_beta(num_orbitals, num_orbitals);
      coeffs_beta << 0.8, 0.6,
                     0.6, -0.8;

      auto unrestricted_orbitals = std::make_shared<Orbitals>(
         coeffs_alpha,
         coeffs_beta,
         std::nullopt, // No orbital energies alpha
         std::nullopt, // No orbital energies beta
         std::nullopt, // No active space specification
         basis_set
      );

      // Create unrestricted integral data (alpha and beta are different)
      Eigen::MatrixXd one_body_alpha(num_orbitals, num_orbitals);
      one_body_alpha << 1.0, 0.2,
                        0.2, 1.5;

      Eigen::MatrixXd one_body_beta(num_orbitals, num_orbitals);
      one_body_beta << 1.1, 0.3,
                     0.3, 1.6;

      // Create spin-separated two-body integrals
      std::random_device rd;
      std::mt19937 gen(42);  // Use seed 42 for reproducibility
      std::uniform_real_distribution<> dis(0.0, 1.0);

      size_t two_body_size = num_orbitals * num_orbitals * num_orbitals * num_orbitals;
      Eigen::VectorXd two_body_aaaa(two_body_size);
      Eigen::VectorXd two_body_aabb(two_body_size);
      Eigen::VectorXd two_body_bbbb(two_body_size);

      for (size_t i = 0; i < two_body_size; i++) {
         two_body_aaaa(i) = dis(gen);
      }
      for (size_t i = 0; i < two_body_size; i++) {
         two_body_aabb(i) = dis(gen);
      }
      for (size_t i = 0; i < two_body_size; i++) {
         two_body_bbbb(i) = dis(gen);
      }

      // Inactive Fock matrices (also spin-separated)
      Eigen::MatrixXd inactive_fock_alpha(num_orbitals, num_orbitals);
      inactive_fock_alpha << 0.5, 0.1,
                           0.1, 0.7;

      Eigen::MatrixXd inactive_fock_beta(num_orbitals, num_orbitals);
      inactive_fock_beta << 0.6, 0.2,
                           0.2, 0.8;

      // Construct unrestricted Hamiltonian directly
      Hamiltonian h_unrestricted(
         one_body_alpha,
         one_body_beta,
         two_body_aaaa,
         two_body_aabb,
         two_body_bbbb,
         unrestricted_orbitals,
         2.0,  // core_energy
         inactive_fock_alpha,
         inactive_fock_beta
      );

      // Check if Hamiltonian is unrestricted
      bool is_unrestricted = h_unrestricted.is_unrestricted();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 96-139

Accessing Hamiltonian data
--------------------------

The ``Hamiltonian`` class provides methods to access the one- and two-electron integrals and other properties. In line
with its immutable design principle, these methods return const references or copies of the internal data:

Two-Electron Integral Storage and Notation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-electron integrals in quantum chemistry can be represented using different notations and storage formats.
QDK/Chemistry uses the physicist notation by default, but it's important to understand the different conventions:

- **Physicist/Dirac notation** :math:`\left\langle ij|kl \right\rangle` or :math:`\left\langle ij|kl \right\rangle`: represents the Coulomb interaction where electron 1 occupies orbitals :math:`i` and :math:`k`, while electron 2 occupies orbitals :math:`j` and :math:`l`.
  This is the default representation in QDK/Chemistry.
  In this notation, the first index of each pair :math:`(i,k)` refers to electron 1, and the second index of each pair :math:`(j,l)` refers to electron 2, following a (1,2,1,2) electron indexing pattern.

- **Chemist/Mulliken notation** :math:`(ij|kl)` or :math:`[ij|kl]`: represents the Coulomb interaction where electron 1 occupies orbitals :math:`i` and :math:`j`, while electron 2 occupies orbitals :math:`k` and :math:`l`.
  In this notation, the first pair of indices :math:`(i,j)` refers to electron 1, and the second pair :math:`(k,l)` refers to electron 2, following a (1,1,2,2) electron indexing pattern.
  The symbols differ (parentheses vs square brackets), but the indexing convention is the same.

The relationship between physicist and chemist notation is:

.. math::

   \left\langle ij | kl \right\rangle = \left(ik|jl \right)

Two-electron integrals with real-valued orbitals possess inherent symmetry properties. From a theoretical perspective,
these symmetries can be expressed as:

.. math::

   \left\langle ij|kl \right\rangle = \left\langle ji|lk \right\rangle = \left\langle kl|ij \right\rangle = \left\langle lk|ji \right\rangle = \left\langle jl|ki \right\rangle = \left\langle lj|ik \right\rangle = \left\langle ki|jl \right\rangle = \left\langle ik|lj \right\rangle

These permutational symmetries arise from the mathematical properties of the two-electron repulsion integrals.
When accessing specific elements with ``get_two_body_element(i, j, k, l)``, the function handles the appropriate index mapping to retrieve the correct value based on the implementation's storage format.

.. tab:: C++ API

   .. code-block:: cpp

      // Access one-electron integrals, returns tuple of const Eigen::MatrixXd&
      // For restricted hamiltonians, these point to the same data
      auto [h1, h1_dup] = hamiltonian->get_one_body_integrals();

      // Access specific one-electron integral element <ij>
      double element_one = hamiltonian->get_one_body_element(0,0)

      // Access two-electron integrals, returns triple of const Eigen::VectorXd&
      // For restricted hamiltonians, these point to the same data
      auto [h2, h2_dup1, h2_dup2] = hamiltonian->get_two_body_integrals();

      // Access a specific two-electron integral <ij|kl>
      double element_two = hamiltonian->get_two_body_element(0, 0, 0, 0);

      // Get core energy (nuclear repulsion + inactive orbital energy)
      auto core_energy = hamiltonian->get_core_energy();

      // Get inactive Fock matrix (if available)
      if (hamiltonian.has_inactive_fock_matrix()) {
          auto [inactive_fock, inactive_fock_dup] = hamiltonian->get_inactive_fock_matrix();
      }

      // Get orbital data
      const auto& orbitals = hamiltonian->get_orbitals();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 145-167

In the unrestricted case we can access the spin-separated one-electron and two-electron integrals and Fock matrices:
.. tab:: C++ API

   .. code-block:: cpp

      // Access one-electron integrals, returns tuple of const Eigen::MatrixXd&
      auto [h1_alpha, h1_beta] = hamiltonian->get_one_body_integrals();

      // Access specific elements of one-electron integrals
      double element_one_aa = hamiltonian->get_one_body_element(0,0, SpinChannel::aa);
      double element_one_bb = hamiltonian->get_one_body_element(0,0, SpinChannel::bb)

      // Access two-electron integrals, returns triple of const Eigen::VectorXd&
      auto [h2_aaaa, h2_aabb, h2_bbbb] = hamiltonian->get_two_body_integrals();

      // Access a specific two-electron integral <ij|kl>
      double element_aaaa = hamiltonian->get_two_body_element(0, 0, 0, 0, SpinChannel::aaaa);
      double element_aabb = hamiltonian->get_two_body_element(0, 0, 0, 0, SpinChannel::aabb);

      // Get inactive Fock matrix (if available)
      if (hamiltonian.has_inactive_fock_matrix()) {
          auto [inactive_fock_alpha, inactive_fock_beta] = hamiltonian->get_inactive_fock_matrix();
      }

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 173-189

Related classes
---------------

- :doc:`Orbitals <orbitals>`: Molecular orbital information used to construct the Hamiltonian
- :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`: Algorithm for constructing Hamiltonians -
  **primary tool** for generating Hamiltonian objects from molecular data
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Uses the Hamiltonian for correlation calculations
- :doc:`Wavefunction <wavefunction>`: Represents the solution of the Hamiltonian eigenvalue problem
- :doc:`Active space methods <../algorithms/active_space>`: Selection and use of active spaces with the Hamiltonian

Related topics
--------------

- :doc:`Serialization <../advanced/serialization>`: Data serialization and deserialization in QDK/Chemistry
- :doc:`Design principles <../advanced/design_principles>`: Design principles for data classes in QDK/Chemistry
- :doc:`Settings <../advanced/settings>`: Configuration options for algorithms operating on Hamiltonians
