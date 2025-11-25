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

Restricted vs. unrestricted Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Hamiltonian`` class supports both restricted and unrestricted representations:

- **Restricted**: Uses the same spatial orbitals for alpha and beta electrons. Suitable for closed-shell systems where alpha and beta electrons occupy the same spatial orbitals with opposite spins.
- **Unrestricted**: Allows different spatial orbitals for alpha and beta electrons.

For unrestricted Hamiltonians, the one-electron and two-electron integrals are stored separately for each spin channel:

- One-electron integrals: :math:`h_{\alpha\alpha}` and :math:`h_{\beta\beta}`
- Two-electron integrals: :math:`h_{\alpha\alpha\alpha\alpha}`, :math:`h_{\alpha\beta\alpha\beta}`, and :math:`h_{\beta\beta\beta\beta}`

Properties
----------

- **One-electron integrals**: Matrix of one-electron integrals (h₁)
- **Two-electron integrals**: Vector of two-electron integrals (h₂) in physicist notation :math:`\left\langle ij|kl \right\rangle`
- **Core energy**: Constant energy term combining nuclear repulsion and inactive orbital contributions
- **Inactive Fock matrix**: Matrix representing interactions between active and inactive orbitals
- **Orbitals**: Molecular orbital information for the system (see the :doc:`Orbitals <orbitals>` documentation for detailed information about orbital properties and representations)
- **Selected orbital indices**: Indices defining the active space orbitals
- **Number of electrons**: Count of electrons in the active space

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

      // Create a Hamiltonian constructor
      auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

      // Set active orbitals if needed
      std::vector<size_t> active_orbitals = {4, 5, 6, 7}; // Example indices
      hamiltonian_constructor->settings().set("active_orbitals", active_orbitals);

      // Construct the Hamiltonian from orbitals
      // (assuming 'orbitals' object exists from prior calculation)
      // auto hamiltonian = hamiltonian_constructor->run(orbitals);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 15-24

Creating an unrestricted Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unrestricted Hamiltonians can be created using a specialized constructor that accepts separate integral tensors for each spin channel:

.. tab:: C++ API

   .. code-block:: cpp

      // Create unrestricted integral data (alpha and beta are different)
      Eigen::MatrixXd one_body_alpha(2, 2);
      one_body_alpha << 1.0, 0.2, 0.2, 1.5;
      Eigen::MatrixXd one_body_beta(2, 2);
      one_body_beta << 1.1, 0.3, 0.3, 1.6;

      // Create spin-separated two-body integrals
      size_t num_orbitals = 2;
      Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Random(num_orbitals * num_orbitals * num_orbitals * num_orbitals);
      Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Random(num_orbitals * num_orbitals * num_orbitals * num_orbitals);
      Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Random(num_orbitals * num_orbitals * num_orbitals * num_orbitals);

      // Inactive Fock matrices (also spin-separated)
      Eigen::MatrixXd inactive_fock_alpha(2, 2);
      inactive_fock_alpha << 0.5, 0.1, 0.1, 0.7;
      Eigen::MatrixXd inactive_fock_beta(2, 2);
      inactive_fock_beta << 0.6, 0.2, 0.2, 0.8;

      // Construct unrestricted Hamiltonian directly
      Hamiltonian h_unrestricted(
          one_body_alpha,
          one_body_beta,
          two_body_aaaa,
          two_body_aabb,
          two_body_bbbb,
          unrestricted_orbitals,
          2.0,                           // Core energy
          inactive_fock_alpha,
          inactive_fock_beta
      );

      // Check if Hamiltonian is unrestricted
      bool is_unrestricted = h_unrestricted.is_unrestricted();
      bool is_restricted = h_unrestricted.is_restricted();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 30-78

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
      auto [h1_alpha, h1_beta] = hamiltonian.get_one_body_integrals();

      // Access two-electron integrals, returns triple of const Eigen::VectorXd&
      // For restricted hamiltonians, these point to the same data
      auto [h2_aaaa, h2_aabb, h2_bbbb] = hamiltonian.get_two_body_integrals();

      // Access a specific two-electron integral <ij|kl>
      double element = hamiltonian.get_two_body_element(i, j, k, l);

      // Get core energy (nuclear repulsion + inactive orbital energy)
      auto core_energy = hamiltonian.get_core_energy();

      // Get inactive Fock matrix (if available)
      if (hamiltonian.has_inactive_fock_matrix()) {
          auto [inactive_fock_alpha, inactive_fock_beta] = hamiltonian.get_inactive_fock_matrix();
      }

      // Get orbital data
      const auto& orbitals = hamiltonian.get_orbitals();

      // Get active space information
      auto active_indices = hamiltonian.get_selected_orbital_indices();
      auto num_electrons = hamiltonian.get_num_electrons();
      auto num_orbitals = hamiltonian.get_num_orbitals();

      // For unrestricted Hamiltonians, access specific one-electron integral channels
      double integral_aa = h_unrestricted.get_one_body_element(0, 0, SpinChannel::aa);
      double integral_bb = h_unrestricted.get_one_body_element(0, 0, SpinChannel::bb);

      // For unrestricted Hamiltonians, access specific two-electron integral channels
      double integral_aaaa = h_unrestricted.get_two_body_element(0, 0, 0, 0, SpinChannel::aaaa);
      double integral_aabb = h_unrestricted.get_two_body_element(0, 0, 0, 0, SpinChannel::aabb);
      double integral_bbbb = h_unrestricted.get_two_body_element(0, 0, 0, 0, SpinChannel::bbbb);

      // Access fock matrices for alpha and beta
      auto [fock_alpha, fock_beta] = h_unrestricted.get_inactive_fock_matrix();

      // Get orbital data
      const auto& orbitals_unrestricted = h_unrestricted.get_orbitals();

      // Get active space information
      auto active_indices_unrestricted = h_unrestricted.get_selected_orbital_indices();
      auto num_electrons_unrestricted = h_unrestricted.get_num_electrons();
      auto num_orbitals_unrestricted = h_unrestricted.get_num_orbitals();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 84-110

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 112-127

File formats
~~~~~~~~~~~~

QDK/Chemistry supports multiple serialization formats for Hamiltonian data:

JSON format
^^^^^^^^^^^

JSON representation of a ``Hamiltonian`` object has the following structure (showing simplified content):

.. code-block:: json

  {
    "core_energy":0.0,
    "has_one_body_integrals":true,
    "has_orbitals":true,
    "has_two_body_integrals":true,
    "num_electrons":2,
    "num_orbitals":2,
    "one_body_integrals":[[-0.7789220366556091,-1.1102230246251565e-16],
      [-1.6653345369377348e-16,-0.6702666733672852]],
    "orbitals":{"..."},
    "selected_orbital_indices":[0,1],
    "two_body_integrals":["..."],
    "is_restricted":true,
  }

.. note::
   The ``orbitals`` field contains a nested ``Orbitals`` object with its own serialization structure.
   For detailed information about the serialization format of the ``Orbitals`` data contained within the Hamiltonian, please refer to the :ref:`Orbitals Serialization <orbitals-serialization>` section.

HDF5 format
^^^^^^^^^^^

HDF5 representation of a ``Hamiltonian`` object has the following structure (showing groups and datasets):

.. code-block:: text
/
├── selected_orbital_indices  # Dataset: uint32, 1D array active space orbital indices
├── one_body_integrals        # Dataset: float64, 1D array of one-electron integrals
├── two_body_integrals        # Dataset: float64, 2D array of one-electron integrals
├── metadata/                     # Group
│   ├── core_energy           # Attribute: float64, core energy
│   ├── has_orbitals          # Attribute: uint8, 0 if false, 1 if true
│   ├── is_restricted         # Attribute: uint8, 0 if false, 1 if true
│   ├── num_electrons         # Attribute: uint32, number of electrons (in the active space)
│   └── num_orbitals          # Attribute: uint32, number of orbitals (in the active space)
└── orbitals/                     # Group
      └── json_data             # Dataset: (), binary representation of the json orbital data

.. note::
   The ``orbitals/`` group follows the same structure and organization as an independent ``Orbitals`` HDF5 file.
   For complete details on the structure and content of this group, see the :ref:`Orbitals Serialization <orbitals-serialization>` section in the Orbitals documentation.

.. tab:: C++ API

   .. code-block:: cpp

      // Serialize to JSON file
      hamiltonian.to_json_file("molecule.hamiltonian.json");

      // Deserialize from JSON file
      auto hamiltonian_from_json_file = Hamiltonian::from_json_file("molecule.hamiltonian.json");

      // Serialize to HDF5 file
      hamiltonian.to_hdf5_file("molecule.hamiltonian.h5");

      // Deserialize from HDF5 file
      auto hamiltonian_from_hdf5_file = Hamiltonian::from_hdf5_file("molecule.hamiltonian.h5");

      // Generic file I/O based on type parameter
      hamiltonian.to_file("molecule.hamiltonian.json", "json");
      auto hamiltonian_from_file = Hamiltonian::from_file("molecule.hamiltonian.h5", "hdf5");

      // Convert to/from JSON object
      nlohmann::json j = hamiltonian.to_json();
      auto hamiltonian_from_json = Hamiltonian::from_json(j);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 133-153

Active space Hamiltonian
------------------------

When constructed with active orbital specifications, the ``Hamiltonian`` represents an active space Hamiltonian, which is a projection of the full electronic Hamiltonian into a smaller subspace.
This is essential for tractable multi-configuration calculations.
The :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm handles the complex process of generating an appropriate active space Hamiltonian based on your specifications.

Validation methods
------------------

The ``Hamiltonian`` class provides methods to check the validity and consistency of its data:

.. tab:: C++ API

   .. code-block:: cpp

      // Check if the Hamiltonian data is complete and consistent
      bool valid = hamiltonian.is_valid();

      // Check if specific components are available
      bool has_one_body = hamiltonian.has_one_body_integrals();
      bool has_two_body = hamiltonian.has_two_body_integrals();
      bool has_orbitals = hamiltonian.has_orbitals();
      bool has_inactive_fock = hamiltonian.has_inactive_fock_matrix();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian.py
      :language: python
      :lines: 159-167

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
