Basis set
=========

The ``BasisSet`` class in QDK/Chemistry represents a collection of atomic orbital basis functions used to describe the electronic structure of molecules.
It organizes basis functions into shells and provides methods for managing, querying, and serializing basis set data.

Overview
--------

In quantum chemistry, a basis set is a collection of mathematical functions used to represent molecular orbitals.
The ``BasisSet`` class in QDK/Chemistry uses a shell-based organization, where each shell contains basis functions with the same atom, angular momentum, and primitive Gaussian functions.

Key features of the ``BasisSet`` class include:

- Shell-based storage for memory efficiency
- Support for both spherical and Cartesian basis functions
- Mapping between shells/basis functions and atoms
- Mapping between shells/basis functions and orbital types
- Basis set metadata (name, parameters)
- Integration with molecular structure information
- On-demand expansion of shells to individual basis functions

Usage
-----

The ``BasisSet`` class is a fundamental component in quantum chemistry calculations, providing the mathematical foundation for representing molecular orbitals.
It's typically used as input for :term:`SCF` calculations and is usually created automatically when selecting a :doc:`predefined basis set <basis_sets>` for a calculation.

.. note::
   QDK/Chemistry provides a collection of predefined basis sets that can be accessed through the appropriate factory functions.
   For common calculations, you typically won't need to construct basis sets manually.

Core concepts
-------------

Shells and primitives
~~~~~~~~~~~~~~~~~~~~~

A shell represents a group of basis functions that share the same atom, angular momentum, and primitive functions, but differ in magnetic quantum numbers.
For example, a :math:`p`-shell contains :math:`p_x, p_y, p_z` functions.

Shells contain primitives, which are Gaussian functions defined by:

- Exponent: Controls how diffuse or tight the function is
- Coefficient: Controls the weight of the primitive in the contracted function

Orbital types
~~~~~~~~~~~~~

The ``BasisSet`` class supports various orbital types with different angular momentum:

- S orbital (angular momentum :math:`l=0`) : 1 function per shell (spherical or Cartesian)
- P orbital (angular momentum :math:`l=1`): 3 functions per shell (spherical or Cartesian)
- D orbital (angular momentum :math:`l=2`): 5 functions (spherical) or 6 functions (Cartesian) per shell
- F orbital (angular momentum :math:`l=3`): 7 functions (spherical) or 10 functions (Cartesian) per shell
- G, H, I orbitals: Higher angular momentum orbitals

Basis types
~~~~~~~~~~~

The ``BasisSet`` class supports two types of basis functions:

- **Spherical**: Uses spherical harmonics with :math:`2l+1` functions per shell
- **Cartesian**: Uses Cartesian coordinates with :math:`(l+1)(l+2)/2` functions per shell

Creating a basis set
--------------------

.. note::
   In most cases, you should use the built-in basis set library rather than creating basis sets manually.
   Manual creation is primarily for advanced use cases or when working with custom basis sets not available in the library.

.. tab:: C++ API

   .. code-block:: cpp

      // Create an empty basis set with a name
      BasisSet basis_set("6-31G", BasisType::Spherical);

      // Add a shell with multiple primitives
      size_t atom_index = 0;  // First atom
      OrbitalType orbital_type = OrbitalType::P;  // p orbital
      Eigen::VectorXd exponents(2);
      exponents << 0.16871439, 0.62391373;
      Eigen::VectorXd coefficients(2);
      coefficients << 0.43394573, 0.56604777;
      basis_set.add_shell(atom_index, orbital_type, exponents, coefficients);

      // Add a shell with a single primitive
      basis_set.add_shell(1, OrbitalType::S, 0.5, 1.0);

      // Set molecular structure
      basis_set.set_structure(structure);

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/basis_set.py
      :language: python
      :lines: 7-20

Accessing basis set data
------------------------

Following the :doc:`immutable design principle <../advanced/design_principles>` used throughout QDK/Chemistry, all getter methods return const references or copies of the data.
This ensures that the basis set data remains consistent and prevents accidental modifications that could lead to inconsistent states.

.. note::
   If you need to modify a basis set after creation, you should create a new BasisSet object with the desired
   changes rather than trying to modify an existing one.

.. tab:: C++ API

   .. code-block:: cpp

      // Get basis set type and name (returns BasisType)
      auto basis_type = basis_set.get_basis_type();
      // Get basis set name (returns std::string)
      auto name = basis_set.get_name();

      // Get all shells (returns const std::vector<Shell>&)
      auto all_shells = basis_set.get_shells();
      // Get shells for specific atom (returns const std::vector<const Shell>&)
      auto shells_for_atom = basis_set.get_shells_for_atom(0);
      // Get specific shell by index (returns const Shell&)
      const Shell& specific_shell = basis_set.get_shell(3);

      // Get counts
      size_t num_shells = basis_set.get_num_shells();
      size_t num_basis_functions = basis_set.get_num_basis_functions();
      size_t num_atoms = basis_set.get_num_atoms();

      // Get basis function information (returns std::pair<size_t, int>)
      auto [shell_index, m_quantum_number] = basis_set.get_basis_function_info(5);
      size_t atom_index = basis_set.get_atom_index_for_basis_function(5);

      // Get indices for specific atoms or orbital types
      // Returns std::vector<size_t>
      auto basis_indices = basis_set.get_basis_function_indices_for_atom(1);
      // Returns std::vector<size_t>
      auto shell_indices = basis_set.get_shell_indices_for_orbital_type(OrbitalType::P);
      // Returns std::vector<size_t>
      auto shell_indices_specific = basis_set.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::D);

      // Validation
      bool is_valid = basis_set.is_valid();
      bool is_consistent = basis_set.is_consistent_with_structure();

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/basis_set.py
      :language: python
      :lines: 22-45

Working with shells
-------------------

The ``Shell`` structure contains information about a group of basis functions:

.. tab:: C++ API

   .. code-block:: cpp

      // Get shell by index (returns const Shell&)
      const Shell& shell = basis_set.get_shell(0);
      size_t atom_idx = shell.atom_index;
      OrbitalType orb_type = shell.orbital_type;
      // Get exponents (returns const Eigen::VectorXd&)
      const Eigen::VectorXd& exps = shell.exponents;
      // Get coefficients (returns const Eigen::VectorXd&)
      const Eigen::VectorXd& coeffs = shell.coefficients;

      // Get information from shell
      size_t num_primitives = shell.get_num_primitives();
      size_t num_basis_funcs = shell.get_num_basis_functions(BasisType::Spherical);
      int angular_momentum = shell.get_angular_momentum();

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/basis_set.py
      :language: python
      :lines: 47-61

Serialization
-------------

The ``BasisSet`` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../advanced/serialization>` documentation.

.. note::
   All basis set-related files require the ``.basis_set`` suffix before the file type extension, for example ``molecule.basis_set.json`` and ``h2.basis_set.h5`` for JSON and HDF5 files respectively.
   This naming convention is enforced to maintain consistency across the QDK/Chemistry ecosystem.

File formats
~~~~~~~~~~~~

QDK/Chemistry supports multiple serialization formats for basis set data:

JSON format
^^^^^^^^^^^

JSON representation of a ``BasisSet`` has the following structure (showing simplified content):

.. code-block:: json

   {
     "atoms": [
       {
         "atom_index": 0,
         "shells": [
           {
             "coefficients": [0.1543289673, 0.5353281423, 0.4446345422],
             "exponents": [3.425250914, 0.6239137298, 0.168855404],
             "orbital_type": "s"
           },
           {
             "coefficients": [0.1559162750, 0.6076837186],
             "exponents": [0.7868272350, 0.1881288540],
             "orbital_type": "p"
           }
         ]
       },
       {
         "atom_index": 1,
         "shells": ["..."]
       }
     ],
     "basis_type": "spherical",
     "name": "6-31G",
     "num_atoms": 2,
     "num_basis_functions": 9,
     "num_shells": 3
   }

HDF5 format
^^^^^^^^^^^

HDF5 representation of a ``BasisSet`` has the following structure (showing groups and datasets):

.. code-block:: text

   /
   ├── shells/             # Group
   │   ├── atom_indices    # Dataset: uint32, 1D Array of atom indices
   │   ├── coefficients    # Dataset: float64, 1D Array of orbital coefficients
   │   ├── exponents       # Dataset: float64, 1D Array of orbital exponents
   │   ├── num_primitives  # Dataset: uint32, 1D Array of number of primitives per orbital
   │   └── orbital_types   # Dataset: int32, 1D Array of orbital type per orbital
   └── metadata/           # Group
       └── name            # Attribute: string value of the basis set name

.. tab:: C++ API

   .. code-block:: cpp

      // Generic serialization with format specification
      basis_set.to_file("molecule.basis.json", "json");
      basis_set.from_file("molecule.basis.json", "json");

      // JSON serialization
      basis_set.to_json_file("molecule.basis.json");
      basis_set.from_json_file("molecule.basis.json");

      // Direct JSON conversion
      nlohmann::json j = basis_set.to_json();
      basis_set.from_json(j);

      // HDF5 serialization
      basis_set.to_hdf5_file("molecule.basis.h5");
      basis_set.from_hdf5_file("molecule.basis.h5");

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/basis_set.py
      :language: python
      :lines: 63-79

Utility functions
-----------------

The ``BasisSet`` class provides several static utility functions:

.. tab:: C++ API

   .. code-block:: cpp

      // Convert orbital type to string (returns std::string)
      std::string orbital_str = BasisSet::orbital_type_to_string(OrbitalType::D);  // "d"
      // Convert string to orbital type (returns OrbitalType)
      OrbitalType orbital_type = BasisSet::string_to_orbital_type("f");  // OrbitalType::F

      // Get angular momentum (returns int)
      int l_value = BasisSet::get_angular_momentum(OrbitalType::P);  // 1
      // Get number of orbitals for angular momentum (returns int)
      int num_orbitals = BasisSet::get_num_orbitals_for_l(2, BasisType::Spherical);  // 5

      // Convert basis type to string (returns std::string)
      std::string basis_str = BasisSet::basis_type_to_string(BasisType::Cartesian);  // "cartesian"
      // Convert string to basis type (returns BasisType)
      BasisType basis_type = BasisSet::string_to_basis_type("spherical");  // BasisType::Spherical

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/basis_set.py
      :language: python
      :lines: 88-95

Predefined basis sets
---------------------

QDK/Chemistry provides access to a library of standard basis sets commonly used in quantum chemistry calculations.
These predefined basis sets can be easily loaded without having to manually specify the basis functions.
For a complete list of available basis sets and their specifications, see the :doc:`Supported Basis Sets <../data/basis_sets>` documentation.

.. tab:: C++ API

   .. code-block:: cpp

      // Create a basis set from a predefined library (returns std::unique_ptr<BasisSet>)
      auto basis_set = BasisSet::create("6-31G");

      // List all available basis sets (returns std::vector<std::string>)
      auto available_basis_sets = BasisSet::get_available_basis_sets();

      // Check if a basis set exists in the library (returns bool)
      bool has_basis = BasisSet::has_basis_set("cc-pvdz");

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/basis_set.py
      :language: python
      :lines: 81-90

.. note::
   The basis set library includes popular basis sets such as STO-nG, Pople basis sets (3-21G, 6-31G, etc.), correlation-consistent basis sets (cc-pVDZ, cc-pVTZ, etc.), and more.
   The availability may depend on your QDK/Chemistry installation.

Related classes
---------------

- :doc:`Structure <structure>`: Molecular structure representation
- :doc:`Orbitals <orbitals>`: Molecular orbitals constructed using the basis set
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that uses basis sets to produce orbitals

Related topics
--------------

- :doc:`Serialization <../advanced/serialization>`: Data serialization and deserialization
- :doc:`Settings <../advanced/settings>`: Configuration settings for algorithms
- :doc:`Supported basis sets <basis_sets>`: List of pre-defined basis sets available in QDK/Chemistry
