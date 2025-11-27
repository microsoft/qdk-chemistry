Serialization
=============

QDK/Chemistry provides serialization capabilities for all its data classes, allowing to save and load computational results in various formats.
This document explains the serialization mechanisms and formats supported by QDK/Chemistry.

Overview
--------

Serialization is the process of converting complex data structures into a format that can be stored or transmitted.
In QDK/Chemistry, this is crucial for:

- Saving intermediate results of calculations
- Sharing data between different programs or languages
- Preserving computational results for future analysis
- Implementing checkpoint and restart capabilities

Supported formats
-----------------

QDK/Chemistry supports multiple serialization formats:

- **JSON**: Human-readable text format, suitable for small to medium data
- **HDF5**: Hierarchical binary format, suitable for large data sets
- **XYZ**: Standard format for molecular geometries (for ``Structure`` only)
- **FCIDUMP**: Format for Hamiltonian integrals (for ``Hamiltonian`` only)

Common serialization interface
------------------------------

All QDK/Chemistry data classes implement a consistent serialization interface as described below.

JSON serialization
~~~~~~~~~~~~~~~~~~

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::data;

      // Structure data class example
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
      std::vector<std::string> symbols = {"H", "H"};
      std::vector <double> custom_masses {1.001, 0.999};
      std::vector<double> custom_charges = {0.9, 1.1};
      Structure structure(coords, elements, custom_masses, custom_charges);

      // Serialize to JSON object
      auto structure_data = structure.to_json();

      // Deserialize from JSON object
      // "Structure" is the data type to de-serialize into (will throw, if it doesn't match)
      auto structure_from_json = Structure::from_json(json_data);

      // Write to json file
      structure.to_json_file("filename.structure.json"); // Extension depends on object type

      // Read from json file
      auto structure_from_json_file = Structure::from_json_file("filename.structure.json");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/serialization.py
      :language: python
      :lines: 17-36

HDF5 serialization
~~~~~~~~~~~~~~~~~~

.. tab:: C++ API

   .. code-block:: cpp

    // Hamiltonian data class example 
    // Create dummy data for Hamiltonian class 
    Eigen::MatrixXd one_body = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd two_body = 2 * Eigen::VectorXd::Ones(16);
    auto orbitals = std::make_shared<ModelOrbitals>(2, true); // 2 orbitals, restricted
    double core_energy = 1.5;
    Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Zero(0, 0);

    Hamiltonian h_example(one_body, two_body, orbitals, core_energy, inactive_fock);

    h_example.to_hdf5_file("h_example.hamiltonian.h5"); // Extension depends on object type

    // Deserialize from HDF5 file
    auto h_example_from_hdf5_file = Hamiltonian::from_hdf5_file("h_example.hamiltonian.h5");


.. tab:: Python API

   .. literalinclude:: ../../../../examples/serialization.py
      :language: python
      :lines: 44-57

File extensions
---------------

QDK/Chemistry enforces specific file extensions to ensure clarity about the content type:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Data class
     - JSON extension
     - HDF5 extension
     - Other formats
   * - :doc:`Structure <../data/structure>`
     - ``.structure.json``
     - ``.structure.h5``
     - ``.structure.xyz``
   * - :doc:`BasisSet <../data/basis_set>`
     - ``.basis_set.json``
     - ``.basis_set.h5``
     - -
   * - :doc:`Orbitals <../data/orbitals>`
     - ``.orbitals.json``
     - ``.orbitals.h5``
     - -
   * - :doc:`Hamiltonian <../data/hamiltonian>`
     - ``.hamiltonian.json``
     - ``.hamiltonian.h5``
     - ``hamiltonian.fcidump``
   * - :class:`~qdk_chemistry.data.Wavefunction`
     - ``.wavefunction.json``
     - ``.wavefunction.h5``
     - -

Related topics
--------------

- :doc:`Structure <../data/structure>`: Molecular geometry and atomic information
- :doc:`BasisSet <../data/basis_set>`: Quantum chemistry basis set definitions
- :doc:`Orbitals <../data/orbitals>`: Molecular orbital coefficients and properties
- :doc:`Hamiltonian <../data/hamiltonian>`: Electronic Hamiltonian operator
- :class:`~qdk_chemistry.data.Wavefunction`: Quantum mechanical wavefunction data
