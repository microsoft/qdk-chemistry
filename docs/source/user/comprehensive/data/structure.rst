Structure
=========

The :class:`~qdk_chemistry.data.Structure` class in QDK/Chemistry represents a molecular structure, which always includes 3d coordinates and element information, and optionally includes related properties like atomic masses and nuclear charges. 
As a core :doc:`data class <../design/index>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

The :class:`~qdk_chemistry.data.Structure` class is a fundamental data container in QDK/Chemistry that represents the geometric arrangement of atoms in a molecular system.
It provides the foundation for all quantum chemistry calculations by defining the nuclear framework on which electronic structure calculations are performed.

Properties
~~~~~~~~~~

- **Coordinates**: 3D Cartesian coordinates for each atom
- **Elements**: Chemical elements of the atoms
- **Masses**: Atomic masses of each of the atoms 
- **Nuclear charges**: Nuclear charges (atomic numbers) of each of the atoms

Units
-----

All internal coordinates in the :class:`~qdk_chemistry.data.Structure` class are in Bohr by default.
This applies to all methods that return or accept coordinates.
The only time Angstrom units can be found by default, are in the *xyz* file format, where Angstrom is default (see below).

Usage
-----

The :class:`~qdk_chemistry.data.Structure` class is typically the starting point for any calculation workflow in QDK/Chemistry.
It is used to define the molecular system before performing electronic structure calculations.

.. note::
   Coordinates are in Bohr by default when creating or importing a Structure. See the `Units`_ section below for more details on unit conversions.

Creating a structure object manually
------------------------------------

A :class:`~qdk_chemistry.data.Structure` object can be created manually as follows:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::data;

      // Specify a structure using coordinates, and either symbols or elements
      std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}}; // Bohr 
      std::vector<std::string> symbols = {"H", "H"};

      Structure structure(coords, symbols);

      // element enum alternative 
      std::vector<Element> elements = {Element::H, Element::H};
      Structure structure_alternative(coords, elements);

      // Can specify custom masses and/or charges
      std::vector <double> custom_masses {1.001, 0.999};
      std::vector<double> custom_charges = {0.9, 1.1};
      Structure structure_custom(coords, elements, custom_masses, custom_charges);


.. tab:: Python API

   .. literalinclude:: ../../../../examples/structure.py
      :language: python
      :lines: 15-27

Accessing structure data
------------------------

The :class:`~qdk_chemistry.data.Structure` class provides methods to access atomic data:

Functions that deal with specific atoms include the word "atom" in their name (e.g., ``get_atom_coordinates``), while functions that return properties for all atoms omit this word (e.g., ``get_coordinates``).
All atomic data is const and immutable once set, following QDK/Chemistry's :doc:`immutable data pattern <../design/index>`.
If you need to modify coordinates or other properties, you must create a new Structure object with the desired changes.

.. tab:: C++ API

   .. code-block:: cpp

      // Get coordinates of a specific atom
      Eigen::Vector3d coords = structure.get_atom_coordinates(0);  // First atom

      // Get element of a specific atom
      std::string element = structure.get_atom_element(0);  // First atom

      // Get all coordinates as a matrix
      Eigen::MatrixXd all_coords = structure.get_coordinates();

      // Get all elements as a vector
      std::vector<std::string> all_elements = structure.get_elements();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/structure.py
      :language: python
      :lines: 33-43

File formats
~~~~~~~~~~~~

QDK/Chemistry supports multiple serialization formats for molecular structures:

JSON format
^^^^^^^^^^^

JSON representation of a :class:`~qdk_chemistry.data.Structure` looks like:

.. code-block:: json

    {
      "coordinates":[[0.0,0.0,0.0],[0.0,0.0,1.4]],
      "elements":[1,1],
      "masses":[1.001,0.999],
      "nuclear_charges":[0.9,1.1],
      "num_atoms":2,
      "symbols":["H","H"],
      "units":"bohr",
      "version":"0.1.0",
    }

XYZ format
^^^^^^^^^^

`XYZ representation <https://en.wikipedia.org/wiki/XYZ_file_format>`_ of the same :class:`~qdk_chemistry.data.Structure`:

.. code-block:: text

    2

    H      0.000000    0.000000    0.000000
    H      0.000000    0.000000    1.400000

Note that here the coordinates are in Angstrom, since this is the standard in xyz files. 

.. tab:: C++ API

   .. code-block:: cpp

      // Serialize to JSON object
      auto json_data = structure.to_json();

      // Deserialize from JSON object
      auto structure_from_json = Structure::from_json(json_data);

      // Serialize to JSON file
      structure.to_json_file("molecule.structure.json");  // Required .structure.json suffix

      // Get XYZ format as string
      std::string xyz_string = structure.to_xyz();

      // Load from XYZ string
      auto structure_from_xyz = Structure::from_xyz(xyz_string);

      // Serialize to XYZ file
      structure.to_xyz_file("molecule.structure.xyz");  // Required .structure.xyz suffix

.. tab:: Python API

   .. literalinclude:: ../../../../examples/serialization.py
      :language: python
      :lines: 1-18

Molecular manipulation
----------------------

The :class:`~qdk_chemistry.data.Structure` class provides methods for basic molecular manipulations:

.. tab:: C++ API

   .. code-block:: cpp

      // Add an atom with coordinates and element
      structure.add_atom(Eigen::Vector3d(1.0, 0.0, 0.0), "O");  // Add an oxygen atom

      // Remove an atom
      structure.remove_atom(2);  // Remove the third atom

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/structure.py
      :language: python
      :lines: 19-24

Related classes
---------------

- :doc:`Orbitals <orbitals>`: Molecular orbitals calculated from the structure
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that performs calculations on the structure

Related topics
--------------

- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
