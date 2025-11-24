Structure
=========

The :class:`~qdk_chemistry.data.Structure` class in QDK/Chemistry represents a molecular structure, storing information about atomic positions, elements, and related properties for chemical systems of interest.
As a core :doc:`data class <../advanced/design_principles>`, it follows QDK/Chemistry's immutable data pattern.

Overview
--------

The :class:`~qdk_chemistry.data.Structure` class is a fundamental data container in QDK/Chemistry that represents the geometric arrangement of atoms in a molecular system.
It provides the foundation for all quantum chemistry calculations by defining the nuclear framework on which electronic structure calculations are performed.

Properties
~~~~~~~~~~

- **Coordinates**: 3D Cartesian coordinates for each atom
- **Elements**: Chemical elements of the atoms

.. note::
   By default, the total charge of a new structure is 0 (neutral) and the spin multiplicity is set to the
   lowest possible value (singlet for even-electron systems, doublet for odd-electron systems).

Usage
-----

The :class:`~qdk_chemistry.data.Structure` class is typically the starting point for any calculation workflow in QDK/Chemistry.
It is used to define the molecular system before performing electronic structure calculations.

.. note::
   Coordinates are in Angstrom by default when creating or importing a Structure. See the `Units`_ section below for more details on unit conversions.

Creating a structure object manually
------------------------------------

A :class:`~qdk_chemistry.data.Structure` object can be created manually by adding atoms one by one:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::data;

      // Create an empty structure
      Structure structure;

      // Add atoms with their 3D coordinates and element symbols (coordinates in Bohr/atomic units)
      structure.add_atom(Eigen::Vector3d(0.0, 0.0, 0.0), "H");
      structure.add_atom(Eigen::Vector3d(0.0, 0.0, 1.4), "H");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/structure.py
      :language: python
      :lines: 1-11

Loading from files
------------------

The :class:`~qdk_chemistry.data.Structure` class can load molecular structures from various file formats.
For detailed format specifications, see the `File Formats`_ section below.

.. note::
   All structure-related files require the ``.structure`` suffix before the file type extension, for example
   ``molecule.structure.xyz`` and ``h2.structure.json`` for XYZ and JSON files respectively.

.. tab:: C++ API

   .. code-block:: cpp

      // Load from XYZ file
      auto structure = Structure::from_xyz_file("molecule.structure.xyz"); // Required .structure.xyz suffix

      // Load from JSON file
      auto structure = Structure::from_json_file("molecule.structure.json"); // Required .structure.json suffix

.. tab:: Python API

   .. note::
      These examples show the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/structure.py
      :language: python
      :lines: 13-17

Accessing structure data
------------------------

The :class:`~qdk_chemistry.data.Structure` class provides methods to access atomic data:

.. note::
   Functions that deal with specific atoms include the word "atom" in their name (e.g., ``get_atom_coordinates``), while functions that return properties for all atoms omit this word (e.g., ``get_coordinates``).
   All atomic data is const and immutable once set, following QDK/Chemistry's :doc:`immutable data pattern <../advanced/design_principles>`.
   If you need to modify coordinates or other properties, you must create a new Structure object with the desired changes.

.. tab:: C++ API

   .. code-block:: cpp

      // Get coordinates of a specific atom in angstrom
      Eigen::Vector3d coords = structure.get_atom_coordinates(0);  // First atom

      // Get element of a specific atom
      std::string element = structure.get_atom_element(0);  // First atom

      // Get all coordinates (in angstrom) as a matrix
      Eigen::MatrixXd all_coords = structure.get_coordinates();

      // Get all elements as a vector
      std::vector<std::string> all_elements = structure.get_elements();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/structure.py
      :language: python
      :lines: 13-23

Serialization
-------------

The :class:`~qdk_chemistry.data.Structure` class supports serialization to and from various formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../advanced/serialization>` documentation.

.. note::
   All structure-related files require the ``.structure`` suffix before the file type extension, for example ``molecule.structure.xyz`` and ``h2.structure.json`` for XYZ and JSON files respectively.
   This naming convention is enforced to maintain consistency across the QDK/Chemistry ecosystem.

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
      "masses":[1.008,1.008],
      "nuclear_charges":[1.0,1.0],
      "num_atoms":2,
      "symbols":["H","H"],
      "total_charge":0,  // TODO: Verify these field names
      "spin_multiplicity":1  // TODO: Verify these field names
    }

XYZ format
^^^^^^^^^^

`XYZ representation <https://en.wikipedia.org/wiki/XYZ_file_format>`_ of the same :class:`~qdk_chemistry.data.Structure`:

.. note::
   QDK/Chemistry uses the comment field (second line) of the XYZ format to store the charge and spin multiplicity information, as this data is not part of the standard XYZ specification.

.. code-block:: text

    2
    charge = 0, spin_multiplicity = 1  (TODO)
    H      0.000000    0.000000    0.000000
    H      0.000000    0.000000    1.400000

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

Units
-----

All internal coordinates in the :class:`~qdk_chemistry.data.Structure` class are in Bohr by default.
This applies to all methods that return or accept coordinates.

.. TODO:  restore the code snippets with working examples.

Related classes
---------------

- :doc:`Orbitals <orbitals>`: Molecular orbitals calculated from the structure
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that performs calculations on the structure

Related topics
--------------

- :doc:`Serialization <../advanced/serialization>`: Data serialization and deserialization
