Structure
=========

The :class:`~qdk_chemistry.data.Structure` class in QDK/Chemistry represents a molecular structure, which always includes 3D coordinates and element information, and optionally includes related properties like atomic masses and nuclear charges.
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
The only time Angstrom units can be found by default is in the *xyz* file format, where Angstrom is default (see below).

Usage
-----

The :class:`~qdk_chemistry.data.Structure` class is typically the starting point for any calculation workflow in QDK/Chemistry.
It is used to define the molecular system before performing electronic structure calculations.

.. note::
   Coordinates are in Bohr by default when creating or importing a Structure. See the `Units`_ section below for more details on unit conversions.

Creating a structure object
---------------------------

A :class:`~qdk_chemistry.data.Structure` object can be created manually as follows:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/structure.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/structure.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

Accessing structure data
------------------------

The :class:`~qdk_chemistry.data.Structure` class provides methods to access atomic data:

Functions that deal with specific atoms include the word "atom" in their name (e.g., ``get_atom_coordinates``), while functions that return properties for all atoms omit this word (e.g., ``get_coordinates``).
All atomic data is const and immutable once set, following QDK/Chemistry's :doc:`immutable data pattern <../design/index>`.
If you need to modify coordinates or other properties, you must create a new Structure object with the desired changes.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/structure.cpp
      :language: cpp
      :start-after: // start-cell-data
      :end-before: // end-cell-data

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/structure.py
      :language: python
      :start-after: # start-cell-data
      :end-before: # end-cell-data

Serialization
-------------

The :class:`~qdk_chemistry.data.Structure` class supports serialization to and from various formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../data/serialization>` documentation.

.. note::
   All structure-related files require the ``.structure`` suffix before the file type extension, for example ``molecule.structure.xyz`` and ``h2.structure.json`` for XYZ and JSON files respectively.
   This naming convention is enforced to maintain consistency across the QDK/Chemistry ecosystem.

File formats
------------

As well as the JSON and HDF5 file formats supported for all data classes of QDK/Chemistry, there is the dedicated XYZ format for structures.

XYZ format
~~~~~~~~~~

`XYZ representation <https://en.wikipedia.org/wiki/XYZ_file_format>`_ of a :class:`~qdk_chemistry.data.Structure`:

.. code-block:: text

    2

    H      0.000000    0.000000    0.000000
    H      0.000000    0.000000    1.400000

Note that here the coordinates are in Angstrom, since this is the standard in xyz files.


Related classes
---------------

- :doc:`Orbitals <orbitals>`: Molecular orbitals calculated from the structure
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that performs calculations on the structure

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/structure.cpp>`_ and `Python <../../../_static/examples/python/structure.py>`_ scripts.
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
