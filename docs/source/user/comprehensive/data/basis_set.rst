BasisSet
=========

The :class:`~qdk_chemistry.data.BasisSet` class in QDK/Chemistry represents a collection of basis functions used to describe the electronic structure of molecules.
It organizes atomic orbitals into shells and provides methods for managing, querying, and serializing basis set data.

Overview
--------

In quantum chemistry, a basis set is a collection of mathematical functions used to represent molecular orbitals.
The :class:`~qdk_chemistry.data.BasisSet` class in QDK/Chemistry uses a shell-based organization, where each shell contains basis functions with the same atom, angular momentum, and primitive Gaussian functions.

Key features of the :class:`~qdk_chemistry.data.BasisSet` class include:

- Shell-based storage for memory efficiency
- Support for both spherical and Cartesian basis functions
- Mapping between shells/basis functions and atoms
- Mapping between shells/basis functions and orbital types
- Basis set metadata (name, parameters)
- Integration with molecular structure information
- On-demand expansion of shells to individual basis functions

Usage
-----

The :class:`~qdk_chemistry.data.BasisSet` class is a fundamental component in quantum chemistry calculations, providing the mathematical foundation for representing molecular orbitals.
It's typically used as input for :term:`SCF` calculations and is usually created automatically when selecting a :doc:`predefined basis set <../basis_functionals>` for a calculation.

.. note::
   QDK/Chemistry provides a collection of :doc:`predefined basis sets <../basis_functionals>` that can be accessed through the appropriate factory functions.
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

The :class:`~qdk_chemistry.data.BasisSet` class supports various orbital types with different angular momentum:

- S orbital (angular momentum :math:`l=0`): 1 function per shell (spherical or Cartesian)
- P orbital (angular momentum :math:`l=1`): 3 functions per shell (spherical or Cartesian)
- D orbital (angular momentum :math:`l=2`): 5 functions (spherical) or 6 functions (Cartesian) per shell
- F orbital (angular momentum :math:`l=3`): 7 functions (spherical) or 10 functions (Cartesian) per shell
- G, H, I orbitals: Higher angular momentum orbitals

Basis types
~~~~~~~~~~~

The :class:`~qdk_chemistry.data.BasisSet` class supports two types of basis functions:

Spherical
   Uses spherical harmonics with :math:`2l+1` functions per shell

Cartesian
   Uses Cartesian coordinates with :math:`(l+1)(l+2)/2` functions per shell

Creating a basis set
--------------------

.. note::
   In most cases, you should use the built-in basis set library rather than creating basis sets manually.
   Manual creation is primarily for advanced use cases or when working with custom basis sets not available in the library.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/basis_set.cpp
      :language: cpp
      :start-after: // start-cell-basis-set-create
      :end-before: // end-cell-basis-set-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/basis_set.py
      :language: python
      :start-after: # start-cell-basis-set-create
      :end-before: # end-cell-basis-set-create

Accessing basis set data
------------------------

Following the :doc:`immutable design principle <../design/index>` used throughout QDK/Chemistry, all getter methods return const references or copies of the data.
This ensures that the basis set data remains consistent and prevents accidental modifications that could lead to inconsistent states.

.. note::
   If you need to modify a basis set after creation, you should create a new BasisSet object with the desired
   changes rather than trying to modify an existing one.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/basis_set.cpp
      :language: cpp
      :start-after: // start-cell-basis-set-get
      :end-before: // end-cell-basis-set-get

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/basis_set.py
      :language: python
      :start-after: # start-cell-basis-set-get
      :end-before: # end-cell-basis-set-get

Working with shells
-------------------

The ``Shell`` structure contains information about a group of basis functions:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/basis_set.cpp
      :language: cpp
      :start-after: // start-cell-shells
      :end-before: // end-cell-shells

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/basis_set.py
      :language: python
      :start-after: # start-cell-shells
      :end-before: # end-cell-shells

Serialization
-------------

The :class:`~qdk_chemistry.data.BasisSet` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../data/serialization>` documentation.

.. note::
   All basis set-related files require the ``.basis_set`` suffix before the file type extension, for example ``molecule.basis_set.json`` and ``h2.basis_set.h5`` for JSON and HDF5 files respectively.
   This naming convention is enforced to maintain consistency across the QDK/Chemistry ecosystem.

File formats
~~~~~~~~~~~~

QDK/Chemistry supports multiple serialization formats for basis set data:

JSON format
^^^^^^^^^^^

JSON representation of a :class:`~qdk_chemistry.data.BasisSet` has the following structure (showing simplified content):

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

HDF5 representation of a :class:`~qdk_chemistry.data.BasisSet` has the following structure (showing groups and datasets):

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

   .. literalinclude:: ../../../_static/examples/cpp/basis_set.cpp
      :language: cpp
      :start-after: // start-cell-serialization
      :end-before: // end-cell-serialization

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/basis_set.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization

Utility functions
-----------------

The :class:`~qdk_chemistry.data.BasisSet` class provides several static utility functions:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/basis_set.cpp
      :language: cpp
      :start-after: // start-cell-utility-functions
      :end-before: // end-cell-utility-functions

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/basis_set.py
      :language: python
      :start-after: # start-cell-utility-functions
      :end-before: # end-cell-utility-functions

Predefined basis sets
---------------------

QDK/Chemistry provides access to a library of standard basis sets commonly used in quantum chemistry calculations.
These predefined basis sets can be easily loaded without having to manually specify the basis functions.
For a complete list of available basis sets and their specifications, see the :doc:`Supported Basis Sets <../basis_functionals>` documentation.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/basis_set.cpp
      :language: cpp
      :start-after: // start-cell-library
      :end-before: // end-cell-library

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/basis_set.py
      :language: python
      :start-after: # start-cell-library
      :end-before: # end-cell-library

.. note::
   The basis set library includes popular basis sets such as :term:`STO`-nG, Pople basis sets (3-21G, 6-31G, etc.), correlation-consistent basis sets (cc-pVDZ, cc-pVTZ, etc.), and more.
   The availability may depend on your QDK/Chemistry installation.

Related classes
---------------

- :doc:`Structure <structure>`: Molecular structure representation
- :doc:`Orbitals <orbitals>`: Molecular orbitals constructed using the basis set
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that uses basis sets to produce orbitals

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/basis_set.cpp>`_ and `Python <../../../_static/examples/python/basis_set.py>`_ scripts.
- :doc:`Serialization <serialization>`: Data serialization and deserialization
- :doc:`Settings <../algorithms/settings>`: Configuration settings for algorithms
- :doc:`Supported basis sets <../basis_functionals>`: List of pre-defined basis sets available in QDK/Chemistry
