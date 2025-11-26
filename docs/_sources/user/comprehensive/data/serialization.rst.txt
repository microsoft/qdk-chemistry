Serialization
=============

QDK/Chemistry provides serialization capabilities for all its data classes, allowing you to save and load computational results in various formats.
This document explains the serialization mechanisms and formats supported by QDK/Chemistry.

Overview
--------

Serialization is the process of converting complex data structures into a format that can be stored or transmitted.
In QDK/Chemistry, this is crucial for:

- Saving intermediate results of calculations
- Sharing data between different programs or languages
- Preserving computational results for future analysis
- Implementing checkpoint and restart capabilities

.. note::
   For detailed information about the structure and organization of serialized data for each class, refer to the corresponding class documentation.
   Each data class page includes examples of the JSON and HDF5 schema used for serialization.

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

      // Serialize to JSON object
      auto json_data = object.to_json();

      // Deserialize from JSON object
      auto object_from_json = ObjectType::from_json(json_data);

      // Serialize to JSON file
      object.to_json_file("filename.ext.json"); // Extension depends on object type

      // Deserialize from JSON file
      auto object_from_json_file = ObjectType::from_json_file("filename.ext.json");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/serialization.py
      :language: python
      :lines: 14-20,26-32

HDF5 serialization
~~~~~~~~~~~~~~~~~~

.. tab:: C++ API

   .. code-block:: cpp

      // Serialize to HDF5 file
      object.to_hdf5_file("filename.ext.h5"); // Extension depends on object type

      // Deserialize from HDF5 file
      auto object_from_hdf5_file = ObjectType::from_hdf5_file("filename.ext.h5");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/serialization.py
      :language: python
      :lines: 35-41

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
