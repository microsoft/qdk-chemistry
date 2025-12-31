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

JSON
  Human-readable text format, suitable for small to medium data

HDF5
  Hierarchical binary format, suitable for large data sets

XYZ
  Standard format for molecular geometries (for ``Structure`` only)

:term:`FCIDUMP`
  Format for Hamiltonian integrals (for ``Hamiltonian`` only)

Common serialization interface
------------------------------

All QDK/Chemistry data classes implement a consistent serialization interface as described below.

JSON serialization
~~~~~~~~~~~~~~~~~~

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/serialization.cpp
      :language: cpp
      :start-after: // start-cell-json
      :end-before: // end-cell-json

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/serialization.py
      :language: python
      :start-after: # start-cell-json
      :end-before: # end-cell-json

HDF5 serialization
~~~~~~~~~~~~~~~~~~

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/serialization.cpp
      :language: cpp
      :start-after: // start-cell-hdf5
      :end-before: // end-cell-hdf5

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/serialization.py
      :language: python
      :start-after: # start-cell-hdf5
      :end-before: # end-cell-hdf5

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
   * - :doc:`Wavefunction <../data/wavefunction>`
     - ``.wavefunction.json``
     - ``.wavefunction.h5``
     - -

The same patterns are observed for other data classes in QDK/Chemistry.

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/serialization.cpp>`_ and `Python <../../../_static/examples/python/serialization.py>`_ scripts.
- :doc:`Structure <../data/structure>`: Molecular geometry and atomic information
- :doc:`BasisSet <../data/basis_set>`: Basis set definitions
- :doc:`Orbitals <../data/orbitals>`: Molecular orbital coefficients and properties
- :doc:`Hamiltonian <../data/hamiltonian>`: Electronic Hamiltonian operator
- :doc:`Wavefunction <../data/wavefunction>`: Wavefunction data
