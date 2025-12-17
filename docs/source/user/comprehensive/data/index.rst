Data classes
============

QDK/Chemistry uses immutable data classes to represent molecular information, electronic structure results, and quantum objects.
These classes serve as the inputs and outputs for :doc:`algorithm classes <../algorithms/index>`, enabling a clean flow of data through the computational pipeline.
All data classes support :doc:`serialization <serialization>` to JSON and HDF5 formats for persistence and interoperability.

Comprehensive details on each data class can be found in the :ref:`API documentation <apidocs>`.
Here, we provide a quick reference guide to help users understand the purpose and typical sources of commonly encountered data classes.
Each of the links below leads to a detailed description of the data class, including its attributes, methods, and usage examples.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Data Class
     - Purpose
     - Typical Source
   * - :doc:`Structure <structure>`
     - Molecular geometry (atoms and coordinates)
     - User input
   * - :doc:`BasisSet <basis_set>`
     - Atomic orbital basis definitions
     - Library lookup, User input
   * - :doc:`Orbitals <orbitals>`
     - Molecular orbital coefficients and energies
     - :doc:`ScfSolver <../algorithms/scf_solver>`
   * - :doc:`Hamiltonian <hamiltonian>`
     - One- and two-electron integrals
     - :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`
   * - ``Wavefunction``
     - Electronic state (orbitals + CI coefficients)
     - :doc:`MCCalculator <../algorithms/mc_calculator>`
   * - ``QubitHamiltonian``
     - Pauli operator representation
     - :doc:`QubitMapper <../algorithms/qubit_mapper>`

.. toctree::
   :maxdepth: 1
   :caption: Contents

   serialization
   basis_set
   hamiltonian
   orbitals
   structure
