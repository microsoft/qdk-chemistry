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

   .. literalinclude:: ../../../../examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-1
      :end-before: // end-cell-1

.. tab:: Python API

   .. literalinclude:: ../../../../examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-2
      :end-before: # end-cell-2

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

   .. literalinclude:: ../../../../examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-2
      :end-before: // end-cell-2

.. tab:: Python API

   .. literalinclude:: ../../../../examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-2
      :end-before: # end-cell-2

      # Get orbital data
      orbitals = hamiltonian.get_orbitals()

      # Get active space information
      active_indices = hamiltonian.get_selected_orbital_indices()
      num_electrons = hamiltonian.get_num_electrons()
      num_orbitals = hamiltonian.get_num_orbitals()

Serialization
-------------

The ``Hamiltonian`` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../advanced/serialization>` documentation.

.. note::
   All Hamiltonian-related files should follow a consistent naming convention, such as
   ``molecule.hamiltonian.json`` and ``molecule.hamiltonian.h5`` for JSON and HDF5 files respectively.

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
  │   ├── num_electrons         # Attribute: uint32, number of electrons (in the active space)
  │   └── num_orbitals          # Attribute: uint32, number of orbitals (in the active space)
  └── orbitals/                     # Group
      └── json_data             # Dataset: (), binary representation of the json orbital data

.. note::
   The ``orbitals/`` group follows the same structure and organization as an independent ``Orbitals`` HDF5 file.
   For complete details on the structure and content of this group, see the :ref:`Orbitals Serialization <orbitals-serialization>` section in the Orbitals documentation.

.. tab:: C++ API

   .. literalinclude:: ../../../../examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-3
      :end-before: // end-cell-3

.. tab:: Python API

   .. literalinclude:: ../../../../examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-3
      :end-before: # end-cell-3

Active space Hamiltonian
------------------------

When constructed with active orbital specifications, the ``Hamiltonian`` represents an active space Hamiltonian, which is a projection of the full electronic Hamiltonian into a smaller subspace.
This is essential for tractable multi-configuration calculations.
The :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm handles the complex process of generating an appropriate active space Hamiltonian based on your specifications.

Validation methods
------------------

The ``Hamiltonian`` class provides methods to check the validity and consistency of its data:

.. tab:: C++ API

   .. literalinclude:: ../../../../examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-4
      :end-before: // end-cell-4

.. tab:: Python API

   .. literalinclude:: ../../../../examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-4
      :end-before: # end-cell-4

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
