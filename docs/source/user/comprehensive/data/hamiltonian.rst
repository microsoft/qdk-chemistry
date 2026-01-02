Hamiltonian
===========

The :class:`~qdk_chemistry.data.Hamiltonian` class in QDK/Chemistry represents the electronic Hamiltonian operator, which describes the physics of a quantum system.
It contains the one- and two-electron integrals that are essential for quantum chemistry calculations, particularly for active space methods.

Overview
--------

In quantum chemistry, the electronic Hamiltonian is the operator that gives the energy of a system of electrons.
The :class:`~qdk_chemistry.data.Hamiltonian` class in QDK/Chemistry stores the matrix elements of this operator in the basis of molecular orbitals.
These matrix elements consist of one-electron integrals (representing kinetic energy and electron-nucleus interactions) and two-electron integrals (representing electron-electron repulsion).

Design principles
~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.data.Hamiltonian` class follows an immutable data model design principle as described in the :doc:`QDK/Chemistry Design Principles <../design/index>` document.
Once properly constructed, the Hamiltonian data is typically not modified during calculations.
This const-correctness approach ensures data integrity throughout computational workflows and prevents accidental modifications of the core quantum system representation.
While setter methods are available for construction and initialization purposes, in normal operation the Hamiltonian object should be treated as immutable after it has been fully populated.

Properties
----------

One-electron integrals
   Matrix of one-electron integrals (h₁)

Two-electron integrals
   Vector of two-electron integrals (h₂) in physicist notation :math:`\left\langle ij|kl \right\rangle`

Core energy
   Constant energy term combining nuclear repulsion and inactive orbital contributions

Inactive Fock matrix
   Matrix representing interactions between active and inactive orbitals

Orbitals
   Molecular orbital information for the system (see the :doc:`Orbitals <orbitals>` documentation for detailed information about orbital properties and representations)

Selected orbital indices
   Indices defining the active space orbitals

Number of electrons
   Count of electrons in the active space

Restricted vs. unrestricted Hamiltonians
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Hamiltonian`` class supports both restricted and unrestricted representations:

- **Restricted**: Uses the same spatial orbitals for alpha and beta electrons. Suitable for closed-shell systems where alpha and beta electrons occupy the same spatial orbitals with opposite spins.
- **Unrestricted**: Allows different spatial orbitals for alpha and beta electrons.

For unrestricted Hamiltonians, the one-electron and two-electron integrals are stored separately for each spin channel:

- One-electron integrals: :math:`h_{\alpha\alpha}` and :math:`h_{\beta\beta}`
- Two-electron integrals: :math:`h_{\alpha\alpha\alpha\alpha}`, :math:`h_{\alpha\beta\alpha\beta}`, and :math:`h_{\beta\beta\beta\beta}`

Usage
-----

The :class:`~qdk_chemistry.data.Hamiltonian` class is typically used as input to correlation methods such as Configuration Interaction (:term:`CI`) and Multi-Configuration Self-Consistent Field (:term:`MCSCF`) calculations.
The :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm is the primary tool for generating :class:`~qdk_chemistry.data.Hamiltonian` objects from molecular data.

Creating a Hamiltonian object
-----------------------------

The :class:`~qdk_chemistry.data.Hamiltonian` object is typically created using the :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm (recommended approach for most users), or it can be created directly with the appropriate integral data.
Once properly constructed with all required data, the Hamiltonian object should be considered constant and not modified:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-hamiltonian-creation
      :end-before: // end-cell-hamiltonian-creation

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-hamiltonian-creation
      :end-before: # end-cell-hamiltonian-creation

Accessing Hamiltonian data
--------------------------

The :class:`~qdk_chemistry.data.Hamiltonian` class provides methods to access the one- and two-electron integrals and other properties.
In line with its immutable design principle, these methods return const references or copies of the internal data:

Two-electron integral storage and notation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-electron integrals in quantum chemistry can be represented using different notations and storage formats.
QDK/Chemistry uses the physicist notation by default, but it's important to understand the different conventions:

Physicist/Dirac notation :math:`\left\langle ij|kl \right\rangle` or :math:`\left\langle ij|kl \right\rangle`
   Represents the Coulomb interaction where electron 1 occupies orbitals :math:`i` and :math:`k`, while electron 2 occupies orbitals :math:`j` and :math:`l`.
   This is the default representation in QDK/Chemistry.
   In this notation, the first index of each pair :math:`(i,k)` refers to electron 1, and the second index of each pair :math:`(j,l)` refers to electron 2, following a (1,2,1,2) electron indexing pattern.

Chemist/Mulliken notation :math:`(ij|kl)` or :math:`[ij|kl]`
   Represents the Coulomb interaction where electron 1 occupies orbitals :math:`i` and :math:`j`, while electron 2 occupies orbitals :math:`k` and :math:`l`.
   In this notation, the first pair of indices :math:`(i,j)` refers to electron 1, and the second pair :math:`(k,l)` refers to electron 2, following a (1,1,2,2) electron indexing pattern.
   The symbols differ (parentheses vs square brackets), but the indexing convention is the same.

The relationship between physicist and chemist notation is:

.. math::

   \left\langle ij | kl \right\rangle = \left(ik|jl \right)

Two-electron integrals with real-valued orbitals possess inherent symmetry properties.
From a theoretical perspective, these symmetries can be expressed as:

.. math::

   \left\langle ij|kl \right\rangle = \left\langle ji|lk \right\rangle = \left\langle kl|ij \right\rangle = \left\langle lk|ji \right\rangle = \left\langle jl|ki \right\rangle = \left\langle lj|ik \right\rangle = \left\langle ki|jl \right\rangle = \left\langle ik|lj \right\rangle

These permutational symmetries arise from the mathematical properties of the two-electron repulsion integrals.
When accessing specific elements with ``get_two_body_element(i, j, k, l)``, the function handles the appropriate index mapping to retrieve the correct value based on the implementation's storage format.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-properties
      :end-before: // end-cell-properties

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-properties
      :end-before: # end-cell-properties

Serialization
-------------

The :class:`~qdk_chemistry.data.Hamiltonian` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../data/serialization>` documentation.

Active space Hamiltonian
------------------------

When constructed with active orbital specifications, the :class:`~qdk_chemistry.data.Hamiltonian` represents an active space Hamiltonian, which is a projection of the full electronic Hamiltonian into a smaller subspace.
This is essential for tractable multi-configuration calculations.
The :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` algorithm handles the complex process of generating an appropriate active space Hamiltonian based on your specifications.

Validation methods
------------------

The :class:`~qdk_chemistry.data.Hamiltonian` class provides methods to check the validity and consistency of its data:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian.cpp
      :language: cpp
      :start-after: // start-cell-validation
      :end-before: // end-cell-validation

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../_static/examples/python/hamiltonian.py
      :language: python
      :start-after: # start-cell-validation
      :end-before: # end-cell-validation

Related classes
---------------

- :doc:`Orbitals <orbitals>`: Molecular orbital information used to construct the Hamiltonian
- :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`: Algorithm for constructing Hamiltonians -
  **primary tool** for generating Hamiltonian objects from molecular data
- :doc:`../algorithms/mc_calculator`: Uses the Hamiltonian for correlation calculations
- :class:`~qdk_chemistry.data.Wavefunction`: Represents the solution of the Hamiltonian eigenvalue problem
- :doc:`Active space methods <../algorithms/active_space>`: Selection and use of active spaces with the Hamiltonian

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/hamiltonian.cpp>`_ and `Python <../../../_static/examples/python/hamiltonian.py>`_ scripts.
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization in QDK/Chemistry
- :doc:`Design principles <../design/index>`: Design principles for data classes in QDK/Chemistry
- :doc:`Settings <../design/index>`: Configuration options for algorithms operating on Hamiltonians
