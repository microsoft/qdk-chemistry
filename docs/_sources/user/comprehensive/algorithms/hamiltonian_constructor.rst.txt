Classical Hamiltonian construction
==================================

The ``HamiltonianConstructor`` algorithm in QDK/Chemistry is responsible for constructing the electronic Hamiltonian, which is essential for quantum chemistry calculations.
It generates the one- and two-electron integrals that define the energy operator for the electronic structure.

Overview
--------

The electronic Hamiltonian describes the energy of a system of electrons in the field of atomic nuclei.
It consists of kinetic energy terms, electron-nucleus attraction terms, and electron-electron repulsion terms.
The ``HamiltonianConstructor`` algorithm computes the matrix elements of this operator in a given orbital basis, which can be the full orbital space or an active subspace.

The sole purpose of the ``HamiltonianConstructor`` class is to transform a set of molecular orbitals into a Hamiltonian representation that can be used for subsequent quantum chemistry calculations.
It acts as a bridge between the orbital representation of a molecular system and its Hamiltonian operator formulation.

Capabilities
------------

The ``HamiltonianConstructor`` in QDK/Chemistry provides:

- **Full-space Hamiltonian**: Computation of the Hamiltonian in the full orbital space
- **Active-space Hamiltonian**: Projection of the Hamiltonian into a selected active space
- **Integral Transformation**: Transformation of integrals from atomic orbital (AO) basis to molecular orbital (MO)
  basis
- **Orbitals Support**: Works with both restricted and unrestricted orbitals

Creating a HamiltonianConstructor
---------------------------------

The ``HamiltonianConstructor`` is created using the factory pattern.
This constructor is used to create a :doc:`Hamiltonian <../data/hamiltonian>` object from a set of :doc:`Orbitals <../data/orbitals>`.
The orbitals provide the necessary information about the molecular system including the basis set, orbital coefficients, and electron occupations.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::algorithms;

      // Create the default HamiltonianConstructor instance
      auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

.. tab:: Python API

   .. literalinclude:: ../../../../examples/hamiltonian_constructor.py
      :language: python
      :lines: 3-

Configuring the Hamiltonian construction
----------------------------------------

The ``HamiltonianConstructor`` can be configured using the ``Settings`` object with the following parameters.

.. note::
   All orbital indices in QDK/Chemistry are 0-based, following the convention used in most programming languages.

.. note::
   **Note on active orbitals:** When specifying active orbitals, the indices must be unique (no duplicates).

.. note::
   **Note on orbital spaces:** If no active orbitals are specified, the entire orbital space is used.
   If no inactive orbitals are specified and active orbitals are provided, orbitals from index 0 up to the first active orbital are considered inactive.

.. tab:: C++ API

   .. code-block:: cpp

      // Specify active orbitals for active space Hamiltonian
      std::vector<int> active_orbitals = {4, 5, 6, 7}; // Example indices (0-based)
      hamiltonian_constructor->settings().set("active_orbitals", active_orbitals);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 4-12

Constructing the Hamiltonian
----------------------------

Once configured, the Hamiltonian can be constructed from a set of orbitals:

.. tab:: C++ API

   .. code-block:: cpp

      // Obtain a valid Orbitals instance
      Orbitals orbitals;
      /* orbitals = ... */

      // Construct the Hamiltonian
      auto hamiltonian = hamiltonian_constructor->run(orbitals);

      // Access the resulting integrals
      auto h1 = hamiltonian.get_one_body_integrals();
      auto h2 = hamiltonian.get_two_body_integrals();

.. tab:: Python API

   .. note::
      This example shows the API pattern.
      For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/factory_pattern.py
      :language: python
      :lines: 1-9

Available settings
------------------

The ``HamiltonianConstructor`` accepts a range of settings to control its behavior.
These settings are divided into base settings (common to all Hamiltonian construction) and specialized settings (specific to certain construction variants).

Available settings
~~~~~~~~~~~~~~~~~~

These settings are available in the default ``HamiltonianConstructor``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``active_orbitals``
     - vector<int>
     - Indices of active orbitals (0-based, empty for full space)
   * - ``inactive_orbitals``
     - vector<int>
     - Indices of inactive orbitals (0-based, empty for automatic detection)

Related classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input orbitals for Hamiltonian construction
- :doc:`Hamiltonian <../data/hamiltonian>`: Output Hamiltonian representation
- :doc:`ActiveSpaceSelector <active_space>`: Provides active orbital indices
- :doc:`MCCalculator <mc_calculator>`: Uses the Hamiltonian for correlation calculations
