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

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

Configuring the Hamiltonian construction
----------------------------------------

The ``HamiltonianConstructor`` can be configured using the ``Settings`` object to control how integrals are computed.

.. note::
   All orbital indices in QDK/Chemistry are 0-based, following the convention used in most programming languages.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

Constructing the Hamiltonian
----------------------------

Once configured, the Hamiltonian can be constructed from a set of orbitals:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-construct
      :end-before: // end-cell-construct

.. tab:: Python API

   .. note::
      This example shows a complete working workflow including structure creation, SCF calculation, and Hamiltonian construction.

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-construct
      :end-before: # end-cell-construct

Available settings
------------------

The ``HamiltonianConstructor`` accepts a range of settings to control its behavior.
These settings are divided into base settings (common to all Hamiltonian construction) and specialized settings (specific to certain construction variants).

These settings are available in the default ``HamiltonianConstructor``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``eri_method``
     - string
     - Method for computing electron repulsion integrals ("direct" or "incore")
   * - ``force_unrestricted``
     - bool
     - Force unrestricted calculation even for closed-shell systems (default: false)

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/hamiltonian_constructor.py>`_ or `C++ <../../../_static/examples/cpp/hamiltonian_constructor.cpp>`_ scripts.
- :doc:`Orbitals <../data/orbitals>`: Input orbitals for Hamiltonian construction
- :doc:`Hamiltonian <../data/hamiltonian>`: Output Hamiltonian representation
- :doc:`ActiveSpaceSelector <active_space>`: Provides active orbital indices
- :doc:`MCCalculator <mc_calculator>`: Uses the Hamiltonian for correlation calculations
