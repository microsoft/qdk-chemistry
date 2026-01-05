Hamiltonian construction
========================

The ``HamiltonianConstructor`` algorithm in QDK/Chemistry constructs electronic Hamiltonians for quantum chemistry calculations.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes an :doc:`Orbitals <../data/orbitals>` instance as input and produces a :doc:`Hamiltonian <../data/hamiltonian>` instance as output.
It generates the one- and two-electron integrals that define the energy operator for the electronic structure.

Overview
--------

The electronic Hamiltonian describes the energy of a system of electrons in the field of atomic nuclei.
It consists of kinetic energy terms, electron-nucleus attraction terms, and electron-electron repulsion terms.
The ``HamiltonianConstructor`` algorithm computes the matrix elements of this operator in a given orbital basis, which can be the full orbital space or an active subspace.

Using the HamiltonianConstructor
---------------------------------

This section demonstrates how to create, configure, and run a Hamiltonian construction. The ``run`` method returns a :doc:`Hamiltonian <../data/hamiltonian>` object containing the one- and two-electron integrals.

Input requirements
~~~~~~~~~~~~~~~~~~

The ``HamiltonianConstructor`` requires the following input:

Orbitals
   An :doc:`Orbitals <../data/orbitals>` instance describing the single orbital basis in which to express the many-body Hamiltonian. This object contains information about the molecular structure, basis set, and orbital coefficients.

.. note::

   The Orbitals object carries all the information needed for integral transformation, including the basis set and molecular structure. Active space indices, if present, determine which orbitals are included in the output Hamiltonian.

.. rubric:: Creating a constructor

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

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

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

.. rubric:: Running the calculation

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-construct
      :end-before: // end-cell-construct

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-construct
      :end-before: # end-cell-construct

Available implementations
-------------------------

QDK/Chemistry's ``HamiltonianConstructor`` provides a unified interface for Hamiltonian construction methods.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

QDK (Native)
~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk"`` (default)

The native QDK/Chemistry implementation for Hamiltonian construction. Transforms molecular orbitals from :term:`AO` to :term:`MO` basis and computes one- and two-electron integrals.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``eri_method``
     - string
     - Method for computing electron repulsion integrals ("direct" or "incore")
   * - ``scf_type``
     - string
     - Type of :term:`SCF` reference ("rhf", "rohf", or "uhf")

Related classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input orbitals for Hamiltonian construction
- :doc:`Hamiltonian <../data/hamiltonian>`: Output Hamiltonian representation

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/hamiltonian_constructor.py>`_ or `C++ <../../../_static/examples/cpp/hamiltonian_constructor.cpp>`_ scripts.
- :doc:`ActiveSpaceSelector <active_space>`: Provides active orbital indices
- :doc:`MCCalculator <mc_calculator>`: Uses the Hamiltonian for correlation calculations
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
