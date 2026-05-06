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
This implementation produces a ``CanonicalFourCenterHamiltonianContainer`` with explicit four-center integrals.

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
     - Type of :term:`SCF` reference ("auto", "unrestricted" or "restricted"). Default: "auto" (automatically detected from orbitals)

QDK Cholesky
~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_cholesky"``

A Cholesky decomposition-based implementation for Hamiltonian construction.
This method uses Cholesky decomposition of the electron repulsion integral (ERI) tensor to reduce memory requirements and computational cost while maintaining high accuracy.
The decomposition represents the four-center ERIs as products of three-center integrals (Cholesky vectors), which are transformed to the MO basis.
The output Hamiltonian stores the MO three-center integrals directly in a ``ThreeCenterHamiltonianContainer``, avoiding expansion to the full four-center representation.
Additionally, the original AO Cholesky vectors are preserved in the container when ``store_ao_cholesky_vectors`` is enabled, and can be retrieved via ``get_ao_cholesky_vectors()``.
Four-center integrals are lazily computed from the three-center integrals on demand.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``scf_type``
     - string
     - Type of :term:`SCF` reference ("auto", "unrestricted" or "restricted"). Default: "auto" (automatically detected from orbitals)
   * - ``cholesky_tolerance``
     - float
     - Tolerance for Cholesky decomposition accuracy. Smaller values give higher accuracy but more Cholesky vectors. Default: 1e-8
   * - ``eri_threshold``
     - float
     - ERI screening threshold for skipping negligible shell quartets during Cholesky decomposition. Default: 1e-12
   * - ``store_ao_cholesky_vectors``
     - bool
     - Whether to store the AO three-center integrals in a ``ThreeCenterHamiltonianContainer`` in addition to the MO three-center integrals, which are always saved. Default: false

QDK Density-Fitted
~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_density_fitted_hamiltonian"``

A memory-efficient implementation that uses density fitting (also known as the resolution-of-the-identity, RI) to approximate the two-electron integrals.
The four-center electron repulsion integrals (ERIs) are factorized through an auxiliary basis as

.. math::

    (ij|kl) \approx \sum_P (ij|P)(P|kl)

where :math:`P` index an auxiliary (fitting) basis and :math:`(ij|P)` is the three-center integrals.
The constructor transforms :math:`(ij|P)` from the AO to the MO basis using the active orbital coefficients and stores the result in a ``ThreeCenterHamiltonianContainer``, reducing the storage from :math:`O(N^4)` to :math:`O(N_{\text{aux}} N^2)`.
Four-center integrals are reconstructed on the fly when consumers request them.

.. rubric:: Requirements

- The input :doc:`Orbitals <../data/orbitals>` must reference a :doc:`BasisSet <../data/basis_set>` that carries an auxiliary basis. Auxiliary shells are attached when the basis set is constructed (e.g. via ``BasisSet::from_basis_name(basis_name, aux_basis_name, structure)``); ``run()`` throws ``std::runtime_error`` if ``has_aux_basis()`` is ``false``.
- An active space must be defined on the orbitals. For unrestricted orbitals, the alpha and beta active spaces must contain the same number of orbitals.

.. rubric:: When to use

- Large active space calculations where memory is a concern
- Systems where the density fitting approximation provides acceptable accuracy


.. tab:: C++ API

   .. code-block:: cpp

      // Build a basis set that includes an auxiliary fitting basis.
      auto basis = data::BasisSet::from_basis_name("cc-pVDZ", "cc-pVDZ-RIFIT", *structure);

      // ... run SCF / orbital localization / active-space selection on `orbitals`
      // such that orbitals->get_basis_set() == basis and an active space is set.

      // Create the density-fitted Hamiltonian constructor.
      auto constructor =
          algorithms::HamiltonianConstructor::create("qdk_density_fitted_hamiltonian");

      // Build the active-space, density-fitted Hamiltonian.
      auto hamiltonian = constructor->run(orbitals);

.. tab:: Python API

   .. code-block:: python

      # Build a basis set that includes an auxiliary fitting basis.
      basis = data.BasisSet.from_basis_name("cc-pVDZ", "cc-pVDZ-RIFIT", structure)

      # ... run SCF / orbital localization / active-space selection on `orbitals`
      # such that orbitals.get_basis_set() is `basis` and an active space is set.

      # Create the density-fitted Hamiltonian constructor.
      constructor = algorithms.create(
          "hamiltonian_constructor", "qdk_density_fitted_hamiltonian"
      )

      # Build the active-space, density-fitted Hamiltonian.
      hamiltonian = constructor.run(orbitals)

See ``examples/language/sample_mp2_reference_energy.py`` for an end-to-end example combining a density-fitted SCF, active-space selection, and ``"qdk_density_fitted_hamiltonian"``.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``scf_type``
     - string
     - Type of :term:`SCF` reference ("auto", "unrestricted" or "restricted"). Default: "auto" (automatically detected from orbitals)

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
