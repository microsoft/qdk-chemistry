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

Auxiliary bases (optional)
   A separate :class:`~qdk_chemistry.data.AuxiliaryBasisCollection`. The density-fitted implementation requires an ``RIFIT`` entry; the canonical and Cholesky implementations do not require auxiliary bases.

.. note::

   The Orbitals object supplies the primary basis, molecular structure, and MO coefficients. Active space indices determine which orbitals are included in the output Hamiltonian; auxiliary fitting bases are supplied separately.

.. rubric:: Creating a constructor

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. note::
   All orbital indices in QDK/Chemistry are 0-based, following the convention used in most programming languages.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. rubric:: Running the calculation

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-construct
      :end-before: # end-cell-construct

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-construct
      :end-before: // end-cell-construct

Available implementations
-------------------------

QDK/Chemistry's ``HamiltonianConstructor`` provides a unified interface for Hamiltonian construction methods.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

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

.. _hamiltonian-constructor-cholesky:

QDK Cholesky
~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_cholesky"``

A Cholesky decomposition-based implementation for Hamiltonian construction.
This method uses Cholesky decomposition of the electron repulsion integral (ERI) tensor to reduce memory requirements and computational cost while maintaining high accuracy.
The decomposition represents the four-center ERIs as products of three-center integrals (Cholesky vectors), which are transformed to the MO basis.
The output Hamiltonian stores the MO three-center integrals directly in a ``ThreeCenterHamiltonianContainer``, avoiding expansion to the full four-center representation.
Additionally, the original AO three-center vectors are preserved in the container when ``store_ao_cholesky_vectors`` is enabled, and can be retrieved via ``ao_three_center_vectors()``.
Four-center integrals are lazily computed from the three-center integrals on demand.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Setting
     - Type
     - Description
   * - ``cholesky_tolerance``
     - float
     - Tolerance for Cholesky decomposition accuracy. Smaller values give higher accuracy but more Cholesky vectors. Default: 1e-8
   * - ``eri_threshold``
     - float
     - ERI screening threshold for skipping negligible shell quartets during Cholesky decomposition. Default: 1e-12
   * - ``store_ao_cholesky_vectors``
     - bool
     - Whether to store the AO three-center integrals in a ``ThreeCenterHamiltonianContainer`` in addition to the MO three-center integrals, which are always saved. Default: false

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-cholesky
      :end-before: # end-cell-cholesky

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-cholesky
      :end-before: // end-cell-cholesky

.. _hamiltonian-constructor-density-fitted:

QDK Density-Fitted
~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_density_fitted_hamiltonian"``

A memory-efficient implementation that uses density fitting (also known as the resolution-of-the-identity, RI) to approximate the two-electron integrals.
For raw three-center integrals :math:`E_{ij,P}=(ij|P)` and auxiliary Coulomb metric :math:`M_{PQ}=(P|Q)`, the constructor computes a Cholesky factor :math:`M=LL^T` and stores the metric-orthonormalized factors

.. math::

   B_{ij}^{Q} = \sum_P (ij|P)\left(L^{-T}\right)_{PQ}.

The resulting approximation is

.. math::

   (ij|kl) \approx
   \sum_{P,Q} (ij|P)\left(M^{-1}\right)_{PQ}(Q|kl)
   = \sum_Q B_{ij}^{Q} B_{kl}^{Q}.

The constructor forms these factors in the AO basis, transforms them to the active MO basis, and stores them in a :ref:`three-center Hamiltonian container <hamiltonian-three-center-container>`.
Raw three-center integrals cannot be contracted directly without the inverse metric. Four-center integrals are materialized lazily when consumers request them.

.. rubric:: Requirements

- The input :doc:`Orbitals <../data/orbitals>` must reference the primary :doc:`BasisSet <../data/basis_set>`.
- Pass an :class:`~qdk_chemistry.data.AuxiliaryBasisCollection` containing an exact ``RIFIT`` association as the second argument to ``run()``. The auxiliary and primary bases must describe the same molecular structure.
- An active space must be defined on the orbitals. For unrestricted orbitals, the alpha and beta active spaces must contain the same number of orbitals.
- The auxiliary Coulomb metric must be positive definite. A failed Cholesky factorization, including exact linear dependence in the auxiliary basis, is rejected.

.. note::

   This constructor uses density fitting for **both** the Coulomb (J) and exchange (K) contributions to the Fock matrix during the integral transformation.
   Unlike hybrid DF-J / exact-K schemes, all two-electron interactions are approximated through the auxiliary basis, making the method uniformly RI-accelerated but reliant on a well-matched fitting basis for accuracy.

.. rubric:: When to use

- Large active space calculations where memory is a concern
- Systems where the density fitting approximation provides acceptable accuracy

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/hamiltonian_constructor.py
      :language: python
      :start-after: # start-cell-density-fitted
      :end-before: # end-cell-density-fitted

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/hamiltonian_constructor.cpp
      :language: cpp
      :start-after: // start-cell-density-fitted
      :end-before: // end-cell-density-fitted

See ``examples/language/sample_mp2_reference_energy.py`` for an end-to-end example combining a four-center SCF, active-space selection, ``"qdk_density_fitted_hamiltonian"``, and density-fitted MP2.

.. rubric:: Settings

This implementation has no configurable settings.

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
