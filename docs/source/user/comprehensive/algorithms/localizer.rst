Orbital localization
====================

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm in QDK/Chemistry performs various orbital transformations to create localized or otherwise transformed molecular orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.Wavefunction` instance with reference orbitals as input and produces a new :class:`~qdk_chemistry.data.Wavefunction` instance with localized orbitals as output.
For more information about this pattern, see the :doc:`Factory Pattern <factory_pattern>` documentation.

Overview
--------

Canonical molecular orbitals from :term:`SCF` calculations are typically delocalized over the entire molecule, which can complicate chemical interpretation and slow the convergence of post-:term:`SCF` correlation methods.
The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm applies unitary transformations to these orbitals to obtain alternative representations that may be more physically intuitive or computationally advantageous.

Orbital transformation techniques broadly fall into two categories:

Optimization-Based Methods
   The vast majority of localization methods define a cost function and iteratively minimize it to yield localized orbitals :cite:`Lehtola2013`.
   Popular choices include **Pipek-Mezey** :cite:`Pipek1989`, **Foster-Boys** :cite:`Foster1960`, and **Edmiston-Ruedenberg** :cite:`Edmiston1963` localization.
   These methods produce orbitals that are maximally localized on specific atoms or bonds.

Analytical Methods
   These methods transform orbitals in a single step through analytical techniques rather than iterative optimization.
   **Natural orbitals** :cite:`Lowdin1956`, which diagonalize the one-particle reduced density matrix, are a prominent example and can be particularly useful for :doc:`active space selection <active_space>`.
   **Cholesky localization** :cite:`Aquilante2006` provides efficient approximate localization via Cholesky decomposition of the density matrix.
   The :ref:`VVHV separation <vvhv-algorithm>` also falls into this category, using projection and orthogonalization to partition virtual orbitals into valence and hard-virtual subspaces.

The specific methods available depend on the backend implementation selected.
QDK/Chemistry provides native implementations of key algorithms and extends these capabilities through :doc:`plugins <../plugins>` that integrate external quantum chemistry packages.
See `Available implementations`_ below for details on each backend and its supported methods.

Running orbital localization
----------------------------

This section demonstrates how to create, configure, and run orbital localization.
The ``run`` method takes a :class:`~qdk_chemistry.data.Wavefunction` instance and returns a new :class:`~qdk_chemistry.data.Wavefunction` object with transformed orbitals.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` requires the following inputs:

Wavefunction
   A :class:`~qdk_chemistry.data.Wavefunction` instance containing the molecular orbitals to be localized.

Alpha orbital indices (``loc_indices_a``)
   A list/vector of indices specifying which alpha orbitals to include in the localization. Indices must be sorted in ascending order.

Beta orbital indices (``loc_indices_b``)
   A list/vector of indices specifying which beta orbitals to include in the localization. Indices must be sorted in ascending order.


.. rubric:: Creating a localizer

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/localizer.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/localizer.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/localizer.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/localizer.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running localization

.. note::
   For restricted calculations, ``loc_indices_a`` and ``loc_indices_b`` must be identical.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/localizer.cpp
      :language: cpp
      :start-after: // start-cell-localize
      :end-before: // end-cell-localize

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/localizer.py
      :language: python
      :start-after: # start-cell-localize
      :end-before: # end-cell-localize

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` provides a unified interface to orbital localization methods.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/localizer.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/localizer.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

.. _localizer-qdk-pipek-mezey:

QDK Pipek-Mezey
~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_pipek_mezey"`` (default)

Native implementation of Pipek-Mezey localization :cite:`Pipek1989`, which maximizes the sum of squared Mulliken charges on each atom for each orbital.
This produces orbitals that are maximally localized on specific atoms or bonds, making them well-suited for chemical interpretation and local correlation methods.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``tolerance``
     - float
     - ``1e-6``
     - Convergence criterion for localization iterations
   * - ``max_iterations``
     - int
     - ``10000``
     - Maximum number of localization iterations
   * - ``small_rotation_tolerance``
     - float
     - ``1e-12``
     - Threshold for small rotation detection

QDK MP2 Natural Orbitals
~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"qdk_mp2_natural_orbitals"``

Computes natural orbitals :cite:`Lowdin1956` from the :term:`MP2` one-particle density matrix.
These orbitals diagonalize the correlation effects and provide occupation numbers that can guide :doc:`active space selection <active_space>`.

.. rubric:: Settings

This implementation has no configurable settings.

.. _localizer-qdk-vvhv:

QDK VVHV
~~~~~~~~

.. rubric:: Factory name: ``"qdk_vvhv"``

The :term:`VVHV` (Valence Virtual--Hard Virtual) localizer addresses the numerical challenges of orbital localization in near-complete basis sets by partitioning the virtual space into chemically meaningful subspaces.
See :ref:`VVHV Algorithm <vvhv-algorithm>` below for a detailed description.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``tolerance``
     - float
     - ``1e-6``
     - Convergence criterion for iterative localization optimization
   * - ``max_iterations``
     - int
     - ``10000``
     - Maximum number of localization iterations
   * - ``small_rotation_tolerance``
     - float
     - ``1e-12``
     - Threshold for small rotation detection
   * - ``minimal_basis``
     - string
     - ``"sto-3g"``
     - Minimal basis set for valence virtual projection
   * - ``weighted_orthogonalization``
     - bool
     - ``True``
     - Use weighted orthogonalization in hard virtual construction

.. _vvhv-algorithm:

VVHV Algorithm
^^^^^^^^^^^^^^

Localization of molecular orbitals expressed in near-complete :doc:`basis sets <../data/basis_set>` is numerically ill-posed and challenging for most localizers.
This can lead to orbitals that do not vary smoothly with molecular geometry, numerically unstable results, and reproducibility difficulties across architectures and compute environments.
The Valence Virtual--Hard Virtual (:term:`VVHV`) separation :cite:`Subotnik2005` addresses these problems by partitioning the virtual orbital space into chemically meaningful subspaces before localization.

.. rubric:: The Problem with Standard Localization

Standard orbital localization methods optimize a cost function (e.g., Pipek-Mezey, Foster-Boys) over blocks of orbitals simultaneously (e.g. occupied, virtual, active).
In large basis sets, the virtual space contains orbitals of vastly different character:

Valence-virtual orbitals
   Low-lying virtual orbitals that are chemically relevant for describing bond breaking/formation and correlation effects
Hard-virtual orbitals
   High-energy orbitals that primarily describe core-valence polarization and basis set completeness

When localization is applied to the full virtual space, the optimizer may mix these distinct orbital types, leading to non-physical results that are sensitive to numerical precision and can vary discontinuously along reaction coordinates.

.. rubric:: The VVHV Separation Procedure

The :term:`VVHV` algorithm proceeds in three stages:

1. **Minimal Basis Projection**: Project the canonical virtual orbitals onto a minimal basis set (e.g., :term:`STO`-3G) to identify the valence-virtual subspace.
   Given the overlap matrix :math:`\mathbf{S}_{\text{min}}` between the computational basis and minimal basis, the valence-virtual orbitals span the range of:

   .. math::

      \mathbf{P}_{\text{VV}} = \mathbf{S}_{\text{min}} (\mathbf{S}_{\text{min}}^T \mathbf{S}_{\text{min}})^{-1} \mathbf{S}_{\text{min}}^T

2. **Orthogonalization**: Construct orthonormal valence-virtual and hard-virtual orbital sets.
   The hard-virtual orbitals are obtained as the orthogonal complement to the valence-virtual space.
   QDK/Chemistry implements both standard and weighted orthogonalization :cite:`Wang2025` procedures; weighted orthogonalization improves numerical stability for near-linear dependencies.

3. **Subspace Localization**: Apply the chosen localization method (e.g., Pipek-Mezey) separately within each subspace.
   This ensures that the optimization landscape is well-behaved and that orbitals vary smoothly with geometry.

.. rubric:: Benefits for Active Space Selection

The :term:`VVHV` separation is particularly valuable for multi-configuration calculations where a consistent active space must be maintained along a reaction pathway.
By localizing only the valence-virtual orbitals, which are the chemically relevant virtual orbitals for active space construction, the :term:`VVHV` procedure ensures:

- Orbitals that track smoothly as bonds stretch and form
- Numerically stable and reproducable results
- Well-defined orbital character that aids chemical interpretation

For details on using localized orbitals in active space selection, see :doc:`ActiveSpaceSelector <active_space>`.

.. _localizer-pyscf-multi:

PySCF Multi
~~~~~~~~~~~

.. rubric:: Factory name: ``"pyscf_multi"``

The PySCF :doc:`plugin <../plugins>` provides access to additional localization algorithms through `PySCF <https://pyscf.org/>`_:

Pipek-Mezey :cite:`Pipek1989`
   Maximizes atomic charge localization
Foster-Boys :cite:`Foster1960`
   Minimizes the spatial extent of orbitals
Edmiston-Ruedenberg :cite:`Edmiston1963`
   Maximizes self-repulsion energy
Cholesky :cite:`Aquilante2006`
   Efficient analytical localization via Cholesky decomposition

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``method``
     - string
     - ``"pipek-mezey"``
     - Localization algorithm: ``"pipek-mezey"``, ``"foster-boys"``, ``"edmiston-ruedenberg"``, or ``"cholesky"``
   * - ``population_method``
     - string
     - ``"mulliken"``
     - Population analysis method for Pipek-Mezey localization
   * - ``occupation_threshold``
     - float
     - ``1e-10``
     - Threshold for classifying orbitals as occupied vs virtual


Related classes
---------------

- :class:`~qdk_chemistry.data.Wavefunction`: Input and output container for orbitals and electronic state information
- :doc:`Orbitals <../data/orbitals>`: Molecular orbital data accessible via the wavefunction

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/localizer.py>`_ or `C++ <../../../_static/examples/cpp/localizer.cpp>`_ code.
- :doc:`ScfSolver <scf_solver>`: Produces initial orbitals for localization
- :doc:`ActiveSpaceSelector <active_space>`: Often used with localized orbitals for better active space selection
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Can build Hamiltonians using localized orbitals
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory pattern <factory_pattern>`: Creating algorithm instances
