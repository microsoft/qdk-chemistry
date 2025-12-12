Orbital localization
====================

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm in QDK/Chemistry performs various orbital transformations to create localized or otherwise transformed molecular orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes an :doc:`Orbitals <../data/orbitals>` instance as input and produces a new :doc:`Orbitals <../data/orbitals>` instance as output.
For more information about this pattern, see the :doc:`Factory Pattern <factory_pattern>` documentation.

These transformations preserve the overall electronic state but provide orbitals with different properties that are useful for chemical analysis or subsequent calculations.

Overview
--------

Canonical molecular orbitals from :term:`SCF` calculations are often delocalized over the entire molecule, which can make chemical interpretation difficult and lead to slow convergence in post-:term:`HF` methods.
The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm applies unitary transformations to these orbitals to obtain alternative representations that may be more physically intuitive or computationally advantageous.
Multiple localization methods are available through a unified interface, each optimizing different criteria to achieve localization.

Localization methods
--------------------

QDK/Chemistry provides several orbital transformation methods through the :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` interface:

**Pipek-Mezey Localization**
   Maximizes the sum of squared Mulliken charges on each atom for each orbital, creating orbitals that are maximally localized on specific atoms or bonds :cite:`Boughton1993, Edmiston1963`.

**MP2 Natural Orbitals**
   Transforms canonical orbitals into natural orbitals based on MP2 density matrices, providing orbitals that diagonalize the correlation effects.

**Valence Virtual Hard Virtual (VVHV) Orbitals**
   Separates orbitals into valence, virtual, and hard virtual categories for more efficient treatment in correlation methods.

Running orbital localization
----------------------------

This section demonstrates how to create, configure, and run orbital localization.
The ``run`` method takes an :doc:`Orbitals <../data/orbitals>` instance (typically from an :doc:`ScfSolver <scf_solver>` calculation) and returns a new :doc:`Orbitals <../data/orbitals>` object with transformed orbitals.

**Creating a localizer:**

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

**Configuring settings:**

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

**Running localization:**

The ``run`` method requires three parameters:

1. **orbitals**: The input :doc:`Orbitals <../data/orbitals>` instance to be localized
2. **loc_indices_a**: Vector/list of indices specifying which alpha orbitals to localize
3. **loc_indices_b**: Vector/list of indices specifying which beta orbitals to localize

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

   .. code-block:: cpp

      auto names = LocalizerFactory::available();
      for (const auto& name : names) {
          std::cout << name << std::endl;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import registry
      print(registry.available("orbital_localizer"))
      # ['pyscf_multi', 'qdk_vvhv', 'qdk_mp2_natural_orbitals', 'qdk_pipek_mezey']

QDK Pipek-Mezey
~~~~~~~~~~~~~~~

**Factory name:** ``"qdk_pipek_mezey"`` (default)

Maximizes the sum of squared Mulliken charges on each atom for each orbital, creating orbitals that are maximally localized on specific atoms or bonds.

**Settings:**

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

QDK MP2 natural orbitals
~~~~~~~~~~~~~~~~~~~~~~~~

**Factory name:** ``"qdk_mp2_natural_orbitals"``

Transforms canonical orbitals into natural orbitals based on MP2 density matrices, providing orbitals that diagonalize the correlation effects.

**Settings:** This implementation has no configurable settings.

QDK VVHV
~~~~~~~~

**Factory name:** ``"qdk_vvhv"``

Valence Virtual Hard Virtual (VVHV) orbital separation—separates orbitals into valence, virtual, and hard virtual categories for more efficient treatment in correlation methods.

**Settings:**

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
   * - ``minimal_basis``
     - string
     - ``"sto-3g"``
     - Minimal basis set for valence virtual projection
   * - ``weighted_orthogonalization``
     - bool
     - ``True``
     - Use weighted orthogonalization in hard virtual construction

PySCF Multi
~~~~~~~~~~~

**Factory name:** ``"pyscf_multi"``

The PySCF plugin provides access to `PySCF's <https://pyscf.org/>`_ orbital localization routines, supporting multiple localization algorithms.

**Capabilities:**

- Pipek-Mezey (PM) localization
- Foster-Boys (FB) localization
- Edmiston-Ruedenberg (ER) localization
- Cholesky-based localization

**Settings:**

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

**Example:**

.. code-block:: python

   from qdk_chemistry.algorithms import create

   localizer = create("orbital_localizer", "pyscf_multi")
   localizer.settings().set("method", "foster-boys")

   # Localize occupied orbitals
   occ_indices = list(range(n_occupied))
   localized_orbs = localizer.run(orbitals, occ_indices, occ_indices)

For more details on how to extend QDK/Chemistry with additional implementations, see the :doc:`plugin system <../plugins>` documentation.

Related classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input and output orbitals
- :doc:`Wavefunction <../data/wavefunction>`: Container for orbitals and electronic state information

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/localizer.py>`_ or `C++ <../../../_static/examples/cpp/localizer.cpp>`_ code.
- :doc:`ScfSolver <scf_solver>`: Produces initial orbitals for localization
- :doc:`ActiveSpaceSelector <active_space>`: Often used with localized orbitals for better active space selection
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Can build Hamiltonians using localized orbitals
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory pattern <factory_pattern>`: Creating algorithm instances
