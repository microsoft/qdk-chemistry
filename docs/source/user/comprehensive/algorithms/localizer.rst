Orbital localization
====================

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm in QDK/Chemistry performs various orbital transformations to create localized or otherwise transformed molecular orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes an :doc:`Orbitals <../data/orbitals>` instance as input and produces a new :doc:`Orbitals <../data/orbitals>` instance as output.
For more information about this pattern, see the :doc:`Factory Pattern <../design/factory_pattern>` documentation.

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
   Maximizes the sum of squared Mulliken charges on each atom for each orbital, creating orbitals that are maximally localized on specific atoms or bonds.

**MP2 Natural Orbitals**
   Transforms canonical orbitals into natural orbitals based on MP2 density matrices, providing orbitals that diagonalize the correlation effects.

**Valence Virtual Hard Virtual (VVHV) Orbitals**
   Separates orbitals into valence, virtual, and hard virtual categories for more efficient treatment in correlation methods.

Usage
-----

Before performing localization, you need an :doc:`Orbitals <../data/orbitals>` instance as input.
This is typically obtained from an :doc:`ScfSolver <scf_solver>` calculation, as localization is usually applied to converged :term:`SCF` orbitals.

The most common use case is localizing occupied orbitals after an SCF calculation:

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


Configuring the localizer
-------------------------

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` can be configured using the :doc:`Settings <../design/settings>` object:

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

Performing orbital localization
-------------------------------

Before performing localization, you need an :doc:`Orbitals <../data/orbitals>` instance as input.
This is typically obtained from an :doc:`ScfSolver <scf_solver>` calculation, as localization is usually applied to converged :term:`SCF` orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, the :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm takes an :doc:`Orbitals <../data/orbitals>` object as input and produces a new :doc:`Orbitals <../data/orbitals>` object as output, preserving the original orbitals while creating a transformed representation.

The ``run`` method requires three parameters:

1. **orbitals**: The input :doc:`Orbitals <../data/orbitals>` instance to be localized
2. **loc_indices_a**: Vector/list of indices specifying which alpha orbitals to localize
3. **loc_indices_b**: Vector/list of indices specifying which beta orbitals to localize

.. note::
   For restricted calculations, ``loc_indices_a`` and ``loc_indices_b`` must be identical.
   If empty vectors/lists are provided, no orbitals of that spin type will be localized.

Once configured, the localization can be performed on a set of orbitals:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/localizer.cpp
      :language: cpp
      :start-after: // start-cell-localize
      :end-before: // end-cell-localize

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../_static/examples/python/localizer.py
      :language: python
      :start-after: # start-cell-localize
      :end-before: # end-cell-localize

Available localization methods
------------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` provides a unified interface for localization methods.

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **QDK/Chemistry**: Native implementation of Pipek-Mezey and MP2 natural orbital localization
- **QDK/Chemistry**: VVHV orbital separation method

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **PySCF**: Interface to PySCF's orbital localization methods (Pipek-Mezey, Boys, etc.)

The factory pattern allows seamless selection between these implementations.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.


Available settings
------------------

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` settings vary by implementation. Different localization algorithms have their own specific configuration parameters.

Specialized settings
~~~~~~~~~~~~~~~~~~~~

These settings apply only to specific variants of localization:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 30 20

   * - Setting
     - Type
     - Default
     - Description
     - Applicable To
   * - ``tolerance``
     - float
     - 1.0e-6
     - Convergence criterion for localization iterations
     - Pipek-Mezey, Valence Virtual Hard Virtual (VVHV)
   * - ``max_iterations``
     - int
     - 10000
     - Maximum number of localization iterations
     - Pipek-Mezey, :term:`VVHV`
   * - ``small_rotation_tolerance``
     - float
     - 1.0e-12
     - Threshold for small rotation detection
     - Pipek-Mezey, :term:`VVHV`
   * - ``minimal_basis``
     - string
     - "sto-3g"
     - Name of the minimal basis set used for valence virtual projection
     - :term:`VVHV`
   * - ``weighted_orthogonalization``
     - bool
     - true
     - Whether to use weighted orthogonalization in hard virtual construction
     - :term:`VVHV`
   * - ``method``
     - string
     - "pipek-mezey"
     - Localization algorithm to use ("pipek-mezey", "foster-boys", "edmiston-ruedenberg", "cholesky")
     - PySCF
   * - ``population_method``
     - string
     - "mulliken"
     - Population analysis method for Pipek-Mezey localization
     - PySCF
   * - ``occupation_threshold``
     - float
     - 1.0e-10
     - Threshold for classifying orbitals as occupied vs virtual
     - PySCF

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/localizer.py>`_ or `C++ <../../../_static/examples/cpp/localizer.cpp>`_ code.
- :doc:`Orbitals <../data/orbitals>`: Input and output orbitals
- :doc:`ScfSolver <scf_solver>`: Produces initial orbitals for localization
- :doc:`ActiveSpaceSelector <active_space>`: Often used with localized orbitals for better active space selection
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Can build Hamiltonians using localized orbitals
- :doc:`Wavefunction <../data/wavefunction>`: Container for orbitals and electronic state information

Related topics
--------------

- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Factory pattern <../design/factory_pattern>`: Creating algorithm instances
