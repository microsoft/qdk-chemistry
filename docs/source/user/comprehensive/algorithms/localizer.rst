Orbital localization
====================

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm in QDK/Chemistry performs various orbital transformations to create localized or otherwise transformed molecular orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes an :doc:`Orbitals <../data/orbitals>` instance as input and produces a new :doc:`Orbitals <../data/orbitals>` instance as output.
These transformations preserve the overall electronic state but provide orbitals with different properties that are useful for chemical analysis or subsequent calculations.

Overview
--------

Canonical molecular orbitals from :term:`SCF` calculations are often delocalized over the entire molecule, which can make chemical interpretation difficult and lead to slow convergence in post-:term:`HF` methods.
The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` algorithm applies unitary transformations to these orbitals to obtain alternative representations that may be more physically intuitive or computationally advantageous.
Multiple localization methods are available through a unified interface, each optimizing different criteria to achieve localization.

Localization Methods
--------------------

QDK/Chemistry provides several orbital transformation methods through the :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` interface:

- **Pipek-Mezey Localization**
- **Natural Orbitals**
- **Second-order Møller-Plesset (MP2) Natural Orbitals**

Creating a localizer
--------------------

As an algorithm class in QDK/Chemistry, the :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` follows the :doc:`factory pattern design principle <../design/index>`.
It is created using its corresponding factory, which provides a unified interface for different localization method implementations.
For more information about this pattern, see the :doc:`Factory Pattern <../design/factory_pattern>` documentation.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::algorithms;

      // Create an MP2 natural orbital localizer
      auto mp2_localizer = LocalizerFactory::create("mp2_natural_orbitals");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/localizer.py
      :language: python
      :lines: 3-6


Configuring the localizer
-------------------------

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` can be configured using the :doc:`Settings <../design/settings>` object:

.. tab:: C++ API

   .. code-block:: cpp

      // Set the convergence threshold
      localizer->settings().set("tolerance", 1.0e-6);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/localizer.py
      :language: python
      :lines: 10-10

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

   .. code-block:: cpp

      // Obtain a valid Orbitals instance
      Orbitals orbitals;
      /* orbitals = ... */

      // Configure electron counts in settings for methods that require them
      localizer->settings().set("n_alpha_electrons", n_alpha);
      localizer->settings().set("n_beta_electrons", n_beta);

      // Create indices for orbitals to localize
      std::vector<size_t> loc_indices_a = {0, 1, 2, 3}; // Alpha orbital indices
      std::vector<size_t> loc_indices_b = {0, 1, 2, 3}; // Beta orbital indices

      // Localize the specified orbitals
      auto localized_orbitals = localizer->run(orbitals, loc_indices_a, loc_indices_b);

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/factory_pattern.py
      :language: python
      :lines: 1-9

Available localization methods
------------------------------

qdk_mp2_natural_orbitals
   :term:`MP2` Natural Orbitals

qdk_pipek_mezey
   Pipek-Mezey Localized Orbitals

qdk_vvhv
   Valence Virtual Hard Virtual (:term:`VVHV`) Orbitals

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` accepts a range of settings to control its behavior. These settings are divided into base settings
(common to all localization methods) and specialized settings (specific to certain localization variants).

Base settings
~~~~~~~~~~~~~

.. note::
   TODO:  The base settings table is currently under construction.
   Please see `online examples <https://github.com/microsoft/qdk-chemistry/blob/main/examples/factory_list.ipynb>`_ for the most up-to-date information.

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
   * - ``n_alpha_electrons``
     - int
     - Required
     - Number of alpha electrons.
       Orbital indices < n_alpha_electrons are treated as occupied, indices >= n_alpha_electrons are treated as virtual.
     - :term:`MP2` Natural Orbitals, :term:`VVHV`
   * - ``n_beta_electrons``
     - int
     - Required
     - Number of beta electrons.
       Orbital indices < n_beta_electrons are treated as occupied, indices >= n_beta_electrons are treated as virtual.
     - :term:`MP2` Natural Orbitals, :term:`VVHV`
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

Implemented interface
---------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.OrbitalLocalizer` provides a unified interface for localization methods.

QDK/Chemistry implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **QDK/Chemistry**: Native implementation of Pipek-Mezey, and :term:`MP2` natural orbital localization

Third-party interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **PySCF**: Interface to PySCF's orbital localization methods

The factory pattern allows seamless selection between these implementations.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

Related classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input and output orbitals
- :doc:`ScfSolver <scf_solver>`: Produces initial orbitals for localization
- :doc:`ActiveSpaceSelector <active_space>`: Often used with localized orbitals
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Can build Hamiltonians using localized orbitals
