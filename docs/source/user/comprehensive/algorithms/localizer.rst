Localizer
=========

The ``Localizer`` algorithm in QDK/Chemistry performs various orbital transformations to create localized or otherwise transformed
molecular orbitals. Following QDK/Chemistry's :doc:`algorithm design principles <../advanced/design_principles>`, it takes an
:doc:`Orbitals <../data/orbitals>` instance as input and produces a new :doc:`Orbitals <../data/orbitals>` instance as
output. These transformations preserve the overall electronic state but provide orbitals with different properties that
are useful for chemical analysis or subsequent calculations.

Overview
--------

Canonical molecular orbitals from SCF calculations are often delocalized over the entire molecule, which can make
chemical interpretation difficult and lead to slow convergence in post-HF methods. The ``Localizer`` algorithm applies
unitary transformations to these orbitals to obtain alternative representations that may be more physically intuitive or
computationally advantageous. Multiple localization methods are available through a unified interface, each optimizing
different criteria to achieve localization.

Localization Methods
--------------------

QDK/Chemistry provides several orbital transformation methods through the ``Localizer`` interface:

- **Pipek-Mezey Localization**
- **Natural Orbitals**
- **MP2 Natural Orbitals**

.. todo::
   TODO (NAB):  add names for invoking localizer methods to list (e.g., "mp2_natural_orbitals")

Creating a Localizer
--------------------

As an algorithm class in QDK/Chemistry, the ``Localizer`` follows the
:doc:`factory pattern design principle <../advanced/design_principles>`. It is created using its corresponding factory,
which provides a unified interface for different localization method implementations. For more information about this
pattern, see the :doc:`Factory Pattern <../advanced/factory_pattern>` documentation.

.. todo::
   TODO (NAB):  Check Localizer code examples after finalizing API.
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41366

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::algorithms;

      // Create an MP2 natural orbital localizer
      auto mp2_localizer = LocalizerFactory::create("mp2_natural_orbitals");

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.algorithms import create_localizer

      # Create an MP2 natural orbital localizer
      mp2_localizer = create_localizer("mp2_natural_orbitals")

Configuring the Localizer
-------------------------

The ``Localizer`` can be configured using the ``Settings`` object:

.. note::
   🔧 **TODO**: Check implementational details

.. todo::
   TODO (NAB):  finish Localizer documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41380

.. tab:: C++ API

   .. code-block:: cpp

      // Set the convergence threshold
      localizer->settings().set("tolerance", 1.0e-6);

.. tab:: Python API

   .. code-block:: python

      # Set the convergence threshold
      localizer.settings().set("tolerance", 1.0e-6)

Performing Orbital Localization
-------------------------------

Before performing localization, you need an :doc:`Orbitals <../data/orbitals>` instance as input. This is typically
obtained from an :doc:`ScfSolver <scf_solver>` calculation, as localization is usually applied to converged SCF orbitals.
Following QDK/Chemistry's :doc:`algorithm design principles <../advanced/design_principles>`, the ``Localizer`` algorithm takes an
``Orbitals`` object as input and produces a new ``Orbitals`` object as output, preserving the original orbitals while
creating a transformed representation.

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

   .. code-block:: python

      # Obtain a valid Orbitals instance
      orbitals = Orbitals()
      # orbitals = ...

      # Configure electron counts in settings for methods that require them
      localizer.settings().set("n_alpha_electrons", n_alpha)
      localizer.settings().set("n_beta_electrons", n_beta)

      # Create indices for orbitals to localize
      loc_indices_a = [0, 1, 2, 3]  # Alpha orbital indices
      loc_indices_b = [0, 1, 2, 3]  # Beta orbital indices

      # Localize the specified orbitals
      localized_orbitals = localizer.run(orbitals, loc_indices_a, loc_indices_b)

Available Localization Methods
------------------------------

.. todo::
   🔧 **TODO**: Add detailed descriptions of each localization method with theory and practical considerations

   TODO (NAB):  finish Localizer documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41380

Available Settings
------------------

The ``Localizer`` accepts a range of settings to control its behavior. These settings are divided into base settings
(common to all localization methods) and specialized settings (specific to certain localization variants).

.. todo::
   🔧 **TODO**: Verify and complete the settings tables with accurate parameters and descriptions

   TODO (NAB):  finish Localizer documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41380

Base Settings
~~~~~~~~~~~~~

These settings apply to all localization methods:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description

Specialized Settings
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
     - Pipek-Mezey, VVHV
   * - ``max_iterations``
     - int
     - 10000
     - Maximum number of localization iterations
     - Pipek-Mezey, VVHV
   * - ``small_rotation_tolerance``
     - float
     - 1.0e-12
     - Threshold for small rotation detection
     - Pipek-Mezey, VVHV
   * - ``n_alpha_electrons``
     - int
     - Required
     - Number of alpha electrons. Orbital indices < n_alpha_electrons are treated as occupied, indices >= n_alpha_electrons are treated as virtual.
     - MP2 Natural Orbitals, VVHV
   * - ``n_beta_electrons``
     - int
     - Required
     - Number of beta electrons. Orbital indices < n_beta_electrons are treated as occupied, indices >= n_beta_electrons are treated as virtual.
     - MP2 Natural Orbitals, VVHV
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
     - VVHV
   * - ``weighted_orthogonalization``
     - bool
     - true
     - Whether to use weighted orthogonalization in hard virtual construction
     - VVHV

Computational Scaling
---------------------

.. todo::
   🔧 **TODO**: Add computational scaling for all localization methods

The computational cost of orbital localization methods scales with the size of the system:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Method
     - Scaling
     - Notes
   * - MP2 Natural Orbitals
     - O(N⁵)
     - Requires MP2 density matrix construction

Implemented Interface
---------------------

QDK/Chemistry's ``Localizer`` provides a unified interface for localization methods::

QDK/Chemistry Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **QDK/Chemistry**: Native implementation of Pipek-Mezey, and MP2 natural orbital localization

Third-Party Interfaces
~~~~~~~~~~~~~~~~~~~~~~

- **PySCF**: Interface to PySCF's orbital localization methods (TODO: is this true?)

.. todo::
   TODO (NAB):  correct Localizer documentation if needed and remove TODO
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41380

The factory pattern allows seamless selection between these implementations.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../advanced/interfaces>`
documentation.

Related Classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input and output orbitals
- :doc:`ScfSolver <scf_solver>`: Produces initial orbitals for localization
- :doc:`ActiveSpaceSelector <active_space>`: Often used with localized orbitals
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Can build Hamiltonians using localized orbitals
