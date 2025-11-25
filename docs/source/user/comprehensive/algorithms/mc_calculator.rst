Multi-configuration calculations
================================

The ``MultiConfigurationCalculator`` algorithm in QDK/Chemistry performs Multi-Configurational (MC) calculations to solve the electronic structure problem beyond the mean-field approximation.
It provides access to various Configuration Interaction (CI) methods to account for static electron correlation effects, which are critical for accurately describing systems with near-degenerate electronic states.

Overview
--------

:term:`MC` methods represent the electronic wavefunction as a linear combination of many electron configurations (Slater determinants).
These methods can accurately describe systems with strong static correlation effects where single-reference methods like Hartree-Fock are inadequate.
Static correlation arises when multiple electronic configurations contribute significantly to the wavefunction, such as in bond-breaking processes, transition states, excited states, and open-shell systems.
The ``MultiConfigurationCalculator`` algorithm implements various :term:`CI` approaches, from full CI (FCI) to selected :term:`CI` methods that focus on the most important configurations.

Capabilities
------------

The ``MultiConfigurationCalculator`` in QDK/Chemistry provides:

- **Full Configuration Interaction (FCI)**: Exact solution within a given orbital space, also known as Complete Active
  Space (**CAS**) when performed within a selected active space of orbitals
- **Selected Configuration Interaction (SCI)**: Adaptive selection of important configurations

Creating an MultiConfigurationCalculator
----------------------------------------

The ``MultiConfigurationCalculator`` is created using the :doc:`factory pattern <../design/factory_pattern>`.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::algorithms;

      // Create the default MultiConfigurationCalculator instance (MACIS implementation)
      auto mc_calculator = MultiConfigurationCalculatorFactory::create();

      // Create a specific type of CI calculator
      auto selected_ci = MultiConfigurationCalculatorFactory::create("macis_cas");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/mc_calculator.py
      :language: python
      :lines: 3-

Configuring the :term:`MC` calculation
--------------------------------------

The ``MultiConfigurationCalculator`` can be configured using the ``Settings`` object:

.. note::
   The examples below show commonly used settings.
   For a complete list of available settings with descriptions, see the `Available Settings`_ section.

.. tab:: C++ API

   .. code-block:: cpp

      // Set the number of states to solve for (ground state + two excited states)
      mc_calculator->settings().set("num_roots", 3);

      // Set the convergence threshold for the CI iterations
      mc_calculator->settings().set("ci_residual_threshold", 1.0e-6);

      // Set the maximum number of Davidson iterations
      mc_calculator->settings().set("davidson_iterations", 200);

      // Calculate one-electron reduced density matrix
      mc_calculator->settings().set("calculate_one_rdm", true);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 4-12

Running a :term:`CI` calculation
---------------------------------

Once configured, the :term:`CI` calculation can be executed using a :doc:`Hamiltonian <../data/hamiltonian>` object as input, which returns energy values and a :class:`qdk_chemistry.data.Wavefunction` object as output:

.. tab:: C++ API

   .. code-block:: cpp

      // Obtain a valid Hamiltonian
      Hamiltonian hamiltonian;
      /* hamiltonian = ... */

      // Run the CI calculation
      auto [E_ci, wavefunction] = mc_calculator->calculate(hamiltonian);

      // For multiple states, access the energies and wavefunctions
      auto energies = mc_calculator->get_energies();
      auto wavefunctions = mc_calculator->get_wavefunctions();

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see the test suite.

   .. literalinclude:: ../../../../examples/factory_pattern.py
      :language: python
      :lines: 1-9

Available :term:`MC` calculators
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Description
     - Typical Use Cases
   * - ``macis_cas``
     - Full Configuration Interaction
     - Small active spaces, benchmark calculations
   * - ``macis_asci``
     - Selected Configuration Interaction
     - Larger active spaces, efficient correlation treatment

Available settings
------------------

The ``MultiConfigurationCalculator`` accepts a range of settings to control its behavior.
These settings are divided into base settings (common to all :term:`MC` calculations) and specialized settings (specific to certain :term:`MC` variants).

Base settings
~~~~~~~~~~~~~

These settings apply to all :term:`MC` calculation methods:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``num_roots``
     - int
     - 3
     - Number of states to solve for (ground state + excited states)
   * - ``ci_residual_threshold``
     - float
     - 1.0e-6
     - Convergence threshold for :term:`CI` iterations
   * - ``davidson_iterations``
     - int
     - 200
     - Maximum number of Davidson iterations
   * - ``calculate_one_rdm``
     - bool
     - true
     - Whether to calculate one-electron reduced density matrix

.. Specialized settings
.. ~~~~~~~~~~~~~~~~~~~~

.. TODO:  The specialized :term:`MC` calculator settings documentation is currently under construction.

.. These settings apply only to specific variants of :term:`MC` calculations.

Implemented interface
---------------------

QDK/Chemistry's ``MultiConfigurationCalculator`` provides a unified interface for :term:`MC` calculations.

- **MACIS**: QDK/Chemistry's native Many-body Adaptive Configuration Interaction Solver library
- **PySCF**: Interface to PySCF's :term:`FCI` and :term:`CAS`-:term:`CI` implementations

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

:term:`MACIS` implementation
-----------------------------

The default ``MultiConfigurationCalculator`` implementation in QDK/Chemistry is based on the :term:`MACIS` library, which provides efficient algorithms for :term:`SCI` calculations.
The :term:`MACIS` implementation automatically determines electron numbers from the orbital occupations in the Hamiltonian.

Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for CI calculation
- :class:`qdk_chemistry.data.Wavefunction`: Output CI wavefunction
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Produces the Hamiltonian for CI
- :doc:`ActiveSpaceSelector <active_space>`: Helps identify important orbitals for the active space
