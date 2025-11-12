MCCalculator
============

The ``MCCalculator`` algorithm in QDK/Chemistry performs Multi-Configurational (MC) calculations to solve the electronic structure
problem beyond the mean-field approximation. It provides access to various Configuration Interaction (CI) methods to
account for static electron correlation effects, which are critical for accurately describing systems with
near-degenerate electronic states.

Overview
--------

Multi-Configurational methods represent the electronic wavefunction as a linear combination of many electron
configurations (Slater determinants). These methods can accurately describe systems with strong static correlation
effects where single-reference methods like Hartree-Fock are inadequate. Static correlation arises when multiple
electronic configurations contribute significantly to the wavefunction, such as in bond-breaking processes, transition
states, excited states, and open-shell systems. The ``MCCalculator`` algorithm implements various CI approaches, from full
CI (FCI) to selected CI methods that focus on the most important configurations.

Capabilities
------------

The ``MCCalculator`` in QDK/Chemistry provides:

- **Full Configuration Interaction (FCI)**: Exact solution within a given orbital space, also known as Complete Active
  Space (**CAS**) when performed within a selected active space of orbitals
- **Selected Configuration Interaction (SCI)**: Adaptive selection of important configurations

Creating an MCCalculator
------------------------

The ``MCCalculator`` is created using the :doc:`factory pattern <../advanced/factory_pattern>`:

.. todo::
   TODO (NAB):  Check MCCalculator code examples after finalizing API.
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41366

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      using namespace qdk::chemistry::algorithms;

      // Create the default MCCalculator instance (MACIS implementation)
      auto mc_calculator = MCCalculatorFactory::create();

      // Create a specific type of CI calculator
      auto selected_ci = MCCalculatorFactory::create("macis_cas");

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.algorithms import create_mc_calculator

      # Create the default MCCalculator instance (MACIS implementation)
      mc_calculator = create_mc_calculator()

      # Create a specific type of CI calculator
      selected_ci = create_mc_calculator("macis_cas")

Configuring the MC Calculation
------------------------------

The ``MCCalculator`` can be configured using the ``Settings`` object:

.. note::
   The examples below show commonly used settings. For a complete list of available settings with descriptions,
   see the `Available Settings`_ section.

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

   .. code-block:: python

      # Set the number of states to solve for (ground state + two excited states)
      mc_calculator.settings().set("num_roots", 3)

      # Set the convergence threshold for the CI iterations
      mc_calculator.settings().set("ci_residual_threshold", 1.0e-6)

      # Set the maximum number of Davidson iterations
      mc_calculator.settings().set("davidson_iterations", 200)

      # Calculate one-electron reduced density matrix
      mc_calculator.settings().set("calculate_one_rdm", True)

Running a CI Calculation
------------------------

Once configured, the CI calculation can be executed using a :doc:`Hamiltonian <../data/hamiltonian>` object as input,
which returns energy values and a :doc:`Wavefunction <../data/wavefunction>` object as output:

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

   .. code-block:: python

      # Obtain a valid Hamiltonian
      hamiltonian = Hamiltonian()
      # hamiltonian = ...

      # Run the CI calculation
      E_ci, wavefunction = mc_calculator.calculate(hamiltonian)

      # For multiple states, access the energies and wavefunctions
      energies = mc_calculator.get_energies()
      wavefunctions = mc_calculator.get_wavefunctions()

Available MC calculators
------------------------

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

Available Settings
------------------

The ``MCCalculator`` accepts a range of settings to control its behavior. These settings are divided into base settings
(common to all MC calculations) and specialized settings (specific to certain MC variants).

Base Settings
~~~~~~~~~~~~~

These settings apply to all MC calculation methods:

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
     - Convergence threshold for CI iterations
   * - ``davidson_iterations``
     - int
     - 200
     - Maximum number of Davidson iterations
   * - ``calculate_one_rdm``
     - bool
     - true
     - Whether to calculate one-electron reduced density matrix

Specialized Settings
~~~~~~~~~~~~~~~~~~~~

.. todo::
   🔧 **TODO**: Add specialized settings for each calculator type with accurate parameter names and default values

   TODO (NAB):  finish documentation
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41381

These settings apply only to specific variants of MC calculations:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 30 20

   * - Setting
     - Type
     - Default
     - Description
     - Applicable To

Implemented Interface
---------------------

QDK/Chemistry's ``MCCalculator`` provides a unified interface for Multi-Configurational calculations:
~~~~~~~~~~~~~~~~~~

- **MACIS**: QDK/Chemistry's native Many-body Adaptive Configuration Interaction Solver library
- **PySCF**: Interface to PySCF's FCI and CASCI implementations

The factory pattern allows seamless selection between these implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../advanced/interfaces>`
documentation.

MACIS Implementation
--------------------

The default ``MCCalculator`` implementation in QDK/Chemistry is based on the MACIS library (Many-body Adaptive Configuration
Interaction Solver), which provides efficient algorithms for selected CI calculations. The MACIS implementation
automatically determines electron numbers from the orbital occupations in the Hamiltonian.

Related Classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for CI calculation
- :doc:`Wavefunction <../data/wavefunction>`: Output CI wavefunction
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Produces the Hamiltonian for CI
- :doc:`DynamicalCorrelation <dynamical_correlation>`: Can add dynamical correlation to CI results
- :doc:`ActiveSpaceSelector <active_space>`: Helps identify important orbitals for the active space
