Multi-configuration calculations
================================

The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` algorithm in QDK/Chemistry performs Multi-Configurational (:term:`MC`) calculations to solve the electronic structure problem beyond the mean-field approximation.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Hamiltonian <../data/hamiltonian>` instance as input and produces a :class:`~qdk_chemistry.data.Wavefunction` instance as output.
It provides access to various Configuration Interaction (:term:`CI`) methods to account for static electron correlation effects, which are critical for accurately describing systems with near-degenerate electronic states.

Overview
--------

:term:`MC` methods represent the electronic wavefunction as a linear combination of many electron configurations (Slater determinants).
These methods can accurately describe systems with strong static correlation effects where single-reference methods like Hartree-Fock are inadequate.
Static correlation arises when multiple electronic configurations contribute significantly to the wavefunction, such as in bond-breaking processes, transition states, excited states, and open-shell systems.
The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` algorithm implements various :term:`CI` approaches, from full :term:`CI` (:term:`FCI`) to selected :term:`CI` methods that focus on the most important configurations.


Using the MultiConfigurationCalculator
--------------------------------------

This section demonstrates how to create, configure, and run a multi-configuration calculation.
The ``run`` method takes a :doc:`Hamiltonian <../data/hamiltonian>` object as input and returns a :class:`~qdk_chemistry.data.Wavefunction` object along with its associated energy.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` requires the following inputs:

Hamiltonian
   A :doc:`Hamiltonian <../data/hamiltonian>` instance that defines the electronic structure problem.

Number of alpha electrons
   The number of alpha (spin-up) electrons in the active space.

Number of beta electrons
   The number of beta (spin-down) electrons in the active space.


.. rubric:: Creating an MC calculator

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/mc_calculator.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mc_calculator.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/mc_calculator.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mc_calculator.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/mc_calculator.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mc_calculator.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` accepts a range of settings to control its behavior.
All implementations share a common base set of settings from ``MultiConfigurationSettings``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``ci_residual_tolerance``
     - float
     - ``1e-6``
     - Convergence threshold for :term:`CI` Davidson solver
   * - ``davidson_iterations``
     - int
     - ``200``
     - Maximum number of Davidson iterations
   * - ``calculate_one_rdm``
     - bool
     - ``False``
     - Calculate one-electron reduced density matrix
   * - ``calculate_two_rdm``
     - bool
     - ``False``
     - Calculate two-electron reduced density matrix

See :doc:`Settings <settings>` for a more general treatment of settings in QDK/Chemistry.

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` provides a unified interface for multi-configurational calculations.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/mc_calculator.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mc_calculator.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

MACIS CAS
~~~~~~~~~

.. rubric:: Factory name: ``"macis_cas"`` (default)

The :term:`MACIS` (Many-body Adaptive Configuration Interaction Solver) :term:`CAS` implementation provides a reference solver to compute the exact energy within the active space. This module is very memory and compute intensive, and is thus suitable only for small active spaces.

This implementation uses only the common settings described above.

.. _macis-asci:


MACIS ASCI
~~~~~~~~~~

.. rubric:: Factory name: ``"macis_asci"``

The :term:`MACIS` :term:`ASCI` (Adaptive Sampling Configuration Interaction) implementation provides an efficient selected :term:`CI` solver that can handle larger active spaces by adaptively selecting the most important configurations. This method balances accuracy and computational cost, making it suitable for medium-sized active spaces.

.. _asci-algorithm:

ASCI Algorithm
^^^^^^^^^^^^^^^

The Adaptive Sampling Configuration Interaction (:term:`ASCI`) algorithm :cite:`Tubman2016,Tubman2020` is a selected configuration interaction method that enables efficient treatment of large active spaces by iteratively identifying and including only the most important determinants. QDK/Chemistry integrates the high-performance, parallel implementation of :term:`ASCI` in the :term:`MACIS` library :cite:`Williams-Young2023`.

The :term:`ASCI` works by growing the determinant space adaptively: at each iteration, it samples the space of possible determinants and selects those with the largest contributions to the wavefunction. This approach achieves near-:term:`CASCI` accuracy at a fraction of the computational cost, making it possible to treat active spaces that are intractable for conventional :term:`CASCI`.

:term:`ASCI` is especially useful for generating approximate wavefunctions and RDMs for use in automated active space selection protocols (such as AutoCAS), as it provides a good balance between computational cost and accuracy. For best practices, see the :ref:`AutoCAS Algorithm <autocas-algorithm-details>` section in the active space selector documentation.

The :term:`ASCI` algorithm proceeds as a two-phase optimization:

1. **Growth Phase**: The growth phase focuses on rapidly expanding the determinant space to capture the most important configurations. Starting from an initial set of determinants (often just the Hartree-Fock determinant), :term:`ASCI` generates new candidate determinants by estimating their importance to the overall wavefunction through perturbation theory. :term:`ASCI` then ranks their contributions to the wavefunction and selects the most significant ones to add to the determinant space for the subsequent iterations. The Hamiltonian is then projected into this expanded space and diagonalized to produce an improved wavefunction. This process is repeated with a iteratively larger determinant space until a target number of determinants (``ntdets_max``) is reached. The rate at which the determinant space grows is controlled by the ``grow_factor`` setting, which determines how many new determinants are added at each iteration. However, if the search algorithm fails to find enough important determinants, the growth factor is reduced by the ``growth_backoff_rate`` to ensure stability. Conversely, if the search is successful, the growth factor is increased by the ``growth_recovery_rate`` to accelerate convergence in subsequent iterations.

2. **Refinement Phase**: Once the determinant space has reached the target size, the refinement phase begins. In this phase, :term:`ASCI` focuses on fine-tuning the wavefunction by iteratively improving the selection of determinants within the fixed-size space. The algorithm evaluates the contributions of each determinant to the wavefunction and removes those that contribute least, replacing them with new candidates generated through perturbation theory. This selective pruning and replacement process continues until convergence is achieved, as determined by the ``refine_energy_tol`` setting or until the maximum number of refinement iterations (``max_refine_iter``) is reached.

.. rubric:: The ASCI Search Algorithm

In both the growth and refinement phases, the :term:`ASCI` search algorithm is performed to update the current wavefunction.
The key realization of :term:`ASCI` is that the search can be drastically accelerated by only searching for determinants that are connected via the Hamiltonian from a small set of "core" determinants rather than the full wavefunction at any particular iteration.
This module provides several ways to control the size of this core set, including a maximum number of core determinants (``ncdets_max``) as well as allowing the core space to update dynamically as the wavefunction grows by specifying that a fixed percentage of the current wavefunction determinants be included in the core set (``core_selection_threshold``).
The method for selecting the core determinants is controlled by the ``core_selection_strategy`` setting.


.. rubric:: Settings

In addition to the common settings, :term:`MACIS` :term:`ASCI` supports the following implementation-specific settings:


.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description

   * - ``ntdets_max``
     - int
     - ``100000``
     - Maximum number of trial determinants in the variational space

   * - ``ntdets_min``
     - int
     - ``100``
     - Minimum number of trial determinants required

   * - ``core_selection_strategy``
     - str
     - ``"percentage"``
     - Strategy for selecting core determinants ("fixed" or "percentage")

   * - ``core_selection_threshold``
     - float
     - ``0.95``
     - Cumulative weight threshold for core selection (if using percentage strategy)

   * - ``ncdets_max``
     - int
     - ``100``
     - Maximum number of core determinants (if using fixed strategy)

   * - ``grow_factor``
     - float
     - ``8.0``
     - Factor for growing determinant space

   * - ``min_grow_factor``
     - float
     - ``1.01``
     - Minimum allowed growth factor

   * - ``growth_backoff_rate``
     - float
     - ``0.5``
     - Rate to reduce grow_factor on failure

   * - ``growth_recovery_rate``
     - float
     - ``1.1``
     - Rate to restore grow_factor on success

   * - ``max_refine_iter``
     - int
     - ``6``
     - Maximum number of refinement iterations

   * - ``refine_energy_tol``
     - float
     - ``1e-6``
     - Energy tolerance for refinement convergence


Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for :term:`CI` calculation
- :class:`~qdk_chemistry.data.Wavefunction`: Output :term:`CI` wavefunction

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/mc_calculator.py>`_ or `C++ <../../../_static/examples/cpp/mc_calculator.cpp>`_ code.
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Produces the Hamiltonian for :term:`CI`
- :doc:`ActiveSpaceSelector <active_space>`: Helps identify important orbitals for the active space
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
