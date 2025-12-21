Multi-configuration calculations
================================

The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` algorithm in QDK/Chemistry performs Multi-Configurational (MC) calculations to solve the electronic structure problem beyond the mean-field approximation.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Hamiltonian <../data/hamiltonian>` instance as input and produces a :class:`~qdk_chemistry.data.Wavefunction` instance as output.
It provides access to various Configuration Interaction (CI) methods to account for static electron correlation effects, which are critical for accurately describing systems with near-degenerate electronic states.

Overview
--------

:term:`MC` methods represent the electronic wavefunction as a linear combination of many electron configurations (Slater determinants).
These methods can accurately describe systems with strong static correlation effects where single-reference methods like Hartree-Fock are inadequate.
Static correlation arises when multiple electronic configurations contribute significantly to the wavefunction, such as in bond-breaking processes, transition states, excited states, and open-shell systems.
The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` algorithm implements various :term:`CI` approaches, from full CI (FCI) to selected :term:`CI` methods that focus on the most important configurations.

:term:`MC` calculations capture static correlation but typically require additional methods for dynamic correlation.
See :doc:`DynamicalCorrelationCalculator <dynamical_correlation>` or :doc:`MultiConfigurationScf <mcscf>` for complementary approaches.

Using the MultiConfigurationCalculator
--------------------------------------

This section demonstrates how to create, configure, and run a multi-configuration calculation.
The ``run`` method takes a :doc:`Hamiltonian <../data/hamiltonian>` object as input and returns a :class:`~qdk_chemistry.data.Wavefunction` object along with its associated energy.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` requires the following inputs:

**Hamiltonian**
   A :doc:`Hamiltonian <../data/hamiltonian>` instance that defines the electronic structure problem.

**Number of alpha electrons**
   The number of alpha (spin-up) electrons in the active space.

**Number of beta electrons**
   The number of beta (spin-down) electrons in the active space.


**Creating an MC calculator:**

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

**Configuring settings:**

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

**Running the calculation:**

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
     - Convergence threshold for CI iterations
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

**Factory name:** ``"macis_cas"`` (default)

The MACIS (Many-body Adaptive Configuration Interaction Solver) CAS implementation provides exact Full Configuration Interaction within the active space.

This implementation uses only the common settings described above.

.. _macis-asci:


MACIS ASCI
~~~~~~~~~~

**Factory name:** ``"macis_asci"``

The MACIS ASCI (Adaptive Sampling Configuration Interaction) provides selected CI for larger active spaces where FCI is computationally prohibitive.

**Capabilities:**

- Adaptive Sampling Configuration Interaction (ASCI)
- One- and two-electron reduced density matrices

.. _asci-algorithm:

ASCI Algorithm
------------------

The Adaptive Sampling Configuration Interaction (ASCI) algorithm :cite:`Tubman2016,Tubman2020` is a selected configuration interaction method that enables efficient treatment of large active spaces by iteratively identifying and including only the most important determinants. QDK/Chemistry integrates the high-performance, parallel implementation of ASCI in the MACIS library :cite:`Williams-Young2023`.

ASCI works by growing the determinant space adaptively: at each iteration, it samples the space of possible determinants and selects those with the largest contributions to the wavefunction. This approach achieves near-CASCI accuracy at a fraction of the computational cost, making it possible to treat active spaces that are intractable for conventional CASCI.

ASCI is especially useful for generating approximate wavefunctions and RDMs for use in automated active space selection protocols (such as AutoCAS), as it provides a good balance between computational cost and accuracy. For best practices, see the :ref:`AutoCAS Algorithm <autocas-algorithm-details>` section in the active space selector documentation.

**Settings:**

In addition to the common settings, MACIS ASCI supports the following implementation-specific settings:

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
     - Maximum number of trial determinants
   * - ``ntdets_min``
     - int
     - ``100``
     - Minimum number of trial determinants
   * - ``ncdets_max``
     - int
     - ``100``
     - Maximum number of connected determinants
   * - ``grow_factor``
     - float
     - ``8.0``
     - Factor for growing determinant space
   * - ``max_refine_iter``
     - int
     - ``6``
     - Maximum refinement iterations
   * - ``refine_energy_tol``
     - float
     - ``1e-6``
     - Energy tolerance for refinement convergence


Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for CI calculation
- :class:`~qdk_chemistry.data.Wavefunction`: Output CI wavefunction

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/mc_calculator.py>`_ or `C++ <../../../_static/examples/cpp/mc_calculator.cpp>`_ code.
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Produces the Hamiltonian for CI
- :doc:`ActiveSpaceSelector <active_space>`: Helps identify important orbitals for the active space
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
