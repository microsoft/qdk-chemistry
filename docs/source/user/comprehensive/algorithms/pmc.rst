Projected Multi-Configuration calculations
==========================================

The :class:`~qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator` algorithm in QDK/Chemistry performs Projected Multi-Configuration (:term:`PMC`) calculations to solve the electronic structure problem for a specified set of electronic configurations.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :doc:`Hamiltonian <../data/hamiltonian>` instance and a set of configurations as input and produces a :class:`~qdk_chemistry.data.Wavefunction` instance as output.
This calculator provides a flexible interface for projecting the Hamiltonian onto a user-defined determinant space, enabling integration with external determinant selection methods.

Overview
--------

The :class:`~qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator` algorithm projects the Hamiltonian onto a user-specified space of configurations (Slater determinants) and solves the resulting eigenvalue problem to obtain the ground state energy and wavefunction.
This contrasts with :doc:`MultiConfigurationCalculator <mc_calculator>`, where the solver determines which configurations to include.

.. note::
   In contrast to the :doc:`MultiConfigurationCalculator <mc_calculator>`, where the spin and number of particles are explicitly defined via the number of alpha and beta particles,
   the :class:`~qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator` derives these symmetry properties from the provided configurations.
   Hence, all configurations must share the same number of alpha and beta electrons.

Using the ProjectedMultiConfigurationCalculator
-----------------------------------------------

This section demonstrates how to create, configure, and run a :term:`PMC` calculation.
The ``run`` method takes a :doc:`Hamiltonian <../data/hamiltonian>` object and a list of configurations as input, and returns a :class:`~qdk_chemistry.data.Wavefunction` object along with its associated energy.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator` requires the following inputs:

Hamiltonian
   A :doc:`Hamiltonian <../data/hamiltonian>` instance that defines the electronic structure problem.

Configurations
   A collection of :class:`~qdk_chemistry.data.Configuration` objects specifying the determinants to include in the calculation. Each configuration defines the occupation of orbitals in the active space.


.. rubric:: Creating a PMC calculator

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pmc.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pmc.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pmc.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pmc.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pmc.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pmc.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator` accepts a range of settings to control its behavior.
All implementations share a common base set of settings from ``ProjectedMultiConfigurationSettings``:

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

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator` provides a unified interface for projected multi-configurational calculations.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/pmc.cpp
      :language: cpp
      :start-after: // start-cell-list-implementations
      :end-before: // end-cell-list-implementations

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/pmc.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

MACIS PMC
~~~~~~~~~

.. rubric:: Factory name: ``"macis_pmc"`` (default)

The :term:`MACIS` (Many-body Adaptive Configuration Interaction Solver) :term:`PMC` implementation provides a high-performance solver for projecting the Hamiltonian onto a specified set of configurations.
This implementation leverages the same efficient parallel algorithms used in :doc:`MACIS ASCI <mc_calculator>` but applies them to a fixed, user-provided determinant space.

.. rubric:: Settings

In addition to the common settings, :term:`MACIS` :term:`PMC` supports the following implementation-specific settings:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description

   * - ``iterative_solver_dimension_cutoff``
     - int
     - ``100``
     - Matrix size cutoff for using iterative eigensolver. If the number of determinants is below this value, dense diagonalization is used instead

   * - ``H_thresh``
     - float
     - ``1e-16``
     - Hamiltonian matrix entries threshold for dense diagonalization

   * - ``h_el_tol``
     - float
     - ``1e-8``
     - Electron interaction tolerance for Hamiltonian-wavefunction products in iterative solver

   * - ``davidson_res_tol``
     - float
     - ``1e-8``
     - Residual tolerance for Davidson solver convergence

   * - ``davidson_max_m``
     - int
     - ``200``
     - Maximum allowed subspace size for Davidson solver


Related classes
---------------

- :doc:`Hamiltonian <../data/hamiltonian>`: Input Hamiltonian for :term:`PMC` calculation
- :class:`~qdk_chemistry.data.Configuration`: Specifies electronic configurations/determinants
- :class:`~qdk_chemistry.data.Wavefunction`: Output :term:`PMC` wavefunction

Further reading
---------------

- The above examples can be downloaded as complete `Python <../../../_static/examples/python/pmc.py>`_ or `C++ <../../../_static/examples/cpp/pmc.cpp>`_ code.
- :doc:`MultiConfigurationCalculator <mc_calculator>`: Adaptive configuration selection
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Produces the Hamiltonian for :term:`PMC`
- :doc:`ActiveSpaceSelector <active_space>`: Helps identify important orbitals for the active space
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
