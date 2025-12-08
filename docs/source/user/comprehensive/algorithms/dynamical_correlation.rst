Dynamical correlation calculations
==================================

The :class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator` algorithm in QDK/Chemistry performs post-Hartree-Fock calculations to account for dynamical electron correlation effects.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes an :doc:`Ansatz <../data/ansatz>` object (containing a reference wavefunction and Hamiltonian) as input and returns correlation energies.
These methods include Møller-Plesset perturbation theory (MP2) and Coupled Cluster (CC) theory.

Overview
--------

Dynamical correlation arises from instantaneous electron-electron interactions that are not captured by mean-field methods like Hartree-Fock.
The :class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator` systematically improves upon the mean-field approximation by including these correlation effects.

The ``run`` method returns:

- **Correlation energy**: The energy correction beyond the reference method
- **Total energy**: Reference energy plus correlation correction

Using the DynamicalCorrelationCalculator
----------------------------------------

This section demonstrates how to create, configure, and run a dynamical correlation calculation.
The ``run`` method returns the total energy (reference plus correlation) and an updated wavefunction.

**Creating a calculator:**

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/dynamical_correlation.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/dynamical_correlation.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

**Configuring settings:**

Settings can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/dynamical_correlation.cpp
      :language: cpp
      :start-after: // start-cell-configure
      :end-before: // end-cell-configure

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/dynamical_correlation.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

**Running the calculation:**

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/dynamical_correlation.cpp
      :language: cpp
      :start-after: // start-cell-run
      :end-before: // end-cell-run

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/dynamical_correlation.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator` provides a unified interface for post-Hartree-Fock correlation methods.
You can discover available implementations programmatically:

.. tab:: C++ API

   .. code-block:: cpp

      auto names = DynamicalCorrelationCalculatorFactory::available();
      for (const auto& name : names) {
          std::cout << name << std::endl;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import registry
      print(registry.available("dynamical_correlation_calculator"))
      # ['pyscf_coupled_cluster', 'qdk_mp2_calculator']

QDK MP2 Calculator
~~~~~~~~~~~~~~~~~~

**Factory name:** ``"qdk_mp2_calculator"``

Native QDK/Chemistry implementation of second-order Møller-Plesset perturbation theory (MP2). Computes the lowest-order correlation energy correction beyond Hartree-Fock theory.

This implementation has no configurable settings.

PySCF Coupled Cluster
~~~~~~~~~~~~~~~~~~~~~

**Factory name:** ``"pyscf_coupled_cluster"``

Coupled cluster implementations via PySCF integration, providing high-accuracy correlation methods including CCSD and CCSD(T).

**Settings:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``conv_tol``
     - float
     - ``1e-7``
     - Energy convergence tolerance
   * - ``conv_tol_normt``
     - float
     - ``1e-5``
     - Amplitude convergence tolerance
   * - ``max_cycle``
     - int
     - ``50``
     - Maximum number of iterations
   * - ``diis_space``
     - int
     - ``6``
     - DIIS space size
   * - ``diis_start_cycle``
     - int
     - ``0``
     - Cycle to start DIIS
   * - ``direct``
     - bool
     - ``False``
     - Use direct (AO-based) algorithm
   * - ``async_io``
     - bool
     - ``True``
     - Use asynchronous I/O
   * - ``incore_complete``
     - bool
     - ``True``
     - Store all integrals in memory
   * - ``store_amplitudes``
     - bool
     - ``False``
     - Store CC amplitudes


Related classes
---------------

- :doc:`Ansatz <../data/ansatz>`: Input combining reference wavefunction and Hamiltonian
- :doc:`Hamiltonian <../data/hamiltonian>`: Hamiltonian data structure, including unrestricted Hamiltonians
- :doc:`Orbitals <../data/orbitals>`: Orbital data structure, including unrestricted orbitals
- :doc:`Wavefunction <../data/wavefunction>`: Wavefunction container for correlation methods

Further reading
---------------

- :doc:`SCF Solver <scf_solver>`: Generate reference wavefunctions
- :doc:`Hamiltonian Constructor <hamiltonian_constructor>`: Build Hamiltonians from orbitals
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation

.. note::
   For additional examples and validation tests, refer to the test suite in ``python/tests/test_mp2.py``
   and ``python/tests/test_pyscf_plugin.py``.
