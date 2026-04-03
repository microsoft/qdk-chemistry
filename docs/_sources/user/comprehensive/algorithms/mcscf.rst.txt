Multi-Configuration Self-Consistent Field
=========================================

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` algorithm in QDK/Chemistry performs Multi-Configurational Self-Consistent Field (:term:`MCSCF`)
calculations to optimize both molecular orbital coefficients and configuration interaction coefficients simultaneously.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes initial :doc:`Orbitals <../data/orbitals>`, a :doc:`CI calculator <mc_calculator>`,
a :doc:`HamiltonianConstructor <hamiltonian_constructor>` and the number of electrons as input and produces an optimized :doc:`Wavefunction <../data/wavefunction>` as output.
Its primary purpose is to optimize the orbitals and wavefunction for systems with strong electron correlation effects, which cannot be adequately described by single-reference methods.

Overview
--------

:term:`MCSCF` methods extend beyond both mean-field and configuration interaction approaches by simultaneously optimizing molecular orbitals and multi-configurational wavefunctions.
Unlike :doc:`Hartree-Fock <scf_solver>`, which optimizes orbitals for a single configuration, or :doc:`CI calculations <mc_calculator>`, which only optimize configuration coefficients with fixed orbitals,
:term:`MCSCF` performs a full variational optimization of both components.

As a prerequisite, an active space must be defined, typically using an :doc:`ActiveSpaceSelector <active_space>`.
The :term:`MCSCF` procedure alternates between:

- **Configuration interaction**: Solving the :term:`CI` problem in the active space with fixed orbitals
- **Orbital optimization**: Updating molecular orbital coefficients while keeping :term:`CI` coefficients fixed

Due to the relaxation of the orbitals, :term:`MCSCF` can capture both static and some dynamic correlation effects more effectively and hence results in lower energies than :term:`CI` calculations.


Running an :term:`MCSCF` calculation
------------------------------------

.. note::
   This algorithm is currently available only in the Python API.


This section demonstrates how to create, configure, and run an :term:`MCSCF` calculation.
The ``run`` method takes initial :doc:`Orbitals <../data/orbitals>`, a :doc:`HamiltonianConstructor <hamiltonian_constructor>`,
a :doc:`MultiConfigurationCalculator <mc_calculator>`, and the number of electrons as input and returns an optimized :class:`~qdk_chemistry.data.Wavefunction` object along with its associated energy.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` requires the following inputs:

Orbitals
   Initial :doc:`Orbitals <../data/orbitals>` containing orbital coefficients and active space information.

HamiltonianConstructor
   A :doc:`HamiltonianConstructor <hamiltonian_constructor>` instance that builds the Hamiltonian for the :term:`CI` step.

MultiConfigurationCalculator
   A :doc:`MultiConfigurationCalculator <mc_calculator>` for solving the :term:`CI` problem in the active space.

Number of alpha electrons
   The number of alpha (spin-up) electrons in the active space.

Number of beta electrons
   The number of beta (spin-down) electrons in the active space.


.. rubric:: Creating an :term:`MCSCF` solver

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` is created using the :doc:`factory pattern <factory_pattern>`.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

The required algorithms can be configured through their settings, which can be modified using the ``settings()`` object.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the calculation

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available settings
------------------

.. note::
   Because only one implementation of the algorithm is currently available through the PySCF plugin, only its implementation-specific settings (such as ``max_cycle_macro``) are listed in the table below.

See :doc:`Settings <settings>` for a more general treatment of settings in QDK/Chemistry.

.. note::

   Additional settings for the :term:`CI` calculation step are configured through the :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` object and settings
   for the Hamiltonian constructor are configured through the :doc:`Hamiltonian constructor <hamiltonian_constructor>` object.
   See :doc:`MultiConfigurationCalculator settings <mc_calculator>` and :doc:`HamiltonianConstructor settings <hamiltonian_constructor>` for more details.

Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` provides a unified interface for :term:`MCSCF` calculations.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

PySCF
~~~~~

**Factory name:** ``"pyscf"`` (default)

The current :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` implementation in QDK/Chemistry uses PySCF's :term:`CASSCF` framework.

Key features of the PySCF implementation:

- **Restricted orbitals**: Currently requires restricted orbitals with identical alpha/beta active and inactive spaces
- **Flexible CI solver**: Any QDK :term:`MC` calculator can be used as the :term:`CI` solver (e.g., full :term:`CI`, selected :term:`CI`)
- **Standard CASSCF**: Implements the standard :term:`CASSCF` algorithm with micro and macro iteration cycles
- **Optimized output**: Returns a :class:`~qdk_chemistry.data.Wavefunction` object containing both optimized orbital coefficients and :term:`CI` coefficients

.. note::
   The current implementation in QDK/Chemistry uses PySCF's :term:`CASSCF` framework as the underlying engine for the :term:`MCSCF` procedure.
   This implementation does not yet utilize the provided Hamiltonian constructor, and uses the integrals directly from PySCF.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``max_cycle_macro``
     - int
     - ``50``
     - Maximum number of :term:`MCSCF` macro iterations (orbital optimization cycles)

Related classes
---------------

- :doc:`Orbitals <../data/orbitals>`: Input orbitals containing orbital coefficients and active space information
- :class:`~qdk_chemistry.data.Wavefunction`: Output optimized wavefunction
- :doc:`MultiConfigurationCalculator <mc_calculator>`: :term:`CI` solver used within :term:`MCSCF` iterations
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Builds the Hamiltonian for the :term:`CI` step

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/mcscf.py>`_ script.
- :doc:`ActiveSpaceSelector <active_space>`: Helps identify important orbitals for the active space
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
