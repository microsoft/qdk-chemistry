Multi-Configuration Self-Consistent Field
=========================================

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` algorithm in QDK/Chemistry performs Multi-Configurational Self-Consistent Field (:term:`MCSCF`)
calculations to optimize both molecular orbital coefficients and configuration interaction coefficients simultaneously.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes initial :doc:`Orbitals <../data/orbitals>`, a :doc:`CI calculator <mc_calculator>`,
a :doc:`HamiltonianConstructor <hamiltonian_constructor>` and the number of electrons as input and produces an optimized :doc:`Wavefunction <../data/wavefunction>` as output.
Its primary purpose is to optimize the orbitals and wavefunction for systems with strong electron correlation effects, which cannot be adequately described by single-reference methods.

Overview
--------

MCSCF methods extend beyond both mean-field and configuration interaction approaches by simultaneously optimizing molecular orbitals and multi-configurational wavefunctions.
Unlike :doc:`Hartree-Fock <scf_solver>`, which optimizes orbitals for a single configuration, or :doc:`CI calculations <mc_calculator>`, which only optimize configuration coefficients with fixed orbitals,
MCSCF performs a full variational optimization of both components.

As prerequisite, an active space must be defined, typically using an :doc:`ActiveSpaceSelector <active_space>`.
Afterwards, the :term:`MCSCF` procedure alternates between:

- **Configuration interaction**: Solving the :term:`CI` problem in the active space with fixed orbitals
- **Orbital optimization**: Updating molecular orbital coefficients while keeping :term:`CI` coefficients fixed

Due to the relaxation of the orbitals, :term:`MCSCF` can capture both static and some dynamic correlation effects more effectively and hence, results in lower energies than :term:`CI` calculations.

Capabilities
------------

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` in QDK/Chemistry provides:

- **Simultaneous orbital and CI optimization**: Variational optimization of both molecular orbital coefficients and configuration interaction coefficients
- **Flexible CI solver integration**: Support for various :term:`CI` solvers through the factory pattern (e.g., full :term:`CI`, selected :term:`CI`)
- **Customizable Hamiltonian construction**: Compatible with different Hamiltonian constructor implementations

Creating a MultiConfigurationScf
--------------------------------

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` is created using the :doc:`factory pattern <../design/factory_pattern>`.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

Configuring the MCSCF calculation
---------------------------------

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` can be configured using the ``Settings`` object of the :class:`~qdk_chemistry.algorithms.HamiltonianConstructor`, the ``Settings`` object of
the :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator`, and its own ``Settings`` object:

.. note::
   The examples below show commonly used settings.
   For a complete list of available settings with descriptions, see the `Available Settings`_ section.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

Running an MCSCF calculation
----------------------------

Once configured, the :term:`MCSCF` calculation requires initial :doc:`orbitals <../data/orbitals>`, a :doc:`Hamiltonian constructor <hamiltonian_constructor>`,
a :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` for the :term:`CI` solver, and the number of alpha and beta electrons in the active space.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/mcscf.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available MCSCF methods
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Description
     - Typical Use Cases
   * - ``pyscf``
     - PySCF-based CASSCF implementation
     - General MCSCF calculations

Available settings
------------------

The :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` accepts a range of settings to control its behavior.
These settings control the :term:`MCSCF` orbital optimization procedure.

Base settings
~~~~~~~~~~~~~

These settings apply to all :term:`MCSCF` calculation methods:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Setting
     - Type
     - Default
     - Description
   * - ``max_cycle_macro``
     - int
     - 50
     - Maximum number of :term:`MCSCF` macro iterations (orbital optimization cycles)
   * - ``verbose``
     - int
     - 0
     - Verbosity level for output (0 = minimal, higher = more detailed)

.. note::

   Additional settings for the :term:`CI` calculation step are configured through the :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` object and settings
   for the Hamiltonian constructor are configured through the :doc:`Hamiltonian constructor <hamiltonian_constructor>` object.
   See :doc:`MultiConfigurationCalculator settings <mc_calculator>` and :doc:`HamiltonianConstructor settings <hamiltonian_constructor>` for more details.

Implemented interfaces
----------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` provides a unified interface for :term:`MCSCF` calculations:

- **PySCF**: Interface to PySCF's :term:`CASSCF` implementation, using QDK :term:`MC` calculators as :term:`FCI` solvers

The factory pattern allows seamless selection between implementations, with the most appropriate option chosen
based on the calculation requirements and available packages.

For more details on how QDK/Chemistry interfaces with external packages, see the :doc:`Interfaces <../design/interfaces>` documentation.

PySCF implementation
--------------------

The current :class:`~qdk_chemistry.algorithms.MultiConfigurationScf` implementation in QDK/Chemistry uses PySCF's :term:`CASSCF` framework.
The implementation wraps a QDK :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator` to serve as the :term:`FCI` solver within PySCF's :term:`MCSCF` procedure.

Key features of the PySCF implementation:

- **Restricted orbitals**: Currently requires restricted orbitals with identical alpha/beta active and inactive spaces
- **Flexible CI solver**: Any QDK :term:`MC` calculator can be used as the :term:`CI` solver (e.g., full :term:`CI`, selected :term:`CI`)
- **Standard CASSCF**: Implements the standard :term:`CASSCF` algorithm with micro and macro iteration cycles
- **Optimized output**: Returns a :class:`~qdk_chemistry.data.Wavefunction` object containing both optimized orbital coefficients and :term:`CI` coefficients

.. note::
   The current implementation in QDK/Chemistry uses PySCF's :term:`CASSCF` framework as the underlying engine for the :term:`MCSCF` procedure.
   This implementation does not yet utilize the provided Hamiltonian constructor, and uses the integrals directly from PySCF.

Related classes
---------------

- :doc:`Wavefunction <../data/wavefunction>`: Input orbitals and output optimized wavefunction
- :doc:`Orbitals <../data/orbitals>`: Contains orbital information and active space indices
- :doc:`MultiConfigurationCalculator <mc_calculator>`: CI solver used within MCSCF iterations
- :doc:`ActiveSpaceSelector <active_space>`: Defines the active space for MCSCF calculations
- :doc:`HamiltonianConstructor <hamiltonian_constructor>`: Builds the Hamiltonian within the MCSCF

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/mcscf.py>`_ script.
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <../design/factory_pattern>`: Understanding algorithm creation
