Dynamical correlation calculations
==================================

Dynamic correlation calculations in QDK/Chemistry are post-Hartree-Fock methods that build upon the wavefunction with one or several reference
determinants to account for electron correlation effects. These methods include Møller-Plesset perturbation theory (MP2) and Coupled Cluster (CC) theory.

Overview
--------

Dynamical-correlation calculators inherit from the ``DynamicalCorrelationCalculator`` base class and operate on an
:doc:`Ansatz <../data/ansatz>` object that contains the reference wavefunction and a :doc:`Hamiltonian <../data/hamiltonian>`.
These methods systematically improve upon the mean-field approximation by including electron-electron correlation effects.

Available calculators
---------------------

Currently, the following implementations are available:

- **MP2 Calculator** (``qdk_mp2_calculator``): Second-order Møller-Plesset perturbation theory
- **PySCF Coupled Cluster** (``pyscf_coupled_cluster``): Coupled cluster implementations via PySCF integration

MP2 Calculator
--------------

The MP2 (second-order Møller-Plesset perturbation theory) calculator computes the lowest-order correlation energy
correction beyond Hartree-Fock theory.

Here's a complete example showing how to calculate MP2 corrections to a HF wavefunction:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/dynamical_correlation.cpp
      :language: cpp
      :start-after: // start-cell-mp2-example
      :end-before: // end-cell-mp2-example

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/dynamical_correlation.py
      :language: python
      :start-after: # start-cell-mp2-example
      :end-before: # end-cell-mp2-example


Related topics
--------------

- :doc:`SCF Solver <scf_solver>`: Generate reference wavefunctions
- :doc:`Hamiltonian Constructor <hamiltonian_constructor>`: Build Hamiltonians from orbitals
- :doc:`../data/hamiltonian`: Hamiltonian data structure, including unrestricted Hamiltonians
- :doc:`../data/orbitals`: Orbital data structure, including unrestricted orbitals
- :doc:`../data/ansatz`: Ansatz combining reference wavefunction and Hamiltonian
- :doc:`../data/wavefunction`: Wavefunction container for correlation methods

.. note::
   For additional examples and validation tests, refer to the test suite in ``python/tests/test_mp2.py``
   and ``python/tests/test_pyscf_plugin.py``.
