Wavefunction
============

The :class:`~qdk_chemistry.data.Wavefunction` class in QDK/Chemistry represents quantum mechanical wavefunctions for molecular systems.
This class provides access to wavefunction coefficients, determinants, reduced density matrices (RDMs), orbital entropies :cite:`Boguslawski2015`, and other quantum chemical properties.

Overview
--------

A wavefunction in quantum chemistry describes the quantum state of a molecular system.
In QDK/Chemistry, the :class:`~qdk_chemistry.data.Wavefunction` class encapsulates various wavefunction types, from simple single-determinant Hartree-Fock wavefunctions to complex multi-reference wavefunctions.

The class uses a container-based design where different wavefunction types (Slater determinants, configuration interaction, coupled cluster, etc.) are implemented as specialized container classes, while the main :class:`~qdk_chemistry.data.Wavefunction` class provides a unified interface.

Mathematical representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wavefunctions are represented as linear combinations of determinants:

.. math::

   |\Psi\rangle = \sum_I c_I |\Phi_I\rangle

where :math:`c_I` are expansion coefficients and :math:`|\Phi_I\rangle` are Slater determinants.

For post-Hartree-Fock methods like coupled cluster, the wavefunction is expressed in terms of cluster operators:

.. math::

   |\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle

where :math:`\hat{T} = \hat{T}_1 + \hat{T}_2 + ...` is the cluster operator and :math:`|\Phi_0\rangle` is the reference determinant.


Container types
---------------

QDK/Chemistry supports different wavefunction container types for various quantum chemistry methods:

Slater determinant container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-determinant wavefunctions (e.g., from Hartree-Fock calculations).

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-slater
      :end-before: // end-cell-create-slater

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-slater
      :end-before: # end-cell-create-slater

SCI wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse multi-determinant wavefunctions for Selected Configuration Interaction methods.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-sci
      :end-before: // end-cell-create-sci

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-sci
      :end-before: # end-cell-create-sci

CAS wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

A multi-determinant wavefunction from Complete Active Space methods (CASSCF/CASCI).

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-cas
      :end-before: // end-cell-create-cas

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-cas
      :end-before: # end-cell-create-cas

MP2 wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

A wavefunction container for second-order Møller-Plesset perturbation theory (MP2), which stores a reference wavefunction and Hamiltonian.
From these, T1 and T2 amplitudes can be computed on demand.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-mp2
      :end-before: // end-cell-create-mp2

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-mp2
      :end-before: # end-cell-create-mp2

Coupled cluster wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A coupled cluster wavefunction container that stores T1 and T2 cluster amplitudes along with a reference wavefunction.
The container supports reduced density matrices (RDMs), which are available if they are provided at construction or computed and stored; otherwise, RDM-related operations are not available.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-cc
      :end-before: // end-cell-create-cc

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-cc
      :end-before: # end-cell-create-cc

Properties
~~~~~~~~~~

The :class:`~qdk_chemistry.data.Wavefunction` class provides access to various quantum chemical properties. Availability depends on the specific container type:

.. list-table:: Property availability by container type
   :header-rows: 1
   :widths: 30 15 15 15 15 15

   * - Property
     - Slater determinant
     - CAS
     - SCI
     - MP2
     - Coupled cluster
   * - **Coefficients**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
   * - **Determinants**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
   * - **Electron counts**
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - **Orbital occupations**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
   * - **1-RDMs (spin-dependent)**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
   * - **1-RDMs (spin-traced)**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
   * - **2-RDMs (spin-dependent)**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓
   * - **2-RDMs (spin-traced)**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓
   * - **Orbital entropies**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓
   * - **T1/T2 amplitudes**
     - ✗
     - ✗
     - ✗
     - ✓
     - ✓
   * - **Overlap calculations**
     - ✗
     - ✓
     - ✗
     - ✗
     - ✗
   * - **Norm calculations**
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗

Legend:
- ✓ Available and implemented
- ✗ Not available (method not implemented)
- ✓* Implemented and available only if 2-RDMs were provided during construction

Accessing wavefunction data
---------------------------

The :class:`~qdk_chemistry.data.Wavefunction` class provides methods to access coefficients, determinants, and derived properties:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-data
      :end-before: // end-cell-access-data

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-data
      :end-before: # end-cell-access-data

Accessing cluster amplitudes
----------------------------

For MP2 and coupled cluster wavefunctions, one can access T1 and T2 cluster amplitudes:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-amplitudes
      :end-before: // end-cell-access-amplitudes

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-amplitudes
      :end-before: # end-cell-access-amplitudes

Related classes
---------------

- :doc:`Structure <structure>`: Molecular structure that defines the system
- :doc:`Orbitals <orbitals>`: Orbital basis set for the wavefunction
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from wavefunction
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces SCF wavefunctions
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Algorithm for multi-configuration wavefunctions

Further reading
---------------

- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
- :doc:`DynamicalCorrelationCalculator <../algorithms/dynamical_correlation>`: Algorithm for dynamical correlation calculators (includes coupled cluster and MP2)
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Active space methods <../algorithms/active_space>`: Active space selection from wavefunctions
