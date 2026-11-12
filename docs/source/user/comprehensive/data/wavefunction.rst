Wavefunction
============

The :class:`~qdk_chemistry.data.Wavefunction` class in QDK/Chemistry represents quantum mechanical wavefunctions for molecular systems.
This class provides access to wavefunction coefficients, determinants, reduced density matrices (:term:`RDM`), orbital entropies :cite:`Boguslawski2015`, and other quantum chemical properties.

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

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-slater
      :end-before: # end-cell-create-slater

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-slater
      :end-before: // end-cell-create-slater

SCI wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse multi-determinant wavefunctions for Selected Configuration Interaction methods.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-sci
      :end-before: # end-cell-create-sci

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-sci
      :end-before: // end-cell-create-sci

CAS wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

A multi-determinant wavefunction from Complete Active Space methods (:term:`CASSCF`/:term:`CASCI`).

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-cas
      :end-before: # end-cell-create-cas

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-cas
      :end-before: // end-cell-create-cas

Amplitude wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single container, :class:`~qdk_chemistry.data.AmplitudeContainer`, represents amplitude-based correlated wavefunctions such as :term:`MP2` and coupled cluster.
It stores a reference wavefunction together with T1/T2 excitation amplitudes, and records which correlated method produced them via :class:`~qdk_chemistry.data.AmplitudeType` (``MP2``, ``CCSD``, or ``Unspecified``).
It is pure storage: the amplitudes are supplied by the producing algorithm and are not expanded into a determinant/coefficient representation, so determinant-, coefficient-, and :term:`RDM`-based accessors raise.

The following example tags the amplitudes as :term:`MP2` (T1 is zero for MP2):

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-mp2
      :end-before: # end-cell-create-mp2

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-mp2
      :end-before: // end-cell-create-mp2

The same container holds coupled cluster amplitudes when tagged as ``CCSD``:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-cc
      :end-before: # end-cell-create-cc

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-cc
      :end-before: // end-cell-create-cc

MPS wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~qdk_chemistry.data.MPSContainer` stores a matrix product state (MPS), for example the result of a density matrix renormalization group (DMRG) calculation.
It holds one :class:`~qdk_chemistry.data.MPSSite` per active orbital, each an optionally symmetry-blocked rank-3 tensor with left-bond, physical, and right-bond indices, together with the order of the orbitals along the chain and an optional orthogonality center.
It is storage first: the sites are not expanded into determinants, so accessors that need the full determinant list or orbital occupations raise.
Three quantities are contracted directly from the sites: :meth:`~qdk_chemistry.data.MPSContainer.get_coefficient` gives the amplitude of a single determinant and serves as the executable definition of the convention below, ``norm`` gives the norm of the stored state, and ``overlap`` gives the overlap with another MPS that uses the same site order, or with a determinant expansion such as :class:`~qdk_chemistry.data.StateVectorContainer`.
Particle counts and RDMs are available only if they are passed to the constructor.

Fermionic coefficient convention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default four-state physical basis ``(0,u,d,2)`` denotes

.. math::

   |0\rangle = |\mathrm{vac}\rangle,\qquad
   |u\rangle = a^\dagger_\alpha |\mathrm{vac}\rangle,\qquad
   |d\rangle = a^\dagger_\beta |\mathrm{vac}\rangle,\qquad
   |2\rangle = a^\dagger_\alpha a^\dagger_\beta |\mathrm{vac}\rangle.

Contracting the sites gives coefficients in the product basis whose creation operators are written from left to right in chain-site order, alpha before beta within each site. An explicit ``physical_basis`` may reorder or restrict the local states; it does not change their phases. The binary basis ``(0,1)`` similarly denotes the vacuum and one occupied spinless mode. Producers whose local states use another operator order convert with a site-local sign on the affected physical states; for example, a doubly occupied state defined as :math:`a^\dagger_\beta a^\dagger_\alpha|\mathrm{vac}\rangle` is the negative of ``2``.

``site_to_orbital_order[s]`` identifies the active-orbital slot represented by chain site ``s``; an omitted order is the identity. These are active-space slots, not indices in the full molecular-orbital basis. The container stores the mapping without rearranging tensors or changing signs.

Determinant coefficients, as stored in :class:`~qdk_chemistry.data.StateVectorContainer`, order the creation operators with all alpha modes before all beta modes, each in ascending orbital order. :meth:`~qdk_chemistry.data.MPSContainer.get_coefficient` contracts the sites for one determinant and multiplies by the parity of the permutation that sorts the chain-ordered operators into that order. The sign is required even for the identity site order: ``|2u>`` creates :math:`a^\dagger_{0\alpha}a^\dagger_{0\beta}a^\dagger_{1\alpha}`, which needs one exchange to reach determinant order. Spinless modes follow the same rule when their site order is permuted. Binary sites are therefore spinless fermionic modes. A chain of non-fermionic qubits or spins can still be stored, but its amplitudes agree with ``get_coefficient`` only for the identity site order, where the sign is trivial.

A small reference state is

.. math::

   |\Psi\rangle = \tfrac12\left(|2u0\rangle + i|u02\rangle - |udu\rangle - i|02u\rangle\right)

in chain order. For ``site_to_orbital_order = [2,0,1]`` its nonzero determinant coefficients are listed below, with occupations written as ``alpha_0 alpha_1 alpha_2 beta_0 beta_1 beta_2``.

.. list-table:: Signed reference amplitudes
   :header-rows: 1

   * - Chain configuration
     - Determinant occupation bits
     - Coefficient
   * - ``2u0``
     - ``101001``
     - :math:`1/2`
   * - ``u02``
     - ``011010``
     - :math:`-i/2`
   * - ``udu``
     - ``011100``
     - :math:`-1/2`
   * - ``02u``
     - ``110100``
     - :math:`i/2`

Properties
~~~~~~~~~~

The :class:`~qdk_chemistry.data.Wavefunction` class provides access to various quantum chemical properties. Availability depends on the specific container type:


.. list-table:: Property availability by container type
   :header-rows: 1
   :widths: 30 15 15 15 15 15 15

   * - Property
     - Slater determinant
     - :term:`CAS`
     - :term:`SCI`
     - :term:`MP2`
     - Coupled cluster
     - MPS

   * - **Coefficients**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✓‡

   * - **Determinants**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗

   * - **Electron counts**
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓†

   * - **Orbital occupations**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
     - ✗

   * - **1-RDMs (spin-dependent)**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
     - ✓†

   * - **1-RDMs (spin-traced)**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
     - ✓†

   * - **2-RDMs (spin-dependent)**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓
     - ✓†

   * - **2-RDMs (spin-traced)**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓
     - ✓†

   * - **Orbital entropies**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓
     - ✓†

   * - **T1/T2 amplitudes**
     - ✗
     - ✗
     - ✗
     - ✓
     - ✓
     - ✗

   * - **Overlap calculations**
     - ✗
     - ✓
     - ✗
     - ✗
     - ✗
     - ✓

   * - **Norm calculations**
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
     - ✓




Legend:

- ✓ Available and implemented

- ✗ Not available (method not implemented)

- ✓* Implemented and available only if 2-RDMs were provided during construction

- ✓† Available only if the required particle counts or RDMs were provided during construction

- ✓‡ Single determinants only, through ``get_coefficient``

Accessing wavefunction data
---------------------------

The :class:`~qdk_chemistry.data.Wavefunction` class provides methods to access coefficients, determinants, and derived properties:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-data
      :end-before: # end-cell-access-data

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-data
      :end-before: // end-cell-access-data

Total spin
-----------

The :meth:`~qdk_chemistry.data.Wavefunction.compute_s_squared` method evaluates
the total-spin expectation value :math:`\langle S^2\rangle` from the real,
spin-resolved active-space one- and two-particle :term:`RDM` blocks. For a pure
spin eigenstate with spin quantum number :math:`S`, the result is
:math:`S(S+1)`.

The calculation requires alpha and beta molecular orbitals to share a common
spatial basis. Restricted closed-shell and restricted open-shell wavefunctions
meet this requirement; unrestricted orbitals are rejected because their
spin-flip contractions additionally require the alpha--beta molecular-orbital
overlap. The method also raises if the required RDM blocks are unavailable,
their symmetry-blocked extents are inconsistent, or they are complex-valued.

.. tab:: Python API

   .. code-block:: python

      s_squared = wavefunction.compute_s_squared()

.. tab:: C++ API

   .. code-block:: cpp

      double s_squared = wavefunction->compute_s_squared();

Accessing cluster amplitudes
----------------------------

For :term:`MP2` and coupled cluster wavefunctions, one can access T1 and T2 cluster amplitudes.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-amplitudes
      :end-before: # end-cell-access-amplitudes

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-amplitudes
      :end-before: // end-cell-access-amplitudes

Related classes
---------------

- :doc:`Serialization <serialization>`: Data serialization and deserialization
- :doc:`Settings <../algorithms/settings>`: Configuration settings for algorithms
- :doc:`Active space methods <../algorithms/active_space>`: Active space selection from wavefunctions
- :doc:`Structure <structure>`: Molecular structure that defines the system
- :doc:`Orbitals <orbitals>`: Orbital basis set for the wavefunction
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from wavefunction
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces :term:`SCF` wavefunctions
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Algorithm for multi-configuration wavefunctions

Further reading
---------------

- :doc:`Settings <../algorithms/settings>`: Configuration settings for algorithms
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Active space methods <../algorithms/active_space>`: Active space selection from wavefunctions
