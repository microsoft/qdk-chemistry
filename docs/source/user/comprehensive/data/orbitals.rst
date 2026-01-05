Orbitals
========

The :class:`~qdk_chemistry.data.Orbitals` class in QDK/Chemistry represents a set of molecular orbitals.
This class stores orbital coefficients, energies, and other properties necessary for quantum chemical calculations.

Overview
--------

Molecular orbitals are a fundamental concept in quantum chemistry.
They are formed through linear combinations of atomic orbitals and provide a framework for understanding chemical bonding and electronic structure.
In QDK/Chemistry, the :class:`~qdk_chemistry.data.Orbitals` class encapsulates all relevant information about these orbitals, including their coefficients, energies, and occupation numbers.

Restricted vs. unrestricted calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.data.Orbitals` class supports both restricted and unrestricted calculations:

Restricted
  Alpha and beta electrons share the same spatial orbitals (:term:`RHF`, :term:`RKS`)

Unrestricted
  Alpha and beta electrons have separate spatial orbitals (:term:`UHF`, :term:`UKS`)

For restricted calculations, the alpha and beta components are identical.
The class maintains separate alpha and beta data internally, but they reference the same underlying data for restricted cases.

Model orbitals
~~~~~~~~~~~~~~

ModelOrbitals are a simpler class in QDK/Chemistry, for model systems without any basis set information.
This class allows to fully specify model Hamiltonians and Wavefunctions.
Several properties present for the Orbitals subclass are missing for ModelOrbitals: coefficients, energies, etc. These are summarized in the properties table below.

Properties
~~~~~~~~~~

The following table summarizes the properties available for the different orbital types:

.. list-table:: Orbital Properties Availability
   :widths: 25 40 15 20
   :header-rows: 1
   :stub-columns: 1

   * - Property
     - Description
     - Orbitals
     - ModelOrbitals
   * - Coefficients
     - Matrix of orbital coefficients [:term:`AO` × :term:`MO`] for alpha and beta spin channels
     - ✓
     - ✗
   * - Energies
     - Vector of orbital energies for alpha and beta spin channels
     - ✓
     - ✗
   * - Active space indices
     - Active space indices for alpha and beta spin channels
     - ✓
     - ✓
   * - Inactive space indices
     - Inactive space indices for alpha and beta spin channels
     - ✓
     - ✓
   * - Virtual space indices
     - Virtual space indices for alpha and beta spin channels
     - ✓
     - ✓
   * - :term:`MO` overlap
     - Overlap matrices between :term:`MO`, for both spin channels
     - ✓
     - ✓*
   * - Basis Set
     - Comprehensive basis set information
     - ✓
     - ✗
   * - :term:`AO` overlap
     - Overlap matrices between :term:`AO`, for both spin channels
     - ✓
     - ✗

.. note::
   \* For ModelOrbitals, :term:`MO` overlap matrices return identity matrices since model systems assume orthonormal orbitals.

For detailed information about basis sets in QDK/Chemistry, including available basis sets, creation, manipulation, and serialization, refer to the :doc:`Basis Set documentation <basis_set>`.


Usage
-----

The :class:`~qdk_chemistry.data.Orbitals` class is typically created as the output of an :term:`SCF` calculation (:doc:`../algorithms/scf_solver`) or :doc:`orbital transformation <../algorithms/localizer>`.
It serves as input to various post-:term:`HF` methods such as :doc:`active space selection <../algorithms/active_space>` and :doc:`Hamiltonian construction <../algorithms/hamiltonian_constructor>`.

Orbital Localization
  Transform delocalized :term:`SCF` orbitals into localized representations for better chemical interpretation and more efficient correlation methods.
  See :doc:`Localizer <../algorithms/localizer>` for details.

Active Space Selection
  Automatically identify important orbitals for multi-reference calculations based on various criteria.
  See :doc:`ActiveSpaceSelector <../algorithms/active_space>` for details.

Hamiltonian Construction
  Build electronic Hamiltonians for post-:term:`HF` methods using the orbital information.
  See :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` for details.

The below example illustrates the typical access to Orbitals (via a :term:`SCF`):

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/orbitals.cpp
      :language: cpp
      :start-after: // start-cell-create
      :end-before: // end-cell-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/orbitals.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

Similar patterns are described below for ModelOrbitals.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/orbitals.cpp
      :language: cpp
      :start-after: // start-cell-model-orbitals-create
      :end-before: // end-cell-model-orbitals-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/orbitals.py
      :language: python
      :start-after: # start-cell-model-orbitals-create
      :end-before: # end-cell-model-orbitals-create

Accessing Orbital data
----------------------

The :class:`~qdk_chemistry.data.Orbitals` class provides methods to access orbital coefficients, energies, and other properties.
Following the :doc:`immutable design principle <../design/index>` used throughout QDK/Chemistry, all getter methods return const references or copies of the data.
For spin-dependent properties, methods return pairs of (alpha, beta) data.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/orbitals.cpp
      :language: cpp
      :start-after: // start-cell-access
      :end-before: // end-cell-access

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/orbitals.py
      :language: python
      :start-after: # start-cell-access
      :end-before: # end-cell-access


Orbital transformations and applications
----------------------------------------

The :class:`~qdk_chemistry.data.Orbitals` class serves as a foundation for several important quantum chemical applications and transformations:

Orbital Localization
  Transform delocalized :term:`SCF` orbitals into localized representations for better chemical interpretation and more efficient correlation methods.
  See :doc:`Localizer <../algorithms/localizer>` for details.

Active Space Selection
  Automatically identify important orbitals for multi-reference calculations based on various criteria.
  See :doc:`ActiveSpaceSelector <../algorithms/active_space>` for details.

Hamiltonian Construction
  Build electronic Hamiltonians for post-:term:`HF` methods using the orbital information.
  Both restricted and unrestricted Hamiltonians are automatically constructed based on the matching orbital type (restricted or unrestricted).
  See :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` and :doc:`Hamiltonian <hamiltonian>` (including the :ref:`Unrestricted Hamiltonians <hamiltonian:unrestricted hamiltonians>` section) for details.


Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/orbitals.cpp>`_ and `Python <../../../_static/examples/python/orbitals.py>`_ scripts.
- :doc:`Serialization <serialization>`: Data serialization and deserialization
- :doc:`Settings <../algorithms/settings>`: Configuration settings for algorithms
- :doc:`Structure <structure>`: Molecular structure representation
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from orbitals
- :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`: Algorithm that builds Hamiltonians from orbitals
- :doc:`ActiveSpaceSelector <../algorithms/active_space>`: Algorithm for selecting active spaces from orbitals
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces orbitals
- :doc:`Localizer <../algorithms/localizer>`: Algorithms for orbital transformations
