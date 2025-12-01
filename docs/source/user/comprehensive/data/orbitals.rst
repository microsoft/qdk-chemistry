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

- **Restricted**: Alpha and beta electrons share the same spatial orbitals (:term:`RHF`, :term:`RKS`)
- **Unrestricted**: Alpha and beta electrons have separate spatial orbitals (:term:`UHF`, :term:`UKS`)

For restricted calculations, the alpha and beta components are identical. The class maintains separate alpha and beta
data internally, but they reference the same underlying data for restricted cases.

Properties
----------

- **Coefficients**: Matrix of orbital coefficients [:term:`AO` × :term:`MO`] for alpha and beta spin channels
- **Energies**: Vector of orbital energies for alpha and beta spin channels
- **Occupations**: Vector of occupation numbers for alpha and beta spin channels
- **AO Overlap**: Atomic orbital overlap matrix [:term:`AO` × :term:`AO`]
- **Basis Set**: Comprehensive basis set information

.. note::
   For detailed information about basis sets in QDK/Chemistry, including available basis sets, creation, manipulation, and serialization, refer to the :doc:`Basis Set documentation <basis_set>`.
   The basis set defines the atomic orbitals used to construct molecular orbitals and is a critical component for accurate quantum chemistry calculations.

Usage
-----

The :class:`~qdk_chemistry.data.Orbitals` class is typically created as the output of an :doc:`SCF calculation <../algorithms/scf_solver>` or :doc:`orbital transformation <../algorithms/localizer>`.
It serves as input to various post-:term:`HF` methods such as :doc:`active space selection <../algorithms/active_space>` and :doc:`Hamiltonian construction <../algorithms/hamiltonian_constructor>`.

Creating an Orbitals object
---------------------------

The :class:`~qdk_chemistry.data.Orbitals` object is typically created by algorithms rather than manually.
However, for advanced use cases, you can create and populate orbitals directly:

.. note::
   The class provides overloaded setter methods for coefficients, energies, and occupations.
   The single-argument versions (restricted setters) set the same values for both alpha and beta channels, while the two-argument versions (unrestricted setters) allow setting different values for alpha and beta channels.

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

Accessing Orbital Data
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

Serialization
-------------

The :class:`~qdk_chemistry.data.Orbitals` class supports serialization to and from JSON and HDF5 formats.
For detailed information about serialization in QDK/Chemistry, see the :doc:`Serialization <../data/serialization>` documentation.

.. note::
   All orbitals-related files require the ``.orbitals`` suffix before the file type extension, for example ``molecule.orbitals.json`` and ``h2.orbitals.h5`` for JSON and HDF5 files respectively.
   This naming convention is enforced to maintain consistency across the QDK/Chemistry ecosystem.

File formats
~~~~~~~~~~~~

QDK/Chemistry supports multiple serialization formats for orbital data, as described below.

JSON format
^^^^^^^^^^^

.. note::
   The ``basis_set`` field in the JSON representation contains a complete serialization of the :class:`~qdk_chemistry.data.BasisSet` object.
   For details on how basis sets are serialized, see the :ref:`JSON Format section in the Basis Set documentation <json-format>`.

JSON representation of an :class:`~qdk_chemistry.data.Orbitals` object has the following structure (showing simplified content):

.. code-block:: json

   {
     "ao_overlap":[[1.0,0.11921652588876666],[0.11921652588876666,1.0]],
     "basis_set":{"..."},
     "coefficients":{
       "alpha":[[0.6683869241842456,0.7534430365674294],
         [0.6683869241842458,-0.7534430365674292]],
       "beta":[[0.6683869241842456,0.7534430365674294],
         [0.6683869241842458,-0.7534430365674292]]
       },
     "energies":{
       "alpha":[-0.26945922362180524,0.10899736829389452],
       "beta":[-0.26945922362180524,0.10899736829389452]
     },
     "has_overlap_matrix":true,
     "is_open_shell":false,
     "is_restricted":true,
     "num_atomic_orbitals":2,
     "num_electrons":{"alpha":1.0,"beta":1.0},
     "num_molecular_orbitals":2,
     "occupations":{"alpha":[1.0,0.0],"beta":[1.0,0.0]}
   }

HDF5 format
^^^^^^^^^^^

.. note::
   The ``basis_set/`` group in the HDF5 representation contains a complete serialization of the :class:`~qdk_chemistry.data.BasisSet` object.
   For details on the HDF5 structure of basis sets, see the :ref:`HDF5 Format section in the Basis Set documentation <hdf5-format>`.

HDF5 representation of an :class:`~qdk_chemistry.data.Orbitals` object has the following structure (showing groups and datasets):

.. code-block:: text

   /
   ├── ao_overlap          # Dataset: 2D array of AO overlap matrix
   ├── basis_set/          # 🔧 **TODO**: Group: Contains basis set information
   ├── coefficients_alpha  # Dataset: 2D array of orbital coefficients (alpha)
   ├── coefficients_beta   # Dataset: 2D array of orbital coefficients (beta)
   ├── energies_alpha      # Dataset: 2D array of orbital energies (alpha)
   ├── energies_beta       # Dataset: 2D array of orbital energies (beta)
   ├── occupations_alpha   # Dataset: 2D array of orbital occupations (alpha)
   ├── occupations_beta    # Dataset: 2D array of orbital occupations (beta)
   └── metadata/           # Group
       ├── has_overlap_matrix       # Dataset: uint8, 0 if false, 1 if true
       ├── has_basis_set        # Dataset: uint8, 0 if false, 1 if true
       ├── is_open_shell        # Dataset: uint8, 0 if false, 1 if true
       ├── is_restricted        # Dataset: uint8, 0 if false, 1 if true
       ├── num_atomic_orbitals              # Dataset: uint32, number of atomic orbitals
       ├── num_electrons_alpha  # Dataset: uint32, number of alpha electrons
       ├── num_electrons_beta   # Dataset: uint32, number of beta electrons
       └── num_molecular_orbitals              # Dataset: uint32, number of molecular orbital coefficients

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/orbitals.cpp
      :language: cpp
      :start-after: // start-cell-serialization
      :end-before: // end-cell-serialization

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/orbitals.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization

Orbital transformations and applications
----------------------------------------

The :class:`~qdk_chemistry.data.Orbitals` class serves as a foundation for several important quantum chemical applications and transformations:

- **Orbital Localization**: Transform delocalized :term:`SCF` orbitals into localized representations for better chemical interpretation and more efficient correlation methods.
  See :doc:`Localizer <../algorithms/localizer>` for details.

- **Active Space Selection**: Automatically identify important orbitals for multi-reference calculations based on
  various criteria.
  See :doc:`ActiveSpaceSelector <../algorithms/active_space>` for details.

- **Hamiltonian Construction**: Build electronic Hamiltonians for post-HF methods using the orbital information.
  See :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>` for details.

Related classes
---------------

- :doc:`Structure <structure>`: Molecular structure representation
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from orbitals
- :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`: Algorithm that builds Hamiltonians from orbitals
- :doc:`ActiveSpaceSelector <../algorithms/active_space>`: Algorithm for selecting active spaces from orbitals
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces orbitals
- :doc:`Localizer <../algorithms/localizer>`: Algorithms for orbital transformations

Further reading
---------------

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/orbitals.cpp>`_ and `Python <../../../_static/examples/python/orbitals.py>`_ scripts.
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Settings <../design/settings>`: Configuration settings for algorithms
