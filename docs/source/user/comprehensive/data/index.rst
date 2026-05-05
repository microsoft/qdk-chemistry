Data classes
============

QDK/Chemistry uses immutable data classes to represent molecular information, electronic structure results, and quantum objects.
These classes serve as the inputs and outputs for :doc:`algorithm classes <../algorithms/index>`, enabling a clean flow of data through the computational pipeline.
All data classes support :doc:`serialization <serialization>` to JSON and HDF5 formats for persistence and interoperability.

Comprehensive details on each data class can be found in the :ref:`API documentation <apidocs>`.
Here, we provide a quick reference guide to help users understand the purpose and typical sources of commonly encountered data classes.
Each of the links below leads to a detailed description of the data class, including its attributes, methods, and usage examples.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   serialization
   ansatz
   basis_set
   hamiltonian
   orbitals
   lattice_graph
   pauli_operator
   structure
   symmetries
   wavefunction
   qpe_result
   circuit


Quick reference
---------------

The following table summarizes the available data classes in QDK/Chemistry and their purposes. For detailed documentation, refer to the linked pages.

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Data Class
     - Purpose
     - Typical Source
   * - :doc:`Structure <structure>`
     - Molecular geometry (atoms and coordinates)
     - User input
   * - :doc:`BasisSet <basis_set>`
     - Atomic orbital basis definitions
     - Library lookup, User input
   * - :doc:`Orbitals <orbitals>`
     - Molecular orbital coefficients and energies
     - :doc:`ScfSolver <../algorithms/scf_solver>`
   * - :doc:`Hamiltonian <hamiltonian>`
     - One- and two-electron integrals
     - :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`
   * - :doc:`Wavefunction <wavefunction>`
     - Electronic state (orbitals + :term:`CI` coefficients)
     - :doc:`MCCalculator <../algorithms/mc_calculator>`
   * - ``QubitHamiltonian``
     - Pauli operator representation
     - :doc:`QubitMapper <../algorithms/qubit_mapper>`
   * - :doc:`Symmetries <symmetries>`
     - Physical symmetries
     - Factory methods, User input
   * - :doc:`LatticeGraph <lattice_graph>`
     - Lattice topology for model Hamiltonians
     - Factory methods, User input
   * - :doc:`PauliOperator <pauli_operator>`
     - Pauli operator expressions with arithmetic
     - User construction
   * - :doc:`QpeResult <qpe_result>`
     - Phase estimation results (phase, energy, aliases)
     - :doc:`PhaseEstimation <../algorithms/phase_estimation>`
   * - :doc:`Circuit <circuit>`
     - Quantum circuit (OpenQASM, Q#, QIR, Qiskit)
     - :doc:`StatePreparation <../algorithms/state_preparation>`, User input

QubitHamiltonian and term partitions
------------------------------------

A :class:`~qdk_chemistry.data.QubitHamiltonian` carries an optional :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition` field describing how its Pauli terms are organised into algorithm-relevant subsets.
The partition is index-based — it stores indices into :attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings` — so it serialises cheaply alongside the Hamiltonian.

The partition is *optional* metadata — ``term_partition is None`` means the partition has not been computed for this Hamiltonian.
Transformations that change term ordering or qubit support (for example :meth:`~qdk_chemistry.data.QubitHamiltonian.to_interleaved`) reset the partition to ``None`` on the new instance.

Algorithms that consume a partition treat its presence as an explicit signal to exploit it — for example, the :doc:`Trotter time-evolution builder <../algorithms/time_evolution_builder>` reads ``term_partition`` and uses it for schedule-level Suzuki recursion and reduction.

FlatPartition
~~~~~~~~~~~~~

:class:`~qdk_chemistry.data.FlatPartition` stores a single-level grouping: each group is a tuple of term indices.
It is suitable for algorithms that only need to know which terms belong together, such as qubit-wise commuting measurement grouping in :class:`~qdk_chemistry.algorithms.QdkEnergyEstimator`.

The ``groups`` field is a tuple of tuples: ``((idx0, idx1, ...), (idx2, ...), ...)``.
Each inner tuple lists the indices of terms in :attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings` that belong to that group.

LayeredPartition
~~~~~~~~~~~~~~~~

:class:`~qdk_chemistry.data.LayeredPartition` stores a two-level hierarchy: groups contain parallelisable layers, and each layer contains term indices.
It is suitable for Trotter-style decompositions where the outer level controls Strang/Suzuki splitting order and each inner layer groups operators with disjoint qubit supports that can be applied simultaneously.

The ``groups`` field is a nested tuple: ``(((idx0, idx1), (idx2,)), ...)``.
The outer level is groups, the middle level is layers within a group, and the innermost level is term indices.

Both classes carry a ``strategy`` label (e.g. ``"geometry_coloring"``, ``"qubit_wise_commuting"``) identifying how the partition was produced.
They serialise as part of :class:`~qdk_chemistry.data.QubitHamiltonian` in both JSON and HDF5 formats.
