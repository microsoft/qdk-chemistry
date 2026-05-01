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
The partition is index-based — it stores indices into :attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings` rather than nested ``QubitHamiltonian`` objects — so it serialises cheaply alongside the Hamiltonian.

Two concrete shapes are supported:

* :class:`~qdk_chemistry.data.FlatPartition` — a single level of *groups*, each group a list of term indices.
  Used by routines such as energy estimation, where the only decision is which terms to evaluate together (for example, qubit-wise commuting groups for measurement basis selection).
* :class:`~qdk_chemistry.data.LayeredPartition` — two levels of structure: each *group* is split into *layers*, and each layer is a list of term indices.
  Used by Trotter decomposition, where the outer level controls the splitting order and the inner level identifies operators that act on disjoint qubit supports and can therefore be applied in parallel.

The partition is *optional* metadata — ``term_partition is None`` means the partition has not been computed for this Hamiltonian, in which case algorithms that exploit groups fall back to computing them on the fly.
Transformations that change term ordering or qubit support (for example :meth:`~qdk_chemistry.data.QubitHamiltonian.to_interleaved`) reset the partition to ``None`` on the new instance.

Two mechanisms populate ``term_partition``:

#. The :doc:`spin model Hamiltonian builders <../model_hamiltonians>` (:func:`~qdk_chemistry.utils.model_hamiltonians.create_heisenberg_hamiltonian`, :func:`~qdk_chemistry.utils.model_hamiltonians.create_ising_hamiltonian`) populate a :class:`~qdk_chemistry.data.LayeredPartition` from the lattice's edge coloring when ``include_term_groups=True`` (the default).
#. The :ref:`term_grouper algorithm <algorithms-term-grouper>` accepts a ``QubitHamiltonian`` and returns a copy whose ``term_partition`` is populated by the requested strategy (``"commuting"``, ``"qubit_wise_commuting"``, or ``"identity"``).

Algorithms that consume a partition treat its presence as an explicit signal to exploit it — for example, the :doc:`Trotter time-evolution builder <../algorithms/time_evolution_builder>` reads ``term_partition`` and uses it for schedule-level Suzuki recursion and reduction.
