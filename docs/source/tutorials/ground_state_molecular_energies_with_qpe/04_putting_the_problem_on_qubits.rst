Putting the problem on qubits
#############################

.. note:: Chapter focus

   How does the active-space electronic problem become a qubit problem?

Learning objectives
===================

After completing this chapter, you will be able to:

- Identify the terms in the active-space electronic Hamiltonian.
- Explain why fermionic operators require an encoding on qubits.
- Describe how the Jordan--Wigner transformation preserves fermionic signs.
- Construct a qubit Hamiltonian with native :term:`QDK`/Chemistry tools.
- Determine the compute-register qubit count for the selected active space.
- Explain why the core energy must be added to a qubit-Hamiltonian result.

.. important:: Lab notebook assignment

   Complete :ref:`lab-notebook-qubits`.
   Derive the compute-register qubit count from the selected active space before verifying it with code.
   Record the core energy separately and identify every quantity excluded from the compute-register count.

Prerequisite concepts
=====================

.. todo::

   Recall spin orbitals, Slater determinants, and the selected active space.
   Provide an external refresher for Pauli operators if needed.

Construct the active-space Hamiltonian
======================================

.. todo::

   Introduce the one- and two-electron terms in second quantization.
   Define creation and annihilation operators and connect their anticommutation to wavefunction antisymmetry.

Explain why an encoding is required
===================================

.. todo::

   Contrast indistinguishable fermionic modes with distinguishable qubits.
   Show why mapping occupation alone does not reproduce the signs introduced by reordering fermionic operators.

Apply the Jordan--Wigner transformation
=======================================

.. todo::

   Explain the occupation representation and parity strings.
   Mention other encodings only as additional reading.
   Use the native :term:`QDK`/Chemistry mapper in the required example.

Count the compute-register qubits
=================================

.. todo::

   Derive one qubit per spin orbital and two compute-register qubits per spatial orbital for the restricted examples in this tutorial.
   Distinguish this count from algorithm ancillas and physical qubits.
   Update the qubit-representation section of the lab notebook.

Track the core energy
=====================

.. todo::

   Explain which nuclear and frozen-orbital contributions are stored as core energy.
   Show that the mapped QubitOperator excludes this contribution and record where the workflow adds it back.

Run the mapping
===============

.. todo::

   Add a standalone native Python example that constructs the active-space Hamiltonian, builds the Jordan--Wigner mapping, creates the qubit operator, and verifies its compute-register qubit count.

Check your understanding
========================

.. todo::

   Add an exercise that derives the qubit count for a new active space and identifies which reported quantities are not physical resource estimates.

Further reading
===============

- :doc:`Hamiltonian construction <../../user/comprehensive/algorithms/hamiltonian_constructor>`
- :doc:`Electronic Hamiltonians <../../user/comprehensive/data/hamiltonian>`
- :doc:`Qubit mapping <../../user/comprehensive/algorithms/qubit_mapper>`
- :doc:`Majorana mappings <../../user/comprehensive/data/majorana_mapping>`
- :doc:`Pauli operators <../../user/comprehensive/data/pauli_operator>`
