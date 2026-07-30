Physical resource estimation
############################

.. note:: Chapter focus

   What fault-tolerant machine could execute a circuit from the required tutorial?

Learning objectives
===================

After completing this chapter, you will be able to:

- Distinguish compute-register, algorithmic, logical, and physical resources.
- Explain why quantum error correction changes the resource requirements.
- Convert a :term:`QDK`/Chemistry circuit into a resource-estimation application.
- Run the :term:`QDK` Quantum Resource Estimator with stated assumptions.
- Interpret tradeoffs between physical qubits and runtime.

Prerequisite concepts
=====================

.. todo::

   Recall the required tutorial's compute-register count and iterative phase-estimation circuits.
   Introduce logical operations and quantum error correction only to the depth needed to interpret an estimate.

From problem size to machine size
=================================

.. todo::

   Define compute-register qubits, algorithm ancillas, logical qubits, physical qubits, logical operations, and runtime.
   Explain why a compute-register count is not a hardware estimate.

State the estimation assumptions
================================

.. todo::

   Introduce the hardware architecture, error rates, error-correction scheme, magic-state factory, and total error budget.
   Explain that the output is conditional on these inputs rather than a prediction of a specific machine.

Build the estimation application
================================

.. todo::

   Select a representative circuit from the required tutorial and convert it with ``Circuit.get_qre_application()``.
   Explain whether the circuit represents one :term:`IQPE` iteration, one trial, or the complete algorithmic workload.

Run and interpret the estimator
===============================

.. todo::

   Add a standalone Python example using the optional ``qre`` dependency.
   Present the Pareto-optimal estimates and interpret physical qubits, runtime, and factory requirements without conflating them with the molecular problem size.

Check your understanding
========================

.. todo::

   Add an exercise that changes one hardware or error-correction assumption and asks the learner to explain the resulting resource tradeoff.
   Consider a second exercise that compares resource estimates derived from the ``cc-pvdz`` and ``cc-pvtz`` basis choices in :doc:`Describing the molecule <../02_describing_the_molecule>` while holding all estimator assumptions fixed.
   Use the comparison to show how a larger molecular representation propagates into compute-register, circuit, and physical-resource requirements.

Further reading
===============

- :doc:`Quantum circuits <../../../user/comprehensive/data/circuit>`
- `Resource-estimation samples <https://github.com/microsoft/qdk/tree/main/samples/qre>`_
