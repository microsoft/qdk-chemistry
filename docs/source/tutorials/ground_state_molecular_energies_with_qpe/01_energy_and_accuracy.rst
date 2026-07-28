Energy and accuracy
###################

.. rubric:: Chapter focus

What energy will we calculate, and how accurate must the result be?

Learning objectives
===================

After completing this chapter, you will be able to:

- Define the molecular electronic ground-state energy used in this tutorial.
- Distinguish accuracy from precision.
- Explain why an energy must be compared with a stated reference.
- Distinguish basis-set sensitivity, active-space convergence, and quantum-algorithm error.
- Identify the main approximations that affect the final energy estimate.
- Describe the complete tutorial workflow at a conceptual level.

.. rubric:: Lab notebook assignment

Complete :ref:`lab-notebook-goal`.
Record the target quantity, the 1 milliHartree teaching target, and the distinct meanings of the basis-set, active-space, and quantum-algorithm comparisons.
This entry defines the criteria you will use to interpret every later result.

Prerequisite concepts
=====================

.. todo::

   Identify the prerequisite ideas from chemistry and quantum mechanics.
   Link to concise refreshers for eigenstates, Hamiltonians, and potential-energy surfaces without teaching the later algorithmic details early.

The quantity we will calculate
==============================

.. todo::

   Define the electronic ground-state energy for fixed nuclear coordinates.
   State what contributions the reported total energy includes.
   Explain why chemically meaningful conclusions often depend on energy differences.

Accuracy requires a reference
=============================

.. todo::

   Define error relative to an explicit reference.
   Explain the conventional uses and limitations of the term chemical accuracy.
   Introduce 1 milliHartree as the teaching target used in this tutorial.
   State that meeting this threshold relative to the selected active-space reference does not demonstrate agreement with experiment to chemical accuracy.
   Keep accuracy distinct from numerical precision and measurement uncertainty.

Where error enters the calculation
==================================

.. todo::

   Introduce basis-set, electronic-structure, active-space, Hamiltonian-simulation, phase-resolution, and statistical errors.
   Name each source without explaining machinery reserved for later chapters.
   Explain that each later comparison isolates a different subset of these errors and therefore requires a different reference.

From a molecule to an energy
============================

.. todo::

   Add a narrated, non-executable diagram of the required workflow.
   Show the molecular input, classical preparation, qubit representation, trial-state preparation, iterative phase estimation, and final comparison with the reference energy.

Tracking the calculation
========================

.. todo::

   Explain how the lab notebook accumulates chemistry and quantum-algorithm results without treating every energy difference as the same error metric.
   Show where later chapters record basis-set sensitivity, active-space convergence, and algorithmic error.
   Leave numerical examples empty until the basis sets, active spaces, and reference values have been benchmarked.

Check your understanding
========================

.. todo::

   Add questions that test the distinction among total energy, energy differences, accuracy, precision, and uncertainty.

Further reading
===============

- :doc:`Features and methods <../../user/features>`
- :doc:`Quickstart <../../user/quickstart>`
- :doc:`References <../../references>`
