Iterative quantum phase estimation
##################################

This chapter introduces iterative quantum phase estimation (:term:`IQPE`).

.. admonition:: Chapter focus
   :class: chapter-focus

   How do we extract the molecular ground-state energy from the prepared quantum state?

Learning objectives
===================

After completing this chapter, you will be able to:

- Relate a Hamiltonian eigenvalue to a time-evolution eigenphase.
- Describe how :term:`IQPE` measures phase bits.
- Explain the roles of the compute register and readout ancilla.
- Relate phase bits, shots per bit, and repeated trials to the energy estimate.
- Run :term:`IQPE` with native :term:`QDK`/Chemistry tools.
- Reconstruct the total molecular energy and evaluate its error.

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

   Complete :ref:`lab-notebook-phase-estimation` and :ref:`lab-notebook-conclusion`.
   Record the algorithm settings before starting the final simulation and record the measured values after it finishes.
   Use the completed notebook to explain whether the result meets the teaching target and which chemistry and algorithm limitations remain.

Prerequisite concepts
=====================

.. todo::

   Recall the qubit Hamiltonian, core energy, trial state, and fidelity.
   Link to an external phase-estimation refresher for learners who need to review the basic circuit model.

Energy-to-phase encoding
========================

.. todo::

   Derive the relationship among the Hamiltonian, time-evolution operator, eigenphase, evolution time, and energy.
   Reconcile the sign convention used by the documentation, circuit implementation, and QpeResult conversion before publishing this derivation.

One phase bit at a time
===============================

.. todo::

   Explain the iterative circuit, controlled evolution, phase feedback, and single readout ancilla.
   Mention standard multi-ancilla phase estimation only as a contrast, not as a second required algorithm.

Numerical controls
=============================

.. todo::

   Explain evolution time, Hamiltonian-simulation approximation, number of phase bits, shots per bit, and number of trials.
   Connect each control to the error taxonomy introduced in the first chapter.

Iterative phase estimation
==============================

.. todo::

   Add a standalone native Python example that composes the trial-state circuit, Jordan--Wigner Hamiltonian, Trotter unitary builder, iterative circuit builder, and native simulator.
   Use short circuit-construction and inspection steps before the final simulation so learners do not repeat the expensive calculation unnecessarily.
   Base the scientific workflow on ``examples/qpe_stretched_n2.ipynb`` without carrying its Qiskit comparison or resource-estimation section into the required tutorial.

Repeated trials
=========================

.. todo::

   Explain why imperfect overlap and finite sampling can produce different trial outcomes.
   Define the aggregation rule and report enough information to distinguish energy resolution from statistical confidence.

Molecular-energy reconstruction
================================

.. todo::

   Add the core energy to the qubit-Hamiltonian result.
   Compare the total energy with the :term:`CASCI` energy of the same selected active-space Hamiltonian.
   Interpret this difference as algorithmic error from Hamiltonian simulation, finite phase resolution, and sampling.
   Evaluate the difference against the 1 milliHartree teaching target.
   State that meeting this target does not remove basis-set or active-space error and does not establish agreement with experiment.
   Update the phase-estimation and conclusion sections of the lab notebook.
   Attribute the remaining error to the approximations used in the workflow.

The complete workflow
=====================

.. todo::

   End with a complete, executable molecule-to-energy calculation assembled from the chapter examples.
   Make the final circuit simulation the only intentionally long required example.
   State that the expected duration is tens of minutes and report a measured duration with reference-hardware context after the final settings are fixed.
   Report progress during the simulation so the learner can distinguish a long calculation from a stalled process.
   State the environment, command, expected output, and criteria for successful completion.
   Mark the final example as a slow integration test and keep shorter tests for the individual components.

Knowledge check
========================

.. todo::

   Add an exercise that changes one precision or sampling parameter and asks the learner to predict and explain its effect.

Further reading
===============

- :doc:`Phase estimation <../../user/comprehensive/algorithms/phase_estimation>`
- :doc:`Phase-estimation circuit builders <../../user/comprehensive/algorithms/qpe_circuit_builder>`
- :doc:`Hamiltonian unitary builders <../../user/comprehensive/algorithms/hamiltonian_unitary_builder>`
- :doc:`Circuit execution <../../user/comprehensive/algorithms/circuit_executor>`
- :doc:`Phase-estimation results <../../user/comprehensive/data/qpe_result>`
