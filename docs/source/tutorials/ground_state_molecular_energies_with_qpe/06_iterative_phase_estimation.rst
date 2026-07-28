Iterative quantum phase estimation
##################################

This chapter introduces iterative quantum phase estimation (:term:`IQPE`).

.. rubric:: Chapter focus

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

.. rubric:: Lab notebook assignment

Complete :ref:`lab-notebook-phase-estimation` and :ref:`lab-notebook-conclusion`.
Record the algorithm settings before starting the final simulation and record the measured values after it finishes.
Use the completed notebook to explain whether the result meets the teaching target and which chemistry and algorithm limitations remain.

Prerequisite concepts
=====================

.. todo::

   Recall the qubit Hamiltonian, core energy, trial state, and fidelity.
   Link to an external phase-estimation refresher for learners who need to review the basic circuit model.

Encode energy as a phase
========================

.. todo::

   Derive the relationship among the Hamiltonian, time-evolution operator, eigenphase, evolution time, and energy.
   Reconcile the sign convention used by the documentation, circuit implementation, and QpeResult conversion before publishing this derivation.

Measure one phase bit at a time
===============================

.. todo::

   Explain the iterative circuit, controlled evolution, phase feedback, and single readout ancilla.
   Mention standard multi-ancilla phase estimation only as a contrast, not as a second required algorithm.

Choose the numerical controls
=============================

.. todo::

   Explain evolution time, Hamiltonian-simulation approximation, number of phase bits, shots per bit, and number of trials.
   Connect each control to the error taxonomy introduced in the first chapter.

Run iterative phase estimation
==============================

.. todo::

   Add a standalone native Python example that composes the trial-state circuit, Jordan--Wigner Hamiltonian, Trotter unitary builder, iterative circuit builder, and native simulator.
   Use short circuit-construction and inspection steps before the final simulation so learners do not repeat the expensive calculation unnecessarily.
   Base the scientific workflow on ``examples/qpe_stretched_n2.ipynb`` without carrying its Qiskit comparison or resource-estimation section into the required tutorial.

Interpret repeated trials
=========================

.. todo::

   Explain why imperfect overlap and finite sampling can produce different trial outcomes.
   Define the aggregation rule and report enough information to distinguish energy resolution from statistical confidence.

Reconstruct the molecular energy
================================

.. todo::

   Add the core energy to the qubit-Hamiltonian result.
   Compare the total energy with the :term:`CASCI` energy of the same selected active-space Hamiltonian.
   Interpret this difference as algorithmic error from Hamiltonian simulation, finite phase resolution, and sampling.
   Evaluate the difference against the 1 milliHartree teaching target.
   State that meeting this target does not remove basis-set or active-space error and does not establish agreement with experiment.
   Update the phase-estimation and conclusion sections of the lab notebook.
   Attribute the remaining error to the approximations used in the workflow.

Complete the workflow
=====================

.. todo::

   End with a complete, executable molecule-to-energy calculation assembled from the chapter examples.
   Make the final circuit simulation the only intentionally long required example.
   State that the expected duration is approximately 20 minutes on a typical student laptop and that the actual duration depends on the computer.
   Report progress during the simulation so the learner can distinguish a long calculation from a stalled process.
   State the environment, command, expected output, and criteria for successful completion.
   Mark the final example as a slow integration test and keep shorter tests for the individual components.

Check your understanding
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
