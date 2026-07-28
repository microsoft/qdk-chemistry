Preparing the trial state
#########################

.. rubric:: Chapter focus

How do we prepare a state that lets phase estimation return the ground-state energy reliably?

Learning objectives
===================

After completing this chapter, you will be able to:

- Explain why phase estimation requires an input state.
- Define state overlap and fidelity.
- Relate ground-state fidelity to the probability of measuring its energy.
- Construct a sparse trial wavefunction from important determinants.
- Generate a native sparse-isometry state-preparation circuit.
- Distinguish trial-state quality from state-preparation circuit cost.

.. rubric:: Lab notebook assignment

Complete :ref:`lab-notebook-trial-state`.
Record trial-state fidelity and state-preparation circuit statistics as separate quantities.
Explain how determinant truncation changes the probability of obtaining the ground-state energy and the cost of preparing the trial state.

Prerequisite concepts
=====================

.. todo::

   Recall the correlated active-space wavefunction, determinant amplitudes, and compute-register encoding.
   Define any additional state-vector notation before introducing overlap.

Phase estimation needs a trial state
====================================

.. todo::

   Explain that phase estimation samples eigenvalues represented in the input state and does not independently search for the ground state.
   Connect this fact to the multi-configurational wavefunction from the active-space chapter.

Measure trial-state quality
===========================

.. todo::

   Define the overlap amplitude and fidelity.
   State precisely how ground-state fidelity affects the probability of obtaining the ground-state energy in an ideal phase-estimation trial.

Build a sparse trial wavefunction
=================================

.. todo::

   Select important determinants from the correlated reference.
   Explain the fidelity lost by truncation and the classical projected calculation used to obtain a normalized trial wavefunction.

Generate the preparation circuit
================================

.. todo::

   Explain the native GF(2)+X sparse-isometry method at the depth needed to interpret its inputs and output.
   Do not require or compare against Qiskit in the required chapter.

Evaluate quality and cost separately
====================================

.. todo::

   Report trial-state fidelity and circuit statistics as different quantities.
   Explain that state preparation does not change the compute-register problem size but can change circuit cost and the number of phase-estimation trials.

Run the preparation
===================

.. todo::

   Add a standalone native Python example that truncates the correlated wavefunction, computes fidelity, builds the sparse-isometry circuit, and checks its register size.

Check your understanding
========================

.. todo::

   Add an exercise that compares two determinant truncations and asks the learner to explain the quality-cost tradeoff.

Further reading
===============

- :doc:`State preparation <../../user/comprehensive/algorithms/state_preparation>`
- :doc:`Projected multi-configuration calculations <../../user/comprehensive/algorithms/pmc>`
- :doc:`Wavefunctions <../../user/comprehensive/data/wavefunction>`
- :doc:`Quantum circuits <../../user/comprehensive/data/circuit>`
