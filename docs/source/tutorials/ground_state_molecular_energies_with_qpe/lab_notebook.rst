Lab notebook
############

Use this lab notebook to record the inputs, decisions, results, and interpretations that lead to the final ground-state energy estimate.
Each required chapter adds information to the same record.
The completed notebook should make the calculation reproducible and should distinguish changes in the molecular model from errors introduced by the quantum algorithm.

.. todo::

   Decide whether to provide this worksheet as a downloadable template in addition to the rendered page.
   Add a completed example after the tutorial calculations and reference values have been finalized.

How to use the notebook
=======================

Create a copy of the headings and prompts below in your preferred note-taking format.
Record values with units and enough significant figures to support later comparisons.
Record the software version and settings associated with each result.
Keep observations separate from interpretations.
Do not overwrite earlier results when a later chapter changes an approximation or algorithm setting.

.. _lab-notebook-setup:

Setup and provenance
====================

Record the environment used for the tutorial.

- Date:
- Operating system:
- Processor or computer model:
- Python version:
- :term:`QDK`/Chemistry version:
- Installation method:

.. _lab-notebook-goal:

Calculation goal and reference plan
===================================

State what the tutorial calculates and how each comparison will be interpreted.

- Molecular system:
- Target quantity:
- Teaching target:
- Basis-set comparison:
- Active-space comparison:
- Quantum-algorithm reference:
- Error sources that remain outside the final algorithmic comparison:

.. _lab-notebook-molecule:

Molecular input and mean-field calculation
==========================================

Record the molecular definition and the first electronic-structure results.

- Geometry source:
- Nitrogen--nitrogen distance and units:
- Molecular charge:
- Spin multiplicity:
- First basis set:
- First Hartree--Fock energy and units:
- Second basis set:
- Second Hartree--Fock energy and units:
- Basis-set sensitivity and units:
- Interpretation of the basis-set sensitivity:

.. _lab-notebook-active-space:

Active-space model
==================

Record how the correlated molecular model was chosen.

- Orbital representation used for selection:
- Initial active electrons and spatial orbitals:
- Initial correlated method:
- Active-space selection evidence:
- Selected active electrons and spatial orbitals:
- Selected active-space energy and units:
- Comparison with the initial active space:
- Interpretation of active-space convergence:
- Algorithmic reference energy and units:

.. _lab-notebook-qubits:

Qubit representation
====================

Record how the selected electronic Hamiltonian was represented on qubits.

- Fermion-to-qubit encoding:
- Number of active spatial orbitals:
- Number of active spin orbitals:
- Number of compute-register qubits:
- Number of Pauli terms in the qubit Hamiltonian:
- Core energy and units:
- Quantities excluded from the compute-register qubit count:

.. _lab-notebook-trial-state:

Trial state
===========

Record the approximation prepared on the compute register.

- Source wavefunction:
- Determinant-selection rule:
- Number of retained determinants:
- Retained norm or fidelity:
- State-preparation method:
- State-preparation circuit statistics:
- Interpretation of the fidelity and circuit cost:

.. _lab-notebook-phase-estimation:

Phase-estimation calculation
============================

Record the algorithm settings and final result.

- Evolution time and units:
- Hamiltonian-simulation method and settings:
- Number of phase bits:
- Shots per bit:
- Number of trials:
- Readout ancillas:
- Measured active-space energy and units:
- Core energy added after phase estimation:
- Total molecular energy estimate and units:
- Difference from the algorithmic reference and units:
- Result relative to the 1 milliHartree teaching target:
- Observed runtime:

.. _lab-notebook-conclusion:

Conclusion
==========

Explain what the final comparison establishes and what it does not establish.

- Main result:
- Evidence that supports the result:
- Basis-set limitations:
- Active-space limitations:
- Quantum-algorithm limitations:
- One change that would strengthen the calculation:
