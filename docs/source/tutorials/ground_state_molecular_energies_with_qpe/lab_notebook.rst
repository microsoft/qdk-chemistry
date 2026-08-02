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

Record the package version used for the tutorial and whether the setup check passed.

- :term:`QDK`/Chemistry version:
- Built-in implementation verification result:

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
- Orbital entropies and entropy-gap evidence used for active-space selection:
- Selected active electrons and spatial orbitals:
- Selected active-space energy and units:
- Comparison with the initial active space:
- Interpretation of the active-space energy change and size tradeoff:
- Algorithmic reference energy and units:

.. _lab-notebook-qubits:

Qubit representation
====================

Record how the selected electronic Hamiltonian was represented on qubits.
The :ref:`compute register <tutorial-compute-register>` contains the qubits that store the encoded active-space fermionic state.
Each qubit in this register is a compute-register qubit.
Ancilla qubits assist with algorithmic tasks such as control, workspace, or readout and are recorded separately.

- Fermion-to-qubit encoding:
- Number of active spatial orbitals:
- Number of active spin orbitals:
- Number of compute-register qubits:
- Number of Pauli terms in the qubit Hamiltonian:
- :ref:`Fixed-electron-number subspace <tutorial-fixed-electron-number-subspace>` (:math:`n_\alpha`, :math:`n_\beta`, and number of basis states):
- Core energy and units:
- Mapped active-space ground-state energy and units:
- Mapped selected-space total energy and units:
- Difference from the :ref:`selected-space algorithmic reference <tutorial-selected-space-reference>` and units:
- Interpretation of the energy comparison:
- Quantities excluded from the compute-register qubit count:

.. _lab-notebook-trial-state:

Trial state
===========

Record the approximation prepared on the compute register.

- Source wavefunction:
- Leading reference determinants, amplitudes, weights, and cumulative weights:
- Determinant-selection rule used before :term:`PMC` re-optimization:
- Determinant counts compared:
- Fidelity for each trial state:
- State-preparation method:
- Compute qubits for each trial state:
- Preparation logical gate count and logical gate-family counts for each trial state:
- Interpretation of the fidelity and circuit cost:

.. _lab-notebook-phase-estimation:

Phase-estimation calculation
============================

Record the algorithm settings and final result.

- Evolution time and units:
- Hamiltonian-simulation method and settings:
- Number of phase bits:
- Shots per bit:
- Number of complete IQPE runs:
- Readout ancillas:
- Complete-run bitstring counts:
- Modal bitstring:
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
