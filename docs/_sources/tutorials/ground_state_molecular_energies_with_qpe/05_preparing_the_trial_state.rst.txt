Preparing the trial state
#########################

.. admonition:: Chapter focus
   :class: chapter-focus

   How do we prepare a state that lets phase estimation return the ground-state energy reliably?

Learning objectives
===================

After completing this chapter, you will be able to:

- Explain why phase estimation requires an input state.
- Define state overlap and fidelity.
- Explain how ground-state fidelity measures target-state weight and influences phase-estimation outcomes.
- Construct a sparse trial wavefunction from important determinants.
- Generate a state-preparation logical circuit with the QDK/Chemistry sparse-isometry implementation.
- Distinguish trial-state quality from state-preparation logical circuit cost.

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

   Complete :ref:`lab-notebook-trial-state`.
   Record trial-state fidelity and state-preparation logical circuit statistics as separate quantities.
   Explain how determinant truncation changes the ground-state weight, its influence on phase-estimation outcomes, and the cost of preparing the trial state.

Example files
=============

Add :download:`tutorial_prepare_trial_state.py <../../_static/examples/python/tutorial_prepare_trial_state.py>` and :download:`tutorial_prepare_trial_state.ipynb <../../_static/examples/python/tutorial_prepare_trial_state.ipynb>` to your tutorial working folder.
Open the new files in Visual Studio Code and review the complete trial-state script, including imports and helper functions omitted from the excerpts below.
The script imports the tested active-space workflow so that this chapter uses the same selected Hamiltonian and :term:`CASCI` reference.
The Jupyter notebook runs that workflow, renders the one-, two-, and four-determinant logical circuits, and validates their reported structure and gate statistics.

Connection to the selected-space workflow
=========================================

The :ref:`selected-space CASCI calculation <tutorial-selected-space-reference>` produced a normalized ground-state wavefunction spanning the :math:`(n_\alpha,n_\beta)=(3,3)` determinant sector introduced in :doc:`Mapping the problem to qubits <04_putting_the_problem_on_qubits>`.
Each determinant represents one pattern of occupations among the selected active spin orbitals, and its coefficient is the corresponding amplitude in the wavefunction.
The :ref:`Jordan--Wigner encoding <tutorial-occupation-encoding>` represents the same occupation patterns on the compute register sized in :doc:`Mapping the problem to qubits <04_putting_the_problem_on_qubits>`.

.. _tutorial-trial-state-definition:

Why phase estimation needs a trial state
========================================

Quantum phase estimation (:term:`QPE`) estimates an eigenphase of a unitary operator.
For molecular energies, that unitary represents evolution under the Hamiltonian: each Hamiltonian eigenstate is also an eigenstate of the time-evolution operator, and its phase depends on its energy.
The phase-to-energy relationship and the :term:`QPE` logical circuit are developed in the next chapter.
For now, the important point is that the compute register must contain a chosen quantum state before phase estimation can begin.

A state-preparation logical circuit initializes the compute register in this chosen normalized quantum state.
Here, *logical* means gates in the generated algorithmic circuit before error-correction code synthesis and hardware mapping.
This input is the trial state.
It is an approximation intended to contain a substantial contribution from the target ground state; :term:`QPE` cannot begin from an unspecified state or create the ground state by searching through all possible wavefunctions.

The trial state can be written as a linear combination of the eigenstates :math:`\{\vert\Psi_j\rangle\}` of the active-space Hamiltonian:

.. math::

   \vert\Psi_{\mathrm{trial}}\rangle
   = \sum_j a_j\vert\Psi_j\rangle,
   \qquad
   \sum_j \left\vert a_j\right\vert^2=1.

To isolate the effect of the input state, first assume that the requested trial state is prepared exactly, time evolution is exact, and phase readout has enough resolution to distinguish the relevant eigenphases.
Under these assumptions, an input eigenstate :math:`\vert\Psi_j\rangle` produces the phase corresponding to :math:`E_j`.
For a trial state containing several eigenstates, a textbook coherent phase-estimation measurement samples the energy :math:`E_j` with probability :math:`\left\vert a_j\right\vert^2` :cite:`vonBurg2021`.
This probability statement assumes that one prepared system state produces one complete phase result.

.. admonition:: Why can phase estimation return an excited-state energy even when the ground-state energy is the target?
   :class: quiz-question
   :collapsible: closed

   A trial state can contain both ground- and excited-state eigenvectors.
   Under the assumptions above, phase estimation returns each represented eigenvalue with probability equal to the squared magnitude of that eigenstate's amplitude in the trial state.

Ground-state fidelity
=====================

As :ref:`introduced in the tutorial overview <tutorial-trial-state-fidelity>`, the ground-state fidelity is the squared overlap

.. math::

   F
   = \left\vert
       \langle\Psi_0\vert\Psi_{\mathrm{trial}}\rangle
     \right\vert^2.

Both states are normalized, so :math:`0\leq F\leq 1`.
The fidelity :math:`F` is the weight of the target ground state in the trial-state eigenstate expansion.
For the textbook coherent measurement described above, it is also the probability of sampling the ground-state eigenphase.

The QDK/Chemistry iterative quantum phase estimation (:term:`IQPE`) implementation used later performs a different sampling procedure.
Each *phase bit* is one binary digit of the estimated phase fraction.
The implementation builds a separate circuit for each phase bit, and every circuit execution freshly prepares the trial state.
Each phase bit is selected by a majority vote over a specified number of circuit executions, then used as feedback for the next bit.
The final bit string therefore combines bitwise decisions from many state preparations rather than recording one eigenstate sample.

Fidelity remains a useful trial-state quality measure because it controls the ground-state contribution to those bit statistics.
However, it is not by itself the probability that one complete implemented :term:`IQPE` run returns the ground-state energy, so it does not determine a trial count.
The complete result also depends on the other eigenstate weights and phases, the number of phase bits, shots per bit, phase feedback, and the Hamiltonian-simulation approximation.
The next chapter develops this bitwise sampling procedure and evaluates repeated complete :term:`IQPE` runs.

Imperfect logical state preparation can also change the state actually loaded.
Residual logical faults after error correction can introduce further errors on a fault-tolerant machine, but they are not modeled by this tutorial's simulator.
These effects should be evaluated separately from the fidelity of the intended trial state.

A sparse trial wavefunction
===========================

The selected-space :term:`CASCI` wavefunction is classically tractable in this teaching example, so it provides a controlled reference for comparing trial states.
The script ranks its determinants by coefficient magnitude and retains the largest one, two, or four.
These determinant counts are examples rather than restrictions of the projected calculation or state-preparation method.
To compare other choices, change ``determinant_counts`` in the Jupyter notebook and rerun its cells.
In a larger problem where exact :term:`CASCI` is unavailable, an approximate classical method must supply the candidate determinants and amplitudes for the trial state.

The script first prints the leading terms in the selected-space wavefunction.
Each occupation string contains one symbol for each selected active spatial orbital: ``2`` means doubly occupied, ``u`` means occupied by one :math:`\alpha` electron, ``d`` means occupied by one :math:`\beta` electron, and ``0`` means unoccupied.
The amplitude is the signed coefficient :math:`c_I` in :math:`\vert\Psi_0\rangle=\sum_I c_I\vert\Phi_I\rangle`, while the weight :math:`\left\vert c_I\right\vert^2` is that determinant's contribution to the squared norm.
The cumulative weight shows how much of the norm is captured by the listed determinants.
The script computes these quantities directly from the leading :term:`CASCI` coefficients.

Simply discarding coefficients and renormalizing would not optimize the wavefunction within the retained determinant space because the full-space amplitudes are not generally the amplitudes that minimize energy after determinants are removed.
A projected multi-configuration (:term:`PMC`) calculation is a configuration-interaction calculation restricted to a user-specified set of determinants.
The QDK/Chemistry :term:`PMC` calculator instead constructs the Hamiltonian matrix in the retained determinant space and solves its eigenvalue problem for the lowest-energy normalized eigenvector.
The resulting projected wavefunction has zero amplitude on every omitted determinant.
When using the projected wavefunction as a trial state, its overlap with the complete selected-space :term:`CASCI` wavefunction therefore quantifies how much fidelity is retained after determinant truncation.

Under the :ref:`Jordan--Wigner encoding <tutorial-occupation-encoding>`, each retained determinant becomes one computational-basis state of the compute register.
The trial wavefunction is therefore

.. math::

   \vert\Psi_{\mathrm{trial}}\rangle
   =\sum_{I=1}^{K}\widetilde{c}_I\vert b_I\rangle,

where :math:`\vert b_I\rangle` is the occupation bitstring for retained determinant :math:`\Phi_I`, and :math:`\widetilde{c}_I` is its reoptimized amplitude.

The script constructs each projected trial state, forms the reference and trial coefficient vectors for the same retained determinants, and evaluates their squared inner product directly:

.. literalinclude:: ../../_static/examples/python/tutorial_prepare_trial_state.py
   :language: python
   :dedent: 8
   :start-after: # start-cell-sparse-trial
   :end-before: # end-cell-sparse-trial

.. admonition:: Why must fidelity be calculated separately from the PMC energy?
   :class: quiz-question
   :collapsible: closed

   The :term:`PMC` calculation chooses the lowest-energy wavefunction within the retained determinant space, but it does not directly maximize overlap with the complete selected-space ground state.
   Energy and overlap measure different properties, so the script evaluates fidelity explicitly.

The trial state preparation logical circuit
===========================================

The QDK/Chemistry sparse-isometry implementation converts the retained determinants into a binary matrix whose rows represent qubits and whose columns represent occupied-or-unoccupied patterns.
The method uses binary row operations to reduce the determinant patterns while recording controlled-NOT (CNOT) and X operations, prepares the reduced set of amplitudes, and reverses the recorded operations to expand the state across the compute register, in an optimized version of approaches introduced by Malvetti et al. :cite:`Malvetti2021`.
Students do not need to reproduce this synthesis by hand; it is implemented natively in QDK/Chemistry.
The important input is the normalized sparse wavefunction and the output is a logical circuit that prepares its amplitudes on the corresponding occupation states.

Some compute-register wires may have no gates in the state-preparation circuit.
The register begins in the all-zero occupation state, so a wire needs no preparation operation when the selected sparse wavefunction does not require that occupation bit to change or become entangled.
This does not make the qubit unnecessary: every compute qubit represents an active spin orbital on which the mapped active-space Hamiltonian acts.
The later controlled time evolution in :term:`QPE` therefore requires the complete compute register and can couple the prepared determinant support to other configurations in the same sector.
Removing a gate-free preparation wire would change the Hamiltonian representation and the molecular problem, rather than merely simplify state preparation.

Before answering the next question, open :download:`tutorial_prepare_trial_state.ipynb <../../_static/examples/python/tutorial_prepare_trial_state.ipynb>` in Visual Studio Code.
Choose **Select Kernel**, select **Python Environments**, and choose the ``.venv`` environment created in :doc:`Before you begin <00_before_you_begin>`.
Then select **Run All** to execute the shared trial-state workflow and render the one-, two-, and four-determinant logical circuits.
Compare the gate types and circuit structure before revealing the answer below.

.. figure:: /_static/diagrams/tutorial_qpe_state_preparation_comparison.png
   :alt: Side-by-side logical state-preparation circuits on twelve compute qubits. The one-determinant circuit on the left contains six X gates that prepare one occupation bit string. The two-determinant circuit on the right contains rotations and controlled operations that prepare a coherent superposition of two occupation bit strings.
   :align: center
   :width: 100%

   Generated logical state-preparation circuits for the one-determinant trial state (left) and two-determinant trial state (right). Both use the same twelve-qubit compute register; the additional operations prepare multiple amplitudes rather than additional spin orbitals.

.. admonition:: Why does one-determinant state preparation look so different from multi-determinant preparation?
   :class: quiz-question
   :collapsible: closed

   One determinant is one occupation bit string and therefore one computational-basis state.
   Starting from the all-zero state, X gates only need to flip the qubits representing occupied spin orbitals.
   A multi-determinant wavefunction is instead a coherent superposition of distinct occupation bit strings.
   Because X gates can only map one basis state to another, rotations are needed to create amplitudes, phase operations establish relative signs or phases, and entangling gates correlate occupation changes across qubits.
   The exact gate sequence depends on the synthesis method, but the distinction between preparing one basis state and preparing a coherent superposition is general.

To measure the generated logical-circuit cost, the script traverses the decomposed :ref:`Q# circuit representation <tutorial-qsharp>`, counts displayed gate records that have no nested child operations, and identifies controlled X gates as CNOT gates.
The script creates the QDK/Chemistry sparse-isometry implementation and inspects the generated Q# logical circuit.
The factory key :ref:`sparse_isometry_gf2x <sparse-isometry-gf2x>` is the implementation's current API identifier using a helper function to count gates:

.. literalinclude:: ../../_static/examples/python/tutorial_prepare_trial_state.py
   :language: python
   :dedent: 8
   :start-after: # start-cell-preparation-circuit
   :end-before: # end-cell-preparation-circuit

The reported *preparation logical gate count* is the number of these childless gate records in the generated Q# logical-circuit representation after the state-preparation operation has been decomposed.
This software-level logical gate count is not logical-circuit depth, a fault-tolerant resource estimate, or a physical-resource estimate.
It can change if the state-preparation or circuit-decomposition implementation changes; error correction affects downstream fault-tolerant and physical costs instead.

Trial-state quality and preparation cost
========================================

Trial-state truncation introduces a separate cost--quality tradeoff within the selected active space.
Retaining more determinants can improve fidelity with the selected-space ground state, but preparing more nonzero amplitudes generally requires more logical gates.
The relevant question is therefore how much fidelity is gained for each increase in state-preparation cost.

All three trial states describe the same selected active spin-orbital space, so they use the same compute register size.
Changing the number of retained determinants changes amplitudes and logical-circuit structure, not the number of spin orbitals represented.

.. admonition:: Does retaining more determinants require more compute qubits?
   :class: quiz-question
   :collapsible: closed

   No.
   Each trial state represents the same selected active spin-orbital space, so the compute-register size is unchanged.
   The number of retained determinants affects state-preparation operations rather than the compute-register size.

Running the preparation
=======================

With the Python environment from :doc:`Before you begin <00_before_you_begin>` active, run the complete script from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_prepare_trial_state.py

.. admonition:: How do fidelity and preparation cost change as determinants are retained?
   :class: quiz-question
   :collapsible: closed

   The one-, two-, and four-determinant fidelities are approximately :math:`0.4825`, :math:`0.5864`, and :math:`0.7324`, respectively.
   Their generated logical circuits have preparation logical gate counts of 6, 14, and 30, respectively, while every logical circuit uses twelve compute qubits.
   From one to two determinants, fidelity increases by approximately :math:`0.104` while the gate count increases by eight.
   From two to four determinants, fidelity increases by approximately :math:`0.146` while the gate count increases by sixteen.
   For these three generated circuits, the second expansion provides less fidelity gain per additional preparation gate than the first.

.. admonition:: What does the single-determinant fidelity reveal about multireference character?
   :class: quiz-question
   :collapsible: closed

   Its fidelity is approximately :math:`0.4825`, equal to the weight of the leading ``222000`` determinant.
   No single determinant therefore carries a majority of the selected-space ground-state weight at this geometry.
   The substantial weight distributed among additional determinants provides direct evidence of multireference character in the selected active-space orbital representation.

Record the leading reference determinants and all three determinant counts, fidelities, compute-qubit counts, preparation logical gate counts, and logical gate-family counts in the :ref:`trial-state section of the lab notebook <lab-notebook-trial-state>`.
Explain what the leading determinant weight reveals about multireference character, and distinguish the fidelity improvement from the increased logical-circuit cost.
The final :term:`IQPE` calculation uses the four-determinant trial state.
In the lab notebook, use your measured fidelities and preparation logical gate counts to evaluate this choice and describe what would be gained or lost by using one of the smaller trial states instead.

Further reading
===============

- :doc:`State preparation <../../user/comprehensive/algorithms/state_preparation>`
- :doc:`Projected multi-configuration calculations <../../user/comprehensive/algorithms/pmc>`
- :doc:`Wavefunctions <../../user/comprehensive/data/wavefunction>`
- :doc:`Quantum circuits <../../user/comprehensive/data/circuit>`
