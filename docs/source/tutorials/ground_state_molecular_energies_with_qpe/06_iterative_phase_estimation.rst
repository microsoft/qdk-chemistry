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
- Relate phase bits, shots per bit, and repeated complete runs to the energy estimate.
- Run :term:`IQPE` with native :term:`QDK`/Chemistry tools.
- Reconstruct the total molecular energy and evaluate its error.

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

   Complete :ref:`lab-notebook-phase-estimation` and :ref:`lab-notebook-conclusion`.
   Record the algorithm settings before starting the final simulation and record the measured values after it finishes.
   Use the completed lab notebook to explain whether the result meets the teaching target and which chemistry and algorithm limitations remain.

Example download
================

Download :download:`tutorial_run_iqpe.py <../../_static/examples/python/tutorial_run_iqpe.py>` and save it in the tutorial working directory alongside :download:`tutorial_orbital_coordinates.py <../../_static/examples/python/tutorial_orbital_coordinates.py>`, :download:`tutorial_choose_active_space.py <../../_static/examples/python/tutorial_choose_active_space.py>`, :download:`tutorial_map_n2_to_qubits.py <../../_static/examples/python/tutorial_map_n2_to_qubits.py>`, and :download:`tutorial_prepare_trial_state.py <../../_static/examples/python/tutorial_prepare_trial_state.py>`.
Also download :download:`tutorial_visualize_iqpe_circuit.ipynb <../../_static/examples/python/tutorial_visualize_iqpe_circuit.ipynb>` to the same directory.
Open the files in Visual Studio Code and review the complete :term:`IQPE` script, including imports, data classes, and helper functions omitted from the excerpts below.
The script imports the mapping and trial-state workflows from previous chapters so every stage uses the same selected Hamiltonian and four-determinant trial state.
The Jupyter notebook constructs and renders the shortest iteration circuit without executing the simulator.

Prerequisite concepts
=====================

An :term:`IQPE` calculation requires four kinds of information:

:ref:`Qubit Hamiltonian <tutorial-qubit-hamiltonian>`
   A Hermitian operator whose eigenvalue is sought, represented on the compute register.
:ref:`Trial state <tutorial-trial-state-definition>`
   A normalized state with nonzero weight on the target Hamiltonian eigenstate.
:ref:`Time-evolution implementation <tutorial-energy-to-phase-encoding>`
   A logical circuit approximating :math:`e^{-i\hat Ht}` and controlled powers of that unitary.
:ref:`Numerical and sampling controls <tutorial-iqpe-numerical-controls>`
   An evolution time, number of phase bits, shots per bit, and number of complete runs.

The previous chapters supplied the qubit Hamiltonian and trial state for the selected molecular problem.
They also recorded the core energy needed to reconstruct the molecular total and the classical :term:`CASCI` reference used to validate the final estimate; those two quantities are bookkeeping and validation data rather than inputs to the phase-estimation circuit.

The compute register stores the encoded active-space wavefunction.
This chapter adds one readout ancilla, a qubit used to extract phase information without representing another spin orbital.
The `quantum phase estimation overview <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`_ provides an optional refresher on the basic controlled-unitary circuit model.

.. _tutorial-energy-to-phase-encoding:

Energy-to-phase encoding
========================

Let :math:`\vert\Psi_j\rangle` be an eigenstate of the qubit Hamiltonian with active-space energy :math:`E_j`:

.. math::

   \hat H_{\mathrm{qubit}}\vert\Psi_j\rangle
   = E_j\vert\Psi_j\rangle.

In atomic units, the evolution time :math:`t` is expressed in inverse Hartree, :math:`E_{\mathrm{h}}^{-1}`, so the product :math:`E_jt` is dimensionless.
The time-evolution unitary is

.. math::

   U(t)=e^{-i\hat H_{\mathrm{qubit}}t}.

Because every power of the Hamiltonian acting on :math:`\vert\Psi_j\rangle` contributes the corresponding power of :math:`E_j`, the exponential acts on that eigenstate as

.. math::

   U(t)\vert\Psi_j\rangle
   =e^{-i\hat H_{\mathrm{qubit}}t}\vert\Psi_j\rangle
   =e^{-iE_jt}\vert\Psi_j\rangle.

The physical eigenphase in the exponential is therefore :math:`-E_jt` modulo :math:`2\pi`.
The phase fraction reported by :term:`QPE` is

.. math::

   \varphi_j
   =\left(\frac{-E_jt}{2\pi}\right)\bmod 1.

The QDK/Chemistry result object handles the modulo wrapping automatically.
It converts the measured phase fraction to a signed angle :math:`\alpha\in(-\pi,\pi]` and returns :math:`-\alpha/t`.
The tutorial script uses this value directly rather than manually applying a sign conversion.
To avoid aliasing, the active-space Hamiltonian energy eigenvalue :math:`E_j` being estimated must lie in the signed interval :math:`(-\pi/t,\pi/t]`; energies outside that interval can produce the same measured phase.
The two boundary energies differ by one complete phase turn and therefore represent the same measured phase; QDK/Chemistry assigns that boundary to :math:`+\pi/t`.

With :math:`m` measured phase bits, the representable fractions are multiples of :math:`2^{-m}`, so adjacent energy-grid points are separated by

.. math::

   \Delta E_{\mathrm{grid}}=\frac{2\pi}{t2^m}.

With :math:`m` measured phase bits, the largest controlled power is :math:`U^{2^{m-1}}`.
Increasing :math:`m` from six to ten therefore raises the largest power from :math:`U^{32}` to :math:`U^{512}`, increasing the size and runtime cost of the largest iteration circuit by a factor of sixteen for this repeated-power strategy.
This tutorial uses six bits to keep the simulation tractable; that choice does not by itself provide :math:`\mathrm{m}E_{\mathrm{h}}` resolution.

The evolution time is computed from quantities already produced by the classically tractable example chosen for this tutorial.
If we write the mapped Hamiltonian as :math:`\hat H_{\mathrm{qubit}}=\sum_\ell h_\ell P_\ell` and define

.. math::

   \lambda=\sum_\ell\lvert h_\ell\rvert.

The QDK/Chemistry application programming interface (:term:`API`) exposes this coefficient sum as ``qubit_hamiltonian.schatten_norm``; the tutorial script uses it to choose the evolution time and reports it in the pre-simulation settings.
For this Hamiltonian, the reported value is :math:`\lambda=19.610172748837\ E_{\mathrm{h}}`.
Because :math:`\lambda` bounds the magnitudes of the Hamiltonian eigenvalues, the initial choice

.. math::

   t_{\mathrm{bound}}=\frac{\pi}{\lambda}
   =0.160202191680\ E_{\mathrm{h}}^{-1}

keeps the spectrum within the signed, unaliased phase interval.
The script reports this value as the ``Initial unaliased time bound`` in its pre-simulation settings.
Using the active-space reference from :doc:`Putting the problem on qubits <04_putting_the_problem_on_qubits>`, :math:`E_{\mathrm{ref}}=-9.653276065987\ E_{\mathrm{h}}`, this initial time gives the implementation phase fraction

.. math::

   \varphi_{\mathrm{bound}}
   =\left(\frac{-t_{\mathrm{bound}}E_{\mathrm{ref}}}{2\pi}\right)\bmod 1
   \approx0.246129297014.

The script reports this value as the ``Reference phase at initial time bound``.
The nearest six-bit fraction is :math:`16/64=0.25`, represented by ``010000``.
Its signed angle is :math:`2\pi(16/64)=+\pi/2`.
Finally, we can choose the evolution time so that this grid point reconstructs an energy :math:`\delta=0.001\ E_{\mathrm{h}}` above the known reference:

.. math::

   t
   =\frac{-\pi/2}{E_{\mathrm{ref}}+\delta}
   =0.162738437655\ E_{\mathrm{h}}^{-1}.

The script reports this adjusted value as the ``Selected evolution time``.
Using this time, the reference phase fraction is approximately :math:`0.250025901`, only about :math:`2.59\times10^{-5}` above the selected grid point.
The grid point therefore reconstructs an active energy exactly :math:`1\ \mathrm{m}E_{\mathrm{h}}` above the classical reference to the displayed precision.

**Please note**:  this use of the already known classical energy is circular.
It is useful for this tutorial, but it is not a generally available strategy when the target energy is unknown.
For the chosen :math:`t`, adjacent energies represented by the six-bit phase grid differ by approximately :math:`0.6033\ E_{\mathrm{h}}`, not :math:`0.001\ E_{\mathrm{h}}`.
The smaller value, :math:`0.001\ E_{\mathrm{h}}` or :math:`1\ \mathrm{m}E_{\mathrm{h}}`, is the accuracy target adopted for this tutorial.
The question below asks why one grid point can nevertheless reconstruct this particular reference energy within that target.

.. admonition:: Why does six-bit phase estimation meet a :math:`1\ \mathrm{m}E_{\mathrm{h}}` target here even though adjacent grid points are much farther apart?
   :class: quiz-question
   :collapsible: closed

   The classically known reference energy was used to tune the evolution time so the target lies almost exactly on one six-bit grid point.
   Six bits do not provide :math:`\mathrm{m}E_{\mathrm{h}}` resolution for arbitrary energies with this evolution time.

.. _tutorial-qubit-measurement:

Qubit measurement and shots
===========================

Immediately before measurement, the readout ancilla has a state

.. math::

   \vert q\rangle
   =\gamma_0\vert0\rangle+\gamma_1\vert1\rangle,
   \qquad
   |\gamma_0|^2+|\gamma_1|^2=1.

The coefficients :math:`\gamma_0` and :math:`\gamma_1` are complex probability amplitudes.
They play the same mathematical role as the determinant coefficients :math:`c_i` and eigenstate amplitudes :math:`a_j` introduced earlier, but here they describe the two computational-basis states of one qubit.

Measuring this qubit in the computational basis returns one classical bit.
The probability of measuring zero is :math:`|\gamma_0|^2`, and the probability of measuring one is :math:`|\gamma_1|^2`.
One preparation, circuit execution, and measurement is called a *shot*.

A single shot samples one outcome; it does not reveal the probabilities themselves.
Repeating the same circuit from a freshly prepared state produces counts whose relative frequencies estimate the probabilities.
For example, 70 zero outcomes among 100 shots estimate the probability of zero as :math:`0.70`.

In this :term:`IQPE` implementation, phase kickback, feedback, and the final H gate make the readout-ancilla probabilities depend on the phase bit being measured.
The circuit uses an odd number of shots and selects the majority outcome as that iteration's bit.
It then uses this classical bit to set the feedback rotation for the next iteration.

The loop executes iteration :math:`k=0` first.
This iteration applies the largest controlled power and estimates the least-significant phase bit.
Later iterations proceed toward the most-significant bit.

.. _tutorial-phase-bits:

One phase bit at a time
===============================

Standard phase estimation uses several readout ancillas and an inverse quantum Fourier transform to obtain a complete phase in one coherent circuit.
The QDK/Chemistry iterative implementation (:term:`IQPE`) instead reuses a single readout ancilla across a sequence of independently executed circuits.

For iterations over :math:`k=0,1,\ldots,m-1`, the circuit builder uses the controlled power

.. math::

   U^{2^{m-k-1}}.

With :math:`m=6`, the six iteration circuits therefore apply powers :math:`32,16,8,4,2,1`.
QDK/Chemistry reverses the measurements from execution order when it constructs the conventional most-significant-bit-first result.
For an input eigenstate, the first H gate prepares the readout ancilla in :math:`(\vert0\rangle+\vert1\rangle)/\sqrt{2}`.
The feedback rotation applies a corrective phase determined by earlier iterations.
The controlled power then produces *phase kickback*: the :math:`\vert1\rangle` branch acquires the eigenphase of :math:`U^{2^{m-k-1}}`, while the :math:`\vert0\rangle` branch does not.
Together, the feedback and kickback phases cancel the contribution already determined by earlier iterations.
After the second H gate, the remaining relative phase changes the probabilities of measuring zero or one, revealing the next phase bit.

If iteration :math:`k` selects bit :math:`b_k`, QDK/Chemistry updates its accumulated feedback angle according to

.. math::

   \Phi_{k+1}=\frac{\Phi_k}{2}+\frac{\pi b_k}{2},
   \qquad \Phi_0=0.

After the last iteration, the reported phase fraction is :math:`\Phi_m/\pi`.
Each circuit performs the following steps:

1. Freshly prepare the selected trial state on the compute register.
2. Apply an H gate to the readout ancilla.
3. Apply a classically determined phase-feedback rotation to that ancilla.
4. Apply the controlled time-evolution power between the readout ancilla and compute register.
5. Apply a second H gate and measure the readout ancilla.

After all iterations, the feedback accumulator determines the final phase fraction.

Each iteration circuit contains twelve compute qubits and one readout ancilla, for thirteen logical qubits in the simulated circuit.
The readout ancilla does not represent an additional molecular spin orbital.

.. admonition:: Why is trial-state preparation included in every IQPE iteration circuit?
   :class: quiz-question
   :collapsible: closed

   Each phase bit is measured by executing a separate circuit, and every shot begins with newly allocated qubits in the all-zero state.
   The state-preparation logical circuit must therefore reload the trial state before each controlled evolution.

The script configures the native iterative circuit builder and first-order Trotter unitary through nested :class:`~qdk_chemistry.data.AlgorithmRef` objects:

.. literalinclude:: ../../_static/examples/python/tutorial_run_iqpe.py
   :language: python
   :start-after: # start-cell-iqpe-settings
   :end-before: # end-cell-iqpe-settings

.. _tutorial-iqpe-numerical-controls:

Numerical controls
=============================

Five controls determine the approximation and sampling behavior of this workflow:

:ref:`Evolution time <tutorial-energy-to-phase-encoding>`
   The value :math:`t` sets the signed energy interval and spacing of the phase grid, as described above.
   Here it is tuned with the known classical reference to produce a :math:`1\ \mathrm{m}E_{\mathrm{h}}` grid error.
:doc:`Hamiltonian simulation <../../user/comprehensive/algorithms/hamiltonian_unitary_builder>`
   The qubit Hamiltonian is a sum of Pauli terms that generally do not commute.
   A first-order `Trotter product formula <https://en.wikipedia.org/wiki/Lie_product_formula>`_ approximates evolution under that sum by applying the exponential of each Pauli term in sequence.
   For :math:`\hat H=\sum_\ell h_\ell P_\ell`, using :math:`r` Trotter divisions gives

   .. math::

      e^{-i\hat Ht}
      \approx
      \left[\prod_\ell e^{-ih_\ell P_\ell t/r}\right]^r.

   One division (:math:`r=1`) is used for each base evolution in this tutorial.
   Increasing :math:`r` shortens each simulated time step and generally reduces product-formula error, but repeats the Pauli-term sequence more times and increases circuit cost.
   The repeated-power strategy implemented by QDK/Chemistry constructs each controlled power :math:`U^{2^{m-k-1}}` by repeating that same approximate base unitary, preserving one consistent approximation across the IQPE iterations.
:ref:`Phase bits <tutorial-phase-bits>`
   Six bits produce six iteration circuits and :math:`2^6=64` representable phase fractions.
   Increasing this count refines the grid but causes the largest controlled power, circuit size, and simulator runtime to grow exponentially for the repeated-power strategy.
:ref:`Shots per bit <tutorial-qubit-measurement>`
   Each iteration circuit is executed three times.
   The odd shot count prevents a tied bit vote, but finite sampling can still select the less probable bit.
:ref:`Complete runs <tutorial-complete-iqpe-run>`
   The full six-bit procedure is repeated twenty times with simulator seeds 42 through 61.
   Each complete run returns one reconstructed bitstring and energy, and the final estimate uses the most frequent complete bitstring.

The default workflow therefore executes :math:`6\times3\times20=360` iteration-circuit shots.
Phase-grid error, Trotter error, and sampling variation have different causes and should not be combined with basis-set or active-space model error.

.. admonition:: Which control changes energy-grid spacing without changing the molecular Hamiltonian?
   :class: quiz-question
   :collapsible: closed

   The number of phase bits changes how finely the phase interval is discretized.
   The evolution time also rescales the grid in energy units, but it simultaneously changes the unaliased energy interval and the simulated evolution.

IQPE circuit visualization
==========================

Before running the simulator, open :download:`tutorial_visualize_iqpe_circuit.ipynb <../../_static/examples/python/tutorial_visualize_iqpe_circuit.ipynb>` in Visual Studio Code.
Choose **Select Kernel**, select **Python Environments**, and choose the ``.venv`` environment created in :doc:`Before you begin <00_before_you_begin>`.
Then select **Run All** to construct the six iteration circuits and render the shortest, power-one circuit.
The Jupyter notebook does not execute a quantum simulation.

The rendered circuit is still long because it contains the four-determinant state preparation and controlled first-order Trotter evolution for a 247-term Hamiltonian.
Before opening the answers below, trace the operations on each wire to infer its role and compare the dimensions reported for all six iteration circuits.
Record the evidence you used in the Jupyter notebook's interpretation task.

.. admonition:: How can you identify the readout ancilla in the rendered circuit?
   :class: quiz-question
   :collapsible: closed

   The q0 wire receives the H gates and feedback rotation, controls the Hamiltonian evolution, and is measured to obtain the phase bit.
   The other twelve wires hold the prepared molecular state and form the compute register.

.. admonition:: Why do all six iteration circuits have the same width but different lengths?
   :class: quiz-question
   :collapsible: closed

   Every iteration uses the same twelve-qubit compute register and one readout ancilla, so each circuit has thirteen logical qubits.
   Different controlled powers repeat the approximate time-evolution unitary different numbers of times, changing the logical gate count rather than the register size.

.. _tutorial-complete-iqpe-run:

Iterative phase estimation
==============================

The script prepares the mapped Hamiltonian and four-determinant trial state, applies the reference-guided evolution-time choice, and constructs all six iteration circuits before starting any simulation.
This separation lets you inspect settings and circuit counts without accidentally repeating the expensive calculation.

The shortest iteration circuit uses controlled :math:`U` rather than a higher repeated power.
It still contains the state-preparation operations, twelve-wire compute register, one-wire readout ancilla, feedback rotation, controlled Trotter evolution, and ancilla measurement shared by every iteration.
The companion Jupyter notebook introduced below renders this circuit without simulating it.

One complete IQPE run then invokes the native phase-estimation algorithm with one simulator seed:

.. literalinclude:: ../../_static/examples/python/tutorial_run_iqpe.py
   :language: python
   :start-after: # start-cell-run-iqpe
   :end-before: # end-cell-run-iqpe

Each iteration contributes one measured phase bit to the complete IQPE result.

.. admonition:: How does IQPE use the result from each iteration to construct the final bitstring and phase fraction?
   :class: quiz-question
   :collapsible: closed

   The majority measurement for one iteration selects a phase bit, which updates the classical phase feedback used by the next iteration.
   After all six iterations, the feedback calculation combines the measured bits into one phase fraction.
   The script writes that fraction as a conventional six-bit string, with the most significant bit first.

Repeated complete runs
======================

One complete run can differ from another because every phase bit is selected from a finite number of simulator shots and the trial state contains several Hamiltonian eigenstates.
The workflow therefore repeats the complete six-bit procedure with twenty deterministic simulator seeds.

The final aggregation rule selects the most frequent complete bitstring, or *mode*.
This differs from the majority vote used inside one complete run: a per-bit majority chooses one bit from three shots, whereas the complete-run mode chooses one reconstructed bitstring from twenty runs.
If several bitstrings tie for the highest count, the script reports that no unique mode exists instead of silently choosing one.

.. admonition:: Why should the final aggregation use complete bitstrings rather than vote on each bit across complete runs?
   :class: quiz-question
   :collapsible: closed

   Each complete bitstring represents one phase-grid point and its corresponding energy.
   Voting independently on bits could assemble a bitstring that was never produced by any complete run and would discard the observed joint distribution.

Molecular energy reconstruction
================================

After the repeated complete runs, the script selects the bitstring observed most often.
Interpret this bitstring as a binary integer :math:`b`.
If the calculation measures :math:`m` phase bits, convert :math:`b` to the phase fraction

.. math::

   \varphi=\frac{b}{2^m}.

QDK/Chemistry converts :math:`2\pi\varphi` to its equivalent signed angle :math:`\alpha` between :math:`-\pi` and :math:`\pi`.
Negating that angle and dividing by the evolution time maps the measured phase to the active-space energy:

.. math::

   E_{\mathrm{active}}^{\mathrm{IQPE}}=\frac{-\alpha}{t}.

This estimates an eigenvalue of the qubit Hamiltonian, not yet the selected-space molecular total.
Finite phase resolution, sampling, and product-formula time evolution all contribute error.
As :doc:`Putting the problem on qubits <04_putting_the_problem_on_qubits>` explains, the mapper does not include the core energy in the qubit Hamiltonian.
The core energy contains the nuclear repulsion and the constant contribution from frozen inactive orbitals.
Because these contributions are not measured by phase estimation, the script adds them classically:

.. math::

   E_{\mathrm{total}}^{\mathrm{IQPE}}
   =E_{\mathrm{active}}^{\mathrm{IQPE}}+E_{\mathrm{core}}.

Finally, compare this reconstructed total with the :term:`CASCI` energy of the same selected-space Hamiltonian:

.. math::

   \Delta E_{\mathrm{algorithm}}
   =E_{\mathrm{total}}^{\mathrm{IQPE}}-E_{\mathrm{CASCI}}.

The workflow meets the teaching target when :math:`\lvert\Delta E_{\mathrm{algorithm}}\rvert\leq1\ \mathrm{m}E_{\mathrm{h}}`.
This comparison evaluates the configured quantum algorithm against its classical reference; it does not measure basis-set or active-space model error.

.. admonition:: Which energy comparison determines whether the IQPE workflow meets the teaching target?
   :class: quiz-question
   :collapsible: closed

   Compare the reconstructed IQPE total energy with the :term:`CASCI` energy of the same selected active-space Hamiltonian.
   Comparing with experiment or a larger orbital space would mix algorithmic error with model error.

The complete workflow
=====================

With the Python environment from :doc:`Before you begin <00_before_you_begin>` active, run the script from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_run_iqpe.py

The script prints its settings before simulation and reports progress for every complete run, including the seed, bitstring, total energy, error, and elapsed time.
A successful run completes all twenty runs and prints the complete-run bitstring counts, most frequent bitstring, component energies, reconstructed total, reference energy, and signed error.

.. admonition:: What bitstring distribution and energy estimate did the script produce?
   :class: quiz-question
   :collapsible: closed

   The bitstring ``010000`` appeared 19 times and ``001111`` appeared once, so ``010000`` was the most frequent result.
   It produced an active-space energy of :math:`-9.652276065987\ E_{\mathrm{h}}` and a reconstructed total of :math:`-108.770051792909\ E_{\mathrm{h}}` after adding the core energy.

.. admonition:: Does the result meet the teaching target, and what does that establish?
   :class: quiz-question
   :collapsible: closed

   The reconstructed total is :math:`+1\ \mathrm{m}E_{\mathrm{h}}` above the selected-space :term:`CASCI` reference, meeting the teaching target at its boundary.
   That offset was deliberately set by the reference-guided phase-grid alignment, while Trotter approximation and finite sampling can still affect which bitstring is selected.
   The result therefore validates this configured teaching workflow; it does not remove molecular-model error, establish agreement with experiment, or demonstrate quantum advantage.

Record all settings, bitstring counts, energies, error, and observed runtime in the :ref:`phase-estimation section of the lab notebook <lab-notebook-phase-estimation>`.
Then complete the :ref:`conclusion <lab-notebook-conclusion>` by separating basis-set, active-space, and quantum-algorithm limitations.

Knowledge check
========================

.. admonition:: What changes if the number of phase bits increases while the repeated-power strategy remains fixed?
   :class: quiz-question
   :collapsible: closed

   The phase grid becomes finer, but an additional iteration circuit is required and the largest controlled-unitary power doubles.
   For repeated-power Trotter evolution, that larger power increases circuit size and simulator runtime substantially.

.. admonition:: Would increasing shots per bit make the phase grid finer?
   :class: quiz-question
   :collapsible: closed

   No.
   More shots can make each bit majority more stable, but grid spacing is controlled by the evolution time and number of phase bits.

What you accomplished
=====================

You completed an end-to-end molecular-energy workflow: defining stretched N\ :sub:`2` in an orbital basis, selecting an active space, mapping its Hamiltonian to qubits, preparing a multiconfigurational trial state, simulating iterative phase estimation, and reconstructing the molecular total energy by adding the core energy.

The final comparison shows that this configured :term:`IQPE` workflow reproduces the matching selected-space :term:`CASCI` reference within the tutorial's :math:`1\ \mathrm{m}E_{\mathrm{h}}` target.
It does not establish agreement with experiment or quantum advantage, because basis-set, active-space, and quantum-algorithm limitations remain distinct.

Your lab notebook records where those choices enter and what evidence supports the result.
A useful next investigation would change one layer at a time: enlarge the molecular model, choose a different trial state, or vary the phase-estimation controls, then identify which accuracy and cost measures respond.

Further reading
===============

- :doc:`Phase estimation <../../user/comprehensive/algorithms/phase_estimation>`
- :doc:`Phase-estimation circuit builders <../../user/comprehensive/algorithms/qpe_circuit_builder>`
- :doc:`Hamiltonian unitary builders <../../user/comprehensive/algorithms/hamiltonian_unitary_builder>`
- :doc:`Circuit execution <../../user/comprehensive/algorithms/circuit_executor>`
- :doc:`Phase-estimation results <../../user/comprehensive/data/qpe_result>`
