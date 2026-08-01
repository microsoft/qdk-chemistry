Ground-state molecular energies with quantum phase estimation
#############################################################

.. todo::

   Revisit this chapter after drafting the later sections, possibly trimming the overview and moving detailed explanations closer to where they are used.

This tutorial uses `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`_ (:term:`QPE`) and the Quantum Development Kit (:term:`QDK`) Chemistry library, called :term:`QDK`/Chemistry in this documentation, to estimate the ground-state electronic energy of a stretched nitrogen molecule, N\ :sub:`2`.
It is intended for advanced undergraduate and early-stage graduate students who have introductory knowledge of quantum computing and chemistry.
:doc:`Before you begin <00_before_you_begin>` describes the prerequisites, software environment, and cumulative lab notebook assignment.

Quantum chemistry background
============================

`Electronic structure theory <https://en.wikipedia.org/wiki/Quantum_chemistry#Electronic_structure>`_ applies quantum mechanics to the electrons in atoms and molecules.
For a system whose Hamiltonian does not depend explicitly on time, its stationary electronic states satisfy the `time-independent Schrödinger equation <https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation#Time-independent_equation>`_:

.. math::

   \hat{H} \vert\Psi\rangle = E \vert\Psi\rangle.

The `wavefunction <https://en.wikipedia.org/wiki/Wave_function>`_ :math:`\vert\Psi\rangle` is a mathematical description of an electronic state.
The Hamiltonian operator :math:`\hat{H}` represents the energy of the system, and the eigenvalue :math:`E` is the energy associated with that wavefunction.
The possible solutions include the ground state, which has the lowest energy, and excited states with higher energies.
This tutorial seeks an approximate solution for the ground-state energy of stretched N\ :sub:`2`.

This tutorial uses the `Born--Oppenheimer approximation <https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation>`_, which holds the nuclei at fixed positions while solving for the electrons.
At a chosen molecular geometry, the electronic Hamiltonian includes the electron kinetic energy, electron--nucleus attraction, and electron--electron repulsion.
The repulsion among the fixed nuclei contributes separately to the total energy.
The computational cost of solving its Schrödinger equation grows rapidly with the number of electrons and orbitals, so practical calculations use approximations.
:doc:`Energy and accuracy <01_energy_and_accuracy>` defines the target energy and the different error comparisons used in the tutorial.
:doc:`Describing the molecule <02_describing_the_molecule>` then specifies the molecular geometry and basis set and obtains the first Hartree--Fock energy.

.. _tutorial-orbitals-and-determinants:

One useful representation starts with molecular orbitals, which describe the spatial part of one-electron states across the molecule.
In the spin basis used in this tutorial, the two possible `spin projections <https://en.wikipedia.org/wiki/Spin_(physics)#Spin_projection_quantum_number_and_multiplicity>`_ are labeled :math:`\alpha` and :math:`\beta`.
A spin orbital combines one spatial molecular orbital with either the :math:`\alpha` or :math:`\beta` spin function.
Each spin orbital can be occupied by at most one electron.
Each spatial orbital therefore corresponds to two spin orbitals and can accommodate at most two electrons, one with each spin projection.
An `electron configuration <https://en.wikipedia.org/wiki/Electron_configuration>`_ specifies which spin orbitals are occupied.
A `Slater determinant <https://en.wikipedia.org/wiki/Slater_determinant>`_ constructs a valid many-electron wavefunction for one configuration and enforces the `Pauli exclusion principle <https://en.wikipedia.org/wiki/Pauli_exclusion_principle>`_.
The `Hartree--Fock method <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_ approximates the molecular wavefunction with one optimized Slater determinant.
This single-configuration approximation is an important starting point, but it becomes inadequate when several electron configurations make substantial contributions to the state.
When several configurations contribute substantially, the ground-state wavefunction must combine their Slater determinants.
A coefficient for each determinant specifies its contribution to the wavefunction.
This need for several important configurations is called `static correlation <https://en.wikipedia.org/wiki/Electronic_correlation>`_, and the resulting wavefunction is called multi-configurational.

The number of possible configurations grows rapidly with the number of electrons and orbitals.
An `active space <https://en.wikipedia.org/wiki/Complete_active_space>`_ selects the electrons and orbitals whose occupations vary among the determinants in the multi-configurational wavefunction :cite:`Stein2016,Stein2019`.
The remaining orbital occupations are fixed, and their energy contributions are tracked separately.
The active space therefore controls a central tradeoff in this tutorial: a larger space can describe more correlation, but it also produces a larger calculation and eventually requires more qubits.
:doc:`Choosing the active space <03_choosing_the_active_space>` develops the correlated molecular model and selects the determinants that can contribute to its wavefunction.

Why quantum computing?
======================

The difficulty of a correlated calculation is not only the cost of evaluating one determinant.
The calculation must represent and combine many possible determinants.
For :math:`N` active electrons distributed among :math:`M` active spin orbitals, the determinant basis contains

.. math::

   N_{\mathrm{det}} = \binom{M}{N}

states before applying additional spin or molecular symmetries.
At a fixed ratio of electrons to orbitals, this number grows exponentially with :math:`M`.
`Full configuration interaction <https://en.wikipedia.org/wiki/Full_configuration_interaction>`_ combines all allowed determinants in a chosen orbital space and therefore becomes impractical as that orbital space grows.
Approximate classical methods can reach larger spaces by retaining or compressing selected information, but their accuracy depends on preserving the correlations that matter.
When many configurations contribute substantially, omitted correlations can cause significant errors in the calculated energy.

.. _tutorial-occupation-encoding:

An occupation-number encoding assigns one qubit to each spin orbital to record whether that orbital is unoccupied or occupied.
The amplitudes of the possible occupations are represented by the quantum state rather than stored as an explicit classical list.

.. _tutorial-compute-register:

A *quantum register* is a group of qubits treated as one part of a computation because they serve the same role.
The occupation-encoding qubits form the *compute register*, which stores the encoded fermionic state and is acted on by the qubit Hamiltonian.
`Ancilla qubits <https://en.wikipedia.org/wiki/Ancilla_bit>`_ are additional qubits used for tasks such as control, temporary workspace, or readout; they do not represent additional spin orbitals and are counted separately.
:doc:`Putting the problem on qubits <04_putting_the_problem_on_qubits>` maps the selected electronic Hamiltonian to a qubit Hamiltonian and determines the size of this register.
During Hamiltonian time evolution, each energy eigenstate acquires a phase determined by its energy.
Quantum phase estimation estimates this phase and converts it to an energy eigenvalue :cite:`AspuruGuzik2005,vonBurg2021`.
This representation does not make the calculation automatically efficient.
Preparing a useful state, implementing time evolution, correcting errors, and repeating measurements can all require substantial resources.

Before running :term:`QPE`, a state-preparation circuit loads an approximate wavefunction, called the trial state, onto the quantum register.
The trial state can be expressed as a combination of the eigenstates of the qubit Hamiltonian.
:term:`QPE` can return the energy of any eigenstate represented in that combination; it does not independently find or prepare the ground state.

.. _tutorial-trial-state-fidelity:

Let :math:`\vert\Psi_0\rangle` denote the exact ground state of the active-space Hamiltonian and :math:`\vert\Psi_{\mathrm{trial}}\rangle` denote the prepared trial state.
Their squared overlap is the ground-state fidelity

.. math::

   F = \left\vert \langle\Psi_0 \vert \Psi_{\mathrm{trial}}\rangle \right\vert^2.

Fidelity measures the weight of the ground state in the trial state.
In an ideal coherent phase-estimation measurement that uses one prepared system state to produce one complete phase result, :math:`F` is the probability of sampling the ground-state eigenphase.
The iterative implementation used by this tutorial instead prepares a new system state for each phase-bit circuit, so fidelity influences its bit statistics but does not by itself determine a complete-run success probability or trial count.
For this classically tractable teaching example, the classical active-space calculation supplies the important determinants and coefficients used to construct the trial state and later validate the quantum result.
:doc:`Preparing the trial state <05_preparing_the_trial_state>` develops overlap and fidelity, and :doc:`Iterative quantum phase estimation <06_iterative_phase_estimation>` explains how the energy is measured.

Why stretched nitrogen?
=======================

In `molecular orbital theory <https://en.wikipedia.org/wiki/Molecular_orbital_theory>`_, occupying a bonding orbital stabilizes a bond, whereas occupying the corresponding antibonding orbital opposes that stabilization.
Near the `equilibrium bond length <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=1000>`_ of :math:`1.097685\ \text{Å}` for N\ :sub:`2`, one electron configuration dominates, and its Slater determinant provides a useful first approximation to the ground-state wavefunction.
As the bond stretches, configurations with different occupations of the bonding and antibonding orbitals become comparable in importance.
Therefore, no single Slater determinant describes the stretched molecule adequately.

Tutorial scope and structure
============================

The stretched N\ :sub:`2` calculation in this tutorial is small enough to solve exactly within its selected active space on a classical computer.
The quantum circuits are also executed on a classical simulator.
The tutorial therefore does not demonstrate quantum advantage.
Instead, the classical result provides a reference for validating each step of a workflow intended for future fault-tolerant quantum computers and larger active spaces.
Complete active space configuration interaction (:term:`CASCI`) performs full configuration interaction within the selected active space rather than across all orbitals in the molecular model.
The final quantum calculation is compared with the :term:`CASCI` energy of the same selected active-space Hamiltonian.
The tutorial uses 1 milliHartree (often referred to as "chemical accuracy") as a teaching target for this algorithmic comparison; however, meeting that target does not establish agreement with experiment or remove basis-set and active-space errors.

Each required chapter introduces one stage of the calculation, provides a testable Python example where appropriate, and ends with questions and an assignment.
Most examples are short.
The final circuit simulation is the only intentionally long required example and is expected to take approximately 20 minutes on a typical student laptop.
The actual duration depends on the computer.
Optional chapters address physical resource estimation, model Hamiltonians, and software interoperability without adding dependencies to the required workflow.

Cumulative assignment
=====================

Maintain a :doc:`lab notebook <lab_notebook>` throughout the tutorial.
Each required chapter asks you to record the inputs, decisions, results, and interpretations needed to reproduce the final calculation.
The notebook turns the chapter learning objectives into evidence that you can inspect and explain.
At the end, use the completed notebook to explain the final energy estimate and the limitations that remain.

.. toctree::
   :maxdepth: 2

   00_before_you_begin
   lab_notebook
   01_energy_and_accuracy
   02_describing_the_molecule
   03_choosing_the_active_space
   04_putting_the_problem_on_qubits
   05_preparing_the_trial_state
   06_iterative_phase_estimation
   optional/index
