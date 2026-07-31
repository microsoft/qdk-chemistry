Putting the problem on qubits
#############################

.. admonition:: Chapter focus
   :class: chapter-focus

   How does the active-space electronic problem become a qubit problem?

Learning objectives
===================

After completing this chapter, you will be able to:

- Identify the terms in the active-space electronic Hamiltonian.
- Explain why fermionic operators require an encoding on qubits.
- Describe how the Jordan--Wigner transformation preserves fermionic signs.
- Construct a qubit Hamiltonian with native :term:`QDK`/Chemistry tools.
- Determine how many qubits encode the selected active-space fermionic state.
- Explain why the core energy must be added to a qubit-Hamiltonian result.

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

   Complete :ref:`lab-notebook-qubits`.
   Calculate the number of qubits needed to encode the selected active-space fermionic state before verifying it with code.
   Record the core energy separately and identify every quantity excluded from this qubit count.

Example download
====================

Download :download:`tutorial_map_n2_to_qubits.py <../../_static/examples/python/tutorial_map_n2_to_qubits.py>` and save it in the tutorial working directory that contains ``tutorial_choose_active_space.py`` from :doc:`Choosing the active space <03_choosing_the_active_space>`.
Open both files in Visual Studio Code and review the complete mapping script, including imports and setup code omitted from the excerpts below.
The script imports the tested Chapter 3 workflow so that both chapters use the same selected active space.

The active-space Hamiltonian
======================================

:ref:`Choosing the active space <tutorial-active-space-definition>` chapter partitioned the `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital_theory>`_ into inactive, active, and virtual spaces.
As introduced in :ref:`Orbitals and determinants <tutorial-orbitals-and-determinants>`, each `spin orbital <https://en.wikipedia.org/wiki/Spin_orbital>`_ combines a spatial orbital with an :math:`\alpha` or :math:`\beta` spin function.
Occupations may vary among the active spin orbitals in the correlated `wavefunction <https://en.wikipedia.org/wiki/Wave_function>`_, while inactive spatial orbitals remain doubly occupied and virtual spatial orbitals remain empty.
The active-space Hamiltonian acts only on the active spin orbitals; interactions with frozen inactive orbitals contribute to its effective one-electron terms and to the separately calculated core energy.
Written in `second quantization <https://en.wikipedia.org/wiki/Second_quantization>`_, this Hamiltonian uses operators that change orbital occupations.
The `creation operator <https://en.wikipedia.org/wiki/Creation_and_annihilation_operators>`_ :math:`\hat{a}_p^\dagger` adds an electron to spin orbital :math:`p`, while the `annihilation operator <https://en.wikipedia.org/wiki/Creation_and_annihilation_operators>`_ :math:`\hat{a}_p` removes one.
The Hamiltonian contains one-electron and two-electron terms:

.. math::

   \hat{H}_{\mathrm{active}}
   = \sum_{pq} h_{pq}\,\hat{a}_p^\dagger \hat{a}_q
   + \frac{1}{2}\sum_{pqrs} g_{pqrs}\,
     \hat{a}_p^\dagger \hat{a}_q^\dagger \hat{a}_s \hat{a}_r.

The indices :math:`p,q,r,s` label active spin orbitals, or equivalently fermionic modes.
The coefficients :math:`h_{pq}` describe one-electron effects, including kinetic energy, attraction to the nuclei, and the effective interaction with frozen inactive electrons.
The coefficients :math:`g_{pqrs}` describe repulsion between pairs of active electrons.

The creation and annihilation operators automatically produce zero when creation acts on an already occupied spin orbital or annihilation acts on an unoccupied one.
Together, these operators connect the `Slater determinants <https://en.wikipedia.org/wiki/Slater_determinant>`_ introduced in :ref:`Orbitals and determinants <tutorial-orbitals-and-determinants>` that can contribute to the active-space wavefunction.

The mapping script reconstructs this selected active-space Hamiltonian from the orbitals produced by Chapter 3:

.. literalinclude:: ../../_static/examples/python/tutorial_map_n2_to_qubits.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-active-hamiltonian
   :end-before: # end-cell-active-hamiltonian

Why orbital occupations are not enough
======================================

The operator products in the active-space Hamiltonian act on Slater determinants.
For example, in the one-electron term :math:`\hat{a}_p^\dagger\hat{a}_q`, the rightmost operator first removes an electron from spin orbital :math:`q`, and then the leftmost operator places it in spin orbital :math:`p`.
The two-electron terms describe corresponding changes involving two electrons.
The orbital occupations determine whether such a change is allowed.
When it is allowed, restoring the resulting determinant to the chosen standard orbital order can introduce a minus sign.

Both this ordering sign and the :ref:`exchange effect discussed in Chapter 2 <tutorial-hartree-fock-wavefunction>` follow from the antisymmetry of fermionic wavefunctions.
Applying the Hamiltonian's operators in a different order can therefore change the sign of the resulting contribution.
These signs are already part of the electronic-structure problem; mapping the Hamiltonian to qubits must preserve them.

As the :ref:`tutorial introduction explains <tutorial-occupation-encoding>`, a qubit can store whether one spin orbital is unoccupied or occupied.
That records the orbital occupations, but operations acting on different qubits commute and therefore do not automatically reproduce fermionic signs.
Simply replacing each creation or annihilation operator with an operation on its corresponding qubit would change the Hamiltonian's matrix elements.
A fermion-to-qubit mapping must represent both the occupations and the signs caused by fermionic operator ordering.

.. rubric:: Optional: the sign rule in symbols

The `anticommutation relations <https://en.wikipedia.org/wiki/Canonical_anticommutation_relation>`_ summarize this sign rule compactly.
With the anticommutator defined as :math:`\{\hat{A},\hat{B}\}=\hat{A}\hat{B}+\hat{B}\hat{A}`, fermionic creation and annihilation operators satisfy

.. math::

   \{\hat{a}_p,\hat{a}_q\}=0,
   \qquad
   \{\hat{a}_p,\hat{a}_q^\dagger\}=\delta_{pq},

For different spin orbitals, the zero on the right means that exchanging the order of two operators changes the sign of the result.
You will not need to manipulate these relations by hand; the Jordan--Wigner transformation enforces them through the parity strings introduced next.

The Jordan--Wigner transformation
=======================================

The `Jordan--Wigner transformation <https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_ :cite:`Jordan-Wigner1928` assigns each fermionic mode to one qubit.
In this molecular problem, each fermionic mode is one active spin orbital.
To keep the two kinds of labels distinct, let :math:`\ell_p` denote the qubit assigned to fermionic mode :math:`p`; its numerical qubit index is :math:`p`.

.. math::

   \vert 0\rangle_{\ell_p} \longleftrightarrow \text{mode }p\text{ unoccupied},
   \qquad
   \vert 1\rangle_{\ell_p} \longleftrightarrow \text{mode }p\text{ occupied}.

The occupation operator becomes

.. math::

   \hat{n}_p=\hat{a}_p^\dagger\hat{a}_p=\frac{I_{\ell_p}-Z_{\ell_p}}{2}.

The creation and annihilation operators become

.. math::

   \hat{a}_p^\dagger
   = \frac{1}{2}\left(\prod_{j=0}^{p-1}Z_{\ell_j}\right)(X_{\ell_p}-iY_{\ell_p}),
   \qquad
   \hat{a}_p
   = \frac{1}{2}\left(\prod_{j=0}^{p-1}Z_{\ell_j}\right)(X_{\ell_p}+iY_{\ell_p}).

The `Pauli operators <https://en.wikipedia.org/wiki/Pauli_matrices>`_ :math:`X_{\ell_p}`, :math:`Y_{\ell_p}`, and :math:`Z_{\ell_p}` act on qubit :math:`\ell_p`.
The combinations :math:`(X_{\ell_p}-iY_{\ell_p})/2` and :math:`(X_{\ell_p}+iY_{\ell_p})/2` raise :math:`\vert 0\rangle_{\ell_p}` to :math:`\vert 1\rangle_{\ell_p}` and lower :math:`\vert 1\rangle_{\ell_p}` to :math:`\vert 0\rangle_{\ell_p}`, respectively.
Within the product, :math:`j` indexes the lower fermionic modes and :math:`Z_{\ell_j}` acts on the qubit assigned to mode :math:`j`.
The product of :math:`Z` operators records the parity of occupied lower-indexed modes.
Each occupied lower-indexed mode contributes an eigenvalue of :math:`-1`, so the product is negative when an odd number of those modes is occupied.
Acting on mode :math:`p` crosses the occupied lower-indexed modes in the chosen fermionic ordering, with each crossing contributing a minus sign.
The parity string supplies their combined sign, so the mapped operators satisfy the fermionic anticommutation relations.

Because the parity strings depend on mode ordering, the ordering must be specified: :term:`QDK`/Chemistry places all active :math:`\alpha` modes before all active :math:`\beta` modes, a convention called *blocked ordering*.
Another ordering would change the parity strings but not the energy spectrum.

.. admonition:: Why does Jordan--Wigner need a string of Pauli Z operators?
   :class: quiz-question
   :collapsible: closed

   The :math:`Z` string records the parity of lower-indexed fermionic modes.
   Its sign ensures that encoded creation and annihilation operators anticommute even though operators on different qubits commute.

Qubits for the encoded fermionic state
======================================

The :ref:`compute register introduced in the tutorial overview <tutorial-compute-register>` stores the encoded active-space fermionic state.
A qubit in this register is a *compute-register qubit*, shortened below to *compute qubit*.
Every spatial orbital corresponds to one :math:`\alpha` spin orbital and one :math:`\beta` spin orbital.
For :math:`n_o` active spatial orbitals, Jordan--Wigner therefore requires

.. math::

   n_{\mathrm{spin}}=2n_o,
   \qquad
   n_{\mathrm{compute}}=n_{\mathrm{spin}}.

Calculate the compute-register count from the active space selected in Chapter 3 before running the script:

Because this restricted calculation has matching :math:`\alpha` and :math:`\beta` active spaces, the ``spatial_indices`` helper reads the :math:`\alpha`-channel entries, one for each active spatial orbital.

.. literalinclude:: ../../_static/examples/python/tutorial_map_n2_to_qubits.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-count-qubits
   :end-before: # end-cell-count-qubits

Its size does not include phase-estimation ancillas, temporary workspace qubits, error-correction overhead, or physical qubits.
Those additional resources depend on later algorithm and hardware choices rather than on the Jordan--Wigner occupation encoding alone.

.. admonition:: How many compute qubits does the selected active space require?
   :class: quiz-question
   :collapsible: closed

   The selected space contains four active spatial orbitals and therefore eight active spin orbitals.
   Jordan--Wigner uses one qubit per spin orbital, so the compute register contains eight qubits.

Record your predicted compute-register qubit count and the excluded qubit categories in the :ref:`qubit-representation section of the lab notebook <lab-notebook-qubits>` before continuing.
You will verify the count when you run the mapping script.

.. admonition:: How many compute qubits would Jordan--Wigner require for six active spatial orbitals?
   :class: quiz-question
   :collapsible: closed

   Six spatial orbitals correspond to twelve spin orbitals, so Jordan--Wigner requires twelve compute qubits.
   This count does not include algorithm ancillas or error-correction overhead.

Qubit Hamiltonian in Pauli form
======================================

Substituting the Jordan--Wigner expressions into :math:`\hat{H}_{\mathrm{active}}` produces a weighted sum of Pauli strings:

.. math::

   \hat{H}_{\mathrm{qubit}}=\sum_k c_k P_k,

where coefficient :math:`c_k` multiplies a tensor product :math:`P_k` of :math:`I`, :math:`X`, :math:`Y`, and :math:`Z` operators.
The index :math:`k` labels Pauli terms, not spin orbitals or qubits.
The number of Pauli terms describes the size of this operator representation; it is not a circuit gate count or a physical-resource estimate.
The exact count also depends on the mapper's numerical thresholds because terms with sufficiently small coefficients are omitted.

The script creates a Jordan--Wigner mapping for the active spin orbitals and passes it to the native :term:`QDK`/Chemistry mapper:

.. literalinclude:: ../../_static/examples/python/tutorial_map_n2_to_qubits.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-map-hamiltonian
   :end-before: # end-cell-map-hamiltonian
   :append: print_representative_pauli_terms(qubit_hamiltonian)

The Pauli terms fall into families that connect back to the determinant picture:

.. _tutorial-all-identity-term:

All-identity term
   Acts in the same way on every occupation-basis state and therefore contributes a constant shift to every eigenvalue of the active Hamiltonian.
   This mapped constant is distinct from the core energy stored outside the qubit Hamiltonian.

:math:`I`- and :math:`Z`-only terms
   Are diagonal in the occupation-number basis because each determinant is an eigenstate of every :math:`Z` operator.
   Their signs depend on which spin orbitals are occupied, so together they contribute occupation-dependent one-electron and electron-interaction energies to the diagonal matrix element of each determinant.

Terms containing :math:`X` or :math:`Y`
   Are off-diagonal in the occupation-number basis and connect basis states with different orbital occupations, corresponding to couplings among Slater determinants.
   A complete mapped hopping or excitation operator generally contains a coordinated sum of several Pauli strings, so one displayed string should not be interpreted as an entire chemical excitation by itself.

Printing every Pauli term would obscure these patterns, so the complete script displays the all-identity term, three of the largest :math:`I`- and :math:`Z`-only terms, and four of the largest terms containing :math:`X` or :math:`Y`.
The preview writes, for example, ``X(qubit 1) X(qubit 2) X(qubit 5) X(qubit 6)`` for a tensor product that applies :math:`X` to qubits 1, 2, 5, and 6 and applies :math:`I` to every unlisted qubit.

Core-energy bookkeeping
=======================

The :ref:`selected-space energy from Chapter 3 <tutorial-selected-space-reference>` contains a constant contribution in addition to the active Hamiltonian:

.. math::

   E_{\mathrm{total}}
   = E_{\mathrm{core}}
   + \langle\Psi_{\mathrm{active}}\vert
     \hat{H}_{\mathrm{active}}
     \vert\Psi_{\mathrm{active}}\rangle.

:term:`QDK`/Chemistry stores nuclear repulsion and the constant contribution from frozen inactive orbitals in :math:`E_{\mathrm{core}}`.
The qubit mapper transforms :math:`\hat{H}_{\mathrm{active}}` but does not include :math:`E_{\mathrm{core}}` in the returned qubit operator.
This scalar must therefore be added to the measured active-space eigenvalue to reconstruct the selected-space total energy.

.. admonition:: Why must the core energy be added to the energy from the qubit Hamiltonian?
   :class: quiz-question
   :collapsible: closed

   The qubit mapper encodes only the active fermionic Hamiltonian.
   Nuclear repulsion and constant frozen-orbital contributions are stored separately in the core energy, so adding that scalar reconstructs the selected-space total energy.

This eight-qubit example is small enough to validate the mapping by exact matrix diagonalization.

.. _tutorial-fixed-electron-number-subspace:

A *fixed-electron-number subspace* contains only the occupation-basis states with specified numbers :math:`n_\alpha` and :math:`n_\beta` of active :math:`\alpha` and :math:`\beta` electrons.
The script restricts :math:`\hat{H}_{\mathrm{qubit}}` to the fixed-electron-number subspace with the same electron counts as the :term:`CASCI` calculation.
Before running the script, use the :ref:`determinant-count formula from the active-space calculation <tutorial-determinant-count>` to calculate the number of occupation-basis states in this subspace as :math:`\binom{n_o}{n_\alpha}\binom{n_o}{n_\beta}`, and record your prediction in the :ref:`qubit-representation section of the lab notebook <lab-notebook-qubits>`.
In the integer label for an occupation-basis state, bit :math:`p` records the occupation of mode :math:`p`, with qubit 0 as the least-significant bit.
Blocked ordering therefore places the :math:`n_o` active :math:`\alpha` occupations in the lowest bits and the :math:`n_o` active :math:`\beta` occupations in the next bits; the mask in the code isolates the :math:`\alpha` occupations so the two electron counts can be checked separately.
The lowest eigenvalue in this subspace is obtained directly from the mapped qubit Hamiltonian:

.. literalinclude:: ../../_static/examples/python/tutorial_map_n2_to_qubits.py
   :language: python
   :dedent: 4
   :start-after: # start-cell-validate-mapping
   :end-before: # end-cell-validate-mapping

Adding :math:`E_{\mathrm{core}}` to this mapped active-space eigenvalue gives a total energy that can be compared with the selected-space :term:`CASCI` reference.
Exact diagonalization is practical for this compact teaching example; it is not a scalable method for solving larger qubit Hamiltonians.

Running the mapping
===================

With the Python environment from :doc:`Before you begin <00_before_you_begin>` active, run the complete script from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_map_n2_to_qubits.py

.. admonition:: Does the script confirm the size of the fixed-electron-number subspace?
   :class: quiz-question
   :collapsible: closed

   The subspace with two active :math:`\alpha` and two active :math:`\beta` electrons contains 36 occupation-basis states, matching the 36 determinants in the selected active space.

.. admonition:: What operator size and core energy does the script report?
   :class: quiz-question
   :collapsible: closed

   The mapped Hamiltonian contains 161 Pauli terms on eight compute qubits.
   The separately stored core energy is approximately :math:`-103.702793099333` Hartree.
   These counts use the mapper's default numerical thresholds.

.. admonition:: Does the mapped qubit Hamiltonian reproduce the selected-space algorithmic reference?
   :class: quiz-question
   :collapsible: closed

   The mapped active-space ground-state energy is :math:`-5.261838965737` Hartree.
   Adding the core energy of :math:`-103.702793099333` Hartree gives :math:`-108.964632065071` Hartree when the unrounded values are used.
   This matches the :ref:`selected-space CASCI reference <tutorial-selected-space-reference>` to the reported precision.
   Manually adding the displayed component values gives :math:`-108.964632065070` Hartree; the final-digit discrepancy is caused only by rounding those displayed components.

This agreement validates the Jordan--Wigner mapping and fixed-electron-number subspace construction for this selected Hamiltonian within numerical precision.
Compare the reported qubit count with your prediction, then complete the :ref:`qubit-representation section of the lab notebook <lab-notebook-qubits>` with the encoding, confirmed orbital and qubit counts, Pauli-term count, fixed-electron-number subspace, mapped energy, reconstructed total, and comparison with the selected-space reference.

Further reading
===============

- :doc:`Hamiltonian construction <../../user/comprehensive/algorithms/hamiltonian_constructor>`
- :doc:`Electronic Hamiltonians <../../user/comprehensive/data/hamiltonian>`
- :doc:`Qubit mapping <../../user/comprehensive/algorithms/qubit_mapper>`
- :doc:`Majorana mappings <../../user/comprehensive/data/majorana_mapping>`
- :doc:`Pauli operators <../../user/comprehensive/data/pauli_operator>`
