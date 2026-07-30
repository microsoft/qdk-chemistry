Choosing the active space
#########################

.. note:: Chapter focus

   Which electrons and orbitals must the calculation treat explicitly?

Learning objectives
===================

After completing this chapter, you will be able to:

- Explain why strongly correlated systems require more than one determinant.
- Distinguish inactive, active, and virtual orbitals.
- Select a valence active space with :term:`QDK`/Chemistry.
- Explain how natural orbitals and reduced density matrices support active-space selection.
- Explain how orbital entropies can refine an active-space choice.
- Evaluate the tradeoff between active-space accuracy and problem size.

.. important:: Lab notebook assignment

   Complete :ref:`lab-notebook-active-space`.
   Record the evidence used to select the active space, not only the final orbital and electron counts.
   Explain what the energy comparison establishes within the chosen basis and identify the energy that will serve as the algorithmic reference.

Download the example
====================

Download :download:`tutorial_choose_active_space.py <../../_static/examples/python/tutorial_choose_active_space.py>` and save it in your tutorial working directory.
Open the file in Visual Studio Code and review the complete script, including imports and setup code omitted from the excerpts below.
The script repeats the stretched N\ :sub:`2` Hartree--Fock calculation from :doc:`Describing the molecule <02_describing_the_molecule>` before constructing and refining an active space.

Recognize the limits of one determinant
=======================================

The Hartree--Fock approximation restricts the wavefunction to one optimized Slater determinant.
This description is often a useful starting point near an equilibrium geometry, where one electron configuration dominates the ground-state wavefunction.
Stretching a chemical bond can make several configurations similar in energy because electrons can no longer be assigned adequately to one fixed pattern of occupied and unoccupied orbitals.
The need to combine these important configurations is called *static correlation*.
Stretched N\ :sub:`2` has enough static correlation to make its one-determinant Hartree--Fock wavefunction inadequate.

A configuration interaction (:term:`CI`) wavefunction addresses this limitation by expanding the wavefunction in multiple Slater determinants:

.. math::

   \vert \Psi \rangle = \sum_I C_I \vert \Phi_I \rangle,

where :math:`\vert \Phi_I \rangle` is determinant :math:`I` and :math:`C_I` is its coefficient.
Allowing every possible determinant in all 28 ``cc-pvdz`` spatial orbitals would be unnecessarily expensive for this tutorial.
An active-space model instead restricts which orbital occupations may vary.

Define the active space
=======================

An active-space calculation partitions the spatial molecular orbitals into three groups:

Inactive orbitals
   Remain doubly occupied in every determinant.
   Their electrons contribute to the energy, but their occupations do not vary.
Active orbitals
   May be empty, singly occupied, or doubly occupied in different determinants.
   The calculation treats correlation among the active electrons explicitly.
Virtual orbitals
   Remain empty in every determinant and do not participate explicitly in the correlated calculation.

A complete active space containing :math:`n_e` active electrons in :math:`n_o` active spatial orbitals is written CAS\ :math:`(n_e,n_o)`.
Complete active space configuration interaction (:term:`CASCI`) forms every determinant consistent with those active electron and orbital counts while keeping the molecular orbitals fixed.
Unlike complete active space self-consistent field (:term:`CASSCF`), CASCI does not reoptimize the orbitals.

A useful first choice is a generous space containing the valence electrons and the occupied and unoccupied valence orbitals associated with bond formation and breaking.
The :func:`~qdk_chemistry.utils.compute_valence_space_parameters` utility determines that neutral N\ :sub:`2` has ten valence electrons and eight valence spatial orbitals.
The ``qdk_valence`` selector places these in an initial CAS\ :math:`(10,8)` space:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :start-after: # start-cell-valence-space
   :end-before: # end-cell-valence-space

The script reports active orbital indices 2 through 9, using zero-based indexing.
At this stage, orbitals 0 and 1 are inactive and the remaining 18 of the 28 ``cc-pvdz`` orbitals are virtual.
This chemically motivated space is intentionally larger than the final model so that the next calculation can measure which valence orbitals carry the strongest static-correlation signal.

Compute a correlated active-space wavefunction
==============================================

The active-space selector labels orbitals but does not determine how strongly each orbital participates in correlation.
That evidence must come from a correlated wavefunction.
The script constructs the molecular Hamiltonian in the initial CAS\ :math:`(10,8)` space and solves it with the :term:`MACIS` CASCI implementation:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :start-after: # start-cell-initial-casci
   :end-before: # end-cell-initial-casci

For the N\ :sub:`2` singlet, the active space contains five :math:`\alpha` and five :math:`\beta` electrons.
Choosing which five of the eight :math:`\alpha` spin orbitals are occupied is independent of choosing which five :math:`\beta` spin orbitals are occupied, so the two counts multiply.
The number of determinants is therefore

.. math::

   \binom{8}{5}\binom{8}{5} = 3136.

This space is small enough to include all 3,136 determinants rather than approximating the wavefunction with a selected subset.
Larger active-space studies often use selected CI to obtain approximate active-space diagnostics at lower cost, but that additional approximation is unnecessary here.

The ``calculate_one_rdm`` and ``calculate_two_rdm`` settings request the one- and two-particle reduced density matrices (:term:`RDMs <RDM>`).
An RDM compresses information from the many-electron wavefunction into expectation values involving one or two particles.
The diagonal of the spin-resolved one-particle RDM gives the expected occupation of each spin orbital, while a corresponding diagonal element of the two-particle RDM gives the joint occupation of a pair of spin orbitals.
Together, these diagonal elements determine the local occupation probabilities needed for the orbital-entanglement diagnostic used below.

Transform to natural orbitals
=============================

Molecular orbitals are not unique: unitary rotations among orbitals in the same active subspace change their individual shapes but not the subspace they span.
For an exact CASCI calculation in a fixed active subspace, such a rotation leaves the total energy unchanged.
Orbital-resolved quantities, however, can change because they describe the chosen orbital representation.

Natural orbitals diagonalize the one-particle RDM :cite:`Lowdin1956`.
Their eigenvalues are natural-orbital occupation numbers between zero and two for spatial orbitals.
Occupations near two identify nearly doubly occupied orbitals, occupations near zero identify nearly empty orbitals, and fractional occupations can reveal orbitals that require multiple electron configurations.
Natural orbitals are not required to compute single-orbital entropies, and they do not make those entropies independent of the orbital basis.
They provide a useful convention in which the one-particle occupations are directly associated with individual orbitals before applying the orbital-resolved entropy criterion.

The supported ``qdk_natural_orbitals`` transformation uses the one-particle RDM from the initial CASCI wavefunction.
Despite being created through the ``orbital_localizer`` algorithm interface, this operation transforms the active orbitals to natural orbitals; it does not localize them in real space.
The script then rebuilds and resolves the CAS\ :math:`(10,8)` Hamiltonian so that both RDMs and the orbital diagnostics are expressed consistently in the natural-orbital representation:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :start-after: # start-cell-natural-orbitals
   :end-before: # end-cell-natural-orbitals

The two CASCI energies should agree to the displayed precision because both calculations span the same complete active space.
The second calculation is needed for the orbital-resolved selection evidence, not to lower the energy.

Refine the active space with orbital entropies
==============================================

The single-orbital entropy measures how uncertain the occupation of one spatial orbital is when that orbital is considered separately from the rest of the correlated system :cite:`Boguslawski2015`.
A spatial orbital has four possible local occupation states: empty, occupied by one :math:`\alpha` electron, occupied by one :math:`\beta` electron, or doubly occupied.
If their probabilities are :math:`\omega_{a,i}` for orbital :math:`i`, its single-orbital entropy is

.. math::

   s_i^{(1)} = -\sum_{a=1}^{4} \omega_{a,i}\ln\omega_{a,i}.

These probabilities come from diagonal elements of the spin-resolved one- and two-particle RDMs.
If :math:`n_{i\alpha}` and :math:`n_{i\beta}` are the one-particle occupations and :math:`d_i` is the probability of simultaneous :math:`\alpha` and :math:`\beta` occupation from the two-particle RDM, then

.. math::

   \begin{aligned}
   \omega_{\mathrm{empty},i} &= 1-n_{i\alpha}-n_{i\beta}+d_i, \\
   \omega_{\alpha,i} &= n_{i\alpha}-d_i, \\
   \omega_{\beta,i} &= n_{i\beta}-d_i, \\
   \omega_{\mathrm{double},i} &= d_i.
   \end{aligned}

An entropy near zero means that one local occupation dominates.
A larger entropy means that several local occupations contribute, indicating that the orbital is more strongly entangled with the rest of the active space and is a stronger candidate for explicit treatment.
Because the complete CASCI wavefunction is a pure state, uncertainty in the reduced state of one orbital reflects quantum entanglement between that orbital and the remaining active orbitals.
QDK/Chemistry evaluates these probabilities and entropies from the RDMs stored in the CASCI wavefunction.

The entropy-difference autoCAS selector, ``qdk_autocas_eos``, sorts the normalized orbital entropies and tests the consecutive gaps against its entropy and difference thresholds :cite:`Stein2016,Stein2019`.
It selects the largest high-entropy group separated by a qualifying gap.
It then repartitions the orbitals according to the selected group:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :start-after: # start-cell-refine
   :end-before: # end-cell-refine

The asterisks in the script output identify the selected orbitals.
For stretched N\ :sub:`2`, orbitals 5 through 8 have entropies near 0.31, separated by a large gap from the remaining values of 0.08 or less.
These four orbitals form the selected high-entropy group and produce a CAS\ :math:`(4,4)` space containing two :math:`\alpha` and two :math:`\beta` electrons.
The refined partition contains five inactive orbitals, four active orbitals, and 19 virtual orbitals.
Three orbitals from the initial active space become inactive, consistent with their remaining nearly doubly occupied, while one becomes virtual, consistent with its remaining nearly empty.

Compute the algorithmic reference
=================================

The script finishes by solving the refined CAS\ :math:`(4,4)` Hamiltonian with CASCI:

.. literalinclude:: ../../_static/examples/python/tutorial_choose_active_space.py
   :language: python
   :start-after: # start-cell-final-casci
   :end-before: # end-cell-final-casci

The final singlet space contains

.. math::

   \binom{4}{2}\binom{4}{2} = 36

determinants.
This substantial reduction in problem size is useful for the quantum-computing stages of the tutorial.
The final CASCI energy is the exact ground-state energy of the selected active-space Hamiltonian, up to numerical solver tolerance, and will be the *algorithmic reference energy* for state preparation and phase estimation.
It is not the exact energy of N\ :sub:`2` in the ``cc-pvdz`` basis, because the inactive and virtual orbitals cannot change occupation in this model.

Evaluate the active-space choice
================================

The CAS\ :math:`(4,4)` determinant space is a subset of the natural-orbital CAS\ :math:`(10,8)` determinant space.
Restricting the wavefunction to a subset of the available determinants cannot produce a lower optimized energy because the larger calculation can use every wavefunction available to the smaller calculation and more.
This consequence of the variational principle applies here because freezing the additional inactive orbitals preserves the same Hamiltonian on that determinant subset.
The script reports the observed energy increase when reducing the active space.

This increase is much larger than the 1 milliHartree teaching target from :doc:`Energy and accuracy <01_energy_and_accuracy>`.
The comparison shows that the compact space does not recover all correlation present in the initial valence space.
It does not measure the total molecular error: even the CAS\ :math:`(10,8)` calculation excludes correlation involving the two inactive core orbitals and 18 external virtual orbitals, and both calculations retain the finite ``cc-pvdz`` basis and fixed geometry.

The CAS\ :math:`(4,4)` model is therefore chosen to preserve the strongest static-correlation degrees of freedom while keeping the later quantum calculation small enough to study directly.
Record both active-space energies and this limitation in the lab notebook.
The next chapter will determine how the four active spatial orbitals are represented by qubits.

Run the calculation
===================

With the Python environment from :doc:`Before you begin <00_before_you_begin>` active, run the complete script from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_choose_active_space.py

Record the orbital representation, initial and refined active-space sizes, selection evidence, determinant counts, and both CASCI energies in the active-space section of the lab notebook.
Use the final CAS\ :math:`(4,4)` energy as the algorithmic reference, while retaining the larger-space result as evidence of the correlation excluded by the compact model.

Check your understanding
========================

.. admonition:: Why can an inactive orbital still contribute to the molecular energy?
   :class: hint
   :collapsible: closed

   An inactive spatial orbital remains doubly occupied in every determinant.
   Its electrons and their interactions contribute to the core part of the active-space Hamiltonian even though the calculation does not vary their occupations.

.. admonition:: Why does autoCAS-EOS require a correlated calculation before it can select orbitals?
   :class: hint
   :collapsible: closed

   The selector uses single-orbital entropies derived from local occupation probabilities.
   Those probabilities require one- and two-particle RDMs from a correlated wavefunction; a Hartree--Fock determinant alone does not provide the required correlation evidence.

.. admonition:: Why do the two CAS(10,8) energies agree even though their orbitals differ?
   :class: hint
   :collapsible: closed

   The natural-orbital transformation is a unitary rotation within the same eight-orbital active subspace.
   Complete CI in that fixed subspace spans the same many-electron space before and after the rotation, so its energy is invariant apart from numerical error.

.. admonition:: Does selecting CAS(4,4) establish 1 milliHartree accuracy for the molecule?
   :class: hint
   :collapsible: closed

   No.
   The increase relative to CAS\ :math:`(10,8)` shows that the compact model omits correlation even within the valence space, while both models also retain fixed-geometry and finite-basis approximations.
   The smaller space is a controlled tutorial model whose exact CASCI energy becomes the reference for testing the later quantum algorithm.

Further reading
===============

- :doc:`Orbital localization and transformation <../../user/comprehensive/algorithms/localizer>`
- :doc:`Active-space selection <../../user/comprehensive/algorithms/active_space>`
- :doc:`Multi-configuration calculations <../../user/comprehensive/algorithms/mc_calculator>`
- :doc:`Wavefunctions <../../user/comprehensive/data/wavefunction>`
