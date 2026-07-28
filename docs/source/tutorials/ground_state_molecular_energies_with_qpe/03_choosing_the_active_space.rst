Choosing the active space
#########################

.. rubric:: Chapter question

Which electrons and orbitals must the calculation treat explicitly?

Learning objectives
===================

After completing this chapter, you will be able to:

- Explain why strongly correlated systems require more than one determinant.
- Explain how an orbital transformation can aid chemical interpretation.
- Distinguish inactive, active, and virtual orbitals.
- Select a valence active space with :term:`QDK`/Chemistry.
- Explain how orbital entropies can refine an active-space choice.
- Evaluate the tradeoff between active-space accuracy and problem size.

.. rubric:: Lab notebook assignment

Complete :ref:`lab-notebook-active-space`.
Record the evidence used to select the active space, not only the final orbital and electron counts.
Explain what the energy comparison establishes within the chosen basis and identify the energy that will serve as the algorithmic reference.

Prerequisite concepts
=====================

.. todo::

   Recall the Hartree--Fock determinant and molecular orbitals from the previous chapter.
   Define static correlation before using multi-reference terminology.

Recognize the limits of one determinant
=======================================

.. todo::

   Use stretched N\ :sub:`2` to motivate a multi-configurational description.
   Introduce determinants and configuration interaction only to the depth needed for active-space selection and trial-state preparation.

Transform the orbitals
======================

.. todo::

   Explain the unitary freedom in the molecular orbitals and why a transformed representation can make orbital selection easier.
   State explicitly which quantities remain unchanged by the transformation used in the example.

Define the active space
=======================

.. todo::

   Define inactive, active, and virtual orbitals.
   Explain the notation for active electrons and spatial orbitals.
   Introduce the valence selector as the first active-space choice.

Compute a correlated active-space wavefunction
==============================================

.. todo::

   Introduce configuration interaction (:term:`CI`), including complete active space configuration interaction (:term:`CASCI`) and selected :term:`CI`, at the level required by the workflow.
   Explain why the entropy-difference autoCAS variant requires one- and two-particle reduced density matrices (:term:`RDMs <RDM>`) from an initial correlated calculation.
   Identify the final :term:`CASCI` energy for the selected active-space Hamiltonian as the algorithmic reference used in the phase-estimation chapter.

Refine the active space with orbital entropies
==============================================

.. todo::

   Define natural occupation numbers, :term:`RDMs <RDM>`, and single-orbital entropy before using them.
   Show the actual entropy-difference autoCAS data flow rather than presenting the selector as a direct Hartree--Fock step.

Evaluate the active-space choice
================================

.. todo::

   Compare nested active spaces with a variational solver.
   State the conditions under which enlarging the space cannot raise the calculated ground-state energy.
   Interpret the energy changes as active-space convergence within the chosen one-particle basis, not as the total error relative to the physical molecule.
   Update the active-space section of the lab notebook and defer the exact qubit count to the next chapter.

Run the calculation
===================

.. todo::

   Add a standalone native Python example covering orbital transformation, valence selection, the initial correlated calculation, entropy-based refinement, and the final active-space energy.
   Keep visualization optional.

Check your understanding
========================

.. todo::

   Add an exercise that asks the learner to interpret occupations or entropies and defend an active-space choice.

Further reading
===============

- :doc:`Orbital localization and transformation <../../user/comprehensive/algorithms/localizer>`
- :doc:`Active-space selection <../../user/comprehensive/algorithms/active_space>`
- :doc:`Multi-configuration calculations <../../user/comprehensive/algorithms/mc_calculator>`
- :doc:`Wavefunctions <../../user/comprehensive/data/wavefunction>`
