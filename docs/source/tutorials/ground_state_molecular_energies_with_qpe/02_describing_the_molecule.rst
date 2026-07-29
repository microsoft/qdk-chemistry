Describing the molecule
#######################

.. note:: Chapter focus

   How do we specify a molecular system and obtain its first energy estimate?

Learning objectives
===================

After completing this chapter, you will be able to:

- Specify a molecule by its geometry, charge, and spin multiplicity.
- Explain the role of a finite atomic-orbital basis set.
- Describe the Hartree--Fock approximation and its Slater determinant.
- Run a native :term:`QDK`/Chemistry self-consistent field calculation.
- Interpret an energy change caused by changing the basis set.

.. important:: Lab notebook assignment

   Complete :ref:`lab-notebook-molecule`.
   Record the molecular inputs and both Hartree--Fock energies before calculating their difference.
   Interpret the difference as basis-set sensitivity rather than as the total error of either energy.

Prerequisite concepts
=====================

.. todo::

   Connect this chapter to the energy definition and error taxonomy from the previous chapter.
   Identify any required refresher material on atomic orbitals, electron spin, and antisymmetric wavefunctions.

Specify the molecular system
============================

.. todo::

   Introduce geometry, element identities, charge, and spin multiplicity.
   Explain the x-, y-, and z-coordinate (:term:`XYZ`) file format and its unit convention.
   Introduce the stretched N\ :sub:`2` structure and justify its use throughout the tutorial.

Represent the orbitals in a finite basis
========================================

.. todo::

   Define basis functions and basis sets at the depth needed by the tutorial.
   Explain the basis-set approximation and select two basis sets that provide a useful, affordable comparison for stretched N\ :sub:`2`.

Obtain a mean-field reference
=============================

.. todo::

   Introduce Hartree--Fock, the single Slater determinant, and the self-consistent field procedure.
   Explain the returned energy and wavefunction without covering convergence algorithms that are not needed by the exercise.

Run the calculation
===================

.. todo::

   Add a standalone native Python example that loads the structure and runs the Hartree--Fock calculation.
   Display the example through a marker-based ``literalinclude`` and verify its numerical output in the script.

Compare two basis sets
======================

.. todo::

   Run the same Hartree--Fock method in two basis sets.
   Interpret the energy difference as basis-set sensitivity, not as the total error of either calculation.
   Do not compare either Hartree--Fock energy directly with the 1 milliHartree teaching target.
   Update the molecular-input and mean-field section of the lab notebook.

Check your understanding
========================

.. todo::

   Add a short exercise that asks the learner to change one molecular input or basis-set choice and predict which quantities will change.

Further reading
===============

- :doc:`Molecular structures <../../user/comprehensive/data/structure>`
- :doc:`Basis sets <../../user/comprehensive/data/basis_set>`
- :doc:`Available basis sets <../../user/comprehensive/basis_functionals>`
- :doc:`Self-consistent field calculations <../../user/comprehensive/algorithms/scf_solver>`
- :doc:`Molecular orbitals <../../user/comprehensive/data/orbitals>`
