Describing the molecule
#######################

.. admonition:: Chapter focus
   :class: chapter-focus

   How do we specify a molecular system and construct the starting wavefunction for the correlated calculation?

Learning objectives
===================

After completing this chapter, you will be able to:

- Specify a molecule by its geometry, charge, and spin multiplicity.
- Explain the role of a finite atomic-orbital basis set.
- Describe the Hartree--Fock approximation and the Slater-determinant form of its wavefunction.
- Generate a Hartree--Fock wavefunction with the built-in :term:`QDK`/Chemistry self-consistent field solver.
- Interpret an energy change caused by changing the basis set.

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

   Complete :ref:`lab-notebook-molecule` in the lab notebook as you work through this chapter.
   Record the molecular inputs and both Hartree--Fock energies before calculating their difference.
   Interpret the difference as basis-set sensitivity rather than as the total error of either energy.

Example download
====================

Download :download:`tutorial_describe_n2.py <../../_static/examples/python/tutorial_describe_n2.py>` and save it in your tutorial working directory.
Open the file in Visual Studio Code and review the complete script, including imports and setup code omitted from the excerpts below.
The sections below explain the inputs and calculations in this complete executable file before you run it.

.. _tutorial-molecular-system:

The molecular system
============================

An electronic-structure calculation requires the identities and positions of the nuclei, the number of electrons, and the target spin state.
This chapter specifies those inputs for the stretched N\ :sub:`2` molecule introduced on the :doc:`tutorial landing page <index>`.

The molecular geometry contains two nitrogen atoms separated by :math:`1.85\ \text{Å}`, compared with the `equilibrium bond length <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=1000>`_ of :math:`1.097685\ \text{Å}`.
The molecule is neutral, so its net charge is zero and it has 14 electrons.
`Spin multiplicity <https://en.wikipedia.org/wiki/Multiplicity_(chemistry)>`_ is defined as :math:`2S+1`, where :math:`S` is the total electron spin produced by combining the spins of all electrons.
Paired electrons contribute no net spin, whereas unpaired electrons can produce :math:`S>0`.
A multiplicity of one, two, or three is called a singlet, doublet, or triplet, respectively.
The labels :math:`\alpha` and :math:`\beta` denote spin projections :math:`+\tfrac{1}{2}` (spin up) and :math:`-\tfrac{1}{2}` (spin down), respectively.
:term:`QDK`/Chemistry uses the molecular charge and spin multiplicity to determine the numbers of :math:`\alpha` and :math:`\beta` electrons in the calculation.
For the target N\ :sub:`2` singlet, :math:`S=0`, so the multiplicity is one and the calculation contains seven :math:`\alpha` and seven :math:`\beta` electrons.

:term:`QDK`/Chemistry represents a molecular geometry with a :class:`~qdk_chemistry.data.Structure` object.
This example constructs the :class:`~qdk_chemistry.data.Structure` object from a string in `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_, which contains an atom count, a comment line, and one element symbol with three Cartesian coordinates for each atom:

.. literalinclude:: ../../_static/examples/python/tutorial_describe_n2.py
   :language: python
   :start-after: # start-cell-molecule
   :end-before: # end-cell-molecule

The :term:`XYZ` format used by :term:`QDK`/Chemistry interprets coordinates in ångström units.
However, this format does not specify molecular charge or spin multiplicity, so the example records these values separately.

.. _tutorial-hartree-fock-wavefunction:

The mean-field wavefunction
================================

The `Hartree--Fock method <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_ approximates the many-electron wavefunction with one `Slater determinant <https://en.wikipedia.org/wiki/Slater_determinant>`_.
As introduced in the :ref:`tutorial overview <tutorial-orbitals-and-determinants>`, each occupied spin orbital combines a spatial molecular orbital with an :math:`\alpha` or :math:`\beta` spin function.
Let :math:`\psi_p(x)` denote occupied spin orbital :math:`p`, where :math:`x=(\mathbf{r},\sigma)` contains a spatial coordinate :math:`\mathbf{r}` and a spin label :math:`\sigma\in\{\alpha,\beta\}`.
For an :math:`N`-electron Hartree--Fock state, the occupied spin orbitals form the Slater determinant

.. math::

   \Phi_{\mathrm{HF}}(x_1,\ldots,x_N)
   =\frac{1}{\sqrt{N!}}
   \det\!\left[\psi_p(x_q)\right]_{q,p=1}^{N}.

Here :math:`\det` denotes the determinant of the matrix whose row :math:`q` evaluates every occupied spin orbital at electron coordinate :math:`x_q`.
The factor :math:`1/\sqrt{N!}` normalizes the wavefunction when the occupied spin orbitals are orthonormal.
Exchanging two electron coordinates swaps two matrix rows and changes the sign of :math:`\Phi_{\mathrm{HF}}`, enforcing fermionic antisymmetry.
This determinant is an *approximate* many-electron wavefunction.
Each electron interacts with the average field generated by the other electrons rather than with their instantaneous correlated motion.
This mean-field treatment makes Hartree--Fock computationally tractable because it avoids representing all possible electron configurations.
The determinant accounts for exchange, an effect of wavefunction antisymmetry that reduces the probability of finding same-spin electrons together, but it omits electron correlation.

The Hartree--Fock energy is the fixed-geometry total energy evaluated with the optimized Hartree--Fock determinant:

.. math::

   E_{\mathrm{HF}}
   = \langle \Phi_{\mathrm{HF}} \vert \hat{H}_{\mathrm{electronic}} \vert \Phi_{\mathrm{HF}} \rangle
   + E_{\mathrm{nuclear}},

where :math:`\Phi_{\mathrm{HF}}` is the optimized determinant and :math:`E_{\mathrm{nuclear}}` is the repulsion among the fixed nuclei.
This energy remains approximate because the determinant cannot represent electron correlation.

.. _tutorial-molecular-orbitals:

Finite-basis orbital representation
========================================

The Hartree--Fock wavefunction :math:`\Phi_{\mathrm{HF}}` is a determinant constructed from occupied spin orbitals.
Each spin orbital combines a spatial molecular orbital with an :math:`\alpha` or :math:`\beta` spin function.
The molecular orbitals are functions of position that provide the spatial parts of these spin orbitals.
Changing the occupied molecular orbitals changes the Hartree--Fock determinant and its many-electron wavefunction.
Because the Hartree--Fock determinant is built from one-electron functions, optimizing the many-electron wavefunction reduces to optimizing these molecular orbitals.
Their spatial shapes and occupations also help identify bonding and antibonding interactions, and the optimized orbitals provide the starting representation for later multi-configurational calculations.

.. admonition:: What is a molecular orbital, and why is it useful?
   :class: quiz-question
   :collapsible: closed

   A molecular orbital is a one-electron spatial function used to construct the spin orbitals in a Hartree--Fock determinant.
   Molecular orbitals make the approximate many-electron wavefunction computationally tractable, help interpret bonding and occupations, and provide the starting representation for later multi-configurational calculations.

Electronic-structure calculations expand each molecular orbital :math:`\phi_p` in a finite collection of known basis functions :math:`\{\chi_\mu\}`:

.. math::

   \phi_p(\mathbf{r}) = \sum_\mu c_{\mu p}\chi_\mu(\mathbf{r}).

The index :math:`p` labels a molecular orbital, and :math:`\mu` labels a basis function.
The coefficients :math:`c_{\mu p}` determine molecular orbital :math:`p`, and the self-consistent field calculation optimizes these coefficients.
The collection of basis functions is called a `basis set <https://en.wikipedia.org/wiki/Basis_set_(chemistry)>`_.
These objects form a hierarchy: nucleus-centered basis functions :math:`\chi_\mu` provide localized radial and angular shapes; molecular orbitals :math:`\phi_p` combine these functions and can extend across multiple nuclei; each spatial molecular orbital combines with an :math:`\alpha` or :math:`\beta` spin function to form a spin orbital :math:`\psi_p`; and the occupied spin orbitals form the many-electron determinant :math:`\Phi_{\mathrm{HF}}`.

A finite basis restricts the shapes available to the molecular orbitals and therefore introduces another approximation in the electronic description of the system.
Adding suitable basis functions gives the orbitals more flexibility, but it also increases computational cost.

This chapter compares the correlation-consistent polarized valence double-zeta basis set, ``cc-pvdz``, with the related triple-zeta basis set, ``cc-pvtz`` :cite:`Dunning1989`.
Here, double-zeta and triple-zeta mean that two or three radial functions, respectively, describe each valence atomic orbital.
The larger basis gives the orbitals more radial flexibility but increases computational cost; both basis sets include polarization functions.

This comparison does not determine the exact basis-set error.
It measures how much the Hartree--Fock energy changes with basis size while the geometry, method, and basis-set family remain fixed.

.. admonition:: Which basis set gives a lower energy and why?
   :class: quiz-question
   :collapsible: closed

   The ``cc-pvtz`` calculation gives the lower energy.
   This result is consistent with its additional radial flexibility, which provides a more flexible representation of the molecular orbitals.
   The atomic elements and coordinates, molecular net charge, spin multiplicity, and Hartree--Fock method remain fixed, so the observed difference measures basis-set sensitivity.

Self-consistent wavefunction optimization
===========================================

The occupied molecular orbitals determine the average distribution of the electrons in the Hartree--Fock determinant.
This distribution generates the effective field that represents the average interaction of each electron with the others, as described above.
Solving the one-electron equations in that field produces new molecular orbitals and therefore a new electron distribution and field.
The `self-consistent field <https://en.wikipedia.org/wiki/Self-consistent_field>`_ (:term:`SCF`) procedure resolves this dependence iteratively:

1. Begin with an initial set of molecular orbitals.
2. Construct the effective one-electron operator generated by those orbitals.
3. Solve for an updated set of orbitals.
4. Repeat until the energy and orbitals satisfy the convergence criteria.

The :term:`QDK`/Chemistry :class:`~qdk_chemistry.algorithms.ScfSolver` returns the converged fixed-geometry total energy and a wavefunction containing the self-consistent molecular orbitals.
These orbitals provide the starting point for the multi-configurational calculations introduced in :doc:`Choosing the active space <03_choosing_the_active_space>`.

.. admonition:: Why does a Hartree--Fock calculation require an iterative :term:`SCF` procedure?
   :class: quiz-question
   :collapsible: closed

   The molecular orbitals determine the mean field experienced by each electron, while that mean field determines the molecular orbitals.
   :term:`SCF` iterations update the orbitals and field until they are mutually consistent.

Running the calculation
=======================

With the Python environment from :doc:`Before you begin <00_before_you_begin>` active, run the complete script from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_describe_n2.py

The following code runs the built-in :term:`QDK`/Chemistry Hartree--Fock solver once for each basis set.
The primary result is the ``cc-pvdz`` wavefunction, whose self-consistent molecular orbitals serve as the starting point for the multi-configurational calculation in :doc:`Choosing the active space <03_choosing_the_active_space>`.
The ``cc-pvdz`` and ``cc-pvtz`` Hartree--Fock energies support the basis-set sensitivity exercise that follows.

.. literalinclude:: ../../_static/examples/python/tutorial_describe_n2.py
   :language: python
   :start-after: # start-cell-hartree-fock
   :end-before: # end-cell-hartree-fock

Run the complete script and record both fixed-geometry total energies and the number of ``cc-pvdz`` molecular orbitals in the lab notebook.
The reported energies include both the electronic energy and the repulsion among the fixed nuclei.

.. admonition:: What should you observe after running the Hartree--Fock calculations?
   :class: quiz-question
   :collapsible: closed

   The script should report one negative total energy for each basis set and the number of molecular orbitals in the ``cc-pvdz`` wavefunction.
   The ``cc-pvtz`` energy should be lower than the ``cc-pvdz`` energy because the larger basis gives the Hartree--Fock orbitals more flexibility in describing the mean-field wavefunction.

The absolute difference between the two energies is the observed basis-set sensitivity.
It does not establish the exact error of either energy because both calculations use finite basis sets.
Therefore, do not compare this basis-set sensitivity with the :math:`1\ \mathrm{m}E_{\mathrm{h}}` teaching target from :doc:`Energy and accuracy <01_energy_and_accuracy>`.

For the rest of this tutorial, we use ``cc-pvdz`` because it includes polarization functions while keeping the correlated examples small enough to run quickly; this is a teaching-model cost choice, not a claim that ``cc-pvdz`` is universally preferable.
The ``cc-pvtz`` calculation is used only for the controlled basis-set comparison in this chapter, avoiding its greater cost in every later stage.
Bond stretching can increase static correlation and weaken a one-determinant description.
:doc:`Choosing the active space <03_choosing_the_active_space>` next evaluates this effect for the selected N\ :sub:`2` geometry and identifies which electrons and orbitals must be treated in a multi-configurational wavefunction.

Further reading
===============

- :doc:`Molecular structures <../../user/comprehensive/data/structure>`
- :doc:`Basis sets <../../user/comprehensive/data/basis_set>`
- :doc:`Available basis sets <../../user/comprehensive/basis_functionals>`
- :doc:`Self-consistent field calculations <../../user/comprehensive/algorithms/scf_solver>`
- :doc:`Molecular orbitals <../../user/comprehensive/data/orbitals>`
