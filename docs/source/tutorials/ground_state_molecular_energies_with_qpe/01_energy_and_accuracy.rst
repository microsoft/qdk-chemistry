Energy and accuracy
###################

.. admonition:: Chapter focus
   :class: chapter-focus

   What energy will we calculate, and how accurate must the result be?

Learning objectives
===================

After completing this chapter, you will be able to:

- Identify the electronic and nuclear contributions to the fixed-geometry ground-state energy reported in this tutorial.
- Explain why small energy errors can cause large errors in predicted equilibria and rates.
- Explain why each energy comparison requires a clearly defined reference.
- Identify the main approximations that affect the final energy estimate.

.. important:: Lab notebook assignment

   Complete :ref:`lab-notebook-goal`.
   Record the target quantity, the 1 milliHartree teaching target, and the reference used to evaluate that target.
   This entry defines the criteria you will use to interpret every later result.

The energy reported in this tutorial
====================================

The :doc:`tutorial introduction <index>` introduced the ground-state eigenvalue of the electronic Hamiltonian for fixed nuclear positions.
The quantity ultimately reported by this tutorial is the corresponding fixed-geometry total energy,

.. math::

   E_{\mathrm{total}}(\mathbf{R})
   = E_{\mathrm{electronic}}(\mathbf{R})
   + E_{\mathrm{nuclear}}(\mathbf{R}),

where :math:`\mathbf{R}` denotes the fixed nuclear coordinates.
The electronic contribution includes the electron kinetic energy, electron--nucleus attraction, and electron--electron repulsion.
The nuclear contribution is the repulsion among the fixed nuclei.
:doc:`Putting the problem on qubits <04_putting_the_problem_on_qubits>` explains how these contributions are tracked when the molecular Hamiltonian is represented on qubits.

:term:`QDK`/Chemistry reports molecular energies in `hartree <https://en.wikipedia.org/wiki/Hartree>`_, with symbol :math:`E_{\mathrm{h}}`.
One milliHartree is :math:`10^{-3}\ E_{\mathrm{h}}`, approximately :math:`2.6255\ \mathrm{kJ\,mol^{-1}}` using the 2022 :term:`CODATA` constants :cite:`Mohr2025`.

Energy differences require context
==================================

A basis set is a finite collection of mathematical functions used to represent molecular orbitals in a calculation.
:doc:`Describing the molecule <02_describing_the_molecule>` explains how basis sets are constructed and used.
Chemically meaningful questions usually compare energies calculated under a consistent set of choices, including geometry, basis set, and electronic-structure method.
Examples include the energy difference between two molecular geometries, the reaction energy between products and reactants, and the barrier between a reactant and a transition state.
Changing the model or numerical method between the two calculations can make the difference difficult to interpret.

Why energy accuracy matters
===========================

Accurate electronic energies are necessary inputs to predictions of chemical equilibria and reaction rates, but they are not sufficient by themselves.
At constant temperature and pressure, these predictions depend on `Gibbs free-energy <https://en.wikipedia.org/wiki/Gibbs_free_energy>`_ differences that also include `zero-point energy <https://en.wikipedia.org/wiki/Zero-point_energy>`_, thermal, entropic, and environmental contributions.
This tutorial calculates only the fixed-geometry electronic and nuclear components of the energy; it does not calculate a free energy.

The dimensionless `equilibrium constant <https://en.wikipedia.org/wiki/Equilibrium_constant>`_ :math:`K` is related to the standard reaction Gibbs free energy :math:`\Delta G^\circ` by

.. math::

   K \propto \exp\left(-\frac{\Delta G^\circ}{RT}\right),

where :math:`R` is the molar gas constant and :math:`T` is the absolute temperature.
Similarly, the `Eyring equation <https://en.wikipedia.org/wiki/Eyring_equation>`_ relates a rate constant :math:`k` to the activation Gibbs free energy :math:`\Delta G^\ddagger`:

.. math::

   k \propto \exp\left(-\frac{\Delta G^\ddagger}{RT}\right).

Both relations depend exponentially on a free-energy difference; therefore, the predicted equilibrium constant or rate can be too large or too small by a factor that scales exponentially with the magnitude of the error.
At :math:`298.15\ \mathrm{K}`, :math:`RT \approx 2.479\ \mathrm{kJ\,mol^{-1}}`, an error of 1 milliHartree can produce a factor of approximately :math:`2.88` in a predicted equilibrium constant or rate if the other free-energy contributions are exact.
An error of approximately :math:`5.71\ \mathrm{kJ\,mol^{-1}}`, or :math:`2.17` milliHartree, produces a factor of ten.
These estimates illustrate why errors of only a few milliHartrees can have a significant impact on accuracy.

Accuracy, precision, and other definitions
==========================================

An energy estimate is accurate only in relation to a stated reference value for the same quantity.

Accuracy
   Closeness to the reference.
Precision
   The spread among repeated estimates under the same conditions.
Resolution
   The smallest energy interval that the chosen numerical representation can distinguish.
Uncertainty
   The range of values plausibly consistent with the available information.

A calculation can be precise but inaccurate, or have fine resolution without a small uncertainty.

The teaching target
===================

This tutorial uses 1 milliHartree as a teaching target for the absolute difference between the final :term:`QPE` total energy and a classical reference energy for the same Hamiltonian.
The threshold is often associated with the term "chemical accuracy" because it is close to the thermal energy :math:`RT` at room temperature.
Definitions of chemical accuracy vary, so the tutorial always states the numerical 1 milliHartree target and the two energies being compared.
Meeting this target shows that the quantum algorithm reproduced its classical reference to the requested tolerance.
It does not show that the molecular model agrees with experiment, nor does it remove errors from the geometry, basis set, electronic-structure model, or omitted free-energy contributions.

Where approximations enter
==========================

Different stages introduce different approximations or uncertainties.
For example:

- The fixed molecular geometry and Born--Oppenheimer approximation define the physical model considered by the tutorial.
- A finite basis set restricts the one-electron functions used to describe the molecule.
- Classical electronic-structure methods use approximations to obtain tractable solutions of the Schrödinger equation.
- Hamiltonian time evolution is approximated when the quantum circuit is constructed.
- Finite phase resolution and measurement sampling limit the reported :term:`QPE` estimate.

.. todo::

   Verify as later chapters are drafted that each approximation is introduced where it first enters the calculation.

Each approximation is introduced where it first enters the calculation, and the corresponding comparison is recorded before proceeding.

Check your understanding
========================

.. admonition:: Which contributions make up the fixed-geometry total energy reported by this tutorial?
   :class: hint
   :collapsible: closed

   The fixed-geometry total energy is the sum of the electronic energy and the repulsion energy among the fixed nuclei.

.. admonition:: Why can a small free-energy error cause a large error in a predicted equilibrium constant or reaction rate?
   :class: hint
   :collapsible: closed

   Equilibrium constants and reaction rates depend exponentially on reaction and activation free-energy differences.
   A small error in a free-energy difference can therefore multiply the predicted equilibrium constant or rate by a large factor.

.. admonition:: Why does an accurate electronic energy not, by itself, determine an equilibrium constant or rate?
   :class: hint
   :collapsible: closed

   Electronic and nuclear energies are only some of the contributions to a free energy.
   Zero-point, thermal, entropic, and environmental contributions can also affect an equilibrium constant or reaction rate.

Further reading
===============

- `Equilibrium constants <https://en.wikipedia.org/wiki/Equilibrium_constant>`_
- `Eyring equation <https://en.wikipedia.org/wiki/Eyring_equation>`_
- `Hartree energy <https://en.wikipedia.org/wiki/Hartree>`_
- :doc:`Features and methods <../../user/features>`
- :doc:`Quickstart <../../user/quickstart>`
- :doc:`References <../../references>`
