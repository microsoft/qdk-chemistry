Plugins and interoperability
############################

.. note:: Chapter focus

   How can another software package implement a stage of a :term:`QDK`/Chemistry workflow?

Learning objectives
===================

After completing this chapter, you will be able to:

- Explain the :term:`QDK`/Chemistry factory and plugin model.
- Discover implementations available in the current environment.
- Replace one native implementation with a compatible plugin implementation.
- Identify the dependency and provenance information needed for reproducibility.

Understand interchangeable implementations
==========================================

.. todo::

   Explain the distinction among an algorithm interface, a named implementation, its settings, and a plugin.
   Connect the explanation to the native factory calls used throughout the required tutorial.

Discover the current environment
================================

.. todo::

   Add a short Python example that lists implementations and settings.
   Explain why optional implementations may be absent when their dependencies are not installed.

Extend the classical workflow
=============================

.. todo::

   Decide whether to teach density functional theory or a Python-based Simulations of Chemistry Framework (:term:`PySCF`) implementation as the primary chemistry plugin example.
   State how the substituted method changes the scientific approximation and provenance.

Consider quantum-framework interoperability
============================================

.. todo::

   Decide whether Qiskit interoperability provides enough educational value to include.
   If included, isolate its installation, version constraints, and application programming interface (:term:`API`) from the required native workflow.
   Do not require Qiskit elsewhere in the tutorial.

Record reproducibility information
==================================

.. todo::

   Define the package versions, implementation names, settings, and citations that a learner must record when combining :term:`QDK`/Chemistry with plugins.

Check your understanding
========================

.. todo::

   Add an exercise that substitutes one compatible implementation and asks the learner to distinguish :term:`API` compatibility from scientific equivalence.

Further reading
===============

- :doc:`Plugins <../../../user/comprehensive/plugins>`
- :doc:`Factory pattern <../../../user/comprehensive/algorithms/factory_pattern>`
- :doc:`Settings <../../../user/comprehensive/algorithms/settings>`
