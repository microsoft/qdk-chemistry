Before you begin
################

.. rubric:: Chapter focus

What background and software do you need to complete this tutorial?

.. rubric:: Lab notebook assignment

Create a working copy of the :doc:`lab notebook <lab_notebook>`.
Complete :ref:`lab-notebook-setup` before running a scientific calculation.
This record identifies the :term:`QDK`/Chemistry version used for the tutorial.

Audience and prerequisites
==========================

This tutorial is intended for advanced undergraduate and early-stage graduate students.
You should be able to read and modify basic scientific Python and interpret numerical output.
You are ready to begin if you can perform the following tasks with reference material available:

- Use NumPy arrays in a short Python calculation and interpret its numerical output.
- Explain how eigenvalues and eigenvectors relate to the time-independent Schrödinger equation.
- Describe molecular orbitals and electron configurations qualitatively.
- Explain how quantum gates change qubit states and how measurement produces classical outcomes.

The tutorial does not assume prior knowledge of active-space methods, fermion-to-qubit encodings, quantum state preparation, or quantum phase estimation.
The introductory page provides the chemistry and quantum-computing context needed to begin, and later chapters develop these methods before using them in calculations.
Use the `Python tutorial <https://docs.python.org/3/tutorial/>`_, the `NumPy fundamentals <https://numpy.org/doc/stable/user/basics.html>`_, and the `quantum-computing concepts overview <https://learn.microsoft.com/azure/quantum/concepts-overview>`_ to refresh prerequisite material.

Required software
=================

The required examples use Python and implementations provided directly by :term:`QDK`/Chemistry.
They do not require Qiskit, OpenFermion, or the Quantum Resource Estimator.
Optional chapters identify any additional packages they require.

Use Python 3.10 or later on a platform supported by :term:`QDK`/Chemistry.
Released packages support Linux on x86-64 and Arm64, macOS on Apple silicon, and Windows on x86-64 and Arm64 through the Windows Subsystem for Linux.
The `installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_ contain the current support matrix.
Check the version before creating the environment:

.. code-block:: console

   python3 --version

Continue if the command reports Python 3.10 or later.

Set up the Python environment
=============================

Use an isolated Python environment so that the tutorial dependencies do not conflict with packages used by other projects.
From a terminal, create and activate a virtual environment:

.. code-block:: console

   python3 -m venv .venv
   source .venv/bin/activate

The activation command above applies to Linux, macOS, and the Windows Subsystem for Linux.
Install the released :term:`QDK`/Chemistry package into the active environment:

.. code-block:: console

   python -m pip install --upgrade pip
   python -m pip install qdk-chemistry==2.0.0

This command installs the :term:`QDK`/Chemistry release that matches this documentation.
The required workflow does not use the ``all``, ``qiskit-extras``, or ``qre`` optional-dependency groups.
Keep the environment active while completing the tutorial.

Check your setup
================

Download :download:`tutorial_qpe_setup.py <../../_static/examples/python/tutorial_qpe_setup.py>`, then run it from the directory where you saved it:

.. code-block:: console

   python tutorial_qpe_setup.py

The script reports the active Python environment to help diagnose setup problems.
It also verifies that the built-in :term:`QDK`/Chemistry implementations required by later chapters are available:

.. literalinclude:: ../../_static/examples/python/tutorial_qpe_setup.py
   :language: python
   :start-after: # start-cell-verify
   :end-before: # end-cell-verify

The check succeeds when it ends with output similar to the following and does not raise an exception:

.. code-block:: text

   Python executable: /path/to/.venv/bin/python
   Python version: 3.12.10
   QDK/Chemistry version: 2.0.0
   Verified 8 built-in implementations.

The path and versions can differ from this example.
Record the :term:`QDK`/Chemistry version and verification result in :ref:`lab-notebook-setup`.
If the import or verification fails, confirm that the ``Python executable`` path contains the virtual-environment directory and compare the installation command with the `installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_.

How to use the tutorial
=======================

Complete each required chapter in order because later chapters use decisions and results recorded earlier.
For each chapter:

1. Read the explanation before running its example.
2. Run the example and compare its output with the stated expectation.
3. Complete the understanding check.
4. Update the linked section of the lab notebook before continuing.

Use links to the reference documentation when you need complete application programming interface (:term:`API`) details.
The final circuit simulation is the only intentionally long required example; the introductory page gives its expected duration, and the final chapter explains how to monitor its progress.

Check your understanding
========================

Before continuing, confirm that you can answer the following questions:

1. Which background topics does the tutorial assume, and which specialized methods does it teach?
2. Which Python environment will run the tutorial examples?
3. What output demonstrates that the required built-in implementations are available?

Further reading
===============

- `Installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_
- :doc:`Quickstart <../../user/quickstart>`
- :doc:`Glossary <../../glossary>`
