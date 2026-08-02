Before you begin
################

.. admonition:: Chapter focus
   :class: chapter-focus

   What background and software do you need to complete this tutorial?

.. admonition:: Lab notebook assignment
   :class: lab-notebook-assignment

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
The :doc:`tutorial introduction <index>` provides the chemistry and quantum-computing context needed to begin, and the specialized methods are developed before they are used in calculations.
Use the `Python tutorial <https://docs.python.org/3/tutorial/>`_, the `NumPy fundamentals <https://numpy.org/doc/stable/user/basics.html>`_, and the `quantum-computing concepts overview <https://learn.microsoft.com/azure/quantum/concepts-overview>`_ to refresh prerequisite material.

Computer requirements
=====================

These are practical requirements for completing the tutorial rather than the full package support matrix.
A supported 64-bit computer with at least 8 GiB of memory and 3 GiB of free storage is the minimum recommended configuration.
For the final simulator workflow, 16 GiB of memory and a recent multicore processor provide more comfortable headroom.
A GPU is not required.

The installed tutorial environment occupies approximately 1.5 GiB.
The short chapter Python scripts generally finish within a minute, while the Jupyter notebooks can take longer.
The final twenty-run :term:`IQPE` simulation took approximately ten minutes in a measured development run.
Allow tens of minutes depending on processor performance and other system activity.

Required software
=================

The required examples use `Visual Studio Code <https://code.visualstudio.com/download>`_, Python, and implementations provided directly by :term:`QDK`/Chemistry.
They do not require Qiskit, OpenFermion, or the Quantum Resource Estimator.

Install the desktop version of Visual Studio Code, the `Microsoft Quantum Development Kit extension <https://marketplace.visualstudio.com/items?itemName=quantum.qsharp-lang-vscode>`_, the `Python extension <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_, and the `Jupyter extension <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_.
These extensions provide the quantum-language, Python, notebook, simulation, debugging, and visualization features used by the tutorial.

.. _tutorial-qsharp:

`Q# <https://learn.microsoft.com/azure/quantum/qsharp-overview>`_ is the :term:`QDK`'s primary quantum programming language for expressing, inspecting, and executing logical circuits.
The :term:`QDK` also supports quantum programs built with Python frameworks such as `Qiskit <https://www.ibm.com/quantum/qiskit>`_ and `Cirq <https://quantumai.google/cirq>`_.
The :term:`QDK` extension provides Q# language support, while the required tutorial's Python scripts use QDK/Chemistry to generate the Q# circuit representations examined in later chapters.
Create a folder for your downloaded tutorial files, then use **File > Open Folder** in Visual Studio Code to open it as your tutorial working folder.

Use Python 3.10 or later on a platform supported by :term:`QDK`/Chemistry.
Released packages support Linux on x86-64 and Arm64, macOS on Apple silicon, and Windows on x86-64 and Arm64 through the Windows Subsystem for Linux.
The `installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_ contain the current support matrix.
Check the version before creating the environment:

.. code-block:: console

   python3 --version

Continue if the command reports Python 3.10 or later; otherwise, install a newer version of Python.

Python environment setup
=============================

Use an isolated Python environment so that the tutorial dependencies do not conflict with packages used by other projects.
In Visual Studio Code, select **Terminal > New Terminal**.
From this integrated terminal, create and activate a virtual environment in the tutorial working folder:

.. code-block:: console

   python3 -m venv .venv
   source .venv/bin/activate

The activation command above applies to Linux, macOS, and the Windows Subsystem for Linux.
Install the released :term:`QDK`/Chemistry package into the active environment:

.. parsed-literal::

   python -m pip install --upgrade pip
   python -m pip install qdk-chemistry==\ |ground-state-tutorial-version| "ipykernel>=6.0"

This command installs the :term:`QDK`/Chemistry release that matches this documentation and the Python kernel used to run the interactive notebook.
QDK/Chemistry already requests QDK's Jupyter support, which supplies the molecular-orbital widget used in :doc:`Choosing the active space <03_choosing_the_active_space>`.
The required workflow does not use the ``all``, ``qiskit-extras``, or ``qre`` optional-dependency groups.
Keep the environment active while completing the tutorial.

Setup check
================

Download :download:`tutorial_qpe_setup.py <../../_static/examples/python/tutorial_qpe_setup.py>` into the tutorial working folder.
Open the file in Visual Studio Code and review the complete script, including imports and setup code that may not appear in the excerpts below.
Then run it from the Visual Studio Code integrated terminal:

.. code-block:: console

   python tutorial_qpe_setup.py

The script reports the active Python environment to help diagnose setup problems.
It also verifies that the built-in :term:`QDK`/Chemistry implementations required by the tutorial calculations are available.

The check succeeds when it finishes without an exception, reports a Python executable from the tutorial virtual environment, reports :term:`QDK`/Chemistry version |ground-state-tutorial-version|, confirms the IPython kernel and molecular viewer are importable, and confirms that all required built-in implementations are available.
Record the :term:`QDK`/Chemistry version and verification result in :ref:`lab-notebook-setup`.
If the import or verification fails, confirm that the ``Python executable`` path contains the virtual-environment directory and compare the installation command with the `installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_.

How to use the tutorial
=======================

Complete the required chapters in the order listed in the :doc:`tutorial introduction <index>` because each stage uses decisions and results recorded earlier.
For each chapter:

1. Download the complete example files and open them in Visual Studio Code.
2. Read the complete file, including imports and setup code omitted from the excerpts in the chapter.
3. Read the chapter explanation before running the example.
4. Run the Python example in the Visual Studio Code integrated terminal, and run any interactive notebook as directed by the chapter.
5. Complete the understanding check, when present.
6. Update the linked section of the lab notebook before continuing.

In a **Check your understanding** section, answer each question before selecting its heading to reveal the suggested answer.
Select the heading again to hide the answer.

.. admonition:: What happens when you click on a question box (after you've answered the question)?
   :class: quiz-question
   :collapsible: closed

   You can check your answer.

Use links to the reference documentation when you need complete application programming interface (:term:`API`) details.
The final circuit simulation is the only intentionally long required example; the :doc:`tutorial introduction <index>` gives its expected duration, and :doc:`Iterative quantum phase estimation <06_iterative_phase_estimation>` explains how to monitor its progress.

Further reading
===============

- `Installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_
- `Visual Studio Code <https://code.visualstudio.com/download>`_
- `Microsoft Quantum Development Kit extension <https://marketplace.visualstudio.com/items?itemName=quantum.qsharp-lang-vscode>`_
- :doc:`Quickstart <../../user/quickstart>`
- :doc:`Glossary <../../glossary>`
