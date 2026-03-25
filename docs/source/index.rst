==============================================
QDK/Chemistry:  A Quantum Applications Toolkit
==============================================

Welcome to QDK/Chemistry
========================

**QDK/Chemistry** provides an end-to-end toolkit for developing, simulating, and running quantum chemistry calculations on quantum computers.
It spans the entire quantum applications pipeline — from molecular setup and classical electronic structure computation through fermion-to-qubit encoding, quantum circuit synthesis, and simulation — all within a single, modular framework.

QDK/Chemistry recognizes that practical quantum chemistry applications depend on the quality of every stage: robust classical preprocessing, efficient quantum algorithm selection, and accurate post-processing.
By treating each stage as an interchangeable module, QDK/Chemistry serves as both a **development platform** for building new quantum algorithms and a **composable framework** for assembling reproducible quantum–classical pipelines from proven components.
Its plugin architecture provides a unified interface to native high-performance C++ implementations alongside established community packages, allowing researchers to mix and match the best tools for each stage of their workflow.

.. note::

   By default, this library collects anonymous usage and performance data to help improve the user experience and product quality. The telemetry implementation can be found in ``python/src/qdk_chemistry/utils/telemetry.py`` and all
   telemetry events are defined in ``python/src/qdk_chemistry/utils/telemetry_events.py``.

   To disable telemetry via bash, set the environment variable ``QSHARP_PYTHON_TELEMETRY`` to one of the
   following values: ``none``, ``disabled``, ``false``, or ``0``. For example:

   .. code-block:: bash

      export QSHARP_PYTHON_TELEMETRY='false'

   Alternatively, telemetry can be disabled within a python script by including the following at the top of the ``.py`` file:

   .. code-block:: python

      import os
      os.environ["QSHARP_PYTHON_TELEMETRY"] = "disabled"


Key Features
============

Quantum Algorithms for Chemistry
  QDK/Chemistry provides a growing collection of quantum algorithms for molecular simulation, including phase estimation, state preparation, fermion-to-qubit encoding, Hamiltonian time evolution, and observable estimation.
  These algorithms are designed to leverage chemical structure — exploiting sparsity, symmetry, and active-space reduction — to minimize quantum resources on both near-term and fault-tolerant hardware.

Classical Electronic Structure as Quantum Input
  The classical methods in QDK/Chemistry — self-consistent field solvers, multi-configuration techniques, orbital localization, and automated active space selection — are essential stages in the quantum pipeline.
  They produce the high-quality molecular orbitals, reference wavefunctions, and compact Hamiltonians that make quantum algorithms practical.

Composable Framework
  Every algorithm in QDK/Chemistry is an interchangeable module with a standardized interface.
  The factory/registry architecture lets users assemble custom quantum–classical pipelines by selecting and combining implementations at each stage — without changing application code.

Extensible-by-Design
  The field of quantum algorithms for chemistry evolves rapidly, and QDK/Chemistry is built to adapt.
  Its plugin system allows new methods and integrations to be registered and used through the same unified API.


.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/quickstart
   user/features
   user/comprehensive/index

.. _apidocs:

.. toctree::
   :maxdepth: 2
   :caption:  API Reference

   api/python_api
   api/cpp_api

.. toctree::
   :maxdepth: 1
   :caption:  Supporting Information

   changelog
   glossary
   references

Companion repositories
----------------------

- `microsoft/qdk-chemistry-data <https://github.com/microsoft/qdk-chemistry-data>`_: Curated datasets and supporting materials for QDK/Chemistry.

Citing QDK/Chemistry
=====================

If you use QDK/Chemistry in your work, please cite our paper :cite:`Baker2026`:

   N. A. Baker *et al.*, "QDK/Chemistry: A Modular Toolkit for Quantum Chemistry Applications,"
   `arXiv:2601.15253 <https://arxiv.org/abs/2601.15253>`_ (2026).
