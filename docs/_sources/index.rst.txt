==============================================
QDK/Chemistry:  A Quantum Applications Toolkit
==============================================

Welcome to QDK/Chemistry
========================

**QDK/Chemistry** provides an end-to-end toolkit for developing, simulating, and running quantum chemistry calculations on quantum computers.
It spans the entire quantum applications pipeline — from molecular setup and classical electronic structure through quantum algorithm execution and simulation — all within a single, modular framework.

QDK/Chemistry recognizes that practical quantum chemistry depends on the quality of every stage, not just the quantum algorithm itself.
By treating each stage as an interchangeable module, QDK/Chemistry serves as both a **development platform** for building new quantum algorithms and a **composable framework** for assembling reproducible quantum–classical pipelines from proven components.
Its plugin architecture provides a unified interface to native high-performance C++ implementations alongside established community packages, so researchers can mix and match the best tools for their problem without being locked into any single approach.

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
  A growing collection of chemistry-aware quantum algorithms that leverage molecular structure to minimize quantum resources on both near-term and fault-tolerant hardware.

Classical Electronic Structure
  Production-quality classical methods that generate the molecular orbitals, reference states, and compact Hamiltonians that quantum algorithms depend on.

Composable, Extensible Architecture
  Every component is an interchangeable module with a standardized interface.
  The plugin system lets users swap implementations, integrate external packages, and assemble custom quantum–classical pipelines — without changing application code.


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

   release-notes/index
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
