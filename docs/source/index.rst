==============================================
QDK/Chemistry:  A Quantum Applications Toolkit
==============================================

Welcome to QDK/Chemistry Documentation
======================================

**QDK/Chemistry** (Quantum Applications Toolkit) provides a comprehensive set of tools and libraries for quantum chemistry calculations and simulations.
It features a unified interface for vast community of quantum chemistry packages.

.. note::
   QDK/Chemistry collects anonymous usage and performance telemetry by default (for production builds) to help improve the product. The implemented telemetry events capture algorithm name and type, calculation execution duration, and number of basis functions (aggregated into buckets to protect sensitive information). Users can disable telemetry by setting the environment variable ``QDK_CHEMISTRY_PYTHON_TELEMETRY=False`` to one of the following: ``none``, ``disabled``, ``false``, or ``0``.

Key Features
============

* **Unified Interface** - Consistent API across quantum chemistry methods and packages
* **Molecular Structure Support** - Work with molecular coordinates, electronic structures, and quantum states
* **Advanced Algorithms** - Implementations of state-of-the-art quantum chemistry methods
* **Plugin System** - Extendable architecture for integrating with other quantum chemistry packages

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/quickstart
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

   glossary
   references

.. todolist::
