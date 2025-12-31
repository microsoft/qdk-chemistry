==============================================
QDK/Chemistry:  A Quantum Applications Toolkit
==============================================

Welcome to QDK/Chemistry
========================

**QDK/Chemistry** provides a set of tools and libraries to enable cutting edge quantum chemistry calculations on quantum computers.
QDK/Chemistry tackles this problem holistically, by recognizing that a significant portion of the quantum applications pipeline heavily relies on the quality, robustness, and efficiency of the classical data preparation and post-processing steps.
QDK/Chemistry also serves as a platform for innovation, by providing a unified interface to a variety of quantum chemistry methods and packages, allowing researchers to focus on their areas of interest while integrating with a vast community of existing tools to accelerate development.

Key Features
============

A Unified Interface for Modular Quantum Applications Development
  The primary feature of QDK/Chemistry is its modular architecture that allows users to easily swap components in and out of their quantum applications workflows.
  This design promotes code reuse and accelerates development by enabling researchers to focus on their specific areas of interest.

Extensible-by-Design
  We recognize that the field of quantum algorithms for chemistry is rapidly evolving, and our goal is to provide a platform that can adapt to these changes.
  Through our plugin system, QDK/Chemistry is built with extensibility in mind, allowing users to easily integrate and modify new quantum chemistry methods, algorithms, and workflows to the framework.

Access to State-of-the-Art Algorithms
  QDK/Chemistry provides access to an ever-evolving collection of cutting-edge methods for solving quantum chemistry problems on quantum computers, available through both built-in modules and our extensible plugin system.
  This comprehensive toolkit spans from robust, best-in-class implementations of classical computational chemistry methods that generate essential input data for quantum algorithms, to sophisticated "chemistry-aware" quantum algorithms that leverage symmetries and other chemical properties to optimize calculations for near-term quantum hardware.
  By maintaining this broad spectrum of capabilities, we ensure researchers have the tools they need to push the boundaries of quantum chemistry while remaining practical for current quantum computing limitations.

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

   glossary
   references

.. todolist::
