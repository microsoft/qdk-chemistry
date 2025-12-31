In-depth user guide
###################

Welcome to the in-depth documentation for the Quantum Development Kit (QDK) Chemistry library.
This documentation provides detailed information about QDK/Chemistry's structure, components, and usage patterns.

The QDK/Chemistry library is designed to facilitate quantum chemistry calculations and simulations.
It provides a set of tools and libraries for working with molecular structures, performing electronic structure calculations, analyzing quantum many-body systems, and running calculations on quantum hardware or simulators.

QDK/Chemistry features a unified interface for community quantum chemistry software packages, allowing seamless interoperability with established software while maintaining a consistent API.
This enables users to leverage specialized capabilities across different packages without changing their workflow.
An example workflow diagram is shown below.

.. graphviz:: /_static/diagrams/workflow.dot
   :alt: QDK/Chemistry Workflow
   :align: center

|

This documentation is organized into the following sections:

.. toctree::
   :maxdepth: 2

   design/index
   data/index
   algorithms/index
   basis_functionals
   model_hamiltonians
   plugins
