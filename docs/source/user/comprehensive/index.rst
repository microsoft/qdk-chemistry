In-depth user guide
###################

Welcome to the in-depth documentation for the Quantum Development Kit (QDK) Chemistry library.
This documentation provides detailed information about QDK/Chemistry's structure, components, and usage patterns.

QDK/Chemistry provides every stage of the quantum chemistry applications pipeline in a single framework:

- **Classical electronic structure**: SCF solvers, orbital localization, active space selection, and multi-configuration methods that produce the high-quality inputs required by quantum algorithms
- **Quantum algorithm building blocks**: fermion-to-qubit mapping, quantum state preparation, phase estimation, Hamiltonian time evolution, and observable estimation
- **Composable workflow design**: every algorithm is an interchangeable module — swap implementations, mix backends, and construct custom quantum–classical pipelines through a unified factory/registry architecture

The library is designed so that each stage feeds naturally into the next, while any individual component can be replaced without affecting the rest of the workflow.
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
