Model Hamiltonians
==================

QDK/Chemistry provides functionality to construct and manipulate model Hamiltonians used in quantum chemistry and condensed matter physics.
These model Hamiltonians serve as simplified representations of complex quantum systems, allowing researchers to study their properties and behaviors using quantum computing techniques.

The definition of model Hamiltonians differs from that of regular Hamiltonians in that they do not need to link to a specific molecular structure.
Hence, they also do not require the use of integrals derived from quantum chemistry calculations.
Instead, model Hamiltonians are defined directly in terms of their parameters.

Creating model Hamiltonians
----------------------------

To define a model Hamiltonian in QDK/Chemistry, users can utilize the ``ModelOrbitals`` class, which is a simplified version of the :doc:`Orbitals <data/orbitals>` class that doesn't require basis set information or molecular orbital coefficients.
This is particularly useful for studying model systems like Hubbard models, Heisenberg models, or other phenomenological Hamiltonians.

Example: Hubbard model Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how to create a simple Hubbard model Hamiltonian using ``ModelOrbitals`` and the ``Hamiltonian`` class:

.. tab:: Python API

   .. literalinclude:: ../../../examples/model_hamiltonian.py
      :language: python
      :start-after: # start-cell-1
      :end-before: # end-cell-1


Using model Hamiltonians with algorithms
-----------------------------------------

Model Hamiltonians created with ``ModelOrbitals`` can be used with any QDK/Chemistry algorithm that accepts :doc:`Hamiltonian <data/hamiltonian>` objects, including:

* :doc:`Multi-configuration calculators <algorithms/mc_calculator>` (:term:`FCI`, :term:`ASCI`, etc.)
* :class:`~qdk_chemistry.algorithms.CoupledClusterCalculator` (:term:`CCSD`, :term:`CCSD(T)`, etc.)
* Quantum algorithm interfaces (:term:`VQE`, :term:`QPE`, etc.)

Example with multi-configuration calculator:

.. tab:: Python API

   .. literalinclude:: ../../../examples/model_hamiltonian.py
      :language: python
      :start-after: # start-cell-2
      :end-before: # end-cell-2

See also
--------

* :doc:`data/orbitals` - Full Orbitals class documentation
* :doc:`data/hamiltonian` - Hamiltonian class documentation
* :doc:`algorithms/mc_calculator` - Multi-configuration calculator usage
