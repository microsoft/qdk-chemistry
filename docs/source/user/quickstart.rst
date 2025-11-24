Quickstart
==========

These quickstart instructions provide a high-level overview of the typical workflow when using the QDK/Chemistry library.
More comprehensive documentation can be found in the :doc:`comprehensive/index`.

Installation
------------

To install QDK/Chemistry, please see the `installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_.

End-to-end example
------------------

This document is intended to provide a brief introduction to the QDK/Chemistry library by walking through a minimal end-to-end example for ground state energy estimation with state preparation and measurement.
The emphasis of this example is optimization:  reducing the resources required for the quantum computer.
The example starts with a molecular structure and ends with a simulated calculation using quantum algorithms.
The focus of this example is on high-level concepts and common coding patterns.

You can also view a related code sample in a Jupyter notebook format, along with several other examples, in the `examples folder <https://github.com/microsoft/qdk-chemistry/blob/main/examples/>`_ of the GitHub repository.

Create a Structure object
^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`comprehensive/data/structure` class represents a molecular structure, i.e. the coordinates of its atoms.
:doc:`comprehensive/data/structure` objects can be constructed manually, or via deserialization from a file.
QDK/Chemistry supports multiple serialization formats for :doc:`comprehensive/data/structure` objects, including the standard `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_, as well as QDK/Chemistry-specific JSON and HDF5 serialization schemes.
Internally to QDK/Chemistry, coordinates are always stored in Bohr/atomic units; however, when reading or writing to files, the units follow the file format convention (XYZ) or can be specified (JSON, HDF5).
See below for language specific examples of creating and serializing :doc:`comprehensive/data/structure` objects.


.. tab:: C++ API

   .. literalinclude:: ../../examples/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-1
      :end-before: // end-cell-1

.. tab:: Python API

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-1
      :end-before: # end-cell-1

Run a self-consistent field (SCF) calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a :doc:`comprehensive/data/structure` is created, an :term:`SCF` calculation can be performed to produce an initial :class:`~qdk_chemistry.data.Wavefunction` as well as an :term:`SCF` energy.
QDK/Chemistry performs :term:`SCF` calculations via instantiations of a :doc:`comprehensive/algorithms/scf_solver` algorithm.
Instantiations of the :doc:`comprehensive/algorithms/scf_solver` algorithm (and all other ``Algorithm`` classes) are managed by a factory.
See the :doc:`comprehensive/design/factory_pattern` documentation for more information on how it is used in the QDK/Chemistry.

The inputs for an :term:`SCF` calculation are a :doc:`comprehensive/data/structure` object, the charge and multiplicity of the molecular system, and information about the single-particle basis to be used.
Optionally, :doc:`comprehensive/design/settings` specific to the particular :doc:`comprehensive/algorithms/scf_solver` can be configured by accessing the ``settings()`` method.
The basis for the :term:`SCF` calculation can be set via a string input, a custom basis set or initial orbitals can also be provided.
See below for language-specific examples.

.. tab:: C++ API

   .. literalinclude:: ../../examples/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-2
      :end-before: // end-cell-2

.. tab:: Python API

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-2
      :end-before: # end-cell-2


Select an active space
^^^^^^^^^^^^^^^^^^^^^^

While a full set of :term:`SCF` orbitals are useful for many applications, they are often not the optimal set for further post-:term:`SCF` calculations.
For this reason, both localization (orbital manipulation) and active space selection (orbital selection) algorithms are provided within QDK/Chemistry.

QDK/Chemistry offers many methods for the selection of active spaces to reduce the problem size:  accurately modeling the quantum many-body problem while avoiding the prohibitive computational scaling of full configuration interaction.
See the :doc:`comprehensive/algorithms/active_space` documentation for a list of supported methods, along with their associated :doc:`comprehensive/design/settings`, which accompany the standard QDK/Chemistry distribution.

The following are language-specific examples of how to select a so-called "valence" active space containing a subset of only those orbitals surrounding the Fermi level.

.. tab:: C++ API

   .. literalinclude:: ../../examples/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-3
      :end-before: // end-cell-3

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see :doc:`comprehensive/algorithms/active_space`.

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-3
      :end-before: # end-cell-3


Calculate the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^

Once an active space has been selected, the electronic Hamiltonian can be computed within that active space to describe the energetic interactions between electrons.
QDK/Chemistry provides flexible Hamiltonian construction capabilities through the :doc:`comprehensive/algorithms/hamiltonian_constructor` algorithm.
The Hamiltonian constructor can generate the one- and two-electron integrals needed for subsequent quantum many-body calculations.

.. tab:: C++ API

   .. literalinclude:: ../../examples/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-4
      :end-before: // end-cell-4

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see :doc:`comprehensive/algorithms/hamiltonian_constructor`.

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-4
      :end-before: # end-cell-4

Compute a multi-configuration wavefunction for the active space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the active space Hamiltonian constructed, quantum many-body calculations
can be performed to obtain accurate electronic energies and wavefunctions.
QDK/Chemistry supports various Multi-Configuration (MC) methods including full
Configuration Interaction (CI) and selected :term:`CI` approaches.
:term:`MC` calculations are performed via instantiations of the :doc:`comprehensive/algorithms/mc_calculator` algorithm.

.. tab:: C++ API

   .. literalinclude:: ../../examples/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-5
      :end-before: // end-cell-5

.. tab:: Python API

   .. note::
      This example shows the API pattern. For complete working examples, see :doc:`comprehensive/algorithms/mc_calculator`.

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-5
      :end-before: # end-cell-5

Select important configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large active spaces, the multi-configuration :class:`~qdk_chemistry.data.Wavefunction` may contain thousands or millions of configurations, but often only a small subset contributes significantly to the overall state.
By identifying and retaining only the dominant configurations (those with the largest amplitudes), we can create a sparse wavefunction that maintains high fidelity with the original while dramatically reducing computational requirements for quantum state preparation.
This truncation is characterized by computing the overlap between the truncated state and the full wavefunction, and by recalculating the energy of the reduced wavefunction using the :doc:`comprehensive/algorithms/pmc`.

.. tab:: C++ API


   .. literalinclude:: ../../examples/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-6-fn
      :end-before: // end-cell-6-fn

.. tab:: Python API

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-6
      :end-before: # end-cell-6

Preparing a qubit representation of the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute quantum algorithms on actual quantum hardware, the Fermionic :doc:`Hamiltonian <comprehensive/data/hamiltonian>` must be mapped to a qubit Hamiltonian composed of Pauli operators.
QDK/Chemistry supports multiple encoding schemes including the Jordan-Wigner and Bravyi-Kitaev transformations.
The resulting qubit Hamiltonian is represented as a sum of Pauli strings (tensor products of Pauli :math:`X, Y, Z`, and identity operators), each with an associated coefficient.

For efficient energy estimation, the qubit Hamiltonian can be optimized in two ways.
First, by filtering out terms that have negligible expectation values given the sparse :class:`~qdk_chemistry.data.Wavefunction`, pre-computing their classical contributions.
Second, by grouping the remaining Pauli operators into commuting sets that can be measured simultaneously, significantly reducing the number of measurement circuits required on the quantum device.

.. tab:: C++ API

   .. code-block:: cpp

      // This step only available in Python API currently.

.. tab:: Python API


   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-7
      :end-before: # end-cell-7

Estimate the ground state energy using a quantum algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final step combines the state preparation circuit with the measurement circuits derived from the grouped qubit :doc:`Hamiltonian <comprehensive/data/hamiltonian>` to estimate the ground state energy.
Each measurement circuit appends specific Pauli basis rotations to the state preparation circuit, followed by computational basis measurements.
The measurement outcomes are repeated many times (shots) to build up statistics for each observable group.

The energy expectation value is computed by combining the measurement statistics from each group with their corresponding Hamiltonian coefficients and the pre-computed classical contributions.
The statistical nature of quantum measurements introduces variance in the energy estimate, which decreases as the number of shots is increased.

.. tab:: C++ API

   .. code-block:: cpp

      // This step only available in Python API currently.

.. tab:: Python API

   .. literalinclude:: ../../examples/quickstart.py
      :language: python
      :start-after: # start-cell-8
      :end-before: # end-cell-8

Additional examples
-------------------

Additional examples of workflows using QDK/Chemistry can be found in the ``examples`` directory of the source repository.
See the :doc:`comprehensive/index` for links to more detailed documentation on specific components of QDK/Chemistry.
