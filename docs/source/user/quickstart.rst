Quickstart
==========

These quickstart instructions provide a high-level overview of the typical workflow when using the QDK/Chemistry library.
More comprehensive documentation can be found in the :doc:`comprehensive/index`. A list of features can be found in the :doc:`features` document.

Installation
------------

To install QDK/Chemistry, please see the `installation instructions <https://github.com/microsoft/qdk-chemistry/blob/main/INSTALL.md>`_.

End-to-end example
------------------

This document is intended to provide a brief introduction to the QDK/Chemistry library by walking through a minimal end-to-end example for ground state energy estimation with state preparation and measurement.
The emphasis of this example is optimization:  reducing the resources required for the quantum computer to run a simple chemistry application.
The example starts with a molecular structure and ends with an energy estimation computed by simulating a quantum circuit.
The focus of this example is on high-level concepts and common coding patterns that can be extended to other applications.

You can also view a related code sample in a Jupyter notebook format, along with several other examples, in the `examples folder <https://github.com/microsoft/qdk-chemistry/blob/main/examples/>`_ of the GitHub repository.

Create a Structure object
^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`comprehensive/data/structure` class represents a molecular structure, i.e. the coordinates of its atoms.
:doc:`comprehensive/data/structure` objects can be constructed manually, or via deserialization from a file.
QDK/Chemistry supports multiple serialization formats for :doc:`comprehensive/data/structure` objects, including the standard `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_, as well as QDK/Chemistry-specific JSON and HDF5 serialization schemes.
Internally to QDK/Chemistry, coordinates are always stored in Bohr/atomic units; however, when reading or writing to files, the units follow the file format convention (Angstrom for XYZ) or can be specified (JSON, HDF5).
See below for language specific examples of creating and serializing :doc:`comprehensive/data/structure` objects.


.. tab:: C++ API

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-structure
      :end-before: // end-cell-structure

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-structure
      :end-before: # end-cell-structure

Run a self-consistent field (SCF) calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once a :doc:`comprehensive/data/structure` is created, an :term:`SCF` calculation can be performed to produce an initial :class:`~qdk_chemistry.data.Wavefunction` as well as an :term:`SCF` energy.
QDK/Chemistry performs :term:`SCF` calculations via instantiations of a :doc:`comprehensive/algorithms/scf_solver` algorithm, and is the first instance of the separation :doc:`Data classes <./comprehensive/data/index>` and :doc:`Algorithm classes <./comprehensive/algorithms/index>` most will encounter in QDK/Chemistry.
See the :doc:`design principles <./comprehensive/design/index>` documentation for more information on this pattern and how data flow is generally treated in QDK/Chemistry.
Instantiations of the :doc:`comprehensive/algorithms/scf_solver` algorithm (and all other :doc:`Algorithm classes <./comprehensive/algorithms/index>`) are managed by a factory.
See the :doc:`comprehensive/algorithms/factory_pattern` documentation for more information on how it is used in the code base.

The inputs for an :term:`SCF` calculation are a :doc:`comprehensive/data/structure` object, the total charge and spin multiplicity of the molecular system, and information about the single-particle basis to be used.
Optionally, :doc:`comprehensive/algorithms/settings` specific to the particular :doc:`comprehensive/algorithms/scf_solver` can be configured to control the execution of the :term:`SCF` algorithm (e.g. convergence tolerances, etc) by accessing the ``settings()`` method.
The basis for the :term:`SCF` calculation can be set via a string input (specifying one of the :ref:`available_basis_sets`), a custom :doc:`comprehensive/data/basis_set` or initial :doc:`comprehensive/data/orbitals` can also be provided.

.. tab:: C++ API

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-scf
      :end-before: // end-cell-scf

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-scf
      :end-before: # end-cell-scf


Select an active space
^^^^^^^^^^^^^^^^^^^^^^

While a full set of :term:`SCF` orbitals are useful for many applications, they are often not the optimal set for further post-:term:`SCF` calculations, including algorithms intended for a quantum computer.
For this reason, both :doc:`orbital localization <comprehensive/algorithms/localizer>` and :doc:`active space selection <comprehensive/algorithms/active_space>` algorithms are provided within QDK/Chemistry.

QDK/Chemistry offers many methods for the selection of active spaces to reduce the problem size:  accurately modeling the quantum many-body problem while avoiding the prohibitive computational scaling of full configuration interaction.
See the :doc:`comprehensive/algorithms/active_space` documentation for a list of supported methods, along with their associated :doc:`comprehensive/algorithms/settings`, which accompany the standard QDK/Chemistry distribution.

The following are language-specific examples of how to select a so-called "valence" active space containing a subset of only those orbitals surrounding the Fermi level - in this case 6 electrons to be distributed in 6 orbitals (6e, 6o).

.. tab:: C++ API

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-active-space
      :end-before: // end-cell-active-space

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-active-space
      :end-before: # end-cell-active-space


Calculate the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^

Once an active space has been selected, the electronic Hamiltonian can be computed within that active space to describe the energetic interactions between electrons.
QDK/Chemistry provides flexible Hamiltonian construction capabilities through the :doc:`comprehensive/algorithms/hamiltonian_constructor` algorithm.
The Hamiltonian constructor generates the one- and two-electron integrals (or factorizations thereof) needed for subsequent quantum many-body calculations and quantum algorithms.

.. tab:: C++ API

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-hamiltonian-constructor
      :end-before: // end-cell-hamiltonian-constructor

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-hamiltonian-constructor
      :end-before: # end-cell-hamiltonian-constructor

Compute a multi-configurational wavefunction for the active space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the active space Hamiltonian constructed, quantum many-body calculations can be performed to obtain multi-configurational wavefunctions that go beyond the single-determinant :term:`SCF` approximation.
QDK/Chemistry supports various Multi-Configuration (:term:`MC`) methods including Complete Active Space Configuration Interaction (:term:`CASCI`) and selected :term:`CI` approaches.
:term:`MC` calculations are performed via instantiations of the :doc:`comprehensive/algorithms/mc_calculator` algorithm, which takes as input an instance of an active space :doc:`comprehensive/data/hamiltonian`, and the number of alpha and beta electrons, and produces as output a :class:`~qdk_chemistry.data.Wavefunction` representing the multi-configurational state as well as its associated energy.

While multi-configurational methods provide more accurate energy estimates than :term:`SCF`, their primary role in the quantum applications workflow is to generate high-quality initial states for quantum algorithms.
On scaled fault-tolerant quantum computers, these classically-computed wavefunctions serve as the foundation for state preparation circuits, enabling algorithms such as quantum phase estimation to achieve chemical accuracy for systems that remain intractable for purely classical methods.
These methods also serve as a critical analysis tool, allowing users to better understand the electronic structure of their systems of interest and the potential for quantum algorithms to provide meaningful utility over classical state of the art.

In the following example, as the aforementioned (6e, 6o) active space is relatively small, we perform a :term:`CASCI` calculation to obtain the exact ground state wavefunction within the active space.

.. tab:: C++ API

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-mc-compute
      :end-before: // end-cell-mc-compute

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-mc-compute
      :end-before: # end-cell-mc-compute

Select important configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For large active spaces, the multi-configuration :class:`~qdk_chemistry.data.Wavefunction` may contain thousands or millions of configurations, but often only a small subset contributes significantly to the overall state.
By identifying and retaining only the dominant configurations (those with the largest amplitudes), we can create a sparse wavefunction that maintains high fidelity with respect to the original Wavefunction while dramatically reducing resource requirements for quantum state preparation.

.. tab:: C++ API

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-wfn-fn-select-configs
      :end-before: // end-cell-wfn-fn-select-configs

   .. literalinclude:: ../_static/examples/cpp/quickstart.cpp
      :language: cpp
      :start-after: // start-cell-wfn-select-configs
      :end-before: // end-cell-wfn-select-configs

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-wfn-select-configs
      :end-before: # end-cell-wfn-select-configs

Preparing a qubit representation of the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute quantum algorithms on actual quantum hardware, the Fermionic :doc:`Hamiltonian <comprehensive/data/hamiltonian>` must be mapped to a qubit Hamiltonian composed of Pauli operators.
QDK/Chemistry supports multiple encoding schemes including the Jordan-Wigner and Bravyi-Kitaev transformations.
The resulting qubit Hamiltonian is represented as a sum of Pauli strings (tensor products of Pauli :math:`X, Y, Z`, and identity operators), each with an associated coefficient.

For efficient energy estimation, the qubit Hamiltonian can be optimized in two ways:


1. Filter out terms that have negligible expectation values given the sparse :class:`~qdk_chemistry.data.Wavefunction`, pre-computing their classical contributions.
2. Group the remaining Pauli operators into commuting sets that can be measured simultaneously, significantly reducing the number of measurement circuits required on the quantum device.

.. tab:: C++ API

   .. code-block:: cpp

      // This step is currently only available in the Python API.

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-qubit-hamiltonian
      :end-before: # end-cell-qubit-hamiltonian

Generate the state preparation circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO: Add more references to other methods

Given the classical representation of the sparse multi-configurational :class:`~qdk_chemistry.data.Wavefunction`, a quantum circuit can be generated to prepare this state on a quantum computer.
This can be done in many ways, including via Isometry encoding :cite:`Christandl2016`, linear combinations of unitaries, and tensor product methods.
However, when the wavefunction is very sparse, these methods can be inefficient.
In QDK/Chemistry, we provide a specialized method for generating state preparation circuits for sparse wavefunctions based on the construction of sparse isometries over GF(2) with X gates, provided as a :doc:`comprehensive/algorithms/state_preparation` algorithm.
See :doc:`comprehensive/algorithms/state_preparation` for more details on the other algorithms provided for state preparation in QDK/Chemistry.

.. tab:: C++ API

   .. code-block:: cpp

      // This step is currently only available in the Python API.

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-state-prep-circuit
      :end-before: # end-state-prep-circuit


Estimate the ground state energy by sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The final step combines the state preparation circuit with the measurement circuits derived from the grouped qubit Hamiltonian to estimate the ground state energy.
Each measurement circuit appends specific Pauli basis rotations to the state preparation circuit, followed by computational basis measurements.
The measurement outcomes are repeated many times (shots) to build up statistics for each observable group.

The energy expectation value is computed by combining the measurement statistics from each group with their corresponding Hamiltonian coefficients and the pre-computed classical contributions.
The statistical nature of quantum measurements introduces variance in the energy estimate, which decreases as the number of shots is increased.

.. tab:: C++ API

   .. code-block:: cpp

      // This step is currently only available in the Python API.

.. tab:: Python API

   .. literalinclude:: ../_static/examples/python/quickstart.py
      :language: python
      :start-after: # start-cell-energy-estimation
      :end-before: # end-cell-energy-estimation

Additional examples
-------------------

For more information, see:

- Above examples as complete `C++ <../_static/examples/cpp/quickstart.cpp>`_ and `Python <../_static/examples/python/quickstart.py>`_ scripts.
- Additional examples of workflows using QDK/Chemistry in the `examples <https://github.com/microsoft/qdk-chemistry/tree/main/examples>`_ directory of the source repository.
- :doc:`comprehensive/index` for links to more detailed documentation on specific components of QDK/Chemistry.
