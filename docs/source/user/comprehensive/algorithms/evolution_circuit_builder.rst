Evolution circuit builder
=========================

The :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilder` is an abstract base class that defines the interface for constructing time-evolution circuits.
It serves as the central component that orchestrates circuit synthesis by composing a :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>` (to construct the time-evolution unitary) with a :doc:`CircuitMapper <circuit_mapper>` (to compile the unitary into a quantum circuit).

The ``EvolutionCircuitBuilder`` mirrors the role of :doc:`QpeCircuitBuilder <qpe_circuit_builder>` for phase estimation: it produces a :class:`~qdk_chemistry.data.circuit.Circuit` suitable for resource estimation or further analysis without requiring a simulator backend.

Overview
--------

Evolution circuit builders are responsible for:

1. **Composing circuits** from state preparation and time-evolution unitaries
2. **Generating strategy-specific circuits** (e.g., Euler stepping with configurable propagators and unitary builders)

The :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilder` abstract class provides the common interface, while concrete implementations handle variant-specific details (e.g., Euler stepping with Magnus propagation).

Use EvolutionCircuitBuilder
---------------------------

.. note::
   This algorithm is currently available only in the Python API.


This section demonstrates how to create, configure, and run an evolution circuit builder.
The ``run`` method returns a :class:`~qdk_chemistry.data.circuit.Circuit` containing the constructed state-prep + evolution circuit.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilder` requires the following inputs:

TimeDependentQubitHamiltonian
   A :class:`~qdk_chemistry.data.TimeDependentQubitHamiltonian` specifying the time-evolution schedule.
   For static Hamiltonians, use :class:`~qdk_chemistry.data.DrivenQubitHamiltonian` with a constant drive ``lambda t: 1.0``.

State preparation circuit
   A :class:`~qdk_chemistry.data.circuit.Circuit` that prepares the initial state.
   Use :func:`~qdk_chemistry.algorithms.state_preparation.identity_state_prep` for the trivial :math:`|0\rangle^{\otimes n}` initial state.


The :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilderSettings` class defines the general configuration parameters shared by all evolution circuit builders.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Setting
     - Type
     - Description
   * - ``evolution_builder``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Reference to the algorithm that constructs the time-evolution unitary. Default: ``"hamiltonian_unitary_builder"`` with method ``"trotter"``.
   * - ``circuit_mapper``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Reference to the algorithm that compiles the unitary to a quantum circuit. Default: ``"circuit_mapper"`` with method ``"pauli_sequence"``.
   * - ``propagator``
     - :class:`~qdk_chemistry.data.AlgorithmRef`
     - Reference to the propagator that evaluates the effective Hamiltonian over each time step. Default: ``"propagator"`` with method ``"magnus"``.
   * - ``total_time``
     - float
     - Total evolution time :math:`T`. Default: ``1.0``.
   * - ``dt``
     - float
     - Time step for evolution discretization. Each step is passed to the propagator. Default: ``0.0`` (invalid; user must set).

Once configured, the builder can be executed:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/evolution_circuit_builder.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Available Implementations
-------------------------

Euler Evolution Circuit Builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Class: ``EulerEvolutionCircuitBuilder``

.. rubric:: Factory name: ``"euler"``

Divides :math:`[0, T]` into Euler steps of size ``dt``.  At each step, the configured propagator (default: Magnus order-1) evaluates the effective Hamiltonian and the evolution builder constructs the time-evolution unitary.

The default propagator (``magnus``) computes the Magnus-expanded Hamiltonian over each interval, giving second-order global accuracy for smooth drives.  Other propagators can be substituted via the ``propagator`` setting.

**Usage example:**

.. literalinclude:: ../../../_static/examples/python/evolution_circuit_builder.py
   :language: python
   :start-after: start-cell-configure-euler
   :end-before: end-cell-configure-euler

Circuit Composition Details
---------------------------

The evolution circuit is built by composing:

1. **State Preparation** — Prepares the initial quantum state on the system register
2. **time-evolution Unitaries** — Applies the time-evolution unitary for each step obtained from the evolution builder

To accomplish this, the :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilder` maintains three key nested algorithm references:

**Nested Algorithm 1: Propagator**
   Reference setting: ``"propagator"``

   Default: :class:`~qdk_chemistry.data.AlgorithmRef` to ``"propagator"`` with method ``"magnus"``

   The propagator evaluates the effective Hamiltonian over each time interval :math:`[t, t+\mathrm{d}t]`.
   The default Magnus propagator computes :math:`H_\text{eff} = H_0 + \bar{f}\,H_1` where :math:`\bar{f}` is the time-averaged drive.

**Nested Algorithm 2: Evolution Builder**
   Reference setting: ``"evolution_builder"``

   Default: :class:`~qdk_chemistry.data.AlgorithmRef` to ``"hamiltonian_unitary_builder"`` with method ``"trotter"``

   The evolution builder (typically :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`) produces a :class:`~qdk_chemistry.data.UnitaryRepresentation` of the time-evolution operator from a :class:`~qdk_chemistry.data.QubitHamiltonian`.

**Nested Algorithm 3: Circuit Mapper**
   Reference setting: ``"circuit_mapper"``

   Default: :class:`~qdk_chemistry.data.AlgorithmRef` to ``"circuit_mapper"`` with method ``"pauli_sequence"``

   The circuit mapper (see :doc:`CircuitMapper <circuit_mapper>`) converts the :class:`~qdk_chemistry.data.UnitaryRepresentation` into an executable :class:`~qdk_chemistry.data.circuit.Circuit`.

Related Classes
---------------

- :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilderSettings`: Settings for evolution circuit builders
- :class:`~qdk_chemistry.data.circuit.Circuit`: Quantum circuit representation
- :class:`~qdk_chemistry.data.UnitaryRepresentation`: Output of unitary builders
- :class:`~qdk_chemistry.data.AlgorithmRef`: Nested algorithm references
- :class:`~qdk_chemistry.algorithms.time_evolution.hamiltonian_simulation.euler_integrator.EulerIntegrator`: High-level Hamiltonian simulation interface that uses evolution circuit builders internally

Further Reading
---------------

- :doc:`hamiltonian_unitary_builder`: Constructs unitary operators for Hamiltonian time evolution
- :doc:`circuit_executor`: Executes quantum circuits
- :doc:`../design/index`: QDK/Chemistry algorithm design principles
- :doc:`settings`: Configuration and settings management
- :doc:`factory_pattern`: Understanding algorithm creation and composition
