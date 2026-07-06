Propagator
==========

The :class:`~qdk_chemistry.algorithms.propagator.base.Propagator` algorithm in QDK/Chemistry converts a time-dependent Hamiltonian over a finite time interval into a single effective (time-independent) Hamiltonian.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.TimeDependentQubitHamiltonian` and a time interval as input and produces a :class:`~qdk_chemistry.data.QubitHamiltonian` as output.

Overview
--------

Many quantum-chemistry workflows need to simulate how a system evolves under a Hamiltonian that changes with time — for example, a molecule driven by a laser pulse.
The time-dependent nature of these systems makes them challenging to simulate directly. The standard approach is to:

1. Divide the total evolution into short time steps.
2. For each step, compute an **effective Hamiltonian** that approximates the time-dependent Hamiltonian over that interval.
3. Feed that effective Hamiltonian into a time-evolution routine (a :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`) to produce the quantum circuit for that step.

Step 2 is what a propagator does.  Given an interval :math:`[t_1, t_2]` and a time-dependent Hamiltonian :math:`H(t)`, the propagator returns a time-independent :math:`H_\text{eff}` that best represents the evolution during that interval.

The propagator's output is divided by the step length :math:`\delta t = t_2 - t_1`, so that the downstream unitary builder — which multiplies by :math:`\delta t` — recovers the correct exponent.
This convention keeps propagator and builder responsibilities strictly separated.

Typical workflow
~~~~~~~~~~~~~~~~

A propagator is not usually called directly.  Instead, an
:doc:`EvolutionCircuitBuilder <evolution_circuit_builder>`
(e.g., ``EulerEvolutionCircuitBuilder``) creates one internally from its
``propagator`` setting and calls it once per time step.  The typical
sequence within each step is:

1. The circuit builder passes the :class:`~qdk_chemistry.data.TimeDependentQubitHamiltonian` and the current interval :math:`[t_1, t_2]` to the propagator
2. The propagator returns an effective :class:`~qdk_chemistry.data.QubitHamiltonian`
3. The :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>` implements the effective evolution as a :class:`~qdk_chemistry.data.UnitaryRepresentation`
4. A :doc:`CircuitMapper <circuit_mapper>` converts the unitary into executable gates

The :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.euler_builder.EulerEvolutionCircuitBuilder`
orchestrates this loop for every time step and combines the per-step
circuits into a single evolution circuit.


Using the Propagator
--------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and use a propagator.
Propagators are typically used as a nested algorithm within a
:class:`~qdk_chemistry.algorithms.time_evolution.hamiltonian_simulation.base.HamiltonianSimulation`,
but can also be created and called independently.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.propagator.base.Propagator` requires the following inputs:

TimeDependentQubitHamiltonian
   A :class:`~qdk_chemistry.data.TimeDependentQubitHamiltonian` describing how the Hamiltonian varies with time.
   Currently the only supported container type is
   :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`,
   which represents Hamiltonians of the form :math:`H(t) = H_0 + f(t)\,H_1` where :math:`f(t)` is a
   user-supplied drive function.

Time interval
   Two floats ``t_start`` and ``t_end`` defining the interval over which the effective Hamiltonian is computed.

.. rubric:: Creating a propagator

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/propagator.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

Settings vary by implementation.
See `Available implementations`_ below for implementation-specific options.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/propagator.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

.. rubric:: Running the propagator

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/propagator.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

When used as a nested algorithm inside an ``EvolutionCircuitBuilder``, the
propagator is configured via the ``propagator`` setting:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/propagator.py
      :language: python
      :start-after: # start-cell-nested
      :end-before: # end-cell-nested


Available implementations
-------------------------

QDK/Chemistry's :class:`~qdk_chemistry.algorithms.propagator.base.Propagator` provides a unified interface for computing effective Hamiltonians.
You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/propagator.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations

Time-averaged propagator
~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"magnus"``

This is the default (and currently only) propagator.  It computes the time-averaged Hamiltonian over each interval.  For a driven Hamiltonian :math:`H(t) = H_0 + f(t)\,H_1` the result is:

.. math::

   H_\text{eff} = H_0 + \bar{f}\,H_1,
   \qquad
   \bar{f} = \frac{1}{\delta t}\int_{t_1}^{t_2} f(t')\,\mathrm{d}t'

where the drive integral is evaluated by numerical quadrature (``scipy.integrate.quad``).

This is the leading-order term of the Magnus expansion.  This approximation gives :math:`O(\delta t^2)` per-step accuracy.

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``order``
     - int
     - Expansion order. Default: ``1``.  Only order 1 (time averaging) is currently implemented; higher values raise :class:`NotImplementedError`.

.. rubric:: Supported Hamiltonian types

Only :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer` Hamiltonians (:math:`H_0 + f(t)\,H_1`) are supported.
Passing any other container type raises :class:`NotImplementedError`.


Related classes
---------------

- :class:`~qdk_chemistry.data.TimeDependentQubitHamiltonian`: Input time-dependent Hamiltonian
- :class:`~qdk_chemistry.data.QubitHamiltonian`: Output effective Hamiltonian
- :class:`~qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base.EvolutionCircuitBuilder`: The circuit builder that calls the propagator each step
- :class:`~qdk_chemistry.algorithms.HamiltonianUnitaryBuilder`: Constructs the time-evolution unitary from the effective Hamiltonian produced by the propagator

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/propagator.py>`_ script.
- :doc:`EvolutionCircuitBuilder <evolution_circuit_builder>`: Time-evolution circuit composition
- :doc:`HamiltonianSimulation <hamiltonian_simulation>`: Full simulation with circuit execution and measurement
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Constructs the time-evolution unitary from the effective Hamiltonian
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
