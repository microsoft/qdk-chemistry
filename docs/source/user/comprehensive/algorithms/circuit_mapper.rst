Controlled circuit mapper
=========================

The :class:`~qdk_chemistry.algorithms.ControlledCircuitMapper` algorithm in QDK/Chemistry converts a :class:`~qdk_chemistry.data.UnitaryRepresentation` into a *controlled* quantum circuit.
Following QDK/Chemistry's :doc:`algorithm design principles <../design/index>`, it takes a :class:`~qdk_chemistry.data.UnitaryRepresentation` as input and produces a :class:`~qdk_chemistry.data.Circuit` as output.
Control and target qubit indices are configured via settings.

Overview
--------

Controlled unitaries — operations of the form :math:`C\text{-}U` that apply :math:`U` to a target register conditioned on the state of a control qubit — are a building block in many quantum algorithms.
Mathematically, for a single control qubit the controlled unitary acts as:

.. math::

   C\text{-}U \;=\; |0\rangle\langle 0| \otimes I \;+\; |1\rangle\langle 1| \otimes U

That is, the target register is left unchanged when the control is :math:`|0\rangle` and :math:`U` is applied when the control is :math:`|1\rangle`.

The :class:`~qdk_chemistry.algorithms.ControlledCircuitMapper` synthesises these controlled operations from the abstract :class:`~qdk_chemistry.data.UnitaryRepresentation` representation produced by a :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`.
This is a core component of algorithms such as :doc:`PhaseEstimation <phase_estimation>`, which requires repeated controlled applications :math:`C\text{-}U^{2^k}`.

The mapper takes inputs:

1. A :class:`~qdk_chemistry.data.UnitaryRepresentation` — produced by a :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`

Control and target qubit indices are configured via the mapper's settings (``control_indices`` and ``target_indices``).

The resulting :class:`~qdk_chemistry.data.Circuit` implements the controlled unitary and can be executed by a :doc:`CircuitExecutor <circuit_executor>`.


Using the ControlledCircuitMapper
---------------------------------

.. note::
   This algorithm is currently available only in the Python API.

This section demonstrates how to create, configure, and run the circuit mapper.

Input requirements
~~~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.algorithms.ControlledCircuitMapper` requires:

UnitaryRepresentation
   A :class:`~qdk_chemistry.data.UnitaryRepresentation` specifying the unitary to be controlled.

Settings
   ``control_indices`` (list of int): Which qubits serve as controls (default: ``[0]``).
   ``target_indices`` (list of int): Which qubits the unitary acts on (default: auto-filled).

.. rubric:: Creating a mapper

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Running the mapper

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run


Available implementations
-------------------------

You can discover available implementations programmatically:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-list-implementations
      :end-before: # end-cell-list-implementations


Pauli sequence mapper
~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"pauli_sequence"`` (default)

Given a time-evolution unitary expressed as a :class:`~qdk_chemistry.data.PauliProductFormulaContainer` — a sequence of exponentiated Pauli terms :math:`e^{-i\theta_j P_j}` — this mapper constructs a controlled version by:

1. Rotating each Pauli operator :math:`P_j` into the Z basis
2. Entangling the target qubits with a CNOT ladder
3. Applying a controlled :math:`R_z(2\theta_j)` rotation from the control qubit
4. Uncomputing the basis rotations and entangling operations

.. note::
   The current implementation supports a single control qubit.

Controlled SWAP Pauli sequence mapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Factory name: ``"cswap_pauli_sequence"``

This mapper avoids controlling every gate. It allocates an internal *vacuum* register in
:math:`|0\ldots0\rangle`, conditionally swaps it with the system register, applies the
**uncontrolled** evolution to the vacuum register, and uncomputes the swap.
When the control is :math:`|0\rangle` the evolution hits the vacuum and the system is untouched;
when it is :math:`|1\rangle` the system is parked in the vacuum register and evolved.
A single layer of controlled-:math:`\mathrm{SWAP}` gates thus replaces a fully controlled circuit.

.. rubric:: Why grouping is required

The construction is exact **only if** the vacuum is an eigenstate of the evolution,
:math:`U|0\ldots0\rangle = \lambda|0\ldots0\rangle`.
Otherwise the vacuum register stays entangled with the control, and resetting it at the end
destroys the control coherence — the phase the algorithm is trying to measure is silently lost.

For a molecular Hamiltonian this looks automatic: every fermionic term
(:math:`a_p^\dagger a_q` and :math:`a_p^\dagger a_r^\dagger a_s a_q`) annihilates the vacuum, so
:math:`H|0\ldots0\rangle = 0` once the core energy is excluded, as all supported
:doc:`qubit mappers <qubit_mapper>` do.
It is *not* automatic after Trotterization. A single Pauli string is unitary and can never
annihilate a state — only the weighted **sum** of the strings coming from one fermionic term
cancels. So the cancellation survives Trotterization only when those strings are exponentiated
as one contiguous block:

.. math::

   U|0\ldots0\rangle = e^{-it\sum_i P_i}|0\ldots0\rangle
   \approx \prod_i e^{-it P_i}|0\ldots0\rangle = |0\ldots0\rangle .

The :ref:`qubit-flip term grouper <qubit-flip-term-grouper>` produces exactly that ordering:
it groups the Pauli strings that flip the same qubits, which are precisely the ones whose
amplitudes can cancel on the vacuum.
The mapper validates its input product formula and raises a :class:`ValueError` when the
ordering is not vacuum preserving, rather than returning a wrong result.

.. rubric:: Worked example

Take the two-mode Hamiltonian

.. math::

   H = a_0^\dagger a_1 + a_1^\dagger a_0 + a_0^\dagger a_0
     = \tfrac12 (XX + YY) + \tfrac12 (I - Z_0),
   \qquad H|00\rangle = 0 .

For one Trotter step at :math:`\Delta t = \pi/2`, the interleaved ordering ``XX, Z0, YY, I``
splits the ``XX``/``YY`` pair and gives

.. math::

   U_{\text{interleaved}}|00\rangle
   = \tfrac{1-i}{2}|00\rangle + \tfrac{-1+i}{2}|11\rangle ,
   \qquad
   \bigl|\langle 11|U_{\text{interleaved}}|00\rangle\bigr|^2 = \tfrac12 ,

so half of the vacuum amplitude leaks. Keeping the partners together instead gives

.. math::

   U_{\text{grouped}}
   = e^{-i\frac{\pi}{4}(I - Z_0)}\, e^{-i\frac{\pi}{4}(XX + YY)},
   \qquad
   U_{\text{grouped}}|00\rangle = |00\rangle .

.. rubric:: End-to-end usage

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/circuit_mapper.py
      :language: python
      :start-after: # start-cell-cswap
      :end-before: # end-cell-cswap

Related classes
---------------

- :class:`~qdk_chemistry.data.UnitaryRepresentation`: The underlying unitary representation (input)
- :class:`~qdk_chemistry.data.Circuit`: Output circuit
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Produces the :class:`~qdk_chemistry.data.UnitaryRepresentation` that this mapper consumes

Further reading
---------------

- The above examples can be downloaded as a complete `Python <../../../_static/examples/python/circuit_mapper.py>`_ script.
- :doc:`PhaseEstimation <phase_estimation>`: Uses the circuit mapper to build controlled-:math:`U` operations
- :doc:`HamiltonianUnitaryBuilder <hamiltonian_unitary_builder>`: Constructs the input unitaries
- :doc:`Settings <settings>`: Configuration settings for algorithms
- :doc:`Factory Pattern <factory_pattern>`: Understanding algorithm creation
