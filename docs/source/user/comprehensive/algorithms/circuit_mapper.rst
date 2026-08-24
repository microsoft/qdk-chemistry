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
This mapper applies to **particle-conserving** Hamiltonians only. It relies on
:math:`|0\ldots0\rangle` staying in its own particle-number sector.

The :math:`|0\rangle` branch picks up :math:`U|0\ldots0\rangle = e^{i\varphi_0}|0\ldots0\rangle`
with :math:`\varphi_0 = -E_0 t` and :math:`E_0 = \langle 0\ldots0|H|0\ldots0\rangle`.
Only the diagonal (:math:`I`/:math:`Z`) terms of the product formula phase the vacuum, so
:math:`\varphi_0` is known classically. The mapper cancels it with a phase gate
:math:`R_1(\varphi_0) = \mathrm{diag}(1, e^{i\varphi_0})` on the control, giving a genuine
controlled-:math:`U` up to a global phase for any :math:`E_0`.

.. rubric:: Why grouping is required

The vacuum must remain an eigenstate of the evolution, which is exactly what particle
conservation buys: :math:`H` cannot connect :math:`|0\ldots0\rangle` to any other occupation
number. Leaked amplitude entangles the vacuum register with the control and destroys the control
coherence, losing the very phase the algorithm is trying to measure.

For a molecular Hamiltonian every fermionic term (:math:`a_p^\dagger a_q` and
:math:`a_p^\dagger a_r^\dagger a_s a_q`) annihilates the vacuum. A single Pauli string is unitary
and cannot, so the cancellation survives Trotterization only when the strings coming from one
fermionic term are exponentiated as one contiguous block:

.. math::

   U|0\ldots0\rangle = e^{-it\sum_i P_i}|0\ldots0\rangle
   \approx \prod_i e^{-it P_i}|0\ldots0\rangle = |0\ldots0\rangle .

The :ref:`vacuum-annihilating term grouper <algorithms-term-grouper>` (factory name
``"vacuum_annihilating"``) produces that ordering. A Pauli factor *flips* a qubit when it
exchanges :math:`|0\rangle` and :math:`|1\rangle`, which :math:`X` and :math:`Y` do and
:math:`I` and :math:`Z` do not. Terms flipping the same qubits are the only ones that can
cancel, so the grouper walks them in order and closes a group as soon as the accumulated
amplitude :math:`\sum_j c_j i^{n_Y^{(j)}}` vanishes — each emitted group therefore annihilates
the vacuum outright. Groups are restricted to one :math:`Y`-count parity so that their members
commute and can be exponentiated term by term.

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
