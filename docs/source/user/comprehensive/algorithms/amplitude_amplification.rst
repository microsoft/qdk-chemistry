Amplitude amplification
=======================

The :class:`~qdk_chemistry.algorithms.amplitude_amplification.AmplitudeAmplification`
algorithm increases the probability of measuring a state in a chosen subspace.
It takes two :class:`~qdk_chemistry.data.Circuit` objects:

- ``state_prep_oracle`` prepares the initial state.
- ``good_state_oracle`` flips a flag qubit for the good subspace.

The circuit first prepares the initial state. Each round then flags the
good subspace and reflects about the prepared state. If the initial probability
of the marked subspace is :math:`a`, the probability after :math:`k` rounds is

.. math::

   p_k = \sin^2\!\big((2k+1)\arcsin\sqrt{a}\big).

This gives the :math:`O(1/\sqrt{a})` query scaling. More rounds are not
always better: after the first maximum, additional rounds reduce the success
probability. Choose ``rounds`` from an estimate of state overlap :math:`a`.

Using amplitude amplification
-----------------------------

.. note::
   This algorithm is currently available only in the Python API.

.. rubric:: Creating an amplitude amplification algorithm

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/amplitude_amplification.py
      :language: python
      :start-after: # start-cell-create
      :end-before: # end-cell-create

.. rubric:: Configuring settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/amplitude_amplification.py
      :language: python
      :start-after: # start-cell-configure
      :end-before: # end-cell-configure

``run`` takes the width of the register both oracles act on from
``state_prep_oracle.num_qubits``. A :class:`~qdk_chemistry.data.Circuit` reports that from the
width its builder recorded, or from a ``numQubits`` Q# parameter, or by counting the qubit wires
of the built circuit. Pass ``num_qubits`` explicitly to override it, or if none of those apply.

The returned circuit measures the whole register. It also carries the same amplification
without measurement as a Q# callable, so a caller can append its own measurement instead.

Amplitude amplified QPE
-----------------------

From an initial state with some overlap with the target state, QPE
coherently writes an estimated phase to the leading phase register; the good
state oracle then checks the phase register, if it's in the target range, and
flips a flag qubit.

The :func:`~qdk_chemistry.algorithms.amplitude_amplification.phase_marking_oracle`
helper builds such a :class:`~qdk_chemistry.data.Circuit`. It reads the first
``num_phase_qubits`` qubits as a little-endian integer and marks the half-open range
``[start, stop)``. For a phase bin ``j``, use ``(j, j + 1)`` to mark only that bin,
``(0, j + 1)`` to mark values at or below it, or ``(j, 2**num_phase_qubits)`` to mark
values at or above it.

A block-encoded walk operator only certifies its phase estimate when its signal ancillas
return to :math:`|0\rangle`. Pass their indices, counted from the first qubit after the
phase register, as a third argument to require that too.

Build a measurement-free QPE circuit, mark the target phase bin, and amplify:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/amplitude_amplification.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``rounds``
     - ``int``
     - Number of Grover iterates (default 1). Must be non-negative.

Further Reading
---------------

- :doc:`PhaseEstimation <phase_estimation>`: the un-amplified algorithm.
- :doc:`QpeCircuitBuilder <qpe_circuit_builder>`: builds the coherent preparation this algorithm amplifies.
- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
