Amplitude amplification
=======================

The :class:`~qdk_chemistry.algorithms.amplitude_amplification.AmplitudeAmplification`
algorithm increases the probability of measuring a state in a chosen subspace.
It takes two :class:`~qdk_chemistry.data.Circuit` objects:

- ``state_prep_oracle`` prepares the initial state.
- ``good_state_oracle`` flips a flag qubit when the prepared state satisfies the
  success criterion.

The circuit first prepares the initial state. Each round then flags the
good subspace and reflects about the prepared state. If the initial probability
of the marked subspace is :math:`a`, the probability after :math:`k` rounds is

.. math::

   p_k = \sin^2\!\big((2k+1)\arcsin\sqrt{a}\big).

This gives the :math:`O(1/\sqrt{a})` query scaling. More rounds are not
always better: after the first maximum, additional rounds reduce the success
probability. Choose ``rounds`` from an estimate of state overlap :math:`a`.

Amplitude amplified QPE
-----------------------

From an initial state with some overlap with the target state, QPE
coherently writes an estimated phase to the leading phase register; the good
state oracle then checks the phase register, if it's in the target range, and
flips a flag qubit.

The :func:`~qdk_chemistry.algorithms.amplitude_amplification.phase_marking_oracle`
helper builds a :class:`~qdk_chemistry.data.Circuit` that marks a half-open
range of phase bins:

.. code-block:: python

   from qdk_chemistry.algorithms import phase_marking_oracle

   selected_bins = phase_marking_oracle(8, (12, 15))
   lower_bins = phase_marking_oracle(8, (0, 32))
   upper_bins = phase_marking_oracle(8, (224, 256))

The helper interprets the first ``num_phase_qubits`` qubits as a little-endian
integer and marks values in ``[start, stop)``. For a phase bin ``j``, use
``(j, j + 1)`` to mark only that bin, ``(0, j + 1)`` to mark values at or below
it, or ``(j, 2**num_phase_qubits)`` to mark values at or above it.

A block-encoded walk operator only certifies its phase estimate when its signal
ancillas return to :math:`|0\rangle`. Pass their indices, counted from the first
qubit after the phase register, as a third argument to require that too:

.. code-block:: python

   trusted_bins = phase_marking_oracle(8, (12, 15), [2, 3])

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
