Amplitude amplification
=======================

The :class:`~qdk_chemistry.algorithms.amplitude_amplification.AmplitudeAmplification` is an algorithm that boosts the probability that a prepared state is found in a marked subspace.
Given a guiding state with overlap :math:`a` on the target eigenstate, it succeeds after :math:`O(1/\sqrt{a})` repetitions:

.. math::

   Q = -\,U_\psi S_0 U_\psi^\dagger \, S_G

Here :math:`U_\psi` prepares the guiding state;
:math:`S_0=I-2\lvert0\rangle\!\langle0\rvert` changes the phase of the all-zero state; and
:math:`S_G=I-2\Pi_G` changes the phase of every state in the marked subspace, whose projector is :math:`\Pi_G`.
Thus :math:`-U_\psi S_0 U_\psi^\dagger` is a reflection about :math:`\lvert\psi\rangle`.
Together, the two reflections rotate amplitude from the unmarked component toward the marked component by :math:`2\theta` per application of :math:`Q`.

With :math:`\theta = \arcsin\sqrt{a}`, the acceptance probability after :math:`k` rounds is

.. math::

   p_k = \sin^2\!\big((2k+1)\theta\big).

More rounds are not always better: after the first maximum, additional Grover
iterations reduce the acceptance probability. The caller must choose ``rounds``
from an estimate of :math:`a`.

QPE phase markers
-----------------

The :func:`~qdk_chemistry.algorithms.amplitude_amplification.phase_marking_oracle`
helper builds a marking-oracle :class:`~qdk_chemistry.data.Circuit` from either
explicit QPE phase-bin indices or an inclusive upper threshold:

.. code-block:: python

   from qdk_chemistry.algorithms import phase_marking_oracle

   selected_bins = phase_marking_oracle(8, target_indices=[12, 13, 14])
   lower_bins = phase_marking_oracle(8, threshold=31)

The oracle interprets the first ``num_phase_qubits`` qubits as a little-endian
integer. Thus ``threshold=31`` marks phase-bin values :math:`j \le 31`. Exactly
one of ``target_indices`` or ``threshold`` must be supplied.

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
