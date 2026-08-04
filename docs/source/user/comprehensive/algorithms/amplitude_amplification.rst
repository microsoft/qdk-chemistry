Amplitude amplification
=======================

The :class:`~qdk_chemistry.algorithms.amplitude_amplification.base.AmplitudeAmplification` is an algorithm that boosts the probability that a prepared state is found in a marked subspace.
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

More rounds are not always better, and could result in overshoot when we past the first maximum acceptance falls again.
The round count is therefore taken from the Yoder-Low-Chuang **fixed-point** schedule, which takes in the ``min_overlap`` *lower* bound plus a ``tolerance`` :math:`\delta`.

Setting ``rounds`` to a non-negative value runs that many plain Grover iterates instead.

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
     - Explicit number of plain Grover iterates. Negative (the default) derives a fixed-point schedule instead.
   * - ``min_overlap``
     - ``double``
     - Lower bound on the overlap :math:`a`. Required unless ``rounds`` is set.
   * - ``tolerance``
     - ``double``
     - Fixed-point tolerance :math:`\delta`; success exceeds :math:`1-\delta^2` (default 0.1).
     
Further Reading
---------------

- :doc:`PhaseEstimation <phase_estimation>`: the un-amplified algorithm.
- :doc:`QpeCircuitBuilder <qpe_circuit_builder>`: builds the coherent preparation this algorithm amplifies.
- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
- Yoder, T. J., Low, G. H., and Chuang, I. L. *Fixed-point quantum search with an optimal number of queries*, Phys. Rev. Lett. **113**, 210501 (2014), `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`_.
