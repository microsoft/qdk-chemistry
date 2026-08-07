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


``run`` takes the two oracles and reads the register width from a resource estimate of
the state preparation. The returned circuit measures the whole register. It also carries
the same amplification without measurement as a Q# callable, so a caller can append its
own measurement instead. See below for an example of amplifying a measurement-free QPE circuit.

Amplitude amplified QPE
-----------------------

Build a measurement-free QPE circuit, mark the target phase bin, and amplify.
:func:`~qdk_chemistry.algorithms.amplitude_amplification.phase_marking_oracle` reads the
register layout from the QPE circuit, so only the target bins have to be given:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/amplitude_amplification.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

The target can also be named as an energy window.
This only applies to a QPE circuit built on a qubitization walk, whose eigenvalues are
:math:`e^{\pm i\arccos(E/\lambda)}` for :math:`\lambda` the L1 norm of the Hamiltonian.

Alternatively, the marked phase window can be replaced by a reflection onto the target
eigenspace built with quantum signal processing on a block encoding of the Hamiltonian,
amplifying an initial state preparation directly. See Lin and Tong,
`arXiv:2002.12508 <https://arxiv.org/abs/2002.12508>`_, and its use in
`arXiv:2510.07273 <https://arxiv.org/abs/2510.07273>`_, Section 2.

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
- Lin, L. and Tong, Y. *Near-optimal ground state preparation*, `arXiv:2002.12508 <https://arxiv.org/abs/2002.12508>`_: the signal-processing eigenspace reflection.
