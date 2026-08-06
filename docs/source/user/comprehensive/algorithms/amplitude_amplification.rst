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

Where the QPE workspace has to live
-----------------------------------

Amplification rotates within the plane spanned by the good and bad components of the
prepared state, so it raises the probability of *accepting* but leaves the accepted state
itself unchanged. Marking a phase window therefore grows the weight of that window, not the
fidelity of the system register with the target eigenvector: QPE spreads each eigenvector
over several bins, and a neighbouring eigenvector leaking into the window is amplified by
exactly the same factor. Widen the phase register to suppress that leak, because more
rounds will not.

Folding QPE into the ``good_state_oracle`` instead, and passing the bare state preparation
as ``state_prep_oracle``, describes the same algorithm. Writing :math:`V` for the QPE
circuit, :math:`\Pi_W` for the phase window and :math:`A` for the state preparation, the
two Grover iterates are related by

.. math::

   Q_{\text{oracle}} = V^\dagger\, Q_{\text{prep}}\, V,

so they amplify the same weight, need the same rounds, and cost the same number of QPE
applications. The catch is that :math:`V^\dagger \Pi_W V` is a reflection only on the
register it is defined over. An oracle that allocates its own phase register, runs QPE,
compares, and then uncomputes leaves that register entangled with the system whenever an
eigenphase is not exactly representable, so it is not a reflection and the closed form
above stops holding. The QPE workspace has to remain part of the amplified register, which
is what passing the QPE circuit as ``state_prep_oracle`` does.

For high-fidelity eigenstate preparation, a reflection built by quantum signal processing
on a block encoding of the Hamiltonian removes the phase register, and its leakage, from
the problem entirely.

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
