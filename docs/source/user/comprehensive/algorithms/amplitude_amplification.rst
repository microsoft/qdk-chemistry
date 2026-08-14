Amplitude amplification
=======================

The :class:`~qdk_chemistry.algorithms.amplitude_amplification.amplitude_amplification.AmplitudeAmplification`
algorithm increases the probability of measuring a state in a chosen subspace.
It takes two :class:`~qdk_chemistry.data.Circuit` objects:

- ``state_prep_oracle`` prepares the initial state from :math:`|0\rangle`.
- ``good_state_oracle`` marks the good subspace: applied to ``(register, flag)`` it flips the
  flag qubit when the register holds a good state, and leaves the register itself untouched.

The circuit first prepares the initial state. Each round then flags the
good subspace and reflects. If the initial probability of the marked
subspace is :math:`a`, the probability after :math:`k` rounds is

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

``run`` takes the two oracles as :class:`~qdk_chemistry.data.Circuit`. See below for an
example of amplifying an eigenstate found by QPE.

QPE subspace marking oracle
---------------------------

The :class:`~qdk_chemistry.algorithms.amplitude_amplification.qpe_subspace.QPESubspaceMarking`
algorithm (``amplitude_amplification_oracle`` / ``qdk_qpe_subspace``) builds a
``good_state_oracle`` for an eigenspace. It runs the nested phase estimation on the register
it is handed, flips the flag when the phase lands in a bin whose energy is at least
``target_energy``, then undoes the estimation so the register comes back as it was found.
``run`` takes only the ``qubit_hamiltonian``: the register already holds the state under
test, so no state preparation is needed.

The oracle reflects about the marked eigenspaces only when the estimation is exact, that is
when every eigenphase of the state under test is a multiple of :math:`2^{-n}` for
``num_bits`` :math:`n`. Off a bin the phase register comes back spread rather than to
:math:`|0\rangle`, so the ancillas the oracle releases carry away part of the state.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/amplitude_amplification.py
      :language: python
      :start-after: # start-cell-run
      :end-before: # end-cell-run

Alternatively, the qpe subspace marking oracle can be replaced by a quantum signal processing oracle
on a block encoding of the Hamiltonian. See Lin and Tong,
`arXiv:2002.12508 <https://arxiv.org/abs/2002.12508>`_, and its use in
`arXiv:2510.07273 <https://arxiv.org/abs/2510.07273>`_, Section 2.

Settings
--------

``amplitude_amplification`` / ``qdk_base``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``rounds``
     - ``int``
     - Number of Grover iterates (default 1). Must be non-negative.

``amplitude_amplification_oracle`` / ``qdk_qpe_subspace``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``target_energy``
     - ``float``
     - Lowest energy the marked subspace may hold. Must be finite. Default: ``nan`` (invalid; user must set).
   * - ``qpe_circuit_builder``
     - ``AlgorithmRef``
     - The phase estimation to mark. Default: ``AlgorithmRef("qpe_circuit_builder", "qdk_standard")``.

Further Reading
---------------

- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
- Lin, L. and Tong, Y. *Near-optimal ground state preparation*, `arXiv:2002.12508 <https://arxiv.org/abs/2002.12508>`_: the signal-processing eigenspace reflection.
