Amplitude amplification
=======================

Amplitude amplification boosts the probability that a computation lands in a designated "good" subspace.
Its main use in QDK/Chemistry is to rescue a :doc:`phase estimation <phase_estimation>` run whose guiding state has poor overlap :math:`a` with the target eigenstate: instead of repeating the whole QPE circuit :math:`O(1/a)` times, amplification succeeds after :math:`O(1/\sqrt{a})` repetitions of a marked QPE circuit, at a cost of :math:`2k+1` coherent preparations per attempt.

The algorithm is a **circuit transform**, not a solver. It takes a measurement-free preparation :math:`U` and a marking oracle, and returns the amplified circuit :math:`Q^k U`; executing it and deciding which shots are good are the caller's job. That keeps it independent of phase estimation — any adjointable preparation works.

.. math::

   Q = -\,U S_0 U^\dagger \, S_G

The state reflection is :math:`I - 2|\psi\rangle\langle\psi| = U S_0 U^\dagger`, because :math:`|0\cdots 0\rangle` is the only state hardware can cheaply recognise. That is why :math:`U` is re-run every round and why :math:`Q^k U` contains :math:`2k+1` preparations — nothing cancels, since :math:`S_G` sits between :math:`U^\dagger` and :math:`U`. It also forces the preparation to be exactly invertible: no mid-circuit measurement, no garbage ancillas.

Choosing the number of rounds
-----------------------------

Writing the prepared state as :math:`\sin\theta\,|G\rangle + \cos\theta\,|B\rangle` with :math:`\theta = \arcsin\sqrt{a}`, the acceptance probability after :math:`k` rounds is

.. math::

   p_k = \sin^2\!\big((2k+1)\theta\big).

Because :math:`p_k` is periodic, more rounds are not always better: past the first maximum the acceptance falls again and vanishes at :math:`(2k+1)\theta = \pi`. This is *overshoot*, the regime a chemistry guiding state lives in — small overlap, known only to within an order of magnitude.

**Overshoot is controlled by an upper bound on the overlap, not a lower one:** underestimating :math:`a` makes :math:`\theta` too small, the round count too large, and the rotation overshoots. Worse, the failure is silent — low acceptance looks exactly like small overlap.

QDK/Chemistry therefore derives :math:`k` from the Yoder-Low-Chuang **fixed-point** schedule (:meth:`~qdk_chemistry.algorithms.amplitude_amplification.base.AmplitudeAmplification.fixed_point_rounds`), which needs only the ``min_overlap`` *lower* bound you actually have from a classical estimate, plus a ``tolerance`` :math:`\delta`. Its phase sequence replaces the sinusoid with a plateau: acceptance is :math:`\ge 1-\delta^2` for every overlap above the threshold and never falls back, so a conservative bound costs queries but cannot destroy the signal. The guarantee is worth roughly a 2x query overhead against an oracle that knows :math:`a` exactly.

Setting ``rounds`` to a non-negative value overrides the schedule and runs that many plain Grover iterates instead — the textbook loop, overshoot and all.

Worked example
--------------

Build a measurement-free QPE circuit, mark the accepted energy bins, amplify, then execute:

.. code-block:: python

    from qdk_chemistry.algorithms import create
    from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import split_coherent_qpe_bitstring
    from qdk_chemistry.data import AlgorithmRef
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS

    num_bits, accepted = 4, [4]

    builder = create("qpe_circuit_builder", "qdk_standard")
    builder.settings().update("num_bits", num_bits)
    builder.settings().update("controlled_circuit_mapper", AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"))
    builder.settings().update("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0))
    builder.settings().update("measurement", "none")
    preparation = builder.run(state_preparation=prep, qubit_hamiltonian=ham)[0]

    # Trotter has no block-encoding ancillas, so the good subspace is the phase window alone.
    marker = QSHARP_UTILS.AmplitudeAmplification.MakeQpeAcceptanceMarkerOp(num_bits, [], accepted)

    algorithm = create("amplitude_amplification")
    algorithm.settings().update("rounds", 1)
    circuit = algorithm.run(preparation, marker, num_qubits=num_bits + ham.num_qubits)

    counts = create("circuit_executor", "qdk_sparse_state_simulator").run(circuit, shots=200).bitstring_counts
    good = [b for b in counts if split_coherent_qpe_bitstring(b, num_bits)[0] == "0100"]

Acceptance is decided classically from the returned bits:
:func:`~qdk_chemistry.algorithms.phase_estimation.circuit_builder.base.split_coherent_qpe_bitstring`
separates the phase register from the block-encoding ancillas, and a shot is good when its phase index is
in the window *and* every signal ancilla measured zero. For a 0.3-overlap guiding state on
:math:`H = (\pi/4)(ZI + IZ)` the dominant accepted bitstring is ``"0100"``.

Swapping the encoding is a matter of swapping the nested refs on the *builder* (for example a
``prepare_select_prepare`` mapper with an ``lcu`` walk builder); amplitude amplification itself is unchanged,
except that the block-encoding ancilla indices must then be passed to the marker.

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
     - Explicit number of plain Grover iterates. Negative (the default) derives a fixed-point schedule from ``min_overlap`` and ``tolerance`` instead.
   * - ``min_overlap``
     - ``double``
     - Lower bound on the overlap :math:`a` of the prepared state with the good subspace. Required unless ``rounds`` is set explicitly.
   * - ``tolerance``
     - ``double``
     - Fixed-point tolerance :math:`\delta`; success is guaranteed to exceed :math:`1-\delta^2` (default 0.1).

``run`` takes the preparation circuit, the marking oracle, ``num_qubits``, and an optional ``measured_indices``
(defaulting to the whole register, little-endian). The preparation must carry an adjointable Q# operation;
iterative QPE cannot, and says so.

.. warning::
   Amplitude amplification changes how *often* the phase-estimation window is accepted; it does not change *what* is accepted.
   It cannot repair a mis-specified energy window or the leakage caused by using too few phase bits — choose the window from the QPE resolution first, then amplify.

References
----------

- :doc:`PhaseEstimation <phase_estimation>`: the un-amplified algorithm.
- :doc:`QpeCircuitBuilder <qpe_circuit_builder>`: builds the coherent preparation this algorithm amplifies.
- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
- Yoder, T. J., Low, G. H., and Chuang, I. L. *Fixed-point quantum search with an optimal number of queries*, Phys. Rev. Lett. **113**, 210501 (2014), `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`_.
