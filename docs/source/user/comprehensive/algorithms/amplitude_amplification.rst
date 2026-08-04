Amplitude amplification
=======================

Amplitude amplification boosts the probability that a computation lands in a designated "good" subspace.
Its main use in QDK/Chemistry is to rescue a :doc:`phase estimation <phase_estimation>` run whose guiding state has poor overlap :math:`a` with the target eigenstate: instead of repeating the whole QPE circuit :math:`O(1/a)` times, amplification succeeds after :math:`O(1/\sqrt{a})` repetitions of a marked QPE circuit, at a cost of :math:`2k+1` coherent preparations per attempt.

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

Fill ``reflect_to_good_space`` with a :doc:`QPE circuit builder <qpe_circuit_builder>` to get amplified phase estimation.
It is configured exactly like :doc:`PhaseEstimation <phase_estimation>` — the same nested builder and ``circuit_executor`` refs and the same :class:`~qdk_chemistry.data.QpeResult` output — plus the settings that define the good subspace and the round count:

.. code-block:: python

    from qdk_chemistry.algorithms import create
    from qdk_chemistry.data import AlgorithmRef

    algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
    algorithm.settings().update(
        "reflect_to_good_space",
        AlgorithmRef(
            "qpe_circuit_builder", "qdk_standard", num_bits=4,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        ),
    )
    algorithm.settings().update("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
    algorithm.settings().update("shots", 200)
    algorithm.settings().update("rounds", 1)
    algorithm.settings().update("accepted_phase_indices", [4])
    result = algorithm.run(state_preparation=prep, qubit_hamiltonian=ham)

``run`` returns a single :class:`~qdk_chemistry.data.QpeResult`, built from the dominant bitstring among the *accepted* shots. For a 0.3-overlap guiding state on :math:`H = (\pi/4)(ZI + IZ)` the observed run gives ``result.bitstring_msb_first == "0100"``.

Swapping the encoding is a matter of swapping the nested refs (for example a ``prepare_select_prepare`` mapper with an ``lcu`` walk builder); the good subspace can also be given as an energy window (``min_energy`` / ``max_energy``) instead of explicit ``accepted_phase_indices``.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``reflect_to_good_space``
     - ``algorithm_ref``
     - Nested builder producing the coherent preparation to reflect about. Locked to the ``qpe_circuit_builder`` algorithm type, so any conforming builder is accepted. Its ``measurement`` setting, if it has one, is forced to ``"none"``; either way it must return a measurement-free circuit whose inverse the loop can apply. Iterative QPE cannot, and says so.
   * - ``circuit_executor``
     - ``algorithm_ref``
     - Backend that executes the amplified circuit.
   * - ``shots``
     - ``int``
     - Number of shots (default 100).
   * - ``rounds``
     - ``int``
     - Explicit number of plain Grover iterates. Negative (the default) derives a fixed-point schedule from ``min_overlap`` and ``tolerance`` instead.
   * - ``min_overlap``
     - ``double``
     - Lower bound on the overlap :math:`a` of the guiding state with the good subspace. Required unless ``rounds`` is set explicitly.
   * - ``tolerance``
     - ``double``
     - Fixed-point tolerance :math:`\delta`; success is guaranteed to exceed :math:`1-\delta^2` (default 0.1).
   * - ``accepted_phase_indices``
     - ``vector<int>``
     - Phase-register indices that count as good. Empty (the default) derives them from the energy window.
   * - ``min_energy`` / ``max_energy``
     - ``double``
     - Accepted energy window, converted to phase indices through the encoding's phase-to-eigenvalue map, so the same window works for Trotter and qubitization alike. Used only when ``accepted_phase_indices`` is empty.

.. warning::
   Amplitude amplification changes how *often* the phase-estimation window is accepted; it does not change *what* is accepted.
   It cannot repair a mis-specified energy window or the leakage caused by using too few phase bits — choose the window from the QPE resolution first, then amplify.

References
----------

- :doc:`PhaseEstimation <phase_estimation>`: the un-amplified algorithm this one wraps.
- :doc:`QpeCircuitBuilder <qpe_circuit_builder>`: the nested algorithm that builds the coherent preparation.
- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
- Yoder, T. J., Low, G. H., and Chuang, I. L. *Fixed-point quantum search with an optimal number of queries*, Phys. Rev. Lett. **113**, 210501 (2014), `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`_.
