Amplitude amplification
=======================

Amplitude amplification boosts the probability that a quantum computation lands in a designated "good" subspace.
In QDK/Chemistry its main use is to rescue a :doc:`phase estimation <phase_estimation>` run whose guiding state has poor overlap with the eigenstate of interest: instead of repeating the whole QPE circuit :math:`O(1/a)` times, amplitude amplification succeeds after :math:`O(1/\sqrt{a})` repetitions of a marked QPE circuit.

.. note::
   Amplitude amplification is a registry algorithm type (``amplitude_amplification``) with one implementation, ``qdk_amplified_qpe``.
   The quantum operations live in the Q# namespace ``QDKChemistry.Utils.AmplitudeAmplification``; the round-count policy lives in :mod:`qdk_chemistry.algorithms.amplitude_amplification.schedule` and is reachable through the algorithm's settings.

Overview
--------

Write the state produced by a preparation unitary :math:`U_\psi` as

.. math::

   |\psi\rangle = U_\psi|0\rangle = \sin\vartheta\,|G\rangle + \cos\vartheta\,|B\rangle,
   \qquad
   \vartheta = \arcsin\sqrt{a},

where :math:`|G\rangle` spans the good subspace and :math:`a` is the initial success probability.
The Grover iterate

.. math::

   Q = -S_\psi S_G,
   \qquad
   S_G = I - 2\Pi_G,
   \qquad
   S_\psi = I - 2|\psi\rangle\langle\psi|

acts as a rotation by :math:`2\vartheta` in the two-dimensional subspace spanned by :math:`|G\rangle` and :math:`|B\rangle`, so after :math:`k` rounds

.. math::

   p_k = \sin^2\!\big((2k+1)\vartheta\big).

The module is deliberately split into two swappable halves:

Reflection about the target
   ``ReflectAboutTargetState`` implements :math:`I - 2|t\rangle\langle t|` for a target given as a preparation circuit, and ``ReflectAboutMarkedSubspace`` implements :math:`I - 2\Pi_G` for a target given as a *predicate*.
   Both satisfy the same signature that the amplification loops consume, so any oracle — including a phase-estimation energy window — can be substituted without touching the amplification code.

Amplification loop
   ``ApplyAmplitudeAmplification`` runs plain Grover iterates; ``ApplyFixedPointAmplitudeAmplification`` runs the phase-matched (generalized) iterates required for a fixed-point schedule.

Choosing the number of rounds
-----------------------------

Because :math:`p_k` is periodic in :math:`k`, more rounds are not always better.
Past the first maximum the success probability falls again, and it vanishes entirely at :math:`(2k+1)\vartheta = \pi`, that is at an overlap of :math:`a = \sin^2(\pi/(2k+1))`.
This is the *overshoot* problem, and it is exactly the regime a chemistry guiding state lives in: the overlap is small and known only to within an order of magnitude.

The key observation is that **overshoot is controlled by an upper bound on the overlap, not a lower bound.**
Underestimating :math:`a` makes :math:`\vartheta` too small, which makes the round count too large, which overshoots.
Choosing

.. math::

   k_{\text{safe}} = \left\lfloor \frac{\pi}{4\vartheta_{\max}} - \frac{1}{2} \right\rfloor,
   \qquad
   \vartheta_{\max} = \arcsin\sqrt{a_{\max}},

guarantees :math:`(2k+1)\vartheta \le \pi/2` for every admissible overlap, so the success probability is a monotonically increasing function of the true overlap.
Being luckier than expected can then only help.

The module offers four policies, in increasing order of robustness:

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Function
     - What you must know
     - Guarantee
   * - :func:`~qdk_chemistry.algorithms.amplitude_amplification.schedule.optimal_rounds`
     - The overlap :math:`a` exactly
     - Success probability :math:`\ge 1 - a` after :math:`\approx \pi/(4\sqrt a)` rounds. Overshoots if :math:`a` was underestimated.
   * - :func:`~qdk_chemistry.algorithms.amplitude_amplification.schedule.safe_rounds`
     - An upper bound :math:`a_{\max}`
     - Never overshoots; success probability increases monotonically with the true overlap.
   * - :func:`~qdk_chemistry.algorithms.amplitude_amplification.schedule.robust_rounds`
     - An interval :math:`[a_{\min}, a_{\max}]`
     - Maximizes the worst-case success probability over the interval; never worse than ``safe_rounds``.
   * - :func:`~qdk_chemistry.algorithms.amplitude_amplification.schedule.fixed_point_phases`
     - A lower bound :math:`a_{\min}`
     - Success probability :math:`\ge 1-\delta^2` for **every** overlap above the threshold; overshoot is impossible.

When nothing at all is known about the overlap, :func:`~qdk_chemistry.algorithms.amplitude_amplification.schedule.exponential_schedule` reproduces the Brassard–Høyer–Mosca–Tapp strategy: draw the round count uniformly from :math:`\{0,\dots,m-1\}` with :math:`m = \lceil (6/5)^{\text{stage}} \rceil` and retry on failure.
Randomizing the round count averages the acceptance probability over the rotation, which removes the overshoot cliff while preserving the :math:`O(1/\sqrt a)` expected cost.

Example::

    from qdk_chemistry.algorithms.amplitude_amplification import (
        optimal_rounds,
        overshoot_overlap,
        robust_rounds,
        safe_rounds,
        success_probability,
        worst_case_success_probability,
    )

    # The guiding state overlap is believed to be between 1% and 5%.
    a_min, a_max = 0.01, 0.05

    optimal_rounds(a_min)                             # 7 -- optimal *if* a is really 1%
    success_probability(a_max, 7)                     # 0.057 -- but it collapses if a is 5%
    overshoot_overlap(7)                              # 0.0432 -- total failure at this overlap
    worst_case_success_probability(7, a_min, a_max)   # 0.0 -- the interval straddles that zero

    safe_rounds(a_max)                                # 2 -- cannot overshoot anywhere
    success_probability(a_min, 2)                     # 0.231
    success_probability(a_max, 2)                     # 0.816

    robust_rounds(a_min, a_max)                       # 4 -- best worst case over the interval
    worst_case_success_probability(4, a_min, a_max)   # 0.615

Fixed-point amplification
-------------------------

If only a lower bound on the overlap is available, the Yoder–Low–Chuang phase sequence removes the overshoot problem altogether.
Replacing the two reflections by the partial rotations

.. math::

   G(\alpha,\beta)
   = \big(I - (1 - e^{i\beta})|\psi\rangle\langle\psi|\big)
     \big(I - (1 - e^{i\alpha})\Pi_G\big)

and choosing the phases as in :func:`~qdk_chemistry.algorithms.amplitude_amplification.schedule.fixed_point_phases` yields, after :math:`L = 2l+1` queries,

.. math::

   p = 1 - \delta^2\,T_L\!\big(T_{1/L}(1/\delta)\sqrt{1-a}\big)^2 ,

which climbs monotonically and then stays inside :math:`[1-\delta^2, 1]` for every larger overlap.
The required query count is :math:`L \ge \log(2/\delta)/\sqrt{a_{\min}}`, so the quadratic speedup is preserved up to a constant factor.

Example::

    from qdk_chemistry.algorithms.amplitude_amplification import (
        fixed_point_phases,
        fixed_point_rounds,
        fixed_point_success_probability,
    )

    rounds = fixed_point_rounds(min_overlap=0.01, tolerance=0.1)  # 15 iterates, 31 queries
    mark_phases, state_phases = fixed_point_phases(rounds, tolerance=0.1)

    # Passed straight to the Q# operation
    # QDKChemistry.Utils.AmplitudeAmplification.ApplyFixedPointAmplitudeAmplification
    fixed_point_success_probability(0.01, rounds, 0.1)  # 0.996
    fixed_point_success_probability(0.90, rounds, 0.1)  # 0.997 -- no overshoot

Amplified phase estimation
--------------------------

.. note::
   This algorithm is currently available only in the Python API.

:class:`~qdk_chemistry.algorithms.amplitude_amplification.amplified_phase_estimation.AmplifiedPhaseEstimation` (registered as ``amplitude_amplification/qdk_amplified_qpe``) composes the amplification loop with a :doc:`QpeCircuitBuilder <qpe_circuit_builder>`.
It is configured exactly like :doc:`PhaseEstimation <phase_estimation>` — same nested ``qpe_circuit_builder`` and ``circuit_executor`` refs, same :class:`~qdk_chemistry.data.QpeResult` output — plus the settings that define the good subspace and the round count:

.. code-block:: python

    from qdk_chemistry.algorithms import create
    from qdk_chemistry.data import AlgorithmRef

    qpe = create("amplitude_amplification", "qdk_amplified_qpe")
    qpe.settings().update(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=8,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        ),
    )
    qpe.settings().update("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
    qpe.settings().update("shots", 200)

    # The guiding state overlap is believed to be at most 5%; amplify the energies
    # near the expected ground state.
    qpe.settings().update("round_policy", "safe")
    qpe.settings().update("max_overlap", 0.05)
    qpe.settings().update("min_energy", -1.20)
    qpe.settings().update("max_energy", -1.05)

    result = qpe.run(state_preparation=state_prep, qubit_hamiltonian=hamiltonian)
    result.raw_energy
    result.metadata["acceptance_probability"]
    result.metadata["amplification_rounds"]

.. rubric:: Settings

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Type
     - Description
   * - ``qpe_circuit_builder``
     - ``algorithm_ref``
     - The QPE circuit builder used as the state preparation. Must be a standard (QFT-based) builder; the algorithm switches it into ``coherent`` mode automatically.
   * - ``circuit_executor``
     - ``algorithm_ref``
     - Backend that executes the amplified circuit.
   * - ``shots``
     - ``int``
     - Number of shots (default 100).
   * - ``rounds``
     - ``int``
     - Explicit number of amplification rounds. Negative (the default) means derive it from ``round_policy``.
   * - ``round_policy``
     - ``string``
     - One of ``fixed``, ``safe`` (default), ``robust``, ``optimal``, ``fixed_point``. See `Choosing the number of rounds`_.
   * - ``min_overlap``
     - ``double``
     - Lower bound on the overlap, used by ``robust``, ``optimal`` and ``fixed_point``.
   * - ``max_overlap``
     - ``double``
     - Upper bound on the overlap, used by ``safe`` and ``robust``. Defaults to ``1.0``, for which ``safe`` yields zero rounds — amplification is off until you supply a real bound.
   * - ``tolerance``
     - ``double``
     - Fixed-point tolerance :math:`\delta`; success is guaranteed to exceed :math:`1-\delta^2` (default 0.1).
   * - ``accepted_phase_indices``
     - ``vector<int>``
     - Phase-register indices that count as good. Empty (the default) means derive them from the energy window.
   * - ``min_energy`` / ``max_energy``
     - ``double``
     - Accepted energy window, converted to phase indices through the unitary container's own phase-to-eigenvalue map, so the same window works for Trotter and qubitization alike. Used only when ``accepted_phase_indices`` is empty.

Acceptance statistics are returned in :attr:`~qdk_chemistry.data.QpeResult.metadata`: ``amplification_rounds``, ``round_policy``, ``accepted_phase_indices``, ``accepted_shots``, ``total_shots``, ``acceptance_probability`` and ``preparations_per_shot`` (:math:`2k+1`), plus the predicted acceptance at the overlap bounds when those are set.

.. rubric:: How it works

The ``AmplitudeAmplification`` Q# module ships a ready-made marking oracle for phase estimation.
``ApplyQpeAcceptanceMark`` flips its target when **both** conditions hold:

1. the phase register decodes to an accepted index, and
2. every block-encoding signal ancilla is :math:`|0\rangle`.

The second condition matters: a nonzero signal ancilla means the block encoding did not project onto the signal block, so the phase register carries no eigenvalue information on that branch.
Because a qubitization walk encodes energies through :math:`\mu = \alpha\cos(2\pi\phi)`, an energy window is naturally a *wrapped* interval of phase indices — a prefix plus a suffix of the index range.
``ApplyAcceptedPhaseMark`` detects that shape and marks it with two comparisons; arbitrary accepted sets fall back to one multiply-controlled flip per index.

``ApplyCoherentStandardQPE`` provides the adjointable, measurement-free QPE circuit that serves as the state preparation :math:`U_\psi`, and ``ApplyAmplifiedStandardQPE`` wires the two together.
Because the amplification loop must apply :math:`U_\psi^\dagger`, the controlled walk operators have to be adjointable; setting ``coherent`` on the circuit builder makes it request the adjointable variants (``QDKChemistry.Utils.PrepSelPrep.MakeAdjointableControlledPSPWalkOp`` and ``QDKChemistry.Utils.ControlledPauliExp.MakeAdjointableRepControlledPauliExpOp``) instead of the resource-estimation-cached factories.
Iterative QPE measures and resets its phase qubit on every iteration, so it cannot be used as the preparation and rejects ``coherent``.

Acceptance is decided classically in Python from the measured bits rather than by mid-circuit branching in Q#, which keeps the circuit entry points compatible with the restricted target profiles used for QIR generation and resource estimation.

.. warning::
   Amplitude amplification changes how *often* the phase-estimation window is accepted; it does not change *what* is accepted.
   It cannot repair a mis-specified energy window, and it cannot remove the leakage caused by using too few phase bits.
   Choose the window from the QPE resolution first, then amplify.

References
----------

- :doc:`PhaseEstimation <phase_estimation>`: the un-amplified algorithm this one wraps.
- :doc:`QpeCircuitBuilder <qpe_circuit_builder>`: the nested algorithm that builds the coherent preparation.
- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
- Yoder, T. J., Low, G. H., and Chuang, I. L. *Fixed-point quantum search with an optimal number of queries*, Phys. Rev. Lett. **113**, 210501 (2014), `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`_.
