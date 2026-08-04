Amplitude amplification
=======================

Amplitude amplification boosts the probability that a prepared state is found in a marked subspace.
Given a guiding state with overlap :math:`a` on the target eigenstate, it succeeds after :math:`O(1/\sqrt{a})` repetitions instead of :math:`O(1/a)`.

``run`` takes a measurement-free preparation :math:`U_\psi` and a marking oracle, and returns the amplified circuit :math:`Q^k U_\psi`:

.. math::

   Q = -\,U_\psi S_0 U_\psi^\dagger \, S_G

Executing that circuit and deciding which shots are good are the caller's job.

Because :math:`|0\cdots0\rangle` is the only state hardware recognises cheaply, the state reflection is built as :math:`U_\psi S_0 U_\psi^\dagger`.
Every round therefore costs one :math:`U_\psi` and one :math:`U_\psi^\dagger`, and :math:`Q^k U_\psi` contains :math:`2k+1` preparations; nothing cancels, since :math:`S_G` sits between them.
The preparation must be exactly invertible: no mid-circuit measurement, no garbage ancillas.

Choosing the number of rounds
-----------------------------

With :math:`\theta = \arcsin\sqrt{a}`, the acceptance probability after :math:`k` rounds is

.. math::

   p_k = \sin^2\!\big((2k+1)\theta\big).

More rounds are not always better: past the first maximum acceptance falls again, vanishing at :math:`(2k+1)\theta = \pi`.
Overshoot fails silently: low acceptance looks exactly like small overlap.
Avoiding it needs an *upper* bound on :math:`a`, which a classical overlap estimate does not give.

The round count is therefore taken from the Yoder-Low-Chuang **fixed-point** schedule (:meth:`~qdk_chemistry.algorithms.amplitude_amplification.base.AmplitudeAmplification.fixed_point_rounds`), which needs only the ``min_overlap`` *lower* bound plus a ``tolerance`` :math:`\delta`.
Its phase sequence replaces the sinusoid with a plateau: acceptance stays :math:`\ge 1-\delta^2` for every overlap above the threshold, so a conservative bound costs queries but cannot destroy the signal, at roughly a 2x overhead.

Setting ``rounds`` to a non-negative value runs that many plain Grover iterates instead.

Marking oracle
--------------

The oracle is any adjointable Q# operation ``(Qubit[], Qubit) => Unit is Adj`` that flips its target on the good subspace and leaves the register otherwise unchanged.
None ships with the library, since what counts as good is application-specific.

For QPE the predicate is "the phase register holds an accepted index *and* every block-encoding signal ancilla is :math:`|0\rangle`".
Both halves matter: a nonzero signal ancilla means the block encoding did not project onto the signal block, so that branch's phase register carries no eigenvalue information.
Conjugating the ancillas by ``X`` turns "all zero" into "all one", so the index test can simply run controlled on them:

.. code-block:: text

    operation MarkAcceptedPhase(
        numPhaseQubits : Int, signalAncillaIndices : Int[], accepted : Int[],
        register : Qubit[], target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        let signalAncillas = Subarray(signalAncillaIndices, register[numPhaseQubits...]);
        within { ApplyToEachCA(X, signalAncillas); }
        apply {
            for index in accepted {
                Controlled ApplyControlledOnInt(signalAncillas, (index, X, phaseRegister, target));
            }
        }
    }

The phase register is little-endian after the inverse QFT.

Worked example
--------------

.. code-block:: python

    from qdk_chemistry.algorithms import create
    from qdk_chemistry.data import AlgorithmRef
    from qdk_chemistry.utils.qsharp import get_qsharp_context

    num_bits, accepted = 4, [4]

    builder = create("qpe_circuit_builder", "qdk_standard")
    builder.settings().update("num_bits", num_bits)
    builder.settings().update("controlled_circuit_mapper", AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"))
    builder.settings().update("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0))
    builder.settings().update("measurement", "none")
    preparation = builder.run(state_preparation=prep, qubit_hamiltonian=ham)[0]

    # MarkAcceptedPhase above, partially applied; Trotter has no block-encoding ancillas.
    marker = get_qsharp_context().code.MyOracles.MakeAcceptedPhaseMarkerOp(num_bits, [], accepted)

    algorithm = create("amplitude_amplification")
    algorithm.settings().update("rounds", 1)
    circuit = algorithm.run(preparation, marker, num_qubits=num_bits + ham.num_qubits)

    counts = create("circuit_executor", "qdk_sparse_state_simulator").run(circuit, shots=200).bitstring_counts

Acceptance is applied classically to ``counts``, mirroring the oracle.
For a 0.3-overlap guiding state on :math:`H = (\pi/4)(ZI + IZ)` the dominant accepted bitstring is ``"0100"``.

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

``run(preparation, marking_oracle, num_qubits, measured_indices=None)`` defaults ``measured_indices`` to the whole register.
The preparation must carry an adjointable Q# operation, which iterative QPE does not.

.. warning::
   Amplification changes how *often* the window is accepted, not *what* is accepted.
   It cannot repair a mis-specified window or too few phase bits: choose the window from the QPE resolution first, then amplify.

References
----------

- :doc:`PhaseEstimation <phase_estimation>`: the un-amplified algorithm.
- :doc:`QpeCircuitBuilder <qpe_circuit_builder>`: builds the coherent preparation this algorithm amplifies.
- Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*, `arXiv:2201.08309 <https://arxiv.org/abs/2201.08309>`_, Chapter 2.
- Brassard, G., Høyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude Amplification and Estimation*, `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
- Yoder, T. J., Low, G. H., and Chuang, I. L. *Fixed-point quantum search with an optimal number of queries*, Phys. Rev. Lett. **113**, 210501 (2014), `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`_.
