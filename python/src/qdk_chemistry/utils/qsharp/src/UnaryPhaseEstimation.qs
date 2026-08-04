// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// QFT phase estimation driven by unary iteration over a signed-power schedule.
///
/// Unlike standard QPE, which applies controlled U^(2^k) once per phase qubit and
/// therefore consumes a power-of-two number of queries, this variant applies a
/// single chain of `numQueries` self-inverse blocks and lets unary iteration over
/// the phase register select which reflection to omit. Branch t of the phase
/// register then sees W^(numQueries - 2t), so any positive `numQueries` is allowed.
///
/// The phase register is prepared by `phaseQubitPrep` (a cosine window state)
/// rather than by uniform Hadamards, which suppresses spectral leakage from the
/// truncated, non-power-of-two schedule.
namespace QDKChemistry.Utils.UnaryPhaseEstimation {

    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyToEach;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationPowerSchedule;

    /// Number of phase qubits required to address `numQueries + 1` reflection slots.
    function PhaseRegisterSize(numQueries : Int) : Int {
        Fact(numQueries > 0, "numQueries must be positive");
        return Ceiling(Lg(IntAsDouble(numQueries + 1)));
    }

    /// Build a unary-iteration QPE circuit for an arbitrary (non-power-of-two) query count.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `signedPowerSchedule`: Applies the ENTIRE signed-power schedule on
    ///   (phase register, targets) in a single call. It is not one walk step: the caller
    ///   builds it already bound to `numQueries`, and it internally sweeps all
    ///   `numQueries + 1` reflection slots, applying `numQueries` walk blocks and skipping
    ///   the one reflection selected by the phase register. Fusing that sweep with the
    ///   address decode is what makes the schedule cost O(numQueries) Toffolis instead of
    ///   O(numQueries * log numQueries); do not lift the repetition to this call site.
    ///   The phase register is passed little-endian, matching the unary addressing convention.
    /// - `numQueries`: Total number of block applications; need not be a power of two.
    ///   Used here only to size the phase register - it must equal the query count that
    ///   `signedPowerSchedule` was built with, or the decoded phase will be wrong.
    /// - `ancillas`: An array of indices for the phase ancilla qubits.
    /// - `systems`: An array of indices for the system qubits (state prep target).
    /// - `phaseQubitPrep`: Prepares the window state on the phase register (big-endian).
    /// - `numAncillas`: Number of extra ancillas required by the block encoding.
    /// - `ancillaPrep`: A function to prepare persistent ancillas (e.g., phase gradient state).
    /// # Returns
    /// - `Result[]`: The measurement results of the phase ancilla qubits (MSB first).
    operation MakeUnaryQPECircuit(
        statePrep : Qubit[] => Unit,
        signedPowerSchedule : (Qubit[], Qubit[]) => Unit,
        numQueries : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
        numAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    ) : Result[] {
        let numBits = PhaseRegisterSize(numQueries);
        Fact(
            Length(ancillas) == numBits,
            $"phase register must hold {numBits} qubits for {numQueries} queries",
        );

        let totalQubits = numBits + Length(systems) + numAncillas;
        use qs = Qubit[totalQubits];
        let phaseAncillas = Subarray(ancillas, qs);
        let systemQubits = Subarray(systems, qs);
        let beAncillas = if numAncillas == 0 {
            []
        } else {
            qs[numBits + Length(systems)..Length(qs) - 1]
        };
        let allTargets = systemQubits + beAncillas;

        statePrep(systemQubits);
        ancillaPrep(beAncillas);
        phaseQubitPrep(phaseAncillas);

        // ApplyQFT and the window state are big-endian; unary addressing is little-endian.
        // One call applies all `numQueries` blocks: the schedule owns the repetition so the
        // slot sweep and the phase-register decode share a single unary-iteration ladder.
        signedPowerSchedule(Reversed(phaseAncillas), allTargets);

        Adjoint ApplyQFT(phaseAncillas);

        ResetAll(allTargets);
        mutable results = [Zero, size = numBits];
        // `Std.Canon.ApplyQFT` maps a little-endian input to a big-endian output, so
        // `Adjoint ApplyQFT` leaves the phase little-endian in `phaseAncillas`. Read the
        // register back in reverse to return the documented most-significant-bit-first order.
        for idx in 0..numBits - 1 {
            set results w/= idx <- MResetZ(phaseAncillas[numBits - 1 - idx]);
        }
        return results;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test wrapper
    // ═══════════════════════════════════════════════════════════════════════════

    internal operation NoAncillaPrep(qs : Qubit[]) : Unit is Adj {}

    /// Runs `MakeUnaryQPECircuit` on a synthetic one-qubit walk with an exact eigenphase.
    ///
    /// The two self-inverse reflections are `R = X` and
    /// `B = Rz(theta) X Rz(-theta) = cos(theta) X + sin(theta) Y`, whose product is the
    /// walk `W = B·R = Rz(2*theta)`, with `W|0> = e^{-i*theta}|0>` and
    /// `W|1> = e^{+i*theta}|1>`. A uniform window is used, so `numQueries` must be
    /// `2^b - 1` for the window to exactly fill the phase register and the outcome to
    /// be deterministic.
    ///
    /// With `theta = -pi*k/(numQueries + 1)` the returned bits must read `k` for
    /// `systemState = 1` and `(-k) mod (numQueries + 1)` for `systemState = 0`, which
    /// pins the documented relation `y = -+2*phi mod 1` together with every endianness
    /// convention in the chain: big-endian window state, little-endian unary addressing,
    /// and the bit order of the measured phase register.
    operation TestUnaryQpeSyntheticWalk(numQueries : Int, theta : Double, systemState : Int) : Result[] {
        let numBits = PhaseRegisterSize(numQueries);
        Fact(2^numBits == numQueries + 1, "numQueries must be one less than a power of two");

        return MakeUnaryQPECircuit(
            (systems) => {
                if systemState == 1 {
                    X(systems[0]);
                }
            },
            (address, targets) => {
                UnaryIterationPowerSchedule(address, numQueries, (selected) => {
                    within {
                        X(selected);
                    } apply {
                        CNOT(selected, targets[0]);
                    }
                }, () => {
                    Rz(-theta, targets[0]);
                    X(targets[0]);
                    Rz(theta, targets[0]);
                });
            },
            numQueries,
            Std.Arrays.SequenceI(0, numBits - 1),
            [numBits],
            ApplyToEach(H, _),
            0,
            NoAncillaPrep
        );
    }
}
