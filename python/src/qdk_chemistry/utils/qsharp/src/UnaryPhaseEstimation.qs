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
/// The phase register is prepared by `phaseQubitPrep` (a window state such as
/// Kaiser or cosine) rather than by uniform Hadamards, which suppresses spectral
/// leakage from the truncated, non-power-of-two schedule.
namespace QDKChemistry.Utils.UnaryPhaseEstimation {

    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;

    /// Number of phase qubits required to address `numQueries + 1` reflection slots.
    function PhaseRegisterSize(numQueries : Int) : Int {
        Fact(numQueries > 0, "numQueries must be positive");
        return Ceiling(Lg(IntAsDouble(numQueries + 1)));
    }

    /// Build a unary-iteration QPE circuit for an arbitrary (non-power-of-two) query count.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `unaryIterationEvolution`: Applies the signed-power schedule on (phase register, targets).
    ///   The phase register is passed little-endian, matching the unary addressing convention.
    /// - `numQueries`: Total number of block applications; need not be a power of two.
    /// - `ancillas`: An array of indices for the phase ancilla qubits.
    /// - `systems`: An array of indices for the system qubits (state prep target).
    /// - `phaseQubitPrep`: Prepares the window state on the phase register (big-endian).
    /// - `numAncillas`: Number of extra ancillas required by the block encoding.
    /// - `ancillaPrep`: A function to prepare persistent ancillas (e.g., phase gradient state).
    /// # Returns
    /// - `Result[]`: The measurement results of the phase ancilla qubits (MSB first).
    operation MakeUnaryQPECircuit(
        statePrep : Qubit[] => Unit,
        unaryIterationEvolution : (Qubit[], Qubit[]) => Unit,
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
        unaryIterationEvolution(Reversed(phaseAncillas), allTargets);

        Adjoint ApplyQFT(phaseAncillas);

        ResetAll(allTargets);
        mutable results = [Zero, size = numBits];
        for idx in 0..numBits - 1 {
            set results w/= idx <- MResetZ(phaseAncillas[idx]);
        }
        return results;
    }
}
