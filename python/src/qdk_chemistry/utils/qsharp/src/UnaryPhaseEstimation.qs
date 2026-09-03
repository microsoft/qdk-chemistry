// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryPhaseEstimation {

    import Std.Arrays.Reversed;
    import Std.Canon.ApplyQFT;
    import Std.Diagnostics.Fact;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationWithControl;

    /// Number of phase qubits required to address `numQueries + 1` reflection slots.
    function PhaseRegisterSize(numQueries : Int) : Int {
        Fact(numQueries > 0, "numQueries must be positive");
        return AddressQubits(numQueries + 1);
    }

    /// Applies `numQueries` self-inverse blocks, omitting the one reflection `phaseReg` selects.
    ///
    /// `phaseReg` must be supported on `0..numQueries`; addresses past that alias valid slots
    /// (see `UnaryIterationWithControl`) and realize a wrong walk power.
    internal operation ApplySignedPowerSchedule(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        numQueries : Int,
        phaseReg : Qubit[],
        allQubits : Qubit[],
    ) : Unit is Adj {
        Fact(numQueries > 0, "numQueries must be positive");
        UnaryIterationWithControl(phaseReg, numQueries + 1, (slot, selected) => {
            within {
                X(selected);
            } apply {
                Controlled applyReflection([selected], allQubits);
            }
            // At slot = numQueries only the reflection is run
            if slot < numQueries {
                applyBlockEncoding(allQubits);
            }
        });
    }

    /// Build a unary-iteration QPE circuit for an arbitrary (non-power-of-two) query count.
    /// Lee et al. Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction.
    /// https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305
    operation MakeUnaryQPECircuit(
        statePrep : Qubit[] => Unit,
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        phaseQubitPrep : Qubit[] => Unit,
        numQueries : Int,
        numSystemQubits : Int,
        numAncillas : Int,
    ) : Result[] {
        Fact(numSystemQubits > 0, "numSystemQubits must be positive");
        Fact(numAncillas >= 0, "numAncillas must be non-negative");
        let numPhaseQubits = PhaseRegisterSize(numQueries);

        use qs = Qubit[numPhaseQubits + numSystemQubits + numAncillas];
        let phaseQubits = qs[0..numPhaseQubits - 1];
        let systemQubits = qs[numPhaseQubits..numPhaseQubits + numSystemQubits - 1];
        let allTargets = qs[numPhaseQubits...];

        statePrep(systemQubits);
        phaseQubitPrep(phaseQubits);

        ApplySignedPowerSchedule(
            applyBlockEncoding,
            applyReflection,
            numQueries,
            Reversed(phaseQubits),
            allTargets
        );

        Adjoint ApplyQFT(phaseQubits);

        mutable results = [Zero, size = numPhaseQubits];
        for idx in 0..numPhaseQubits - 1 {
            set results w/= idx <- MResetZ(phaseQubits[idx]);
        }

        ResetAll(allTargets);
        return results;
    }
}
