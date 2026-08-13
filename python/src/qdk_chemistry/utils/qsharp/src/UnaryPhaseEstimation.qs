// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryPhaseEstimation {

    import Std.Arrays.Reversed;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyToEach;
    import Std.Canon.ApplyXorInPlace;
    import Std.Diagnostics.Fact;
    import Std.Math.AbsI;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationWithControl;

    /// Number of phase qubits required to address `numQueries + 1` reflection slots.
    function PhaseRegisterSize(numQueries : Int) : Int {
        Fact(numQueries > 0, "numQueries must be positive");
        return AddressQubits(numQueries + 1);
    }

    /// Applies `numQueries` self-inverse blocks, omitting the one reflection `phaseReg` selects.
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

    /// Checks the generic schedule against the explicit walk power.
    function MakeTestSignedPowerScheduleAgainstWalkOp(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        numQueries : Int,
        addressValue : Int,
        systemAngle : Double,
    ) : (Qubit[] => Unit) {
        return qs => {
            let numAddressQubits = AddressQubits(numQueries + 1);
            let address = qs[0..numAddressQubits - 1];
            let targets = qs[numAddressQubits...];

            ApplyXorInPlace(addressValue, address);
            Ry(systemAngle, targets[0]);

            ApplySignedPowerSchedule(applyBlockEncoding, applyReflection, numQueries, address, targets);

            let walk = (register) => {
                applyBlockEncoding(register);
                applyReflection(register);
            };
            let power = numQueries - 2 * addressValue;
            for _ in 1..AbsI(power) {
                if power > 0 {
                    Adjoint walk(targets);
                } else {
                    walk(targets);
                }
            }

            ApplyXorInPlace(addressValue, address);
        };
    }

    /// Runs `MakeUnaryQPECircuit` on a synthetic one-qubit walk with an exact eigenphase.
    operation TestUnaryQpeSyntheticWalk(numQueries : Int, theta : Double, systemAngle : Double) : Result[] {
        Fact(
            2^PhaseRegisterSize(numQueries) == numQueries + 1,
            "numQueries must be one less than a power of two",
        );

        return MakeUnaryQPECircuit(
            (systems) => Ry(systemAngle, systems[0]),
            (qubits) => {
                Rz(-theta, qubits[0]);
                X(qubits[0]);
                Rz(theta, qubits[0]);
            },
            (qubits) => X(qubits[0]),
            ApplyToEach(H, _),
            numQueries,
            1,
            0
        );
    }
}
