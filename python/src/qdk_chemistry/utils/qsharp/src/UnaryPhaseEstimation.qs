// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryPhaseEstimation {

    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyToEach;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.AbsI;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationWithControl;

    /// Number of phase qubits required to address `numQueries + 1` reflection slots.
    function PhaseRegisterSize(numQueries : Int) : Int {
        Fact(numQueries > 0, "numQueries must be positive");
        return Ceiling(Lg(IntAsDouble(numQueries + 1)));
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
    operation MakeUnaryQPECircuit(
        statePrep : Qubit[] => Unit,
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        phaseQubitPrep : Qubit[] => Unit,
        numQueries : Int,
        ancillas : Int[],
        systems : Int[],
        numAncillas : Int,
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
        phaseQubitPrep(phaseAncillas);

        ApplySignedPowerSchedule(
            applyBlockEncoding,
            applyReflection,
            numQueries,
            Reversed(phaseAncillas),
            allTargets
        );

        Adjoint ApplyQFT(phaseAncillas);

        ResetAll(allTargets);
        mutable results = [Zero, size = numBits];

        for idx in 0..numBits - 1 {
            set results w/= idx <- MResetZ(phaseAncillas[idx]);
        }
        return results;
    }

    /// Checks the generic schedule against the explicit walk power.
    ///
    /// The returned operation acts on `[address | targets]`, so the caller allocates
    /// `AddressQubits(numQueries + 1)` qubits ahead of the target register. The schedule at
    /// `addressValue` is applied, then `W^(numQueries - 2 * addressValue)` is explicitly undone
    /// with the same two callables, so a correct schedule leaves the prepared input untouched.
    function MakeTestSignedPowerScheduleAgainstWalkOp(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        numQueries : Int,
        addressValue : Int,
        systemAngle : Double,
    ) : (Qubit[] => Unit) {
        (qs) => {
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
        }
    }

    /// Runs `MakeUnaryQPECircuit` on a synthetic one-qubit walk with an exact eigenphase.
    ///
    /// The reflection is `R = X` on the single target qubit and the block is
    /// B = Rz(theta)·X·Rz(-theta), a reflection about an axis in the XY plane.
    /// The block is self-inverse, and it does not commute with the reflection it is paired
    /// with, so the walk W = B·X = Rz(2*theta) has genuinely distinct powers. A pair of
    /// commuting factors (two diagonal ones, say) would make every schedule branch collapse
    /// to the same operator and silently pass no matter what the address decode did.
    operation TestUnaryQpeSyntheticWalk(numQueries : Int, theta : Double, systemState : Int) : Result[] {
        let numBits = PhaseRegisterSize(numQueries);
        Fact(2^numBits == numQueries + 1, "numQueries must be one less than a power of two");

        return MakeUnaryQPECircuit(
            (systems) => {
                if systemState == 1 {
                    X(systems[0]);
                }
            },
            (qubits) => {
                Rz(-theta, qubits[0]);
                X(qubits[0]);
                Rz(theta, qubits[0]);
            },
            (qubits) => X(qubits[0]),
            ApplyToEach(H, _),
            numQueries,
            Std.Arrays.SequenceI(0, numBits - 1),
            [numBits],
            0
        );
    }
}
