// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Test-only drivers for `QDKChemistry.Utils.UnaryPhaseEstimation`.
///
/// These are staged into a throwaway Q# project by the Python test layer
/// (see `tests/qsharp_test_project.py`) and are never part of the shipped
/// `qdk_chemistry.utils.qsharp` project. Staging them alongside the shipped
/// sources — rather than as a dependent Q# package — is what keeps the
/// `internal` `ApplySignedPowerSchedule` reachable from here.
namespace QDKChemistry.TestUtils.UnaryPhaseEstimationTests {

    import Std.Canon.ApplyToEach;
    import Std.Canon.ApplyXorInPlace;
    import Std.Diagnostics.Fact;
    import Std.Math.AbsI;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryPhaseEstimation.ApplySignedPowerSchedule;
    import QDKChemistry.Utils.UnaryPhaseEstimation.MakeUnaryQPECircuit;
    import QDKChemistry.Utils.UnaryPhaseEstimation.PhaseRegisterSize;

    /// Checks the generic schedule against the explicit walk power.
    function TestMakeSignedPowerScheduleAgainstWalkOp(
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
    operation TestRunSyntheticWalkQpe(numQueries : Int, theta : Double, systemAngle : Double) : Result[] {
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
