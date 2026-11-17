// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Test-only drivers for `QDKChemistry.Utils.PrepSelPrep`.
///
/// The Python test layer evaluates this file into a throwaway context (see the
/// `qsharp_test_context` fixture in `tests/conftest.py`); it is never part of the
/// shipped `qdk_chemistry.utils.qsharp` project.
namespace QDKChemistry.TestUtils.PrepSelPrepTests {

    import QDKChemistry.Utils.PrepSelPrep.MakePrepSelPrepOp;
    import QDKChemistry.Utils.Select.PauliSelect;
    import QDKChemistry.Utils.Select.PauliSelectParams;
    import QDKChemistry.Utils.Select.UnaryPauliSelect;

    /// Signed SELECT on an explicit address register, including outer controls and adjoints.
    function TestMakePauliSelectOp(controlStates : Int[], numAddress : Int, numControls : Int, unary : Bool, useAdjoint : Bool) : (Qubit[] => Unit) {
        mutable signs = Std.Core.Repeated(1, Std.Core.Length(controlStates));
        for i in 0..Std.Core.Length(controlStates) - 1 {
            set signs w/= i <- 1 - 2 * (i % 2);
        }
        let params = new PauliSelectParams {
            pauliTerms = Std.Core.Repeated([PauliY], Std.Core.Length(controlStates)),
            signs = signs,
            controlStates = controlStates
        };
        let selectOp = unary ? UnaryPauliSelect | PauliSelect;
        return qs => {
            let controls = qs[0..numControls - 1];
            let address = qs[numControls..numControls + numAddress - 1];
            let targets = qs[numControls + numAddress...];
            if useAdjoint {
                Controlled Adjoint selectOp(controls, (params, address, targets));
            } else {
                Controlled selectOp(controls, (params, address, targets));
            }
        };
    }

    /// # Summary
    /// One-system-qubit, one-ancilla block encoding used to drive block-encoding-agnostic
    /// schedules from a test.
    function TestMakeBlockEncodingOp(theta : Double) : (Qubit[] => Unit is Adj + Ctl) {
        MakePrepSelPrepOp(
            (ancilla) => Ry(theta, ancilla[0]),
            (ancilla, system) => Controlled Z(ancilla, system[0]),
            1,
            1
        )
    }
}
