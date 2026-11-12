// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.UnaryIterationTests {

    import Std.Arrays.Mapped;
    import Std.Canon.ApplyToEach;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.ResultAsBool;
    import Std.Diagnostics.Fact;
    import Std.Measurement.MResetEachZ;
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationWithControl;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationActionIndex;

    /// Checks the classical action-index mirror against the circuit on every address state.
    operation TestUnaryIterationActionIndex(numActions : Int) : Bool {
        let numAddressQubits = AddressQubits(numActions);
        let numAddressStates = 1 <<< numAddressQubits;
        use address = Qubit[numAddressQubits];
        use flags = Qubit[numActions];
        mutable allCorrect = true;

        for addressValue in 0..numAddressStates - 1 {
            ApplyXorInPlace(addressValue, address);
            UnaryIteration(address, numActions, index => X(flags[index]));
            ApplyXorInPlace(addressValue, address);

            let actual = Mapped(ResultAsBool, MResetEachZ(flags));
            let expectedIndex = UnaryIterationActionIndex(numActions, addressValue);
            for index in 0..numActions - 1 {
                if actual[index] != (index == expectedIndex) {
                    set allCorrect = false;
                }
            }
        }

        allCorrect
    }

    /// Runs the one-hot iteration on a uniform superposition of every address.
    function MakeTestUnaryIterationSuperposedAddressOp(numActions : Int) : (Qubit[] => Unit) {
        return qs => {
            let numAddressQubits = AddressQubits(numActions);
            Fact(2^numAddressQubits == numActions, "numActions must be a power of two");
            let address = qs[0..numAddressQubits - 1];
            let flags = qs[numAddressQubits...];
            ApplyToEach(H, address);
            UnaryIteration(address, numActions, (index) => {
                X(flags[index]);
            });
        };
    }

    /// Applies `Z` to the exposed unary control for every index flagged in `data`.
    function MakeTestUnaryIterationControlPhasesOp(numActions : Int, data : Bool[]) : (Qubit[] => Unit) {
        return address => {
            ApplyToEach(H, address);
            UnaryIterationWithControl(address, numActions, (index, control) => {
                if data[index] {
                    Z(control);
                }
            });
        };
    }
}
