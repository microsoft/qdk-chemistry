// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Test-only drivers for `QDKChemistry.Utils.UnaryIteration`.
///
/// The Python test layer evaluates this file into a throwaway context (see the
/// `qsharp_test_context` fixture in `tests/conftest.py`); it is never part of the
/// shipped `qdk_chemistry.utils.qsharp` project.
namespace QDKChemistry.TestUtils.UnaryIterationTests {

    import Std.Canon.ApplyToEach;
    import Std.Canon.ApplyXorInPlace;
    import Std.Diagnostics.Fact;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationWithControl;

    /// Flips `flags[index]` for the single selected address.
    function TestMakeOneHotOp(numActions : Int, addressValue : Int) : (Qubit[] => Unit) {
        return qs => {
            let numAddressQubits = AddressQubits(numActions);
            let address = qs[0..numAddressQubits - 1];
            let flags = qs[numAddressQubits...];
            ApplyXorInPlace(addressValue, address);
            UnaryIteration(address, numActions, (index) => {
                X(flags[index]);
            });
            ApplyXorInPlace(addressValue, address);
        }
    }

    /// Runs the one-hot iteration on a uniform superposition of every address.
    function TestMakeSuperposedAddressOp(numActions : Int) : (Qubit[] => Unit) {
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
    function TestMakeControlPhasesOp(numActions : Int, data : Bool[]) : (Qubit[] => Unit) {
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
