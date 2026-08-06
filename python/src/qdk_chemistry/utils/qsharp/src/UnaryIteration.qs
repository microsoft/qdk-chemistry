// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryIteration {

    import Std.Arrays.MostAndTail;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;


    /// Unary iteration
    /// References:
    ///   Babbush et al. (arXiv:1805.03662), Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
    operation UnaryIteration(
        address : Qubit[],
        numActions : Int,
        action : (Int => Unit is Adj + Ctl),
    ) : Unit is Adj {
        Fact(numActions > 0, "actions cannot be empty");
        if numActions == 1 {
            action(0);
        } else {
            UnaryIterationWithControl(address, numActions, (index, control) => {
                Controlled action([control], index);
            });
        }
    }

    /// Applies one action per address value and exposes its active unary control.
    operation UnaryIterationWithControl(
        address : Qubit[],
        numActions : Int,
        action : ((Int, Qubit) => Unit is Adj),
    ) : Unit is Adj {
        Fact(numActions > 0, "actions cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(numActions)));
        Fact(
            Length(address) >= n,
            $"address register is too small, requires at least {n} qubits",
        );

        if numActions == 1 {
            use control = Qubit();
            within {
                X(control);
            } apply {
                action(0, control);
            }
        } else {
            let (most, tail) = MostAndTail(address[...n - 1]);

            within {
                X(tail);
            } apply {
                SinglyControlledUnaryIterationWithControl(tail, most, 2^(n - 1), 0, action);
            }

            SinglyControlledUnaryIterationWithControl(
                tail,
                most,
                numActions - 2^(n - 1),
                2^(n - 1),
                action,
            );
        }
    }

    internal operation SinglyControlledUnaryIterationWithControl(
        ctl : Qubit,
        address : Qubit[],
        numActions : Int,
        actionOffset : Int,
        action : ((Int, Qubit) => Unit is Adj),
    ) : Unit is Adj {
        Fact(numActions > 0, "actions cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(numActions)));
        Fact(
            Length(address) >= n,
            $"address register is too small, requires at least {n} qubits",
        );

        if numActions == 1 {
            action(actionOffset, ctl);
        } else {
            use helper = Qubit();

            let (most, tail) = MostAndTail(address[...n - 1]);

            within {
                X(tail);
            } apply {
                AND(ctl, tail, helper);
            }

            SinglyControlledUnaryIterationWithControl(helper, most, 2^(n - 1), actionOffset, action);

            CNOT(ctl, helper);

            SinglyControlledUnaryIterationWithControl(
                helper,
                most,
                numActions - 2^(n - 1),
                actionOffset + 2^(n - 1),
                action,
            );

            Adjoint AND(ctl, tail, helper);
        }
    }

    /// Number of address qubits needed to enumerate `numActions` values.
    function AddressQubits(numActions : Int) : Int {
        Ceiling(Lg(IntAsDouble(numActions)))
    }

    /// Flips `flags[index]` for the single selected address.
    ///
    /// The returned operation acts on `[address | flags]` and restores the address register,
    /// so the caller allocates `AddressQubits(numActions) + numActions` qubits.
    function MakeTestUnaryIterationOneHotOp(numActions : Int, addressValue : Int) : (Qubit[] => Unit) {
        (qs) => {
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
    function MakeTestUnaryIterationSuperposedAddressOp(numActions : Int) : (Qubit[] => Unit) {
        (qs) => {
            let numAddressQubits = AddressQubits(numActions);
            Fact(2^numAddressQubits == numActions, "numActions must be a power of two");
            let address = qs[0..numAddressQubits - 1];
            let flags = qs[numAddressQubits...];
            ApplyToEach(H, address);
            UnaryIteration(address, numActions, (index) => {
                X(flags[index]);
            });
        }
    }

    /// Applies `Z` to the exposed unary control for every index flagged in `data`.
    function MakeTestUnaryIterationControlPhasesOp(numActions : Int, data : Bool[]) : (Qubit[] => Unit) {
        (address) => {
            let numAddressQubits = AddressQubits(numActions);
            Fact(2^numAddressQubits == numActions, "numActions must be a power of two");
            ApplyToEach(H, address);
            UnaryIterationWithControl(address, numActions, (index, control) => {
                if data[index] {
                    Z(control);
                }
            });
        }
    }
}
