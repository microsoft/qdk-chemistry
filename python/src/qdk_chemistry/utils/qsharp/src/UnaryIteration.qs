// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryIteration {

    import Std.Arrays.MostAndTail;
    import Std.Canon.ApplyToEach;
    import Std.Canon.ApplyXorInPlace;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;


    /// Unary iteration
    ///
    /// Produces the one-hot indicator of the address register one qubit at a time.
    ///
    /// References:
    ///   Babbush et al. Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity
    ///   (arXiv:1805.03662), Sec. III A "Unary Iteration and Indexed Operations", Figs. 3-7.
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

    /// Applies one action per address value recursively.
    operation UnaryIterationWithControl(
        address : Qubit[],
        numActions : Int,
        action : ((Int, Qubit) => Unit is Adj),
    ) : Unit is Adj {
        Fact(numActions > 0, "actions cannot be empty");

        let n = AddressQubits(numActions);
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

        let n = AddressQubits(numActions);
        Fact(
            Length(address) >= n,
            $"address register is too small, requires at least {n} qubits",
        );

        if numActions == 1 {
            action(actionOffset, ctl);
        } else {
            if BeginEstimateCaching("QDKChemistry.Utils.UnaryIteration.SinglyControlledUnaryIterationWithControl", numActions) {
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

                EndEstimateCaching();
            }
        }
    }

    /// Number of address qubits needed to enumerate `numActions` values, i.e.
    /// `Ceiling(Lg(numActions))`.
    function AddressQubits(numActions : Int) : Int {
        Fact(numActions > 0, "numActions must be positive");
        return BitSizeI(numActions - 1);
    }

    /// Flips `flags[index]` for the single selected address.
    internal function MakeTestUnaryIterationOneHotOp(numActions : Int, addressValue : Int) : (Qubit[] => Unit) {
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
    internal function MakeTestUnaryIterationSuperposedAddressOp(numActions : Int) : (Qubit[] => Unit) {
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
    internal function MakeTestUnaryIterationControlPhasesOp(numActions : Int, data : Bool[]) : (Qubit[] => Unit) {
        return address => {
            Fact(2^AddressQubits(numActions) == numActions, "numActions must be a power of two");
            ApplyToEach(H, address);
            UnaryIterationWithControl(address, numActions, (index, control) => {
                if data[index] {
                    Z(control);
                }
            });
        };
    }
}
