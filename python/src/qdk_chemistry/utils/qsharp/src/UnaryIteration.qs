// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryIteration {

    import Std.Arrays.MostAndTail;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;


    /// Unary iteration building blocks shared by QROM-style data loading and
    /// unary-iteration phase estimation
    /// References:
    ///   Babbush et al. (arXiv:1805.03662), Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
    /// Applies `action(index)` for each valid address value.
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
    ///
    /// The control qubit passed to `action` is in state |1⟩ exactly on the branch
    /// where the address register holds that index, so callers may use it as a
    /// positive or negative control.
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

    // The signed-power schedule that used to live here is now
    // `QDKChemistry.Utils.UnaryPhaseEstimation.ApplySignedPowerSchedule`, so that all
    // phase-estimation-specific unary-iteration logic sits in one module.

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
    operation TestUnaryIterationOneHot(numActions : Int, addressValue : Int) : Unit {
        let numAddressQubits = AddressQubits(numActions);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + numActions);
        let address = qs[0..numAddressQubits - 1];
        let flags = qs[numAddressQubits...];
        ApplyXorInPlace(addressValue, address);
        UnaryIteration(address, numActions, (index) => {
            X(flags[index]);
        });
        ApplyXorInPlace(addressValue, address);
    }

    /// Runs the one-hot iteration on a uniform superposition of every address.
    operation TestUnaryIterationSuperposedAddress(numActions : Int) : Unit {
        let numAddressQubits = AddressQubits(numActions);
        Fact(2^numAddressQubits == numActions, "numActions must be a power of two");
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + numActions);
        let address = qs[0..numAddressQubits - 1];
        let flags = qs[numAddressQubits...];
        ApplyToEach(H, address);
        UnaryIteration(address, numActions, (index) => {
            X(flags[index]);
        });
    }

    /// Applies `Z` to the exposed unary control for every index flagged in `data`.
    operation TestUnaryIterationControlPhases(numActions : Int, data : Bool[]) : Unit {
        let numAddressQubits = AddressQubits(numActions);
        Fact(2^numAddressQubits == numActions, "numActions must be a power of two");
        let address = QIR.Runtime.AllocateQubitArray(numAddressQubits);
        ApplyToEach(H, address);
        UnaryIterationWithControl(address, numActions, (index, control) => {
            if data[index] {
                Z(control);
            }
        });
    }
}
