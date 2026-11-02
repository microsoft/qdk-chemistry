// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.UnaryIteration {

    import Std.Arrays.MostAndTail;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.Measurement.MResetEachZ;


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

    /// `UnaryIterationWithControl` under the controls `ctls`.
    operation ControlledUnaryIterationWithControl(
        ctls : Qubit[],
        address : Qubit[],
        numActions : Int,
        action : ((Int, Qubit) => Unit is Adj),
    ) : Unit is Adj {
        let numControls = Length(ctls);
        if numControls == 0 {
            UnaryIterationWithControl(address, numActions, action);
        } elif numControls == 1 {
            SinglyControlledUnaryIterationWithControl(ctls[0], address, numActions, 0, action);
        } else {
            use anc = Qubit[numControls - 1];
            within {
                AND(ctls[0], ctls[1], anc[0]);
                for i in 2..numControls - 1 {
                    AND(anc[i - 2], ctls[i], anc[i - 1]);
                }
            } apply {
                SinglyControlledUnaryIterationWithControl(anc[numControls - 2], address, numActions, 0, action);
            }
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

        use helpers = Qubit[n];
        for (step, control, value, helper) in UnaryIterationSchedule(numActions, actionOffset, -1) {
            let controlQubit = control < 0 ? ctl | helpers[control];
            if step == ComputeStep() {
                within {
                    X(address[value]);
                } apply {
                    ExactAnd(controlQubit, address[value], helpers[helper]);
                }
            } elif step == ToggleStep() {
                CNOT(controlQubit, helpers[helper]);
            } elif step == UncomputeStep() {
                Adjoint ExactAnd(controlQubit, address[value], helpers[helper]);
            } else {
                action(value, controlQubit);
            }
        }
    }

    internal function ComputeStep() : Int { 0 }
    internal function ToggleStep() : Int { 1 }
    internal function UncomputeStep() : Int { 2 }
    internal function ActionStep() : Int { 3 }

    /// Gate sequence of Fig. 5 of arXiv:1805.03662 below the control `control` (-1 for the
    /// caller's qubit, otherwise a helper index), as (step, control, address bit or action
    /// index, helper) tuples.
    ///
    /// The recursion is classical so the callable `action` is never passed recursively, which
    /// QIR generation cannot resolve.
    internal function UnaryIterationSchedule(numActions : Int, actionOffset : Int, control : Int) : (Int, Int, Int, Int)[] {
        if numActions == 1 {
            return [(ActionStep(), control, actionOffset, 0)];
        }
        let n = AddressQubits(numActions);
        let half = 1 <<< (n - 1);
        let helper = control + 1;
        [(ComputeStep(), control, n - 1, helper)]
            + UnaryIterationSchedule(half, actionOffset, helper)
            + [(ToggleStep(), control, 0, helper)]
            + UnaryIterationSchedule(numActions - half, actionOffset + half, helper)
            + [(UncomputeStep(), control, n - 1, helper)]
    }

    /// `AND`, uncomputed by measurement.
    @Config(Adaptive)
    internal operation ExactAnd(control1 : Qubit, control2 : Qubit, target : Qubit) : Unit is Adj {
        AND(control1, control2, target);
    }

    /// Without measurement `AND` is a relative-phase Toffoli, which the iteration's CNOT breaks.
    @Config(not Adaptive)
    internal operation ExactAnd(control1 : Qubit, control2 : Qubit, target : Qubit) : Unit is Adj {
        CCNOT(control1, control2, target);
    }

    /// Number of address qubits needed to enumerate `numActions` values, i.e.
    /// `Ceiling(Lg(numActions))`.
    function AddressQubits(numActions : Int) : Int {
        Fact(numActions > 0, "numActions must be positive");
        return BitSizeI(numActions - 1);
    }

    /// Returns the action selected by `UnaryIteration` for any fitted address state.
    ///
    /// Valid addresses map to themselves. When `numActions` is not a power of two,
    /// the recursive iteration aliases each unused state onto a valid final subtree.
    /// Classical unlookup tables use this function to reproduce that routing exactly.
    internal function UnaryIterationActionIndex(numActions : Int, addressValue : Int) : Int {
        Fact(numActions > 0, "numActions must be positive");
        Fact(addressValue >= 0, "addressValue must be nonnegative");

        if numActions == 1 {
            return 0;
        }

        let n = AddressQubits(numActions);
        let addressState = addressValue % (1 <<< n);
        let lowerSubtreeSize = 1 <<< (n - 1);
        if addressState < lowerSubtreeSize {
            return addressState;
        }

        return lowerSubtreeSize + UnaryIterationActionIndex(
            numActions - lowerSubtreeSize,
            addressState - lowerSubtreeSize
        );
    }
}
