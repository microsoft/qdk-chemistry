// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Unary iteration building blocks shared by QROM-style data loading and
/// unary-iteration phase estimation.
///
/// Operations:
///   UnaryIteration — applies one action per address value.
///   UnaryIterationWithControl — same, but exposes the active one-hot control.
///   UnaryIterationPowerSchedule — signed-power schedule over self-inverse blocks.
///
/// All operations support address ranges that are not powers of two.
///
/// References:
///   Babbush et al. (arXiv:1805.03662), Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
namespace QDKChemistry.Utils.UnaryIteration {

    import Std.Arrays.MostAndTail;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;

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

    /// Applies `numBlocks` self-inverse blocks, omitting one of `numBlocks + 1` reflections.
    ///
    /// With reflection A, block B, and walk W = A·B, the branch selected by address t
    /// applies W^(numBlocks - 2t): pairs before the omitted reflection compose as W†
    /// and pairs after it compose as W.
    operation UnaryIterationPowerSchedule(
        address : Qubit[],
        numBlocks : Int,
        applyReflectionUnlessSelected : (Qubit => Unit is Adj),
        applyBlock : (Unit => Unit is Adj),
    ) : Unit is Adj {
        Fact(numBlocks > 0, "numBlocks must be positive");
        UnaryIterationWithControl(address, numBlocks + 1, (slot, selected) => {
            applyReflectionUnlessSelected(selected);
            if slot < numBlocks {
                applyBlock();
            }
        });
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

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Applies the power schedule with reflection A = X and block B = H for testing.
    operation TestUnaryIterationPowerSchedule(
        numBlocks : Int,
        addressValue : Int,
    ) : Unit {
        let numAddressQubits = Ceiling(Lg(IntAsDouble(numBlocks + 1)));
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + 1);
        let address = qs[0..numAddressQubits - 1];
        let target = qs[numAddressQubits];

        ApplyXorInPlace(addressValue, address);
        UnaryIterationPowerSchedule(address, numBlocks, (selected) => {
            within {
                X(selected);
            } apply {
                Controlled X([selected], target);
            }
        }, () => {
            H(target);
        });
        ApplyXorInPlace(addressValue, address);
    }
}
