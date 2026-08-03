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

    /// Number of address qubits needed to enumerate `numActions` values.
    function AddressQubits(numActions : Int) : Int {
        Ceiling(Lg(IntAsDouble(numActions)))
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test wrappers
    //
    //  These leak their qubits with `QIR.Runtime.AllocateQubitArray` so that the
    //  full statevector survives until the caller dumps it, and they restore the
    //  address register to |0...0> so the dumped state is exactly the payload.
    // ═══════════════════════════════════════════════════════════════════════════

    /// Flips `flags[index]` for the single selected address.
    ///
    /// The dumped state must be the basis state whose only excited flag qubit is
    /// `flags[addressValue]`, which pins down both the selection and the absence of
    /// residual entanglement with the internal AND ancillas.
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
    ///
    /// `numActions` must be a power of two so that all addresses are in range; the
    /// dumped state must be the entangled sum over addresses with no ancilla residue.
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
    ///
    /// The address register carries a uniform superposition, so the dumped
    /// amplitudes are the sign pattern of `data`. This checks that the exposed
    /// control is an exact equality predicate for phase-only actions, which is the
    /// mode used by the reflection schedule.
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

    /// Signed-power schedule with reflection A = Z and block B = X on one target
    /// prepared in Ry(0.7)|0>.
    ///
    /// Branch `addressValue` must apply exactly (Z·X)^(numBlocks - 2*addressValue),
    /// including the relative phase, which distinguishes every power in the schedule.
    operation TestUnaryIterationSignedPower(numBlocks : Int, addressValue : Int) : Unit {
        let numAddressQubits = AddressQubits(numBlocks + 1);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + 1);
        let address = qs[0..numAddressQubits - 1];
        let target = qs[numAddressQubits];
        ApplyXorInPlace(addressValue, address);
        Ry(0.7, target);
        UnaryIterationPowerSchedule(address, numBlocks, (selected) => {
            within {
                X(selected);
            } apply {
                Controlled Z([selected], target);
            }
        }, () => {
            X(target);
        });
        ApplyXorInPlace(addressValue, address);
    }
}
