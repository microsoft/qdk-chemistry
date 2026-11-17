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

        // Walks the leaves of the Fig. 7 tree of arXiv:1805.03662 in order, keeping the helper of
        // the node that tests address bit p in helpers[p]. Leaf `leaf` branches off its
        // predecessor's path at bit b = TrailingZeros(leaf): the predecessor's right-going nodes
        // on bits b-1..0 are uncomputed, node b is toggled, and the leaf's left-going nodes on
        // bits m-1..0 are computed, where m = b unless node b is on the right spine that leads to
        // the last leaf. Each leaf costs O(1) amortized classical work and holds no path, and
        // `action` is never passed recursively, which QIR generation cannot resolve.
        let lastLeaf = numActions - 1;
        use helpers = Qubit[n];
        for bit in n - 1..-1..0 {
            ComputeUnaryNode(UnaryNodeControl(ctl, helpers, UnaryParentBit(lastLeaf, 0, bit)), address[bit], helpers[bit]);
        }
        action(actionOffset, n > 0 ? helpers[0] | ctl);
        for leaf in 1..lastLeaf {
            let branch = TrailingZeros(leaf);
            for bit in 0..branch - 1 {
                UncomputeUnaryNode(helpers[bit + 1], address[bit], helpers[bit]);
            }
            CNOT(UnaryNodeControl(ctl, helpers, UnaryParentBit(lastLeaf, leaf, branch)), helpers[branch]);
            let onSpine = (leaf >>> branch) == (lastLeaf >>> branch);
            let numNew = onSpine ? BitSizeI(lastLeaf - leaf) | branch;
            for bit in numNew - 1..-1..0 {
                ComputeUnaryNode(helpers[bit == numNew - 1 ? branch | bit + 1], address[bit], helpers[bit]);
            }
            action(actionOffset + leaf, helpers[numNew > 0 ? 0 | branch]);
        }
        for bit in 0..n - 1 {
            if ((lastLeaf >>> bit) &&& 1) == 1 {
                UncomputeUnaryNode(UnaryNodeControl(ctl, helpers, UnaryParentBit(lastLeaf, lastLeaf, bit)), address[bit], helpers[bit]);
            }
        }
    }

    /// Helper of the node testing address bit `parentBit`, or the caller's control at the root
    /// (`parentBit` = -1).
    internal function UnaryNodeControl(ctl : Qubit, helpers : Qubit[], parentBit : Int) : Qubit {
        parentBit < 0 ? ctl | helpers[parentBit]
    }

    /// Address bit of the parent of the node testing `bit` on the path of `leaf`, or -1 at the
    /// root. Off the right spine every bit below the turn off it is tested, so the parent tests
    /// `bit + 1`; on the spine the parent is the next set bit of `lastLeaf`.
    internal function UnaryParentBit(lastLeaf : Int, leaf : Int, bit : Int) : Int {
        let higher = lastLeaf >>> (bit + 1);
        if (leaf >>> (bit + 1)) != higher {
            bit + 1
        } elif higher == 0 {
            -1
        } else {
            bit + 1 + TrailingZeros(higher)
        }
    }

    /// Sets `helper` for the left branch of a node (address bit is 0).
    internal operation ComputeUnaryNode(control : Qubit, addressBit : Qubit, helper : Qubit) : Unit is Adj {
        within {
            X(addressBit);
        } apply {
            ExactAnd(control, addressBit, helper);
        }
    }

    /// Clears `helper` after the right branch of a node (address bit is 1).
    internal operation UncomputeUnaryNode(control : Qubit, addressBit : Qubit, helper : Qubit) : Unit is Adj {
        Adjoint ExactAnd(control, addressBit, helper);
    }

    internal function TrailingZeros(value : Int) : Int {
        Fact(value > 0, "value must be positive");
        mutable count = 0;
        while ((value >>> count) &&& 1) == 0 {
            set count += 1;
        }
        count
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
