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

        // Walks the leaves of the Fig. 7 tree of arXiv:1805.03662 in order. Between leaves the
        // right-going nodes below the branch point are uncomputed, the branch node toggled, and
        // the next leaf's left-going nodes computed. Each step builds only the next leaf's
        // O(log L) path: the branch is its deepest right turn, and below it the previous leaf
        // ran down the right edge of a full subtree, whose address bits count down to 0.
        // `action` is never passed recursively, which QIR generation cannot resolve.
        use helpers = Qubit[n];
        for leaf in 0..numActions - 1 {
            let path = UnaryIterationPath(numActions, leaf);
            if leaf == 0 {
                for depth in 0..Length(path) - 1 {
                    ComputeUnaryNode(ctl, helpers, address, depth, Fst(path[depth]));
                }
            } else {
                let branch = UnaryIterationBranchDepth(path);
                let branchBit = Fst(path[branch]);
                for depth in branch + branchBit..-1..branch + 1 {
                    UncomputeUnaryNode(ctl, helpers, address, depth, branchBit - (depth - branch));
                }
                CNOT(UnaryNodeControl(ctl, helpers, branch), helpers[branch]);
                for depth in branch + 1..Length(path) - 1 {
                    ComputeUnaryNode(ctl, helpers, address, depth, Fst(path[depth]));
                }
            }
            action(actionOffset + leaf, UnaryNodeControl(ctl, helpers, Length(path)));
        }
        let last = UnaryIterationPath(numActions, numActions - 1);
        for depth in Length(last) - 1..-1..0 {
            UncomputeUnaryNode(ctl, helpers, address, depth, Fst(last[depth]));
        }
    }

    /// Qubit controlling the tree node at `depth`: the caller's control at the root, otherwise
    /// the helper of its parent.
    internal function UnaryNodeControl(ctl : Qubit, helpers : Qubit[], depth : Int) : Qubit {
        depth == 0 ? ctl | helpers[depth - 1]
    }

    /// Sets the helper of the node at `depth` for its left branch (address bit `bit` is 0).
    internal operation ComputeUnaryNode(ctl : Qubit, helpers : Qubit[], address : Qubit[], depth : Int, bit : Int) : Unit is Adj {
        within {
            X(address[bit]);
        } apply {
            ExactAnd(UnaryNodeControl(ctl, helpers, depth), address[bit], helpers[depth]);
        }
    }

    /// Clears the helper of the node at `depth` after its right branch (address bit `bit` is 1).
    internal operation UncomputeUnaryNode(ctl : Qubit, helpers : Qubit[], address : Qubit[], depth : Int, bit : Int) : Unit is Adj {
        Adjoint ExactAnd(UnaryNodeControl(ctl, helpers, depth), address[bit], helpers[depth]);
    }

    /// Root-to-leaf path of `leaf` in the unary iteration tree over `numActions` leaves, as
    /// (address bit, goes right) per internal node.
    internal function UnaryIterationPath(numActions : Int, leaf : Int) : (Int, Bool)[] {
        mutable path = [];
        mutable remaining = numActions;
        mutable index = leaf;
        while remaining > 1 {
            let n = AddressQubits(remaining);
            let half = 1 <<< (n - 1);
            let goesRight = index >= half;
            set path += [(n - 1, goesRight)];
            if goesRight {
                set remaining -= half;
                set index -= half;
            } else {
                set remaining = half;
            }
        }
        path
    }

    /// Depth where the path of a leaf other than the first splits from its predecessor's: its
    /// deepest right turn, since below it the path only turns left.
    internal function UnaryIterationBranchDepth(path : (Int, Bool)[]) : Int {
        mutable depth = Length(path) - 1;
        while not Snd(path[depth]) {
            set depth -= 1;
        }
        depth
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
