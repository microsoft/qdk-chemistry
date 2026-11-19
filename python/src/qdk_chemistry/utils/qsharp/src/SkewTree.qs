// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Box-restricted skew-tree table lookup (Khattar & Gidney, arXiv:2407.17966, Fig. 10;
/// Low et al., arXiv:2605.30455, App. C).
///
/// Node p of the skew tree is gated on the AND of the address bits set in p, so the target
/// receives the XOR of the subset-XOR (Möbius) coefficients of every p ⊆ x, which is data[x].
/// Every subset of an in-range address is in range, so only the nodes inside the box
/// `node < Length(data)` and `field_j(node) < extent_j` are visited: a table over a composite
/// address costs the product of its extents, not of their power-of-two paddings.
namespace QDKChemistry.Utils.SkewTree {

    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Zipped;
    import Std.Core.Length;
    import Std.Intrinsic.AND;
    import Std.Math.MinI;

    /// Whether `node` lies in the box: below `numData` and each `(bits, extent)` field, low bits first, below its extent.
    function InSkewBox(node : Int, numData : Int, fields : (Int, Int)[]) : Bool {
        mutable inBox = node < numData;
        mutable offset = 0;
        for (bits, extent) in fields {
            if ((node >>> offset) &&& ((1 <<< bits) - 1)) >= extent {
                set inBox = false;
            }
            set offset += bits;
        }
        inBox
    }

    /// Number of box nodes, which is the Toffoli count of a singly controlled lookup plus one.
    function SkewTreeBoxSize(numData : Int, fields : (Int, Int)[], numAddressBits : Int) : Int {
        mutable count = 0;
        for node in 0..MinI(numData, 1 <<< numAddressBits) - 1 {
            if InSkewBox(node, numData, fields) {
                set count += 1;
            }
        }
        count
    }

    /// Subset-XOR (Möbius) transform over GF(2): entry p becomes the XOR of every entry q ⊆ p; it is its own inverse.
    function SubsetXorTransform(words : Bool[][]) : Bool[][] {
        let n = Length(words);
        if n <= 1 {
            return words;
        }
        mutable half = 1;
        while 2 * half < n {
            set half *= 2;
        }
        let low = SubsetXorTransform(words[0..half - 1]);
        let high = SubsetXorTransform(words[half..n - 1]);
        low + MappedOverRange(i -> MappedOverRange(k -> low[i][k] != high[i][k], 0..Length(high[i]) - 1), 0..n - half - 1)
    }

    /// What the lookup writes at every one of the `2^numAddressBits` addresses: `data[x]` inside the box, the
    /// XOR of the in-box coefficients below `x` outside it.
    function SkewTreeOutputTable(data : Bool[][], fields : (Int, Int)[], numAddressBits : Int) : Bool[][] {
        let coefficients = SubsetXorTransform(data);
        let zero = MappedOverRange(_ -> false, 0..Length(data[0]) - 1);
        SubsetXorTransform(
            MappedOverRange(
                node -> if node < Length(data) and InSkewBox(node, Length(data), fields) { coefficients[node] } else { zero },
                0..(1 <<< numAddressBits) - 1
            )
        )
    }

    /// XORs `data[address]` into `target` for every address inside the box.
    ///
    /// Singly controlled it costs one AND per box node beyond the root, as many Toffolis as
    /// `Select` on an unpadded table; uncontrolled the root's children are address bits and cost nothing.
    /// Outside the box the target picks up `SkewTreeOutputTable`, not `Select`'s aliased row.
    operation SkewTreeLookup(
        data : Bool[][],
        fields : (Int, Int)[],
        address : Qubit[],
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        body (...) {
            let coefficients = SubsetXorTransform(data);
            for (bit, qubit) in Zipped(coefficients[0], target) {
                if bit {
                    X(qubit);
                }
            }
            for j in 0..Length(address) - 1 {
                if InSkewBox(1 <<< j, Length(data), fields) {
                    VisitSkewNode(coefficients, fields, address, 1 <<< j, j, address[j], target);
                }
            }
        }
        adjoint self;
        controlled (controls, ...) {
            let coefficients = SubsetXorTransform(data);
            if Length(controls) == 0 {
                SkewTreeLookup(data, fields, address, target);
            } elif Length(controls) == 1 {
                VisitSkewNode(coefficients, fields, address, 0, Length(address), controls[0], target);
            } else {
                use root = Qubit();
                within {
                    Controlled X(controls, root);
                } apply {
                    VisitSkewNode(coefficients, fields, address, 0, Length(address), root, target);
                }
            }
        }
        controlled adjoint self;
    }

    /// Writes node `node`'s coefficient under `control`, then visits its in-box children `node + 2^j`, `j < lowBit`.
    internal operation VisitSkewNode(
        coefficients : Bool[][],
        fields : (Int, Int)[],
        address : Qubit[],
        node : Int,
        lowBit : Int,
        control : Qubit,
        target : Qubit[],
    ) : Unit is Adj {
        for (bit, qubit) in Zipped(coefficients[node], target) {
            if bit {
                CNOT(control, qubit);
            }
        }
        for j in 0..lowBit - 1 {
            let child = node + (1 <<< j);
            if InSkewBox(child, Length(coefficients), fields) {
                use childControl = Qubit();
                within {
                    AND(control, address[j], childControl);
                } apply {
                    VisitSkewNode(coefficients, fields, address, child, j, childControl, target);
                }
            }
        }
    }
}
