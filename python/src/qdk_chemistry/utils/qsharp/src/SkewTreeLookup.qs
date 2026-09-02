// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Clean-ancilla skew-tree QROM from arXiv:2605.30455v1, Section V.A.
namespace QDKChemistry.Utils.SkewTreeLookup {

    import Std.Arrays.All;
    import Std.Arrays.Mapped;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.ResultAsBool;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.Measurement.MResetEachZ;
    import Std.TableLookup.Select;

    internal function XorWords(left : Bool[], right : Bool[]) : Bool[] {
        Fact(Length(left) == Length(right), "words must have the same width");
        mutable result = left;
        for index in 0..Length(left) - 1 {
            set result w/= index <- left[index] != right[index];
        }
        return result;
    }

    internal function FirstTrueIndex(word : Bool[]) : Int {
        for index in 0..Length(word) - 1 {
            if word[index] {
                return index;
            }
        }
        return -1;
    }

    /// Computes the subset-lattice Mobius transform used as skew-tree data.
    internal function SkewTreeData(data : Bool[][]) : Bool[][] {
        Fact(Length(data) > 0, "data cannot be empty");
        let width = Length(data[0]);
        Fact(width > 0, "data words cannot be empty");
        Fact((Length(data) &&& (Length(data) - 1)) == 0, "data length must be a power of two");

        mutable skewData : Bool[][] = [];
        for node in 0..Length(data) - 1 {
            mutable word = [false, size = width];
            for subset in 0..node {
                Fact(Length(data[subset]) == width, "data words must have the same width");
                if (subset &&& node) == subset {
                    set word = XorWords(word, data[subset]);
                }
            }
            set skewData += [word];
        }
        return skewData;
    }

    internal operation ApplyXorWord(word : Bool[], target : Qubit[]) : Unit is Adj + Ctl {
        Fact(Length(word) == Length(target), "word and target must have the same width");
        for index in 0..Length(word) - 1 {
            if word[index] {
                X(target[index]);
            }
        }
    }

    internal operation ApplyControlledXorWord(control : Qubit, word : Bool[], target : Qubit[]) : Unit is Adj + Ctl {
        Fact(Length(word) == Length(target), "word and target must have the same width");
        for index in 0..Length(word) - 1 {
            if word[index] {
                CNOT(control, target[index]);
            }
        }
    }

    /// Applies a doubly controlled XOR word with one Toffoli, using a marked target bit
    /// as a toggle detector. The target may contain arbitrary quantum data.
    internal operation ApplyDoublyControlledXorWord(
        firstControl : Qubit,
        secondControl : Qubit,
        word : Bool[],
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        Fact(Length(word) == Length(target), "word and target must have the same width");
        let pivot = FirstTrueIndex(word);

        if pivot >= 0 {
            for index in 0..Length(word) - 1 {
                if index != pivot and word[index] {
                    CNOT(target[pivot], target[index]);
                }
            }
            CCNOT(firstControl, secondControl, target[pivot]);
            for index in Length(word) - 1..-1..0 {
                if index != pivot and word[index] {
                    CNOT(target[pivot], target[index]);
                }
            }
        }
    }

    internal operation ApplySkewSubtreeRest(
        node : Int,
        highestVariableBit : Int,
        activation : Qubit,
        skewData : Bool[][],
        address : Qubit[],
        target : Qubit[],
        pathAncilla : Qubit[],
        depth : Int,
    ) : Unit is Adj {
        if highestVariableBit > 0 {
            ApplyDoublyControlledXorWord(activation, address[0], skewData[node + 1], target);

            for childBit in 1..highestVariableBit - 1 {
                let child = node + (1 <<< childBit);
                AND(activation, address[childBit], pathAncilla[depth]);
                ApplyControlledXorWord(pathAncilla[depth], skewData[child], target);
                ApplySkewSubtreeRest(
                    child,
                    childBit,
                    pathAncilla[depth],
                    skewData,
                    address,
                    target,
                    pathAncilla,
                    depth + 1,
                );
                Adjoint AND(activation, address[childBit], pathAncilla[depth]);
            }
        }
    }

    internal operation ApplySkewTreeLookupAscending(
        skewData : Bool[][],
        address : Qubit[],
        target : Qubit[],
    ) : Unit is Adj {
        let numAddressQubits = BitSizeI(Length(skewData) - 1);
        ApplyXorWord(skewData[0], target);
        if numAddressQubits > 0 {
            let numPathAncilla = numAddressQubits > 2 ? numAddressQubits - 2 | 0;
            use pathAncilla = Qubit[numPathAncilla];
            for rootBit in 0..numAddressQubits - 1 {
                let root = 1 <<< rootBit;
                ApplyControlledXorWord(address[rootBit], skewData[root], target);
                ApplySkewSubtreeRest(
                    root,
                    rootBit,
                    address[rootBit],
                    skewData,
                    address,
                    target,
                    pathAncilla,
                    0,
                );
            }
        }
    }

    /// Loads `data[address]` in reverse DFS order using at most `n - 2` clean path
    /// ancillas for `n` address qubits.
    operation SkewTreeLookup(data : Bool[][], address : Qubit[], target : Qubit[]) : Unit is Adj {
        let skewData = SkewTreeData(data);
        let numAddressQubits = BitSizeI(Length(data) - 1);
        Fact(Length(address) >= numAddressQubits, "address register is too small");
        Fact(Length(target) == Length(data[0]), "target has the wrong width");
        Adjoint ApplySkewTreeLookupAscending(skewData, address, target);
    }

    internal function SubtreeCorrectionOrder(node : Int, highestVariableBit : Int) : Int[] {
        mutable order : Int[] = [];
        if highestVariableBit > 1 {
            for childBit in highestVariableBit - 1..-1..1 {
                let child = node + (1 <<< childBit);
                set order += [child] + SubtreeCorrectionOrder(child, childBit);
            }
        }
        if highestVariableBit > 0 {
            set order += [node + 1];
        }
        return order;
    }

    /// Returns non-level-1 nodes in the order their CCZ injections complete during
    /// reverse DFS traversal. This order is required when corrections compose.
    internal function CczCorrectionOrder(numAddressQubits : Int) : Int[] {
        Fact(numAddressQubits >= 0, "number of address qubits cannot be negative");
        mutable order : Int[] = [];
        if numAddressQubits > 1 {
            for rootBit in numAddressQubits - 1..-1..1 {
                let root = 1 <<< rootBit;
                set order += SubtreeCorrectionOrder(root, rootBit);
            }
        }
        return order;
    }

    internal function SkewTreeCczCount(numEntries : Int) : Int {
        Fact(numEntries > 0 and (numEntries &&& (numEntries - 1)) == 0, "number of entries must be a power of two");
        let numAddressQubits = BitSizeI(numEntries - 1);
        return numEntries - numAddressQubits - 1;
    }

    internal function SkewTreeCleanAncillaCount(numEntries : Int) : Int {
        Fact(numEntries > 0 and (numEntries &&& (numEntries - 1)) == 0, "number of entries must be a power of two");
        let numAddressQubits = BitSizeI(numEntries - 1);
        return numAddressQubits > 2 ? numAddressQubits - 2 | 0;
    }

    /// Number of serial waits caused specifically by CCZ-injection corrections.
    /// The zero result requires the timing premise of Theorem 3.
    internal function CczCorrectionWaits(
        numEntries : Int,
        useClassicalAbsorption : Bool,
        latticeSurgeryCycles : Int,
        reactionCycles : Int,
    ) : Int {
        Fact(latticeSurgeryCycles > 0, "lattice-surgery duration must be positive");
        Fact(reactionCycles >= 0, "reaction duration cannot be negative");
        let numCcz = SkewTreeCczCount(numEntries);
        if useClassicalAbsorption {
            Fact(latticeSurgeryCycles >= reactionCycles, "classical absorption requires lattice-surgery duration >= reaction duration");
            return 0;
        }
        return numCcz;
    }

    /// Worst-case number of bitwise XORs performed by Theorem 3's streaming
    /// update when every injection has both outcomes set.
    internal function ClassicalAbsorptionBitXorBound(numEntries : Int, wordWidth : Int) : Int {
        Fact(numEntries > 0 and (numEntries &&& (numEntries - 1)) == 0, "number of entries must be a power of two");
        Fact(wordWidth >= 0, "word width cannot be negative");
        let numAddressQubits = BitSizeI(numEntries - 1);
        let descendantVisits = numAddressQubits * (numEntries / 2) - (numEntries - 1);
        return 3 * wordWidth * descendantVisits;
    }

    /// Same-throughput compute-workspace points from Section V.A: one CCZ per d/2 cycles.
    internal function FastSkewTreeComputePatches() : Int { 3 * 24 }
    internal function PriorArtFastLookupComputePatches() : Int { 9 * 72 }

    internal function SkewParent(node : Int) : Int {
        node &&& (node - 1)
    }

    internal function SkewPivotBit(node : Int) : Int {
        node &&& -node
    }

    internal function IsSkewDescendant(node : Int, candidate : Int) : Bool {
        (candidate &&& -SkewPivotBit(node)) == node
    }

    internal function ApplyCczCorrectionAtNode(
        skewData : Bool[][],
        node : Int,
        firstOutcome : Bool,
        secondOutcome : Bool,
    ) : Bool[][] {
        let pivotBit = SkewPivotBit(node);
        let parent = SkewParent(node);
        mutable corrected = skewData;
        for descendant in node..Length(skewData) - 1 {
            if IsSkewDescendant(node, descendant) {
                if firstOutcome {
                    let target = descendant ^^^ pivotBit;
                    set corrected w/= target <- XorWords(corrected[target], skewData[descendant]);
                }
                if secondOutcome {
                    let target = descendant ^^^ parent;
                    set corrected w/= target <- XorWords(corrected[target], skewData[descendant]);
                }
                if firstOutcome and secondOutcome {
                    let target = descendant ^^^ node;
                    set corrected w/= target <- XorWords(corrected[target], skewData[descendant]);
                }
            }
        }
        return corrected;
    }

    /// Applies Theorem 3's streaming classical update to the current not-yet-loaded
    /// skew-data buffer. Outcomes are indexed by skew-tree node.
    internal function AbsorbCczCorrections(
        skewData : Bool[][],
        firstOutcomes : Bool[],
        secondOutcomes : Bool[],
    ) : Bool[][] {
        let numAddressQubits = BitSizeI(Length(skewData) - 1);
        Fact(Length(firstOutcomes) == Length(skewData), "first-outcome array has the wrong length");
        Fact(Length(secondOutcomes) == Length(skewData), "second-outcome array has the wrong length");

        mutable corrected = skewData;
        for node in CczCorrectionOrder(numAddressQubits) {
            set corrected = ApplyCczCorrectionAtNode(
                corrected,
                node,
                firstOutcomes[node],
                secondOutcomes[node],
            );
        }
        return corrected;
    }

    internal function ErroneousSkewActivation(
        node : Int,
        addressValue : Int,
        firstOutcomes : Bool[],
        secondOutcomes : Bool[],
    ) : Bool {
        if node == 0 {
            return true;
        }
        let pivotBit = SkewPivotBit(node);
        let selectionBit = (addressValue &&& pivotBit) != 0;
        let parent = SkewParent(node);
        if parent == 0 {
            return selectionBit;
        }
        return (selectionBit != firstOutcomes[node]) and (
            ErroneousSkewActivation(parent, addressValue, firstOutcomes, secondOutcomes) != secondOutcomes[node]
        );
    }

    internal function EvaluateErroneousSkewLookup(
        skewData : Bool[][],
        addressValue : Int,
        firstOutcomes : Bool[],
        secondOutcomes : Bool[],
    ) : Bool[] {
        mutable output = [false, size = Length(skewData[0])];
        for node in 0..Length(skewData) - 1 {
            if ErroneousSkewActivation(node, addressValue, firstOutcomes, secondOutcomes) {
                set output = XorWords(output, skewData[node]);
            }
        }
        return output;
    }

    internal function TestClassicalCczAbsorption(
        data : Bool[][],
        firstOutcomes : Bool[],
        secondOutcomes : Bool[],
    ) : Bool {
        let corrected = AbsorbCczCorrections(SkewTreeData(data), firstOutcomes, secondOutcomes);
        for addressValue in 0..Length(data) - 1 {
            if EvaluateErroneousSkewLookup(
                corrected,
                addressValue,
                firstOutcomes,
                secondOutcomes,
            ) != data[addressValue] {
                return false;
            }
        }
        return true;
    }

    internal operation TestSkewTreeLookupCorrectness(data : Bool[][]) : Bool {
        let numAddressQubits = BitSizeI(Length(data) - 1);
        let width = Length(data[0]);
        use address = Qubit[numAddressQubits];
        use target = Qubit[width];
        use copy = Qubit[width];
        mutable allCorrect = true;

        for addressValue in 0..Length(data) - 1 {
            ApplyXorInPlace(addressValue, address);
            within {
                SkewTreeLookup(data, address, target);
            } apply {
                ApplyToEachCA(CNOT, Zipped(target, copy));
            }
            ApplyXorInPlace(addressValue, address);

            let actual = Mapped(ResultAsBool, MResetEachZ(copy));
            if actual != data[addressValue] {
                set allCorrect = false;
            }
        }
        return allCorrect;
    }

    internal operation TestSkewTreeLookupPhaseAgreement(data : Bool[][]) : Bool {
        let numAddressQubits = BitSizeI(Length(data) - 1);
        let width = Length(data[0]);
        use address = Qubit[numAddressQubits];
        ApplyToEachA(H, address);

        {
            use target = Qubit[width];
            within {
                SkewTreeLookup(data, address, target);
            } apply {
                Z(target[0]);
            }
        }
        {
            use target = Qubit[width];
            within {
                Select(data, address, target);
            } apply {
                Z(target[0]);
            }
        }

        Adjoint ApplyToEachA(H, address);
        return All(result -> result == Zero, MResetEachZ(address));
    }

    internal operation SkewTreeLookupResourceEstimate(data : Bool[][]) : Unit {
        let numAddressQubits = BitSizeI(Length(data) - 1);
        use address = Qubit[numAddressQubits];
        use target = Qubit[Length(data[0])];
        SkewTreeLookup(data, address, target);
        ResetAll(target);
    }
}