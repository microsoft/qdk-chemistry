// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Tree-structured Givens rotations from arXiv:2605.30455v1, Section V.C
/// and Appendix D.1.
namespace QDKChemistry.Utils.ParallelGivens {

    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.BitSizeI;
    import Std.Math.PI;
    import Std.Measurement.MResetEachZ;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.RzViaPhaseGradient;

    internal function GivensTreeDepth(numOrbitals : Int) : Int {
        Fact(numOrbitals > 0, "number of orbitals must be positive");
        return BitSizeI(numOrbitals - 1);
    }

    /// Returns the breadth-first binary-tree edges grouped into disjoint layers.
    internal function GivensTreeLayers(numOrbitals : Int) : (Int, Int)[][] {
        Fact(numOrbitals > 0, "number of orbitals must be positive");
        mutable layers : (Int, Int)[][] = [];
        mutable parents = [0];

        for depth in 0..GivensTreeDepth(numOrbitals) - 1 {
            let stride = 1 <<< depth;
            mutable layer : (Int, Int)[] = [];
            mutable nextParents : Int[] = [];
            for parent in parents {
                set nextParents += [parent];
                let child = parent + stride;
                if child < numOrbitals {
                    set layer += [(parent, child)];
                    set nextParents += [child];
                }
            }
            set layers += [layer];
            set parents = nextParents;
        }
        return layers;
    }

    internal function GivensTreeEdges(numOrbitals : Int) : (Int, Int)[] {
        mutable edges : (Int, Int)[] = [];
        for layer in GivensTreeLayers(numOrbitals) {
            set edges += layer;
        }
        return edges;
    }

    internal function GivensLayerOffset(numOrbitals : Int, layerIndex : Int) : Int {
        let layers = GivensTreeLayers(numOrbitals);
        Fact(layerIndex >= 0 and layerIndex < Length(layers), "Givens layer index is out of range");
        mutable offset = 0;
        if layerIndex > 0 {
            for index in 0..layerIndex - 1 {
                set offset += Length(layers[index]);
            }
        }
        return offset;
    }

    internal function GivensTreeLayersAreDisjoint(numOrbitals : Int) : Bool {
        for layer in GivensTreeLayers(numOrbitals) {
            for firstIndex in 0..Length(layer) - 1 {
                let (firstLeft, firstRight) = layer[firstIndex];
                for secondIndex in firstIndex + 1..Length(layer) - 1 {
                    let (secondLeft, secondRight) = layer[secondIndex];
                    if firstLeft == secondLeft or firstLeft == secondRight or firstRight == secondLeft or firstRight == secondRight {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /// Equation (14): cycles to process `numFactories` controlled-adder bits in parallel.
    internal function ControlledAdderBatchCycles(
        numFactories : Int,
        factoryCycles : Int,
        codeDistance : Int,
        reactionCycles : Int,
    ) : Int {
        Fact(numFactories > 0, "number of factories must be positive");
        Fact(factoryCycles > 0, "factory duration must be positive");
        Fact(codeDistance > 0, "code distance must be positive");
        Fact(reactionCycles >= 0, "reaction duration cannot be negative");
        let productionLimited = 2 * factoryCycles;
        let correctionLimited = factoryCycles + 3 * codeDistance + numFactories * reactionCycles;
        return productionLimited > correctionLimited ? productionLimited | correctionLimited;
    }

    internal function ControlledAdderCczPerDistance(
        numFactories : Int,
        factoryCycles : Int,
        codeDistance : Int,
        reactionCycles : Int,
    ) : Double {
        let batchCycles = ControlledAdderBatchCycles(
            numFactories,
            factoryCycles,
            codeDistance,
            reactionCycles,
        );
        return IntAsDouble(2 * numFactories * codeDistance) / IntAsDouble(batchCycles);
    }

    internal function PhaseGradientAngles(angleValues : Int[], numBits : Int) : Double[] {
        Fact(numBits > 0, "angle words cannot be empty");
        mutable angles : Double[] = [];
        for angleValue in angleValues {
            Fact(angleValue >= 0 and angleValue < (1 <<< numBits), "angle value does not fit in the word");
            set angles += [4.0 * PI() * IntAsDouble(angleValue) / IntAsDouble(1 <<< numBits)];
        }
        return angles;
    }

    /// Applies Figure 19's two-mode Givens rotation.
    internal operation GivensRotation(
        angle : Double,
        firstQubit : Qubit,
        secondQubit : Qubit,
    ) : Unit is Adj + Ctl {
        within {
            CNOT(secondQubit, firstQubit);
            S(secondQubit);
            H(secondQubit);
        } apply {
            Controlled Rz([firstQubit], (2.0 * angle, secondQubit));
        }
    }

    /// Implements controlled-Rz(2 angle) with Figure 20's two parallel Rz(angle)
    /// rotations. The operations commute and touch disjoint qubits.
    internal operation ControlledRzUsingParallelRotations(
        angle : Double,
        control : Qubit,
        target : Qubit,
    ) : Unit is Adj + Ctl {
        within {
            CNOT(target, control);
        } apply {
            within {
                X(control);
            } apply {
                Rz(angle, control);
            }
            Rz(angle, target);
        }
    }

    internal operation GivensRotationUsingParallelRotations(
        angle : Double,
        firstQubit : Qubit,
        secondQubit : Qubit,
    ) : Unit is Adj + Ctl {
        within {
            CNOT(secondQubit, firstQubit);
            S(secondQubit);
            H(secondQubit);
        } apply {
            ControlledRzUsingParallelRotations(angle, firstQubit, secondQubit);
        }
    }

    /// Phase-gradient implementation of the two parallel half-angle rotations.
    /// Each angle word must contain the same computational-basis value, and each
    /// rotation has a distinct gradient register so both adders can run concurrently.
    internal operation GivensRotationViaPhaseGradients(
        firstAngleWord : Qubit[],
        secondAngleWord : Qubit[],
        firstPhaseGradient : Qubit[],
        secondPhaseGradient : Qubit[],
        firstQubit : Qubit,
        secondQubit : Qubit,
    ) : Unit is Adj + Ctl {
        Fact(Length(firstAngleWord) == Length(secondAngleWord), "angle words must have the same width");
        Fact(Length(firstPhaseGradient) == Length(firstAngleWord), "first phase-gradient register has the wrong width");
        Fact(Length(secondPhaseGradient) == Length(secondAngleWord), "second phase-gradient register has the wrong width");

        within {
            CNOT(secondQubit, firstQubit);
            S(secondQubit);
            H(secondQubit);
        } apply {
            within {
                CNOT(secondQubit, firstQubit);
            } apply {
                within {
                    X(firstQubit);
                } apply {
                    RzViaPhaseGradient(firstQubit, firstAngleWord, firstPhaseGradient);
                }
                RzViaPhaseGradient(secondQubit, secondAngleWord, secondPhaseGradient);
            }
        }
    }

    /// Applies all `N - 1` tree rotations. Edges in each of the `ceil(log2 N)`
    /// layers are disjoint and may be scheduled concurrently.
    internal operation ApplyGivensTree(angles : Double[], target : Qubit[]) : Unit is Adj + Ctl {
        let edges = GivensTreeEdges(Length(target));
        Fact(Length(angles) == Length(edges), "one angle is required for each tree edge");
        for index in 0..Length(edges) - 1 {
            let (first, second) = edges[index];
            GivensRotation(angles[index], target[first], target[second]);
        }
    }

    internal operation ApplyGivensTreeUsingParallelRotations(
        angles : Double[],
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        let edges = GivensTreeEdges(Length(target));
        Fact(Length(angles) == Length(edges), "one angle is required for each tree edge");
        for index in 0..Length(edges) - 1 {
            let (first, second) = edges[index];
            GivensRotationUsingParallelRotations(angles[index], target[first], target[second]);
        }
    }

    /// Applies the multiplexed tree using independent resources for every edge.
    /// Grouping by `GivensTreeLayers` exposes logarithmic target-register depth;
    /// the QROM loads and physical factory throughput are separate architecture costs.
    internal operation ApplyGivensTreeViaPhaseGradients(
        firstAngleWords : Qubit[][],
        secondAngleWords : Qubit[][],
        firstPhaseGradients : Qubit[][],
        secondPhaseGradients : Qubit[][],
        target : Qubit[],
    ) : Unit is Adj + Ctl {
        let edges = GivensTreeEdges(Length(target));
        Fact(Length(firstAngleWords) == Length(edges), "one first angle word is required for each tree edge");
        Fact(Length(secondAngleWords) == Length(edges), "one second angle word is required for each tree edge");
        Fact(Length(firstPhaseGradients) == Length(edges), "one first phase-gradient state is required for each tree edge");
        Fact(Length(secondPhaseGradients) == Length(edges), "one second phase-gradient state is required for each tree edge");

        let layers = GivensTreeLayers(Length(target));
        for treeLayerIndex in 0..Length(layers) - 1 {
            let layer = layers[treeLayerIndex];
            let offset = GivensLayerOffset(Length(target), treeLayerIndex);
            for layerIndex in 0..Length(layer) - 1 {
                let edgeIndex = offset + layerIndex;
                let (first, second) = layer[layerIndex];
                GivensRotationViaPhaseGradients(
                    firstAngleWords[edgeIndex],
                    secondAngleWords[edgeIndex],
                    firstPhaseGradients[edgeIndex],
                    secondPhaseGradients[edgeIndex],
                    target[first],
                    target[second],
                );
            }
        }
    }

    internal operation PrepareGivensComparisonInput(inputKind : Int, qubits : Qubit[]) : Unit is Adj {
        if inputKind < 4 {
            ApplyXorInPlace(inputKind, qubits);
        } elif inputKind == 4 {
            H(qubits[0]);
            H(qubits[1]);
        } elif inputKind == 5 {
            H(qubits[0]);
            S(qubits[0]);
            H(qubits[1]);
        } else {
            fail "unknown Givens comparison input";
        }
    }

    internal operation TestParallelGivensDecomposition(angle : Double, inputKind : Int) : Bool {
        use qubits = Qubit[2];
        within {
            PrepareGivensComparisonInput(inputKind, qubits);
        } apply {
            GivensRotationUsingParallelRotations(angle, qubits[0], qubits[1]);
            Adjoint GivensRotation(angle, qubits[0], qubits[1]);
        }
        All(result -> result == Zero, MResetEachZ(qubits))
    }

    internal operation TestPhaseGradientGivensDecomposition(
        angleValue : Int,
        numBits : Int,
        inputKind : Int,
    ) : Bool {
        let angle = 4.0 * PI() * IntAsDouble(angleValue) / IntAsDouble(1 <<< numBits);
        use qubits = Qubit[2];
        within {
            PrepareGivensComparisonInput(inputKind, qubits);
        } apply {
            use firstAngleWord = Qubit[numBits];
            use secondAngleWord = Qubit[numBits];
            use firstPhaseGradient = Qubit[numBits];
            use secondPhaseGradient = Qubit[numBits];
            ApplyXorInPlace(angleValue, firstAngleWord);
            ApplyXorInPlace(angleValue, secondAngleWord);
            within {
                PreparePhaseGradientState(firstPhaseGradient);
                PreparePhaseGradientState(secondPhaseGradient);
            } apply {
                GivensRotationViaPhaseGradients(
                    firstAngleWord,
                    secondAngleWord,
                    firstPhaseGradient,
                    secondPhaseGradient,
                    qubits[0],
                    qubits[1],
                );
            }
            ApplyXorInPlace(angleValue, firstAngleWord);
            ApplyXorInPlace(angleValue, secondAngleWord);
            Adjoint GivensRotationUsingParallelRotations(angle, qubits[0], qubits[1]);
        }
        All(result -> result == Zero, MResetEachZ(qubits))
    }

    internal operation TestPhaseGradientGivensTree(angleValues : Int[], numBits : Int) : Bool {
        let numEdges = Length(angleValues);
        let angles = PhaseGradientAngles(angleValues, numBits);
        use target = Qubit[numEdges + 1];
        use firstAngleRegister = Qubit[numEdges * numBits];
        use secondAngleRegister = Qubit[numEdges * numBits];
        use firstGradientRegister = Qubit[numEdges * numBits];
        use secondGradientRegister = Qubit[numEdges * numBits];
        let firstAngleWords = Chunks(numBits, firstAngleRegister);
        let secondAngleWords = Chunks(numBits, secondAngleRegister);
        let firstPhaseGradients = Chunks(numBits, firstGradientRegister);
        let secondPhaseGradients = Chunks(numBits, secondGradientRegister);

        X(target[0]);
        for index in 0..numEdges - 1 {
            ApplyXorInPlace(angleValues[index], firstAngleWords[index]);
            ApplyXorInPlace(angleValues[index], secondAngleWords[index]);
        }
        within {
            for index in 0..numEdges - 1 {
                PreparePhaseGradientState(firstPhaseGradients[index]);
                PreparePhaseGradientState(secondPhaseGradients[index]);
            }
        } apply {
            ApplyGivensTreeViaPhaseGradients(
                firstAngleWords,
                secondAngleWords,
                firstPhaseGradients,
                secondPhaseGradients,
                target,
            );
            Adjoint ApplyGivensTreeUsingParallelRotations(angles, target);
        }
        for index in 0..numEdges - 1 {
            ApplyXorInPlace(angleValues[index], firstAngleWords[index]);
            ApplyXorInPlace(angleValues[index], secondAngleWords[index]);
        }
        X(target[0]);
        All(result -> result == Zero, MResetEachZ(target))
    }

    internal operation GivensTreeResourceEstimate(angles : Double[], useParallelRotations : Bool) : Unit {
        use target = Qubit[Length(angles) + 1];
        if useParallelRotations {
            ApplyGivensTreeUsingParallelRotations(angles, target);
        } else {
            ApplyGivensTree(angles, target);
        }
    }

}