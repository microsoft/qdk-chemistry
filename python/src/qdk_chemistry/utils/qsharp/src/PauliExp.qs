// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.PauliExp {

    import Std.Arrays.Subarray;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Performs Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the time evolution on the allocated qubits.
    operation PauliExp(
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        for idx in 0..Length(pauliExponents) - 1 {
            let paulis = pauliExponents[idx];
            let coeff = pauliCoefficients[idx];
            Exp(paulis, -coeff, systems);
        }
    }


    /// Performs repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `pauliExponents`: An array of arrays of Pauli operators representing the Pauli terms.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `repetitions`: The number of times to repeat the evolution.
    struct RepPauliExpParams {
        pauliExponents : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
    }

    /// Performs repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated time evolution on the allocated qubits.
    operation RepPauliExp(
        params : RepPauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for i in 1..params.repetitions {
            PauliExp(params.pauliExponents, params.pauliCoefficients, systems);
        }
    }

    /// A helper operation to create a circuit for repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// - `system`: An array of integers representing the indices of the system qubits.
    /// # Returns
    /// - `Unit`: The operation prepares the repeated time evolution on the allocated qubits.
    operation MakeRepPauliExpCircuit(
        params : RepPauliExpParams,
        system : Int[],
    ) : Unit {
        // If no system indices are provided, there is nothing to do.
        if Length(system) == 0 {
            return ();
        }

        // Determine the maximum index in the system array to size the qubit register safely.
        mutable maxIndex = system[0];
        for idx in 1..Length(system) - 1 {
            let current = system[idx];
            if current > maxIndex {
                set maxIndex = current;
            }
        }

        // Allocate enough qubits so that all indices in `system` are valid.
        use qs = Qubit[maxIndex + 1];
        RepPauliExp(params, Subarray(system, qs));
    }

    /// Uncontrolled entry point for `RepPauliExp`.
    ///
    /// Declared without functors on purpose. Callables built by partially applying an
    /// `Adj + Ctl` operation cannot be resolved by the Q# defunctionalizer once they are
    /// stored in a plain `Qubit[] => Unit` slot (such as the arguments of
    /// `CircuitComposition.ApplySequential` or `MeasurementBasis.MakeMeasurementCircuit`),
    /// which makes lowering those compositions to QIR fail. Forwarding through a plain
    /// operation keeps the composed circuits statically resolvable.
    operation ApplyRepPauliExp(params : RepPauliExpParams, systems : Qubit[]) : Unit {
        RepPauliExp(params, systems);
    }

    /// A helper function to create a callable for repeated Time Evolution for a set of Pauli exponentials.
    /// # Parameters
    /// - `params`: A `RepPauliExpParams` struct containing the parameters for the operation.
    /// # Returns
    /// - `Qubit[] => Unit`: A callable that takes an array of system qubits, and prepares the repeated time evolution on the allocated qubits.
    function MakeRepPauliExpOp(params : RepPauliExpParams) : Qubit[] => Unit {
        ApplyRepPauliExp(params, _)
    }

    /// Sparse form of `RepPauliExpParams`.
    ///
    /// The dense `pauliExponents` carries one Pauli per system qubit for every term,
    /// which is O(terms * qubits). Jordan-Wigner terms are far from dense, so here each
    /// term instead lists only its non-identity positions: `pauliIndices[t]` indexes
    /// into `systems` and `pauliOps[t]` holds the matching axis. A term with no entries
    /// is the identity term.
    ///
    struct SparseRepPauliExpParams {
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
    }

    /// Performs Time Evolution for a sparsely encoded set of Pauli exponentials.
    /// # Parameters
    /// - `pauliIndices`: For each term, the positions in `systems` carrying a non-identity Pauli.
    /// - `pauliOps`: For each term, the Pauli axis at each position in `pauliIndices`.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    operation SparsePauliExp(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        CheckSparseLengths(pauliIndices, pauliOps, pauliCoefficients);
        for idx in 0..Length(pauliCoefficients) - 1 {
            // `Exp` takes the opposite sign to the container's exp(-i theta P) convention.
            Exp(pauliOps[idx], -pauliCoefficients[idx], Subarray(pauliIndices[idx], systems));
        }
    }

    /// Rejects a sparse term list whose parallel arrays disagree in length.
    function CheckSparseLengths(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[]
    ) : Unit {
        if Length(pauliIndices) != Length(pauliCoefficients) or Length(pauliOps) != Length(pauliCoefficients) {
            fail "SparsePauliExp: pauliIndices, pauliOps, and pauliCoefficients must have the same length.";
        }
    }

    /// Performs repeated Time Evolution for a sparsely encoded set of Pauli exponentials.
    operation SparseRepPauliExp(
        params : SparseRepPauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {

        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                SparsePauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    systems
                );
            }
        } else {
            for _ in 1..params.repetitions {
                SparsePauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    systems
                );
            }
        }
    }

    /// Resolves sparse term positions to their system qubits.
    internal function TermTargets(pauliIndices : Int[][], systems : Qubit[]) : Qubit[][] {
        mutable targets : Qubit[][] = [];
        for indices in pauliIndices {
            set targets += [Subarray(indices, systems)];
        }
        return targets;
    }

    /// Returns the qubit carrying each mapped term's phase.
    internal function HammingWeightRepresentatives(targets : Qubit[][]) : Qubit[] {
        mutable representatives : Qubit[] = [];
        for term in targets {
            set representatives += [term[Length(term) - 1]];
        }
        return representatives;
    }

    /// Rotates one nonempty Pauli string onto a single Z representative.
    internal operation MapPauliTermToSingleZ(ops : Pauli[], targets : Qubit[]) : Unit is Adj + Ctl {
        Fact(Length(ops) == Length(targets), "MapPauliTermToSingleZ needs one Pauli axis per target.");
        Fact(Length(ops) > 0, "MapPauliTermToSingleZ needs a non-empty Pauli string.");
        for i in 0..Length(ops) - 1 {
            if ops[i] == PauliX {
                H(targets[i]);
            } elif ops[i] == PauliY {
                Adjoint S(targets[i]);
                H(targets[i]);
            } else {
                Fact(ops[i] == PauliZ, "MapPauliTermToSingleZ does not accept PauliI.");
            }
        }
        let last = Length(targets) - 1;
        for i in 0..last - 1 {
            CNOT(targets[i], targets[last]);
        }
    }

    /// Plans an optimal adder tree for a Hamming-weight computation.
    internal function HammingWeightSchedule(count : Int) : ((Int, Int, Int, Int)[], Int[], Int) {
        if count <= 0 {
            return ([], [], 0);
        }
        let levels = BitSizeI(count);
        mutable initial : Int[] = [];
        for i in 0..count - 1 {
            set initial += [i];
        }
        mutable buckets : Int[][] = [[], size = levels + 1];
        set buckets w/= 0 <- initial;
        mutable schedule : (Int, Int, Int, Int)[] = [];
        mutable next = count;

        for k in 0..levels - 1 {
            mutable current = buckets[k];
            mutable upper = buckets[k + 1];
            while Length(current) >= 3 {
                set schedule += [(current[0], current[1], current[2], next)];
                set current = current[3...] + [current[2]];
                set upper += [next];
                set next += 1;
            }
            if Length(current) == 2 {
                set schedule += [(current[0], current[1], -1, next)];
                set current = [current[1]];
                set upper += [next];
                set next += 1;
            }
            set buckets w/= k <- current;
            set buckets w/= k + 1 <- upper;
        }

        mutable finalBits : Int[] = [];
        for k in 0..levels - 1 {
            set finalBits += [Length(buckets[k]) == 1 ? buckets[k][0] | -1];
        }
        return (schedule, finalBits, next);
    }

    /// Compresses three equal-significance bits into a sum and carry.
    internal operation FullAdderStep(a : Qubit, b : Qubit, c : Qubit, carry : Qubit) : Unit is Adj {
        CNOT(a, b);
        CNOT(a, c);
        AND(b, c, carry);
        CNOT(a, carry);
        CNOT(b, c);
        CNOT(a, c);
    }

    /// Compresses two equal-significance bits into a sum and carry.
    internal operation HalfAdderStep(a : Qubit, b : Qubit, carry : Qubit) : Unit is Adj {
        AND(a, b, carry);
        CNOT(a, b);
    }

    /// Computes a Hamming weight into a little-endian output register.
    internal operation ComputeHammingWeight(
        inputs : Qubit[],
        scratch : Qubit[],
        weight : Qubit[]
    ) : Unit is Adj {
        let (schedule, finalBits, _) = HammingWeightSchedule(Length(inputs));
        let work = inputs + scratch;
        for (a, b, c, carry) in schedule {
            if c < 0 {
                HalfAdderStep(work[a], work[b], work[carry]);
            } else {
                FullAdderStep(work[a], work[b], work[c], work[carry]);
            }
        }
        for k in 0..Length(finalBits) - 1 {
            if finalBits[k] >= 0 {
                CNOT(work[finalBits[k]], weight[k]);
            }
        }
    }

    /// Applies equal-angle Z phases using logarithmically many rotations.
    internal operation HammingWeightPhase(theta : Double, inputs : Qubit[]) : Unit is Adj + Ctl {
        let count = Length(inputs);
        let bits = BitSizeI(count);
        let (schedule, _, _) = HammingWeightSchedule(count);
        use scratch = Qubit[Length(schedule)];
        use weight = Qubit[bits];
        within {
            ComputeHammingWeight(inputs, scratch, weight);
        } apply {
            for k in 0..bits - 1 {
                let scale = IntAsDouble(1 <<< k);
                Rz(2.0 * theta * scale, weight[k]);
                R(PauliI, -2.0 * theta * scale, weight[k]);
            }
            R(PauliI, 2.0 * theta * IntAsDouble(count), inputs[0]);
        }
    }

    /// Applies one equal-angle batch, selecting HWP only at its measured break-even size.
    internal operation HammingWeightPhaseTerms(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][]
    ) : Unit is Adj + Ctl {
        Fact(Length(pauliOps) == Length(targets), "HammingWeightPhaseTerms needs one axis list per term.");
        // For a controlled batch of m terms, HWP costs 3 ceil(log2(m + 1)) + 1
        // rotations and m - w(m) AND operations, versus 2m rotations term by term.
        if Length(targets) < 8 {
            for t in 0..Length(targets) - 1 {
                Exp(pauliOps[t], -theta, targets[t]);
            }
        } else {
            within {
                for t in 0..Length(targets) - 1 {
                    MapPauliTermToSingleZ(pauliOps[t], targets[t]);
                }
            } apply {
                HammingWeightPhase(theta, HammingWeightRepresentatives(targets));
            }
        }
    }

    /// One equal-angle group in a structured sparse product formula.
    ///
    /// A singleton is one Pauli exponential. Multiple entries are a batch whose
    /// lowering policy is owned by `HammingWeightPhaseTerms`.
    struct SparsePauliExpGroupParams {
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        angle : Double,
    }

    /// Applies one plain or batched equal-angle group.
    operation SparsePauliExpGroup(
        params : SparsePauliExpGroupParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        if Length(params.pauliIndices) != Length(params.pauliOps) {
            fail "SparsePauliExpGroup: pauliIndices and pauliOps must have the same length.";
        }
        if Length(params.pauliIndices) == 0 {
            fail "SparsePauliExpGroup: a group must contain at least one Pauli term.";
        }
        if Length(params.pauliIndices) == 1 {
            Exp(params.pauliOps[0], -params.angle, Subarray(params.pauliIndices[0], systems));
        } else {
            HammingWeightPhaseTerms(
                params.angle,
                params.pauliOps,
                TermTargets(params.pauliIndices, systems)
            );
        }
    }

    /// Applies a sequence of plain or batched groups.
    operation SparsePauliExpGroups(
        groups : SparsePauliExpGroupParams[],
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for group in groups {
            SparsePauliExpGroup(group, systems);
        }
    }

    /// One direct group or a structured `within { V } apply { D }` block.
    struct ConjugatedSparsePauliExpParams {
        withinGroups : SparsePauliExpGroupParams[],
        applyGroups : SparsePauliExpGroupParams[],
    }

    /// Applies a direct block or a conjugation whose `within` block stays bare under control.
    operation ConjugatedSparsePauliExp(
        params : ConjugatedSparsePauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        if Length(params.withinGroups) == 0 {
            SparsePauliExpGroups(params.applyGroups, systems);
        } else {
            within {
                SparsePauliExpGroups(params.withinGroups, systems);
            } apply {
                SparsePauliExpGroups(params.applyGroups, systems);
            }
        }
    }

    /// Applies one structured product-formula step.
    operation StructuredSparsePauliExpStep(
        blocks : ConjugatedSparsePauliExpParams[],
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        for block in blocks {
            ConjugatedSparsePauliExp(block, systems);
        }
    }

    /// A repeated structured product formula, optionally conjugated once as a whole.
    struct StructuredSparseRepPauliExpParams {
        conjugatingGroups : SparsePauliExpGroupParams[],
        stepBlocks : ConjugatedSparsePauliExpParams[],
        repetitions : Int,
    }

    /// Applies the repeated structured step, using estimator-native repetition when available.
    operation RepeatedStructuredSparsePauliExp(
        params : StructuredSparseRepPauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                StructuredSparsePauliExpStep(params.stepBlocks, systems);
            }
        } else {
            for _ in 1..params.repetitions {
                StructuredSparsePauliExpStep(params.stepBlocks, systems);
            }
        }
    }

    /// Applies `V step^r V^dagger`; Q# controls only `step^r` automatically.
    operation StructuredSparseRepPauliExp(
        params : StructuredSparseRepPauliExpParams,
        systems : Qubit[],
    ) : Unit is Adj + Ctl {
        if Length(params.conjugatingGroups) == 0 {
            RepeatedStructuredSparsePauliExp(params, systems);
        } else {
            within {
                SparsePauliExpGroups(params.conjugatingGroups, systems);
            } apply {
                RepeatedStructuredSparsePauliExp(params, systems);
            }
        }
    }

    /// Allocates a register and applies structured sparse Pauli evolution.
    operation MakeStructuredSparseRepPauliExpCircuit(
        evoParams : StructuredSparseRepPauliExpParams,
        system : Int[],
    ) : Unit {
        if Length(system) == 0 {
            return ();
        }

        mutable maxIndex = system[0];
        for idx in 1..Length(system) - 1 {
            if system[idx] > maxIndex {
                set maxIndex = system[idx];
            }
        }

        use qs = Qubit[maxIndex + 1];
        StructuredSparseRepPauliExp(evoParams, Subarray(system, qs));
    }

    /// Uncontrolled entry point for structured sparse Pauli evolution.
    operation ApplyStructuredSparseRepPauliExp(
        params : StructuredSparseRepPauliExpParams,
        systems : Qubit[],
    ) : Unit {
        StructuredSparseRepPauliExp(params, systems);
    }

    /// Returns an uncontrolled callable for structured sparse Pauli evolution.
    function MakeStructuredSparseRepPauliExpOp(
        params : StructuredSparseRepPauliExpParams
    ) : Qubit[] => Unit {
        ApplyStructuredSparseRepPauliExp(params, _)
    }

    /// A helper operation to create a circuit for repeated sparse Time Evolution.
    operation MakeSparseRepPauliExpCircuit(
        params : SparseRepPauliExpParams,
        system : Int[],
    ) : Unit {
        // If no system indices are provided, there is nothing to do.
        if Length(system) == 0 {
            return ();
        }

        // Determine the maximum index in the system array to size the qubit register safely.
        mutable maxIndex = system[0];
        for idx in 1..Length(system) - 1 {
            let current = system[idx];
            if current > maxIndex {
                set maxIndex = current;
            }
        }

        // Allocate enough qubits so that all indices in `system` are valid.
        use qs = Qubit[maxIndex + 1];
        SparseRepPauliExp(params, Subarray(system, qs));
    }

    /// Uncontrolled entry point for `SparseRepPauliExp`; see `ApplyRepPauliExp` for why
    /// this forwarding operation carries no functors.
    operation ApplySparseRepPauliExp(params : SparseRepPauliExpParams, systems : Qubit[]) : Unit {
        SparseRepPauliExp(params, systems);
    }

    /// A helper function to create a callable for repeated sparse Time Evolution.
    function MakeSparseRepPauliExpOp(params : SparseRepPauliExpParams) : Qubit[] => Unit {
        ApplySparseRepPauliExp(params, _)
    }

}
