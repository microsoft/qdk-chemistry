// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.PauliExp {

    import Std.Arrays.Subarray;
    import QDKChemistry.Utils.HammingWeightPhasing.BatchSegments;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhaseTerms;
    import QDKChemistry.Utils.HammingWeightPhasing.TermTargets;
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

    /// Returns an `Adj + Ctl` callable for repeated dense Time Evolution.
    internal function MakeRepPauliExpAdjCtlOp(params : RepPauliExpParams) : (Qubit[] => Unit is Adj + Ctl) {
        RepPauliExp(params, _)
    }

    /// Sparse form of `RepPauliExpParams`.
    ///
    /// The dense `pauliExponents` carries one Pauli per system qubit for every term,
    /// which is O(terms * qubits). Jordan-Wigner terms are far from dense, so here each
    /// term instead lists only its non-identity positions: `pauliIndices[t]` indexes
    /// into `systems` and `pauliOps[t]` holds the matching axis. A term with no entries
    /// is the identity term.
    ///
    /// `needsControl[t]` marks whether term `t` must be controlled when the whole
    /// evolution is. A conjugating factor whose partner also appears in the sequence
    /// cancels against that partner when the control is off, so controlling it changes
    /// nothing and only costs gates. An empty array controls every term, which is the
    /// safe default and what a caller that does not reason about exemptions should pass.
    ///
    /// `batchIds[t]` groups consecutive terms that share a rotation angle and act on
    /// disjoint qubits. Such a group is phased together through a Hamming weight
    /// register, so its `m` rotations collapse to `O(log m)` at the cost of `m - w(m)`
    /// Toffolis. `0`, or an empty array, applies the term on its own.
    struct SparseRepPauliExpParams {
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        needsControl : Bool[],
        batchIds : Int[],
        repetitions : Int,
    }

    /// Performs Time Evolution for a sparsely encoded set of Pauli exponentials.
    /// # Parameters
    /// - `pauliIndices`: For each term, the positions in `systems` carrying a non-identity Pauli.
    /// - `pauliOps`: For each term, the Pauli axis at each position in `pauliIndices`.
    /// - `pauliCoefficients`: An array of doubles representing the coefficients for each Pauli term.
    /// - `needsControl`: For each term, whether it must be controlled; empty controls all.
    /// - `systems`: An array of qubits representing the system on which the operation acts.
    operation SparsePauliExp(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        needsControl : Bool[],
        batchIds : Int[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        body ... {
            CheckSparseLengths(pauliIndices, pauliOps, pauliCoefficients, needsControl);
            for (start, count) in SparseSegments(batchIds, Length(pauliCoefficients)) {
                if count == 1 {
                    // `Exp` takes the opposite sign to the container's exp(-i theta P) convention.
                    Exp(pauliOps[start], -pauliCoefficients[start], Subarray(pauliIndices[start], systems));
                } else {
                    // `HammingWeightPhaseTerms` already applies exp(-i theta P), so it needs
                    // no sign flip. Bound to annotated locals so the nested array types infer.
                    let batchOps : Pauli[][] = pauliOps[start..start + count - 1];
                    let batchTargets : Qubit[][] = TermTargets(pauliIndices[start..start + count - 1], systems);
                    HammingWeightPhaseTerms(pauliCoefficients[start], batchOps, batchTargets);
                }
            }
        }
        controlled (ctls, ...) {
            CheckSparseLengths(pauliIndices, pauliOps, pauliCoefficients, needsControl);
            let exemptKnown = Length(needsControl) != 0;
            for (start, count) in SparseSegments(batchIds, Length(pauliCoefficients)) {
                if count == 1 {
                    let targets = Subarray(pauliIndices[start], systems);
                    if exemptKnown and not needsControl[start] {
                        // A conjugating factor: its partner is also in the sequence, so with
                        // the control off the pair cancels and running it bare is exact.
                        Exp(pauliOps[start], -pauliCoefficients[start], targets);
                    } else {
                        Controlled Exp(ctls, (pauliOps[start], -pauliCoefficients[start], targets));
                    }
                } else {
                    let batchOps : Pauli[][] = pauliOps[start..start + count - 1];
                    let batchTargets : Qubit[][] = TermTargets(pauliIndices[start..start + count - 1], systems);
                    Controlled HammingWeightPhaseTerms(ctls, (pauliCoefficients[start], batchOps, batchTargets));
                }
            }
        }
    }

    /// Returns the `(start, count)` blocks a sparse term list is applied in.
    ///
    /// Defers to `BatchSegments` when batch identifiers are supplied, and otherwise
    /// gives every term its own block.
    function SparseSegments(batchIds : Int[], termCount : Int) : (Int, Int)[] {
        if Length(batchIds) == 0 {
            mutable singles : (Int, Int)[] = [];
            for idx in 0..termCount - 1 {
                set singles += [(idx, 1)];
            }
            return singles;
        }
        if Length(batchIds) != termCount {
            fail "SparsePauliExp: batchIds must be empty or as long as pauliCoefficients.";
        }
        return BatchSegments(batchIds);
    }

    /// Rejects a sparse term list whose parallel arrays disagree in length.
    function CheckSparseLengths(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        needsControl : Bool[]
    ) : Unit {
        if Length(pauliIndices) != Length(pauliCoefficients) or Length(pauliOps) != Length(pauliCoefficients) {
            fail "SparsePauliExp: pauliIndices, pauliOps, and pauliCoefficients must have the same length.";
        }
        if Length(needsControl) != 0 and Length(needsControl) != Length(pauliCoefficients) {
            fail "SparsePauliExp: needsControl must be empty or as long as pauliCoefficients.";
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
                    params.needsControl,
                    params.batchIds,
                    systems
                );
            }
        } else {
            for _ in 1..params.repetitions {
                SparsePauliExp(
                    params.pauliIndices,
                    params.pauliOps,
                    params.pauliCoefficients,
                    params.needsControl,
                    params.batchIds,
                    systems
                );
            }
        }
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

    /// Returns an `Adj + Ctl` callable for repeated sparse Time Evolution.
    internal function MakeSparseRepPauliExpAdjCtlOp(
        params : SparseRepPauliExpParams
    ) : (Qubit[] => Unit is Adj + Ctl) {
        SparseRepPauliExp(params, _)
    }
}
