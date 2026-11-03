// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.PauliExp {

    import Std.Arrays.Subarray;
    import Std.Math.Max;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// Non-identity qubit positions and axes for each term; an empty row is identity.
    struct RepPauliExpParams {
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        repetitions : Int,
        // Counts of one-time terms before and after the repeated body.
        numPrefixTerms : Int,
        numSuffixTerms : Int,
    }

    /// Applies one step, accessing only each term's non-identity support.
    operation PauliExp(
        pauliIndices : Int[][],
        pauliOps : Pauli[][],
        pauliCoefficients : Double[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if Length(pauliIndices) != Length(pauliCoefficients) or Length(pauliOps) != Length(pauliCoefficients) {
            fail "PauliExp: inconsistent array lengths.";
        }
        for term in 0..Length(pauliCoefficients) - 1 {
            // Exp uses the opposite sign; empty support retains the phase needed by controlled evolution.
            Exp(pauliOps[term], -pauliCoefficients[term], Subarray(pauliIndices[term], systems));
        }
    }

    /// Repeats a step symbolically during resource estimation, and explicitly during simulation.
    operation RepPauliExp(params : RepPauliExpParams, systems : Qubit[]) : Unit is Adj + Ctl {
        let suffixStart = Length(params.pauliCoefficients) - params.numSuffixTerms;
        let prefix = 0..params.numPrefixTerms - 1;
        let step = params.numPrefixTerms..suffixStart - 1;
        let suffix = suffixStart..Length(params.pauliCoefficients) - 1;
        PauliExp(params.pauliIndices[prefix], params.pauliOps[prefix], params.pauliCoefficients[prefix], systems);
        if IsResourceEstimating() {
            within {
                RepeatEstimates(params.repetitions);
            } apply {
                PauliExp(params.pauliIndices[step], params.pauliOps[step], params.pauliCoefficients[step], systems);
            }
        } else {
            for _ in 1..params.repetitions {
                PauliExp(params.pauliIndices[step], params.pauliOps[step], params.pauliCoefficients[step], systems);
            }
        }
        PauliExp(params.pauliIndices[suffix], params.pauliOps[suffix], params.pauliCoefficients[suffix], systems);
    }

    /// Creates a circuit for repeated sparse Pauli evolution.
    operation MakeRepPauliExpCircuit(params : RepPauliExpParams, system : Int[]) : Unit {
        if Length(system) == 0 {
            return ();
        }
        use qs = Qubit[Max(system) + 1];
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

    /// Returns a composable callable for repeated sparse evolution.
    function MakeRepPauliExpOp(params : RepPauliExpParams) : Qubit[] => Unit {
        ApplyRepPauliExp(params, _)
    }
}
