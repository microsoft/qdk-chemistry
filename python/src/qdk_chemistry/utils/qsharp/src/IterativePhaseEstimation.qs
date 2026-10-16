// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;

    /// A struct to hold parameters for iterative Quantum Phase Estimation (IQPE).
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The phase qubit index, distinct from every system index.
    /// - `systems`: Unique system indices, in the order expected by state preparation and the unitary.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    struct IterativePhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
    }

    /// Validates a register of `Length(systems) + 1 + numAncillaQubits` qubits
    /// and returns its unused indices in ascending order.
    function GetIQPEAncillaIndices(phaseQubit : Int, systems : Int[], numAncillaQubits : Int) : Int[] {
        if numAncillaQubits < 0 {
            fail "numAncillaQubits must be non-negative.";
        }
        let numQubits = Length(systems) + 1 + numAncillaQubits;
        if phaseQubit < 0 or phaseQubit >= numQubits {
            fail "phaseQubit must be within the allocated register.";
        }
        mutable occupied = [false, size = numQubits];
        set occupied w/= phaseQubit <- true;
        for index in systems {
            if index < 0 or index >= numQubits {
                fail "System qubit indices must be within the allocated register.";
            }
            if index == phaseQubit {
                fail "System qubit indices must be distinct from phaseQubit.";
            }
            if occupied[index] {
                fail "System qubit indices must be unique.";
            }
            set occupied w/= index <- true;
        }
        mutable ancillas = [];
        for index in 0..numQubits - 1 {
            if not occupied[index] {
                set ancillas += [index];
            }
        }
        return ancillas;
    }

    /// Runs the iterative Quantum Phase Estimation (IQPE) circuit based on the provided parameters.
    /// Ancillas are the unused register indices in ascending order, appended after the systems.
    /// # Parameters
    /// - `params`: An `IterativePhaseEstimationParams` struct containing the parameters for IQPE.
    /// # Returns
    /// - `Result[]`: The result of measuring the phase qubit after the IQPE circuit is executed.
    operation RunIQPE(params : IterativePhaseEstimationParams) : Result[] {
        let ancillaIndices = GetIQPEAncillaIndices(params.phaseQubit, params.systems, params.numAncillaQubits);
        use qs = Qubit[Length(params.systems) + 1 + params.numAncillaQubits];
        let phaseQubit = qs[params.phaseQubit];
        let systems = Subarray(params.systems, qs);
        let ancillas = Subarray(ancillaIndices, qs);
        let allTargets = systems + ancillas;

        params.statePrep(systems);

        within {
            H(phaseQubit);
        } apply {
            Rz(params.accumulatePhase, phaseQubit);
            params.repControlledUnitary(phaseQubit, allTargets);
        }
        let result = MResetZ(phaseQubit);
        ResetAll(allTargets);
        return [result];
    }

    /// Prepare iterative Quantum Phase Estimation (IQPE) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The phase qubit index, distinct from every system index.
    /// - `systems`: Unique system indices, in the order expected by state preparation and the unitary.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// All indices must be within a register of `Length(systems) + 1 + numAncillaQubits` qubits.
    /// Ancillas are the unused indices in ascending order, appended after the systems.
    /// # Returns
    /// The result of measuring the phase qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
    ) : Result[] {
        return RunIQPE(new IterativePhaseEstimationParams {
            statePrep = statePrep,
            repControlledUnitary = repControlledUnitary,
            accumulatePhase = accumulatePhase,
            phaseQubit = phaseQubit,
            systems = systems,
            numAncillaQubits = numAncillaQubits
        });
    }
}
