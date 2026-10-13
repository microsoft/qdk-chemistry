// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.IterativePhaseEstimation {

    import Std.Arrays.Subarray;

    /// A struct to hold parameters for iterative Quantum Phase Estimation (IQPE).
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// - `prepareSharedOp`: Prepares the shared register around the state preparation and the
    ///   controlled unitary.
    /// - `numSharedAncillas`: Size of the shared register, placed after the unitary's ancillas.
    /// - `statePrepUsesShared`: Append the shared register to the system qubits `statePrep` acts on.
    /// - `unitaryUsesShared`: Append the shared register to the controlled unitary's targets.
    ///   Each consumer leaves the register prepared.
    struct IterativePhaseEstimationParams {
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
        prepareSharedOp : Qubit[] => Unit is Adj + Ctl,
        numSharedAncillas : Int,
        statePrepUsesShared : Bool,
        unitaryUsesShared : Bool,
    }

    /// Runs the iterative Quantum Phase Estimation (IQPE) circuit based on the provided parameters.
    /// # Parameters
    /// - `params`: An `IterativePhaseEstimationParams` struct containing the parameters for IQPE.
    /// # Returns
    /// - `Result[]`: The result of measuring the phase qubit after the IQPE circuit is executed.
    operation RunIQPE(params : IterativePhaseEstimationParams) : Result[] {
        // Fully qualified: importing `Fact` here makes Base-profile partial evaluation panic.
        Std.Diagnostics.Fact(
            params.numSharedAncillas > 0 or not (params.statePrepUsesShared or params.unitaryUsesShared),
            "consuming shared ancillas requires a non-empty shared register"
        );
        use qs = Qubit[Length(params.systems) + 1 + params.numAncillaQubits + params.numSharedAncillas];
        let phaseQubit = qs[params.phaseQubit];
        let systems = Subarray(params.systems, qs);
        let ancillaStart = 1 + Length(params.systems);
        let ancillas = qs[ancillaStart..ancillaStart + params.numAncillaQubits - 1];
        let shared = qs[ancillaStart + params.numAncillaQubits...];
        let unitaryTargets = systems + ancillas + (params.unitaryUsesShared ? shared | []);

        within {
            params.prepareSharedOp(shared);
        } apply {
            params.statePrep(params.statePrepUsesShared ? systems + shared | systems);
            within {
                H(phaseQubit);
            } apply {
                Rz(params.accumulatePhase, phaseQubit);
                params.repControlledUnitary(phaseQubit, unitaryTargets);
            }
        }
        let result = MResetZ(phaseQubit);
        ResetAll(systems + ancillas + shared);
        return [result];
    }

    /// Prepare iterative Quantum Phase Estimation (IQPE) circuit.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state.
    /// - `repControlledUnitary`: A function to perform repeated controlled unitary operations.
    /// - `accumulatePhase`: The phase to accumulate during the evolution.
    /// - `phaseQubit`: The index of the phase qubit (ancilla used for phase readout).
    /// - `systems`: An array of indices representing the system qubits.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// - `prepareSharedOp`: Prepares the shared register around the state preparation and the
    ///   controlled unitary.
    /// - `numSharedAncillas`: Size of the shared register, placed after the unitary's ancillas.
    /// - `statePrepUsesShared`: Append the shared register to the system qubits `statePrep` acts on.
    /// - `unitaryUsesShared`: Append the shared register to the controlled unitary's targets.
    /// # Returns
    /// The result of measuring the phase qubit after the IQPE circuit is executed.
    operation MakeIQPECircuit(
        statePrep : Qubit[] => Unit,
        repControlledUnitary : (Qubit, Qubit[]) => Unit,
        accumulatePhase : Double,
        phaseQubit : Int,
        systems : Int[],
        numAncillaQubits : Int,
        prepareSharedOp : Qubit[] => Unit is Adj + Ctl,
        numSharedAncillas : Int,
        statePrepUsesShared : Bool,
        unitaryUsesShared : Bool,
    ) : Result[] {
        return RunIQPE(new IterativePhaseEstimationParams {
            statePrep = statePrep,
            repControlledUnitary = repControlledUnitary,
            accumulatePhase = accumulatePhase,
            phaseQubit = phaseQubit,
            systems = systems,
            numAncillaQubits = numAncillaQubits,
            prepareSharedOp = prepareSharedOp,
            numSharedAncillas = numSharedAncillas,
            statePrepUsesShared = statePrepUsesShared,
            unitaryUsesShared = unitaryUsesShared
        });
    }
}
