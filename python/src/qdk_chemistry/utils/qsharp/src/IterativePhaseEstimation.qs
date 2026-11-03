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
    /// - `prepareSharedOp`: Prepares the shared register around the state preparation and the
    ///   controlled unitary.
    /// - `numSharedAncillas`: Size of the shared register, allocated after the
    ///   `Length(systems) + 1 + numAncillaQubits` indexed qubits.
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
        // Fully qualified: importing `Fact` here makes Base-profile partial evaluation panic.
        Std.Diagnostics.Fact(
            params.numSharedAncillas > 0 or not (params.statePrepUsesShared or params.unitaryUsesShared),
            "consuming shared ancillas requires a non-empty shared register"
        );
        let ancillaIndices = GetIQPEAncillaIndices(params.phaseQubit, params.systems, params.numAncillaQubits);
        let numIndexedQubits = Length(params.systems) + 1 + params.numAncillaQubits;
        use qs = Qubit[numIndexedQubits + params.numSharedAncillas];
        let phaseQubit = qs[params.phaseQubit];
        let systems = Subarray(params.systems, qs);
        let ancillas = Subarray(ancillaIndices, qs);
        let shared = qs[numIndexedQubits...];
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
    /// - `phaseQubit`: The phase qubit index, distinct from every system index.
    /// - `systems`: Unique system indices, in the order expected by state preparation and the unitary.
    /// - `numAncillaQubits`: Number of ancilla qubits needed by the controlled unitary (0 if none).
    /// - `prepareSharedOp`: Prepares the shared register around the state preparation and the
    ///   controlled unitary.
    /// - `numSharedAncillas`: Size of the shared register, allocated after the
    ///   `Length(systems) + 1 + numAncillaQubits` indexed qubits.
    /// - `statePrepUsesShared`: Append the shared register to the system qubits `statePrep` acts on.
    /// - `unitaryUsesShared`: Append the shared register to the controlled unitary's targets.
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
