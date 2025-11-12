import Std.Math.ArcCos;
import Std.Math.PI;
import Std.Convert.IntAsDouble;
import Std.Arrays.Subarray;
import Std.StatePreparation.PreparePureStateD;

@EntryPoint(Adaptive_RIF)
operation Main() : Double {
    // Run with the initial quantum state |ψ⟩ = 0.8|00⟩ + 0.6|11⟩.
    // This state is close to the Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2, which is an
    // eigenstate of H = 0.5·XX + ZZ with eigenvalue E = 1.5. The high overlap (~0.99)
    // ensures the QPE primarily measures this eigenvalue and returns ~1.5 with high probability.
    IQPEMSB(new PhaseEstimationParams {
        numQubits = 2,
        numIterations = 3,
        rowMap = [0, 1],
        stateVector = [0.8, 0.0, 0.0, 0.6],
        expansionOps = [],
        pauliCoefficients = [0.5, 1.0],
        pauliExponents = [[PauliX, PauliX], [PauliZ, PauliZ]],
        evolutionTime = PI() / 2.0,
        strategy = "repeat"
    })
}

/// # Summary
/// Parameters for the Iterative Quantum Phase Estimation (IQPE) algorithm.
/// This structure encapsulates all necessary inputs for performing IQPE using the MSB-first approach.
///
/// # Fields
/// ## numQubits
/// The number of qubits in the system register representing the quantum state |ψ⟩.
/// ## numIterations
/// The number of iterations (bits) for the IQPE algorithm, determining the precision of the phase estimation.
/// ## rowMap
/// An array mapping the indices of the non-zero, non-duplicate elements in the sparse state vector.
/// ## stateVector
/// The sparse representation of the initial quantum state |ψ⟩ as a vector of doubles.
/// ## expansionOps
/// A list of operations (as arrays of qubit indices) to expand the initial state preparation into the non-sparse |ψ⟩.
    /// ## pauliCoefficients
    /// Real coefficients matching each Pauli term, defining the weighted Hamiltonian H = Σ c_j P_j.
    /// ## pauliExponents
    /// A list of Pauli operator arrays defining the Hamiltonian terms for the unitary evolution.
    /// Each array corresponds to a term in the Hamiltonian and shares an index with `pauliCoefficients`.
/// ## evolutionTime
/// The time parameter t for the unitary evolution U = exp(-iHt).
/// ## strategy
/// The strategy for implementing controlled time evolution:
/// - "repeat": Apply the controlled unitary multiple times for each bit.
/// - "rescaled": Apply a single controlled unitary with rescaled time for efficiency.
struct PhaseEstimationParams {
    numQubits : Int,
    numIterations : Int,
    rowMap : Int[],
    stateVector : Double[],
    expansionOps : Int[][],
    pauliCoefficients : Double[],
    pauliExponents : Pauli[][],
    evolutionTime : Double,
    strategy : String,
}

/// # Summary
/// Perform Iterative Quantum Phase Estimation (IQPE) using the MSB-first approach with Kitaev's phase accumulation method.
///
/// This implementation estimates the eigenphase of a unitary operator U = exp(-iHt) given an initial state |ψ⟩.
/// The method processes bits from the most significant to the least significant, accumulating phase information iteratively
/// without direct bit-string correspondence.
///
/// # Input
    /// ## params
    /// `PhaseEstimationParams` containing all necessary parameters for the IQPE algorithm, including Pauli coefficients.
///
/// # Output
/// A Double representing the estimated eigenvalue corresponding to the input state |ψ⟩.
operation IQPEMSB(params : PhaseEstimationParams) : Double {
    mutable accumulatedPhase = 0.0;

    // Perform IQPE iterations
    for k in params.numIterations.. -1..1 {
        // Allocate qubits
        use ancilla = Qubit();
        use system = Qubit[params.numQubits];

        // Prepare the initial sparse state
        PrepareSparseState(params.rowMap, params.stateVector, params.expansionOps, system);

        IQPEMSBIteration(
            params.pauliExponents,
            params.pauliCoefficients,
            params.evolutionTime,
            k,
            accumulatedPhase,
            params.strategy,
            ancilla,
            system
        );

        // Measure the ancilla qubit
        let result = MResetZ(ancilla);
        accumulatedPhase /= 2.0;
        if result == One {
            accumulatedPhase += PI() / 2.0;
        }

        // Reset system qubits
        ResetAll(system);
    }

    return (2.0 * PI() / params.evolutionTime) * (accumulatedPhase / PI());
}

operation PrepareSparseState(
    rowMap : Int[],
    stateVector : Double[],
    expansionOps : Int[][],
    qs : Qubit[]
) : Unit {
    PreparePureStateD(stateVector, Subarray(rowMap, qs));
    for op in expansionOps {
        if Length(op) == 2 {
            CNOT(qs[op[0]], qs[op[1]]);
        } elif Length(op) == 1 {
            X(qs[op[0]]);
        } else {
            fail "Unsupported operation length in expansionOps.";
        }
    }
}

/// Perform one iteration of MSB-first IQPE with Kitaev's phase accumulation method.
///
/// This implements the phase accumulation approach based on Kitaev's original iterative
/// QPE formulation (PRA 76, 030306, 2007). Unlike direct binary mapping, this method
/// accumulates phase information iteratively without direct bit-string correspondence.
///
/// Mathematical Framework:
///     - Process bits from MSB to LSB: k = n_bits → 1
///     - Phase accumulation: Φ(k) = Φ(k+1)/2 + π×j_k/2
///     - Feedback correction uses accumulated phase Φ(k+1)
///     - Final eigenphase: λ = 2×Φ(1), then φ = λ/(2π)
///
/// Circuit Steps:
///     1. Apply H to ancilla qubit
///     2. If k < n_bits: Apply Rz(Φ(k+1)) for phase kickback correction
///     3. Apply controlled-U^(2^(k-1)) time evolution
///     4. Apply final H for measurement basis
operation IQPEMSBIteration(
    pauliExponents : Pauli[][],
    pauliCoefficients : Double[],
    evolutionTime : Double,
    k : Int,
    accumulatedPhase : Double,
    strategy : String,
    ancilla : Qubit,
    system : Qubit[]
) : Unit {
    // Step 1: Hadamard basis for ancilla
    within {
        H(ancilla);
    } apply {

        // Step 2: Apply phase kickback if not the first iteration
        if accumulatedPhase > 0.0 or accumulatedPhase < 0.0 {
            Rz(accumulatedPhase, ancilla);
        }

        // Step 3: Apply controlled unitary evolution
        let repetitions = 2^(k - 1);
        Message($"Applying controlled evolution with {repetitions} repetitions using strategy '{strategy}'");
                if strategy == "repeat" {
                    for i in 1..repetitions {
                        ControlledEvolution(pauliExponents, pauliCoefficients, evolutionTime, ancilla, system);
                    }
                } elif strategy == "rescaled" {
                    ControlledEvolution(
                        pauliExponents,
                        pauliCoefficients,
                        evolutionTime * IntAsDouble(repetitions),
                        ancilla,
                        system
                    );
                } else {
                    fail "Invalid strategy. Use 'repeat' or 'rescaled'.";
                }
            }

    // Step 4: Final Hadamard on ancilla, automatically done by 'within ... apply' block
}

operation ControlledEvolution(
    pauliExponents : Pauli[][],
    pauliCoefficients : Double[],
    evolutionTime : Double,
    control : Qubit,
    system : Qubit[]
) : Unit {
    for idx in 0..Length(pauliExponents) - 1 {
        let paulis = pauliExponents[idx];
        let coeff = pauliCoefficients[idx];
        Controlled Exp([control], (paulis, -coeff * evolutionTime, system));
    }
}
