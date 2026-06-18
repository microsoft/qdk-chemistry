// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Alias sampling state preparation for block encoding PREPARE oracles.
///
/// Implements the Walker alias method for quantum state preparation:
///   |0⟩ → Σ_ℓ √(p_ℓ) |ℓ⟩|garbage⟩
///
/// Uses O(log N) qubits and O(N) classical preprocessing.
/// The quantum circuit uses:
///   1. PrepareUniformSuperposition over N terms
///   2. Hadamard on μ comparison qubits
///   3. QROM to load (keep_ℓ, alt_ℓ) alias table
///   4. Comparison: flag = (σ ≥ keep_ℓ)
///   5. Conditional swap: if flag, index ↔ alt_ℓ
///
/// Reference: Babbush et al. arXiv:1805.03662, Fig. 11.
namespace QDKChemistry.Utils.AliasSampling {

    import Std.Arithmetic.ApplyIfGreaterLE;
    import Std.Arrays.MappedOverRange;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.AbsD;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.StatePreparation.PrepareUniformSuperposition;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;

    /// Parameters for alias sampling state preparation.
    struct AliasSamplingParams {
        /// Unnormalized probability weights (positive real values), length L.
        coefficients : Double[],
        /// Number of bits μ for keep-coefficient precision.
        bitsPrecision : Int,
        /// Number of qubits for the index register: ⌈log₂ L⌉.
        numIndexQubits : Int,
        /// Total number of qubits to allocate (index + uniform + flag + qromOutput).
        numQubits : Int,
    }

    /// Compute the discretized probability distribution for alias sampling.
    ///
    /// Returns (keepCoefficients, alternateIndices) where:
    ///   keepCoefficients[i] = discretized probability of keeping index i
    ///   alternateIndices[i] = alias index to swap to if comparison fails
    ///
    /// Uses Walker's alias method (Vose's O(n) implementation).
    function DiscretizedProbabilityDistribution(
        bitsPrecision : Int,
        coefficients : Double[],
    ) : (Int[], Int[]) {
        let nCoeffs = Length(coefficients);
        let barHeight = (1 <<< bitsPrecision) - 1;

        // Normalize probabilities
        mutable total = 0.0;
        for i in 0..nCoeffs - 1 {
            set total += AbsD(coefficients[i]);
        }

        // Scale to bar height × nCoeffs
        mutable scaledProbs : Int[] = [];
        mutable remainder = 0;
        for i in 0..nCoeffs - 1 {
            let scaled = Ceiling(AbsD(coefficients[i]) / total * IntAsDouble(barHeight * nCoeffs));
            set scaledProbs += [scaled];
            set remainder += scaled;
        }

        // Initialize keep and alt arrays
        mutable keepCoeff : Int[] = [];
        mutable altIndex : Int[] = [];
        for i in 0..nCoeffs - 1 {
            set keepCoeff += [barHeight];
            set altIndex += [i];
        }

        // Partition into small and large
        mutable small : Int[] = [];
        mutable large : Int[] = [];
        for i in 0..nCoeffs - 1 {
            if scaledProbs[i] < barHeight {
                set small += [i];
            } else {
                set large += [i];
            }
        }

        // Walker alias construction
        mutable si = 0;
        mutable li = 0;
        while si < Length(small) and li < Length(large) {
            let s = small[si];
            let l = large[li];
            set keepCoeff w/= s <- scaledProbs[s];
            set altIndex w/= s <- l;
            set scaledProbs w/= l <- scaledProbs[l] - (barHeight - scaledProbs[s]);
            set si += 1;
            if scaledProbs[l] < barHeight {
                set small += [l];
            } else {
                set large += [l];
            }
            set li += 1;
        }

        return (keepCoeff, altIndex);
    }

    /// Alias sampling PREPARE operation.
    ///
    /// Prepares: |0⟩ → Σ_ℓ √(p̃_ℓ) |ℓ⟩|garbage⟩
    ///
    /// Register layout:
    ///   indexRegister[numIndexQubits] — output: sampled index ℓ
    ///   uniformRegister[bitsPrecision] — ancilla for comparison σ
    ///   flagQubit[1] — ancilla for alias resolution
    ///   qromOutput[bitsPrecision + numIndexQubits] — QROM target (keep_ℓ, alt_ℓ)
    operation AliasSamplingPrepare(
        params : AliasSamplingParams,
        qs : Qubit[],
    ) : Unit is Adj + Ctl {
        let nIndexQubits = params.numIndexQubits;
        let mu = params.bitsPrecision;
        let nCoeffs = Length(params.coefficients);

        let indexRegister = qs[0..nIndexQubits - 1];
        let uniformRegister = qs[nIndexQubits..nIndexQubits + mu - 1];
        let flagQubit = qs[nIndexQubits + mu];
        let qromOutput = qs[nIndexQubits + mu + 1..2 * nIndexQubits + 2 * mu];

        let (keepCoeff, altIndex) = DiscretizedProbabilityDistribution(mu, params.coefficients);

        let nPadded = 1 <<< nIndexQubits;
        let barHeight = (1 <<< mu) - 1;

        // Build QROM data table: (keep_ℓ, alt_ℓ) for each index
        let selectData = MappedOverRange(
            idx -> if idx < nCoeffs {
                IntAsBoolArray(keepCoeff[idx], mu)
                    + IntAsBoolArray(altIndex[idx], nIndexQubits)
            } else {
                IntAsBoolArray(barHeight, mu)
                    + IntAsBoolArray(idx, nIndexQubits)
            },
            0..nPadded - 1
        );

        // Step 1: Uniform superposition over L terms
        PrepareUniformSuperposition(nCoeffs, indexRegister);

        // Step 2: H⊗μ on comparison register
        ApplyToEachCA(H, uniformRegister);

        // Step 3: QROM load via SELECT-SWAP network
        SelectSwap(-1, selectData, indexRegister, qromOutput);

        // Step 4: Compare σ ≥ keep_ℓ → set flag
        let keepLoaded = qromOutput[0..mu - 1];
        let altLoaded = qromOutput[mu..mu + nIndexQubits - 1];
        ApplyIfGreaterLE(X, uniformRegister, keepLoaded, flagQubit);

        // Step 5: Conditional swap index ↔ alt
        for i in 0..nIndexQubits - 1 {
            Controlled SWAP([flagQubit], (indexRegister[i], altLoaded[i]));
        }
    }

    /// Create an alias sampling state preparation callable.
    function MakeAliasSamplingOp(params : AliasSamplingParams) : Qubit[] => Unit is Adj + Ctl {
        AliasSamplingPrepare(params, _)
    }

    /// Circuit entry point for alias sampling (allocates qubits).
    operation MakeAliasSamplingCircuit(
        coefficients : Double[],
        bitsPrecision : Int,
        numIndexQubits : Int,
        numQubits : Int,
    ) : Unit {
        let params = new AliasSamplingParams {
            coefficients = coefficients,
            bitsPrecision = bitsPrecision,
            numIndexQubits = numIndexQubits,
            numQubits = numQubits,
        };
        use qs = Qubit[numQubits];
        AliasSamplingPrepare(params, qs);
    }

    /// Helper to compute the total number of qubits needed for alias sampling.
    /// Returns: numIndexQubits + bitsPrecision + 1 (flag) + bitsPrecision + numIndexQubits (qrom output)
    function ComputeAliasSamplingQubits(numCoefficients : Int, bitsPrecision : Int) : (Int, Int) {
        let numIndexQubits = Ceiling(Lg(IntAsDouble(numCoefficients)));
        let numQubits = 2 * numIndexQubits + 2 * bitsPrecision + 1;
        return (numIndexQubits, numQubits);
    }

}
