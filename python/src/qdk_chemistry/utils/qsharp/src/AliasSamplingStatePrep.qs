// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Alias sampling state preparation.
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
    import Std.Arrays.Mapped;
    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalLambda2D;
    import QDKChemistry.Utils.SelectSwap.Select2DLoad;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;

    /// Parameters for alias sampling state preparation.
    struct AliasSamplingParams {
        /// Unnormalized probability weights, length L.
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

    /// Alias sampling state preparation.
    ///
    /// Prepares: |0⟩ → Σ_ℓ √(p̃_ℓ) |ℓ⟩|garbage_ℓ⟩
    ///
    /// **Warning:** The index register is entangled with the ancilla qubits (garbage).
    /// This operation is only useful as the PREPARE subroutine in a block encoding
    /// (LCU or qubitization), where PREPARE† is applied to uncompute the garbage
    /// and project onto the correct subspace.
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
                IntAsBoolArray(keepCoeff[idx], mu) + IntAsBoolArray(altIndex[idx], nIndexQubits)
            } else {
                IntAsBoolArray(barHeight, mu) + IntAsBoolArray(idx, nIndexQubits)
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

    /// Circuit entry point for alias sampling.
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

    // ════════════════════════════════════════════════════════════════════════════
    //  Conditional alias sampling (2D)
    // ════════════════════════════════════════════════════════════════════════════

    /// Build 3D QROM table for conditional alias sampling with sign bits and
    /// optional free-rider data.
    ///
    /// Returns Bool[][][] of shape [nCond][nPaddedIdx][dataBits], suitable for
    /// Select2DLoad with outerAddress=conditionalRegister, innerAddress=indexRegister.
    ///
    /// Each row encodes: keepCoeff[μ] + altIndex[nIdx] + signOrig[1] + signAlt[1] + freeRider[*].
    function BuildConditionalAliasTable3D(
        coefficients : Double[][],
        freeRiderData : Bool[][],
        bitsPrecision : Int,
        nIndexBits : Int
    ) : Bool[][][] {
        let nCond = Length(coefficients);
        let nCoeffs = Length(coefficients[0]);
        let nPaddedIdx = 1 <<< nIndexBits;
        let nFreeRiderBits = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let barHeight = (1 <<< bitsPrecision) - 1;

        mutable result : Bool[][][] = [];
        for c in 0..nCond - 1 {
            let squaredCoeffs = Mapped(x -> x * x, coefficients[c]);
            let (keepCoeff, altIndex) = DiscretizedProbabilityDistribution(bitsPrecision, squaredCoeffs);
            mutable innerData : Bool[][] = [];
            for b in 0..nPaddedIdx - 1 {
                if b < nCoeffs {
                    let signBit = coefficients[c][b] < 0.0;
                    let signAltBit = coefficients[c][altIndex[b]] < 0.0;
                    set innerData += [
                        IntAsBoolArray(keepCoeff[b], bitsPrecision) + IntAsBoolArray(altIndex[b], nIndexBits) + [signBit, signAltBit] + (if nFreeRiderBits > 0 { freeRiderData[c] } else { [] })
                    ];
                } else {
                    set innerData += [
                        IntAsBoolArray(barHeight, bitsPrecision) + IntAsBoolArray(b, nIndexBits) + [false, false] + (if nFreeRiderBits > 0 { freeRiderData[c] } else { [] })
                    ];
                }
            }
            set result += [innerData];
        }
        return result;
    }

    /// Conditional alias sampling PREPARE (2D) — prepares
    /// |c⟩|0⟩ → |c⟩ Σ_ℓ √(p̃_{c,ℓ}) e^{iπ·sign_{c,ℓ}} |ℓ⟩|garbage⟩.
    ///
    /// Uses Select2DLoad to load per-condition alias tables in a single QROM pass.
    /// Sign bits encode negative amplitudes via Z phase (Von Burg arXiv:2011.03494, Def. 1).
    ///
    /// Register layout:
    ///   conditionalRegister — outer address (x_o)
    ///   indexRegister — output sampled index (b)
    ///   uniformRegister[μ] — ancilla for comparison
    ///   flagQubit[1] — ancilla for alias resolution
    ///   qromOutput[μ + nIdx + 2] — QROM target (keep, alt, signOrig, signAlt)
    operation ConditionalAliasSamplingPrepare(
        coefficients : Double[][],
        bitsPrecision : Int,
        conditionalRegister : Qubit[],
        indexRegister : Qubit[],
        uniformRegister : Qubit[],
        flagQubit : Qubit,
        qromOutput : Qubit[],
        numSwapBits : Int
    ) : Unit is Adj {
        ConditionalAliasSamplingPrepareWithFreeRider(
            coefficients,
            [],
            bitsPrecision,
            conditionalRegister,
            indexRegister,
            uniformRegister,
            flagQubit,
            qromOutput,
            [],
            numSwapBits
        );
    }

    /// Conditional alias sampling state preparation with free-rider data — prepares
    /// |c⟩|0⟩ → |c⟩ Σ_ℓ √(p̃_{c,ℓ}) e^{iπ·sign_{c,ℓ}} |ℓ⟩|garbage⟩ ⊗ |data_c⟩.
    ///
    /// "Free rider" = classical data that depends only on the conditional register
    /// (not on the sampled index), appended to every QROM row for that condition.
    /// Use case: DFTHC inner PREP loads (G, r) bits alongside the b-sampling alias table.
    ///
    /// Circuit (arXiv:2502.15882v1, Table A):
    ///   1. PrepareUniformSuperposition on indexRegister
    ///   2. H⊗μ on uniformRegister
    ///   3. Select2DLoad: (cond, idx) → (keep, alt, signOrig, signAlt, freeRider)
    ///   4. Compare σ ≥ keep → set flag
    ///   5. Conditional swap index ↔ alt
    ///   6. Conditional swap signOrig ↔ signAlt
    ///   7. Z(signOrig) for sign encoding
    operation ConditionalAliasSamplingPrepareWithFreeRider(
        coefficients : Double[][],
        freeRiderData : Bool[][],
        bitsPrecision : Int,
        conditionalRegister : Qubit[],
        indexRegister : Qubit[],
        uniformRegister : Qubit[],
        flagQubit : Qubit,
        qromOutput : Qubit[],
        freeRiderRegister : Qubit[],
        numSwapBits : Int
    ) : Unit is Adj {
        let nIndexBits = Length(indexRegister);
        let nCoeffs = Length(coefficients[0]);

        // Build 3D table for Select2D: shape [nCond][2^nIdx][dataBits]
        let table3D = BuildConditionalAliasTable3D(
            coefficients,
            freeRiderData,
            bitsPrecision,
            nIndexBits
        );

        PrepareUniformSuperposition(nCoeffs, indexRegister);
        ApplyToEachCA(H, uniformRegister);

        // 2D QROM load — outer=conditionalRegister, inner=indexRegister.
        let nCond = Length(table3D);
        let nInnerData = Length(table3D[0]);
        let m = Length(qromOutput) + Length(freeRiderRegister);
        let lambda = if numSwapBits == -1 {
            ComputeOptimalLambda2D(nCond, nInnerData, m)
        } elif numSwapBits > 0 {
            numSwapBits
        } else {
            0
        };
        if lambda == 0 {
            Select2DLoad(table3D, conditionalRegister, indexRegister, 0, qromOutput + freeRiderRegister);
        } else {
            use swapAnc = Qubit[m * ((1 <<< lambda) - 1)];
            Select2DLoad(table3D, conditionalRegister, indexRegister, lambda, qromOutput + freeRiderRegister + swapAnc);
        }

        let keepCoeffLoaded = qromOutput[0..bitsPrecision - 1];
        let altIndexReg = qromOutput[bitsPrecision..bitsPrecision + nIndexBits - 1];
        let signOrigQubit = qromOutput[bitsPrecision + nIndexBits];
        let signAltQubit = qromOutput[bitsPrecision + nIndexBits + 1];
        ApplyIfGreaterLE(X, uniformRegister, keepCoeffLoaded, flagQubit);

        for i in 0..nIndexBits - 1 {
            Controlled SWAP([flagQubit], (indexRegister[i], altIndexReg[i]));
        }

        // Sign encoding (Von Burg arXiv:2011.03494, Def. 1)
        Controlled SWAP([flagQubit], (signOrigQubit, signAltQubit));
        Z(signOrigQubit);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers — allocate qubits via QIR.Runtime so they persist for
    // dump_machine (qubit values cannot cross the Python ↔ Q# boundary).
    // ═══════════════════════════════════════════════════════════════════════════

    /// Test wrapper: run alias sampling and leave state for dump_machine.
    operation RunAliasSamplingPrep(
        coefficients : Double[],
        bitsPrecision : Int,
        numIndexQubits : Int,
        numQubits : Int,
    ) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(numQubits);
        let params = new AliasSamplingParams {
            coefficients = coefficients,
            bitsPrecision = bitsPrecision,
            numIndexQubits = numIndexQubits,
            numQubits = numQubits,
        };
        AliasSamplingPrepare(params, qs);
    }

    /// Test wrapper: run conditional alias sampling and leave state for dump_machine.
    operation RunConditionalAliasSamplingPrep(
        coefficients : Double[][],
        bitsPrecision : Int,
        conditionValue : Int,
    ) : Unit {
        let nCond = Length(coefficients);
        let nCoeffs = Length(coefficients[0]);
        let nIndexBits = Ceiling(Lg(IntAsDouble(nCoeffs)));
        let nCondBits = Ceiling(Lg(IntAsDouble(nCond)));
        let nQromOutput = bitsPrecision + nIndexBits + 2;
        let totalQubits = nCondBits + nIndexBits + bitsPrecision + 1 + nQromOutput;

        let qs = QIR.Runtime.AllocateQubitArray(totalQubits);

        let conditionalReg = qs[0..nCondBits - 1];
        let indexReg = qs[nCondBits..nCondBits + nIndexBits - 1];
        let uniformReg = qs[nCondBits + nIndexBits..nCondBits + nIndexBits + bitsPrecision - 1];
        let flagQubit = qs[nCondBits + nIndexBits + bitsPrecision];
        let qromOut = qs[nCondBits + nIndexBits + bitsPrecision + 1..nCondBits + nIndexBits + bitsPrecision + nQromOutput];

        // Set conditional register to |conditionValue⟩
        ApplyXorInPlace(conditionValue, conditionalReg);

        ConditionalAliasSamplingPrepare(
            coefficients,
            bitsPrecision,
            conditionalReg,
            indexReg,
            uniformReg,
            flagQubit,
            qromOut,
            0
        );
    }

    /// Test wrapper: run conditional alias sampling with free-rider data.
    operation RunConditionalAliasSamplingPrepWithFreeRider(
        coefficients : Double[][],
        freeRiderData : Bool[][],
        bitsPrecision : Int,
        conditionValue : Int,
    ) : Unit {
        let nCond = Length(coefficients);
        let nCoeffs = Length(coefficients[0]);
        let nIndexBits = Ceiling(Lg(IntAsDouble(nCoeffs)));
        let nCondBits = Ceiling(Lg(IntAsDouble(nCond)));
        let nFreeRiderBits = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let nQromOutput = bitsPrecision + nIndexBits + 2;
        let totalQubits = nCondBits + nIndexBits + bitsPrecision + 1 + nQromOutput + nFreeRiderBits;

        let qs = QIR.Runtime.AllocateQubitArray(totalQubits);

        let conditionalReg = qs[0..nCondBits - 1];
        let indexReg = qs[nCondBits..nCondBits + nIndexBits - 1];
        let uniformReg = qs[nCondBits + nIndexBits..nCondBits + nIndexBits + bitsPrecision - 1];
        let flagQubit = qs[nCondBits + nIndexBits + bitsPrecision];
        let qromOut = qs[nCondBits + nIndexBits + bitsPrecision + 1..nCondBits + nIndexBits + bitsPrecision + nQromOutput];
        let freeRiderReg = qs[nCondBits + nIndexBits + bitsPrecision + 1 + nQromOutput..nCondBits + nIndexBits + bitsPrecision + nQromOutput + nFreeRiderBits];

        ApplyXorInPlace(conditionValue, conditionalReg);

        ConditionalAliasSamplingPrepareWithFreeRider(
            coefficients,
            freeRiderData,
            bitsPrecision,
            conditionalReg,
            indexReg,
            uniformReg,
            flagQubit,
            qromOut,
            freeRiderReg,
            0
        );
    }

}
