// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Circuits for isometries and unitary matrices from the C++ unitary synthesis
/// (Rupprecht & Wölk, arXiv:2605.28489).
///
/// Registers are little-endian, and row q · 2^n + a of an isometry is added state q and input
/// state a of the n-qubit input register followed by the added qubits.
///
/// Each `unitary_synthesis_*` circuit mapper uses these entry points with its prefix X
/// (`DenseIsometry`, `SparseIsometry` or `HouseholderIsometry`):
/// - `XSynthesis` holds the circuit data from the C++ synthesis and the register widths;
/// - `ApplyX(synthesis, input, added, phaseGradient, angleReg)` applies the isometry;
/// - `MakeXOp(synthesis)` returns the operation on the target, with its own phase gradient;
/// - `MakeXOpWithPhaseGradient(synthesis)` returns it on the target and a prepared phase gradient;
/// - `MakeXCircuit(synthesis)` runs it.
namespace QDKChemistry.Utils.IsometrySynthesis {

    import Std.Arrays.All;
    import Std.Arrays.Any;
    import Std.Arrays.ForEach;
    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Reversed;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Convert.BoolArrayAsInt;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.ResultArrayAsBoolArray;
    import Std.Diagnostics.Fact;
    import Std.Math.PI;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.Measurement.MResetX;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.SelectSwap.ApplyBranchPhaseFixup;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalLambda2D;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyGivensLayersDeferred;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyMultiplexedRyDeferred;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.ApplySignTable;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.LoadWordDeferred;
    import QDKChemistry.Utils.UnitarySynthesis.MeasuredParity;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizedGivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeTrimmedRyAngles;
    import QDKChemistry.Utils.UnitarySynthesis.XorSigns;

    /// Circuit data of a dense real isometry, mirroring the C++
    /// `qdk::chemistry::utils::DenseIsometrySynthesis`, applied as
    /// V → UCR₀ → CNOT → W₀ → UCR₁ → CNOT → W₁ → UCR₂ → B for two added qubits, V → UCR₀ → B for
    /// one, and B for none.
    struct DenseIsometrySynthesis {
        /// Ry angles of each UCRⱼ, one per input state, with the Givens signs folded in.
        rotationAngles : Double[][],
        /// W₀ and W₁ on the input register; empty unless two qubits are added.
        mixingGivens : GivensDecomposition[],
        /// B on the joint input and added register.
        blockGivens : GivensDecomposition,
        /// V on the input register, or empty when V is omitted or no qubit is added.
        rightFactorGivens : GivensDecomposition[],
        /// Widest swap network of the angle lookups, -1 to let each loader's cost model choose.
        maxSwapBits : Int,
        /// Qubits of the input register, the low qubits of the target.
        numInputQubits : Int,
        /// Qubits added to the input register, the high qubits of the target.
        numAddedQubits : Int,
        rotationBitPrecision : Int,
    }

    /// Circuit data of a sparse real isometry U = P_r · B · P_c, mirroring the C++
    /// `qdk::chemistry::utils::SparseIsometrySynthesis`.
    struct SparseIsometrySynthesis {
        /// P_c|c⟩ = |columnPermutation[c]⟩.
        columnPermutation : Int[],
        /// P_r|v⟩ = |rowPermutation[v]⟩.
        rowPermutation : Int[],
        /// Block-diagonal B.
        blockGivens : GivensDecomposition,
        /// Widest swap network of the angle lookups, -1 to let each loader's cost model choose.
        maxSwapBits : Int,
        /// Qubits of the input register, the low qubits of the target.
        numInputQubits : Int,
        /// Qubits added to the input register, the high qubits of the target.
        numAddedQubits : Int,
        rotationBitPrecision : Int,
    }

    /// Circuit data of a real isometry applied as a product of Householder reflections, one per
    /// column (Low, Kliuchnikov & Schaeffer, arXiv:1812.00954; Hennefarth et al., arXiv:2611.19728,
    /// Sec. 3.1), with each column loaded by `PrepareControlledColumn`.
    struct HouseholderIsometrySynthesis {
        /// Ry angles 2θ of each column, indexed by column, level (most significant bit first) and address.
        columnAngles : Double[][][],
        /// Sign bit of each amplitude of each column, empty for a column without negative amplitudes.
        columnSigns : Bool[][],
        /// Widest swap network of the angle lookups, -1 to let each loader's cost model choose.
        maxSwapBits : Int,
        /// Qubits of the input register, the low qubits of the target.
        numInputQubits : Int,
        /// Qubits added to the input register, the high qubits of the target.
        numAddedQubits : Int,
        rotationBitPrecision : Int,
    }

    /// Prepares Σ_x a_x |x⟩ on the little-endian `target` in |0⟩ when `control` is |1⟩, by layers of
    /// multiplexed Ry rotations from QROAM-loaded angles (Shende, Bullock & Markov) and a one-bit sign
    /// lookup, the controlled LKS state preparation of Hennefarth et al., App. B.1.
    operation PrepareControlledColumn(
        levels : Double[][],
        signs : Bool[],
        maxSwapBits : Int,
        control : Qubit,
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            let msbFirst = Reversed(target);
            for level in IndexRange(levels) {
                // The address is the already-prepared high part of the target, read little-endian.
                let table = QuantizeTrimmedRyAngles(levels[level], Length(angleReg));
                ApplyControlledMultiplexedRy(table, maxSwapBits, Reversed(msbFirst[...level - 1]), [control], msbFirst[level], phaseGradient, angleReg);
            }
            if not IsEmpty(signs) {
                ApplyBranchPhaseFixup([true], Mapped(sign -> [sign], signs), true, [control] + target[...Length(levels) - 1]);
            }
        }
        adjoint (...) {
            if not IsEmpty(signs) {
                ApplyBranchPhaseFixup([true], Mapped(sign -> [sign], signs), true, [control] + target[...Length(levels) - 1]);
            }
            let msbFirst = Reversed(target);
            for level in Length(levels) - 1..-1..0 {
                let table = QuantizeTrimmedRyAngles(levels[level], Length(angleReg));
                Adjoint ApplyControlledMultiplexedRy(table, maxSwapBits, Reversed(msbFirst[...level - 1]), [control], msbFirst[level], phaseGradient, angleReg);
            }
        }
    }

    /// Cache key of a column load: everything that sets its cost, including the trimmed angle tables.
    function ColumnLoadCacheKey(levels : Double[][], signs : Bool[], maxSwapBits : Int, widths : Int[]) : String {
        let lengths = MappedOverRange(level -> Length(QuantizeTrimmedRyAngles(levels[level], widths[0])), IndexRange(levels));
        $"w{widths}_s{maxSwapBits}_l{lengths}_g{not IsEmpty(signs)}"
    }

    /// Reflects about (|0⟩|column⟩ - |1⟩|v⟩)/√2 on [reflect | target] (Eq. 17 of Hennefarth et al.), where
    /// |column⟩ is written to `input` and v is prepared on `target` by `PrepareControlledColumn`.
    operation ApplyHouseholderColumnReflection(
        levels : Double[][],
        signs : Bool[],
        column : Int,
        maxSwapBits : Int,
        reflect : Qubit,
        input : Qubit[],
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            // Every column with the same key has the same cost, as the column index only sets CNOTs.
            if BeginEstimateCaching($"HouseholderColumn_{ColumnLoadCacheKey(levels, signs, maxSwapBits, [Length(angleReg), Length(input), Length(target)])}", 0) {
                within {
                    Adjoint PrepareControlledColumn(levels, signs, maxSwapBits, reflect, target, phaseGradient, angleReg);
                    Adjoint ApplyControlledOnInt(0, ApplyXorInPlace(column, _), [reflect], input);
                    Z(reflect);
                    H(reflect);
                } apply {
                    within {
                        X(reflect);
                    } apply {
                        ApplyControlledOnInt(0, Z, target, reflect);
                    }
                }
                EndEstimateCaching();
            }
        }
        adjoint self;
    }

    /// Applies a Householder isometry to the little-endian `input` and the `added` qubits in |0⟩, with
    /// the signature of `ApplyDenseIsometry`. The reflection ancilla ends in |0⟩ up to rotation
    /// quantization error and is reset before release.
    operation ApplyHouseholderIsometry(
        synthesis : HouseholderIsometrySynthesis,
        input : Qubit[],
        added : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            use reflect = Qubit();
            for column in IndexRange(synthesis.columnAngles) {
                ApplyHouseholderColumnReflection(synthesis.columnAngles[column], synthesis.columnSigns[column], column, synthesis.maxSwapBits, reflect, input, input + added, phaseGradient, angleReg);
            }
            X(reflect);
            Reset(reflect);
        }
        adjoint (...) {
            // The reflections are their own inverses, so the adjoint applies them in reverse order.
            use reflect = Qubit();
            X(reflect);
            for column in Length(synthesis.columnAngles) - 1..-1..0 {
                ApplyHouseholderColumnReflection(synthesis.columnAngles[column], synthesis.columnSigns[column], column, synthesis.maxSwapBits, reflect, input, input + added, phaseGradient, angleReg);
            }
            Reset(reflect);
        }
    }

    /// Returns the bit-string tables of a permutation and of its inverse.
    function PermutationData(permTargets : Int[], numBits : Int) : (Bool[][], Bool[][]) {
        mutable inverse = Repeated(0, Length(permTargets));
        for source in IndexRange(permTargets) {
            set inverse w/= permTargets[source] <- source;
        }
        (Mapped(IntAsBoolArray(_, numBits), permTargets), Mapped(IntAsBoolArray(_, numBits), inverse))
    }

    /// Applies |i⟩ → |P(i)⟩, with P and P⁻¹ given as little-endian bit strings, by a QROAM
    /// load of at most `maxSwapBits` swap bits (-1 to let the loader's cost model choose) and a measurement-based
    /// unlookup.
    operation PermutationViaQROAM(
        permTargets : Bool[][],
        invPermTargets : Bool[][],
        maxSwapBits : Int,
        target : Qubit[]
    ) : Unit is Adj {
        body (...) {
            ApplySignTable(ApplyPermutationDeferred(permTargets, invPermTargets, maxSwapBits, target, [false, size = Length(permTargets)]), target);
        }
        adjoint (...) {
            PermutationViaQROAM(invPermTargets, permTargets, maxSwapBits, target);
        }
    }

    /// `PermutationViaQROAM` without its sign fixups, applied to a register carrying the pending `signs`,
    /// which it returns permuted and with the measured-out loads added.
    internal operation ApplyPermutationDeferred(
        permTargets : Bool[][],
        invPermTargets : Bool[][],
        maxSwapBits : Int,
        target : Qubit[],
        signs : Bool[]
    ) : Bool[] {
        if All(fixed -> fixed, MappedOverRange(state -> BoolArrayAsInt(permTargets[state]) == state, IndexRange(permTargets))) {
            // The identity needs no lookup.
            return signs;
        }
        use loaded = Qubit[Length(target)];
        let swapBits = ComputeOptimalLambda2D(1, Length(permTargets), Length(target), true, maxSwapBits);
        let loadSigns = LoadWordDeferred(permTargets, swapBits, target, [], loaded);
        ApplyToEachA(SWAP, Zipped(target, loaded));
        // `loaded` now holds P⁻¹ of the permuted state.
        let measured = ResultArrayAsBoolArray(ForEach(MResetX, loaded));
        MappedOverRange(
            state -> {
                let source = BoolArrayAsInt(invPermTargets[state]);
                (signs[source] != loadSigns[source % Length(loadSigns)]) != MeasuredParity(measured, invPermTargets[state])
            },
            IndexRange(signs)
        )
    }

    /// Returns the quantized rotations, mixing unitaries, right factors and block unitary of a dense isometry.
    internal function QuantizeDenseIsometry(
        synthesis : DenseIsometrySynthesis,
        numInput : Int,
        numAdded : Int,
        rotationBitPrecision : Int
    ) : (Bool[][][], QuantizedGivensDecomposition[], QuantizedGivensDecomposition[], QuantizedGivensDecomposition) {
        Fact(numAdded <= 2, "Dense isometry synthesis adds at most two qubits.");
        Fact(
            Length(synthesis.rotationAngles) == (numAdded == 2 ? 3 | numAdded),
            "Dense isometry data needs one multiplexed rotation per added qubit, or three for two."
        );
        Fact(
            Length(synthesis.mixingGivens) == (numAdded == 2 ? 2 | 0),
            "Dense isometry data needs two mixing unitaries for two added qubits and none otherwise."
        );
        Fact(Length(synthesis.rightFactorGivens) <= 1, "Dense isometry data has at most one right factor.");
        Fact(numAdded > 0 or IsEmpty(synthesis.rightFactorGivens), "Dense isometry data has no right factor without added qubits.");
        let inputDim = 1 <<< numInput;
        // A Givens layer on an n-qubit register is addressed by its n - 1 upper qubits.
        let inputAddresses = inputDim / 2;
        let blockAddresses = (inputDim <<< numAdded) / 2;
        let maxSwapBits = synthesis.maxSwapBits;
        let quantize = givens -> QuantizeGivensDecomposition(givens, inputAddresses, rotationBitPrecision, maxSwapBits);
        (
            Mapped(QuantizeTrimmedRyAngles(_, rotationBitPrecision), synthesis.rotationAngles),
            Mapped(quantize, synthesis.mixingGivens),
            Mapped(quantize, synthesis.rightFactorGivens),
            QuantizeGivensDecomposition(synthesis.blockGivens, blockAddresses, rotationBitPrecision, maxSwapBits)
        )
    }

    /// Applies a dense isometry to the little-endian `input` and the `added` qubits in |0⟩,
    /// using the prepared `phaseGradient` and a clean `angleReg` of the same length.
    /// The forward circuit fixes the measured-out loads together with the signs of the Givens unitaries in one
    /// final lookup, plus one after UCR₁ for two added qubits.
    operation ApplyDenseIsometry(
        synthesis : DenseIsometrySynthesis,
        input : Qubit[],
        added : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            let numAdded = Length(added);
            let (rotations, mixingGivens, rightFactorGivens, blockGivens) = QuantizeDenseIsometry(synthesis, Length(input), numAdded, Length(phaseGradient));
            let maxSwapBits = synthesis.maxSwapBits;
            // The Givens operations take most-significant-first registers, hence `Reversed(input)`.
            if numAdded == 0 {
                ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(input), phaseGradient, angleReg);
            } else {
                // Pending signs are indexed by the little-endian value of `input`, then of the added qubits they depend on.
                mutable signs = [false, size = 1 <<< Length(input)];
                for givens in rightFactorGivens {
                    set signs = ApplyGivensLayersDeferred(givens, false, [], Reversed(input), phaseGradient, angleReg, signs);
                }
                set signs = XorSigns(signs, ApplyMultiplexedRyDeferred(rotations[0], maxSwapBits, input, [], added[0], phaseGradient, angleReg));
                if numAdded == 2 {
                    let q0 = added[0];
                    let q1 = added[1];
                    CNOT(q1, q0);
                    set signs = ApplyGivensLayersDeferred(mixingGivens[0], false, [q0], Reversed(input), phaseGradient, angleReg, signs + signs);
                    set signs = XorSigns(signs, ApplyMultiplexedRyDeferred(rotations[1], maxSwapBits, input, [q0], q1, phaseGradient, angleReg));
                    // The next CNOT would make the signs depend on q0 while W₁ is controlled by q1.
                    ApplySignTable(signs, input + [q0]);
                    CNOT(q1, q0);
                    let afterMixing = ApplyGivensLayersDeferred(mixingGivens[1], false, [q1], Reversed(input), phaseGradient, angleReg, [false, size = 2 <<< Length(input)]);
                    let pending = XorSigns(afterMixing, ApplyMultiplexedRyDeferred(rotations[2], maxSwapBits, input, [q1], q0, phaseGradient, angleReg));
                    let half = 1 <<< Length(input);
                    set signs = pending[...half - 1] + pending[...half - 1] + pending[half...] + pending[half...];
                } else {
                    set signs = signs + signs;
                }
                // The joint register Reversed(input + added) selects block q, or block 2·q1 + q0, of B.
                ApplySignTable(ApplyGivensLayersDeferred(blockGivens, false, [], Reversed(input + added), phaseGradient, angleReg, signs), input + added);
            }
        }
        adjoint (...) {
            // Each component is inverted on its own, since deferred signs would meet spectator added qubits.
            let numAdded = Length(added);
            let (rotations, mixingGivens, rightFactorGivens, blockGivens) = QuantizeDenseIsometry(synthesis, Length(input), numAdded, Length(phaseGradient));
            let maxSwapBits = synthesis.maxSwapBits;
            Adjoint ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(input + added), phaseGradient, angleReg);
            if numAdded == 2 {
                let q0 = added[0];
                let q1 = added[1];
                Adjoint ApplyControlledMultiplexedRy(rotations[2], maxSwapBits, input, [q1], q0, phaseGradient, angleReg);
                Adjoint ApplyRealUnitaryViaGivens(mixingGivens[1], [q1], Reversed(input), phaseGradient, angleReg);
                CNOT(q1, q0);
                Adjoint ApplyControlledMultiplexedRy(rotations[1], maxSwapBits, input, [q0], q1, phaseGradient, angleReg);
                Adjoint ApplyRealUnitaryViaGivens(mixingGivens[0], [q0], Reversed(input), phaseGradient, angleReg);
                CNOT(q1, q0);
            }
            if numAdded > 0 {
                Adjoint ApplyMultiplexedRy(rotations[0], maxSwapBits, input, added[0], phaseGradient, angleReg);
            }
            for givens in rightFactorGivens {
                if IsEmpty(input) {
                    // V is the 1 × 1 sign ±1, a global sign with no input qubit to carry it.
                    if Any(flip -> flip, givens.phases) {
                        R(PauliI, 2.0 * PI(), added[0]);
                    }
                } else {
                    Adjoint ApplyRealUnitaryViaGivens(givens, [], Reversed(input), phaseGradient, angleReg);
                }
            }
        }
    }

    /// Applies a sparse isometry U = P_r · B · P_c to the little-endian register `input + added`,
    /// with the signature of `ApplyDenseIsometry`.
    operation ApplySparseIsometry(
        synthesis : SparseIsometrySynthesis,
        input : Qubit[],
        added : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        body (...) {
            ApplySparseIsometryInDirection(synthesis, false, input + added, phaseGradient, angleReg);
        }
        adjoint (...) {
            ApplySparseIsometryInDirection(synthesis, true, input + added, phaseGradient, angleReg);
        }
    }

    /// Applies `ApplySparseIsometry`, or its adjoint when `reverse`, fixing all measured-out loads and the signs
    /// of B in one final lookup.
    internal operation ApplySparseIsometryInDirection(
        synthesis : SparseIsometrySynthesis,
        reverse : Bool,
        target : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        let totalBits = Length(target);
        Fact(
            Length(synthesis.columnPermutation) == 1 <<< totalBits and Length(synthesis.rowPermutation) == 1 <<< totalBits,
            "Sparse isometry permutations need one entry per basis state of the target register."
        );
        let (colPermData, colInvPermData) = PermutationData(synthesis.columnPermutation, totalBits);
        let (rowPermData, rowInvPermData) = PermutationData(synthesis.rowPermutation, totalBits);
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, 1 <<< (totalBits - 1), Length(phaseGradient), synthesis.maxSwapBits);
        // U† = P_c⁻¹ · B† · P_r⁻¹.
        let ((firstPerm, firstInv), (lastPerm, lastInv)) = reverse ? ((rowInvPermData, rowPermData), (colInvPermData, colPermData)) | ((colPermData, colInvPermData), (rowPermData, rowInvPermData));
        mutable signs = ApplyPermutationDeferred(firstPerm, firstInv, synthesis.maxSwapBits, target, [false, size = 1 <<< totalBits]);
        // The Givens layers act on the most-significant-first view of the register.
        set signs = ApplyGivensLayersDeferred(blockGivens, reverse, [], Reversed(target), phaseGradient, angleReg, signs);
        set signs = ApplyPermutationDeferred(lastPerm, lastInv, synthesis.maxSwapBits, target, signs);
        ApplySignTable(signs, target);
    }

    /// Applies `isometry` to `qs` = `[input | added | phaseGradient]`, whose last
    /// `rotationBitPrecision` qubits hold a prepared phase gradient, with a clean angle register.
    operation ApplyIsometryOnRegister(
        numInputQubits : Int,
        numAddedQubits : Int,
        rotationBitPrecision : Int,
        isometry : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        qs : Qubit[]
    ) : Unit is Adj {
        let numTargetQubits = numInputQubits + numAddedQubits;
        Fact(
            Length(qs) == numTargetQubits + rotationBitPrecision,
            "The register must hold the target qubits followed by the phase gradient."
        );
        use angleReg = Qubit[rotationBitPrecision];
        isometry(qs[...numInputQubits - 1], qs[numInputQubits..numTargetQubits - 1], qs[numTargetQubits...], angleReg);
    }

    /// `ApplyIsometryOnRegister` on the target qubits alone, with its own phase gradient.
    operation ApplyIsometryOnTarget(
        numInputQubits : Int,
        numAddedQubits : Int,
        rotationBitPrecision : Int,
        isometry : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        target : Qubit[]
    ) : Unit is Adj {
        use phaseGradient = Qubit[rotationBitPrecision];
        within {
            PreparePhaseGradientState(phaseGradient);
        } apply {
            ApplyIsometryOnRegister(numInputQubits, numAddedQubits, rotationBitPrecision, isometry, target + phaseGradient);
        }
    }

    /// Returns `ApplyDenseIsometry` on the target `[input | added]`, with its own phase gradient.
    function MakeDenseIsometryOp(synthesis : DenseIsometrySynthesis) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            synthesis.numInputQubits,
            synthesis.numAddedQubits,
            synthesis.rotationBitPrecision,
            ApplyDenseIsometry(synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplyDenseIsometry` on `[input | added | phaseGradient]`, whose prepared phase
    /// gradient is left in its state.
    function MakeDenseIsometryOpWithPhaseGradient(synthesis : DenseIsometrySynthesis) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            synthesis.numInputQubits,
            synthesis.numAddedQubits,
            synthesis.rotationBitPrecision,
            ApplyDenseIsometry(synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a dense isometry, leaving the output on the allocated target.
    operation MakeDenseIsometryCircuit(synthesis : DenseIsometrySynthesis) : Unit {
        use target = Qubit[synthesis.numInputQubits + synthesis.numAddedQubits];
        MakeDenseIsometryOp(synthesis)(target);
    }

    /// Returns `ApplySparseIsometry` on the target, with its own phase gradient.
    function MakeSparseIsometryOp(synthesis : SparseIsometrySynthesis) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            synthesis.numInputQubits,
            synthesis.numAddedQubits,
            synthesis.rotationBitPrecision,
            ApplySparseIsometry(synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplySparseIsometry` on `[target | phaseGradient]`, as for dense isometries.
    function MakeSparseIsometryOpWithPhaseGradient(synthesis : SparseIsometrySynthesis) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            synthesis.numInputQubits,
            synthesis.numAddedQubits,
            synthesis.rotationBitPrecision,
            ApplySparseIsometry(synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a sparse isometry, leaving the output on the allocated target.
    operation MakeSparseIsometryCircuit(synthesis : SparseIsometrySynthesis) : Unit {
        use target = Qubit[synthesis.numInputQubits + synthesis.numAddedQubits];
        MakeSparseIsometryOp(synthesis)(target);
    }

    /// Returns `ApplyHouseholderIsometry` on the target, with its own phase gradient.
    function MakeHouseholderIsometryOp(synthesis : HouseholderIsometrySynthesis) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            synthesis.numInputQubits,
            synthesis.numAddedQubits,
            synthesis.rotationBitPrecision,
            ApplyHouseholderIsometry(synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplyHouseholderIsometry` on `[target | phaseGradient]`, as for dense isometries.
    function MakeHouseholderIsometryOpWithPhaseGradient(synthesis : HouseholderIsometrySynthesis) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            synthesis.numInputQubits,
            synthesis.numAddedQubits,
            synthesis.rotationBitPrecision,
            ApplyHouseholderIsometry(synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a Householder isometry, leaving the output on the allocated target.
    operation MakeHouseholderIsometryCircuit(synthesis : HouseholderIsometrySynthesis) : Unit {
        use target = Qubit[synthesis.numInputQubits + synthesis.numAddedQubits];
        MakeHouseholderIsometryOp(synthesis)(target);
    }
}
