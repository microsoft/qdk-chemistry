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
/// (`DenseIsometry` or `SparseIsometry`):
/// - `XSynthesis` holds the circuit data from the C++ synthesis;
/// - `ApplyX(synthesis, input, added, phaseGradient, angleReg)` applies the isometry;
/// - `XParams { synthesis, numInputQubits, numAddedQubits, rotationBitPrecision }` bundles it;
/// - `MakeXOp(params)` returns the operation on the target, with its own phase gradient;
/// - `MakeXOpWithPhaseGradient(params)` returns it on the target and a prepared phase gradient;
/// - `MakeXCircuit(synthesis, numInputQubits, numAddedQubits, rotationBitPrecision)` runs it.
namespace QDKChemistry.Utils.IsometrySynthesis {

    import Std.Arrays.Any;
    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.Reversed;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Convert.IntAsBoolArray;
    import Std.Diagnostics.Fact;
    import Std.Math.PI;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.SelectSwap.ComputeOptimalSwapBits;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.TrimZeroWords;

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
        /// Widest swap network of the angle lookups, -1 for the Toffoli-optimal width.
        maxSwapBits : Int,
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
        /// Widest swap network of the angle lookups, -1 for the Toffoli-optimal width.
        maxSwapBits : Int,
    }

    /// Parameters of a composable dense isometry operation.
    struct DenseIsometryParams {
        synthesis : DenseIsometrySynthesis,
        /// Qubits of the input register, the low qubits of the target.
        numInputQubits : Int,
        /// Qubits added to the input register, the high qubits of the target.
        numAddedQubits : Int,
        rotationBitPrecision : Int,
    }

    /// Parameters of a composable sparse isometry operation; see `DenseIsometryParams`.
    struct SparseIsometryParams {
        synthesis : SparseIsometrySynthesis,
        numInputQubits : Int,
        numAddedQubits : Int,
        rotationBitPrecision : Int,
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
    /// load of at most `maxSwapBits` swap bits (-1 for Toffoli-optimal) and a measurement-based
    /// unlookup.
    operation PermutationViaQROAM(
        permTargets : Bool[][],
        invPermTargets : Bool[][],
        maxSwapBits : Int,
        target : Qubit[]
    ) : Unit is Adj {
        body (...) {
            use loaded = Qubit[Length(target)];
            let swapBits = ComputeOptimalSwapBits(Length(permTargets), Length(target), maxSwapBits);
            SelectSwap2D([permTargets], swapBits, true, [], target, [], loaded);
            ApplyToEachA(SWAP, Zipped(target, loaded));
            Adjoint Select(invPermTargets, target, loaded);
        }
        adjoint (...) {
            PermutationViaQROAM(invPermTargets, permTargets, maxSwapBits, target);
        }
    }

    /// Applies a dense isometry to the little-endian `input` and the `added` qubits in |0⟩,
    /// using the prepared `phaseGradient` and a clean `angleReg` of the same length.
    operation ApplyDenseIsometry(
        synthesis : DenseIsometrySynthesis,
        input : Qubit[],
        added : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit is Adj {
        let numAdded = Length(added);
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
        let rotationBitPrecision = Length(phaseGradient);
        let inputDim = 1 <<< Length(input);
        // A Givens layer on an n-qubit register is addressed by its n - 1 upper qubits.
        let inputAddresses = inputDim / 2;
        let blockAddresses = (inputDim <<< numAdded) / 2;
        let maxSwapBits = synthesis.maxSwapBits;
        let quantize = givens -> QuantizeGivensDecomposition(givens, inputAddresses, rotationBitPrecision, maxSwapBits);
        let rotations = Mapped(angles -> TrimZeroWords(QuantizeRyAngles(angles, rotationBitPrecision)), synthesis.rotationAngles);
        let mixingGivens = Mapped(quantize, synthesis.mixingGivens);
        let rightFactorGivens = Mapped(quantize, synthesis.rightFactorGivens);
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, blockAddresses, rotationBitPrecision, maxSwapBits);

        // The Givens operations take most-significant-first registers, hence `Reversed(input)`.
        for givens in rightFactorGivens {
            if IsEmpty(input) {
                // V is the 1 × 1 sign ±1, a global sign with no input qubit to carry it.
                if Any(flip -> flip, givens.phases) {
                    R(PauliI, 2.0 * PI(), added[0]);
                }
            } else {
                ApplyRealUnitaryViaGivens(givens, [], Reversed(input), phaseGradient, angleReg);
            }
        }
        if numAdded > 0 {
            ApplyMultiplexedRy(rotations[0], maxSwapBits, input, added[0], phaseGradient, angleReg);
        }
        if numAdded == 2 {
            let q0 = added[0];
            let q1 = added[1];
            CNOT(q1, q0);
            ApplyRealUnitaryViaGivens(mixingGivens[0], [q0], Reversed(input), phaseGradient, angleReg);
            ApplyControlledMultiplexedRy(rotations[1], maxSwapBits, input, [q0], q1, phaseGradient, angleReg);
            CNOT(q1, q0);
            ApplyRealUnitaryViaGivens(mixingGivens[1], [q1], Reversed(input), phaseGradient, angleReg);
            ApplyControlledMultiplexedRy(rotations[2], maxSwapBits, input, [q1], q0, phaseGradient, angleReg);
        }
        // The joint register Reversed(input + added) selects block q, or block 2·q1 + q0, of B.
        ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(input + added), phaseGradient, angleReg);
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
        let target = input + added;
        let totalBits = Length(target);
        Fact(
            Length(synthesis.columnPermutation) == 1 <<< totalBits and Length(synthesis.rowPermutation) == 1 <<< totalBits,
            "Sparse isometry permutations need one entry per basis state of the target register."
        );
        let (colPermData, colInvPermData) = PermutationData(synthesis.columnPermutation, totalBits);
        let (rowPermData, rowInvPermData) = PermutationData(synthesis.rowPermutation, totalBits);
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, 1 <<< (totalBits - 1), Length(phaseGradient), synthesis.maxSwapBits);

        PermutationViaQROAM(colPermData, colInvPermData, synthesis.maxSwapBits, target);
        // The Givens layers act on the most-significant-first view of the register.
        ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(target), phaseGradient, angleReg);
        PermutationViaQROAM(rowPermData, rowInvPermData, synthesis.maxSwapBits, target);
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
    function MakeDenseIsometryOp(params : DenseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            params.numInputQubits,
            params.numAddedQubits,
            params.rotationBitPrecision,
            ApplyDenseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplyDenseIsometry` on `[input | added | phaseGradient]`, whose prepared phase
    /// gradient is left in its state.
    function MakeDenseIsometryOpWithPhaseGradient(params : DenseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            params.numInputQubits,
            params.numAddedQubits,
            params.rotationBitPrecision,
            ApplyDenseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a dense isometry, leaving the output on the allocated target.
    operation MakeDenseIsometryCircuit(
        synthesis : DenseIsometrySynthesis,
        numInputQubits : Int,
        numAddedQubits : Int,
        rotationBitPrecision : Int
    ) : Unit {
        use target = Qubit[numInputQubits + numAddedQubits];
        ApplyIsometryOnTarget(numInputQubits, numAddedQubits, rotationBitPrecision, ApplyDenseIsometry(synthesis, _, _, _, _), target);
    }

    /// Returns `ApplySparseIsometry` on the target, with its own phase gradient.
    function MakeSparseIsometryOp(params : SparseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            params.numInputQubits,
            params.numAddedQubits,
            params.rotationBitPrecision,
            ApplySparseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplySparseIsometry` on `[target | phaseGradient]`, as for dense isometries.
    function MakeSparseIsometryOpWithPhaseGradient(params : SparseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            params.numInputQubits,
            params.numAddedQubits,
            params.rotationBitPrecision,
            ApplySparseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a sparse isometry, leaving the output on the allocated target.
    operation MakeSparseIsometryCircuit(
        synthesis : SparseIsometrySynthesis,
        numInputQubits : Int,
        numAddedQubits : Int,
        rotationBitPrecision : Int
    ) : Unit {
        use target = Qubit[numInputQubits + numAddedQubits];
        ApplyIsometryOnTarget(numInputQubits, numAddedQubits, rotationBitPrecision, ApplySparseIsometry(synthesis, _, _, _, _), target);
    }
}
