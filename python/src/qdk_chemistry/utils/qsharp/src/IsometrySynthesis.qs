// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center
// (DLR), licensed under the Apache License, Version 2.0, and modified for QDK
// Chemistry.

/// Circuits for real isometries and orthogonal matrices from the C++ unitary synthesis
/// (Rupprecht & Wölk, arXiv:2605.28489).
///
/// Registers are little-endian: basis state |r⟩ of a register holds bit k of r in qubit k.
/// An isometry with 2^k · 2^n rows acts on an n-qubit input register followed by k added
/// qubits, so row q · 2^n + a is added state q and input state a. Column a is input state
/// a with the added qubits in |0⟩.
namespace QDKChemistry.Utils.IsometrySynthesis {

    import Std.Arrays.Any;
    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.Mapped;
    import Std.Arrays.Reversed;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.PI;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyMultiplexedRy;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;

    /// Circuit data of a dense real isometry, mirroring the C++
    /// `qdk::chemistry::utils::DenseIsometrySynthesis`.
    ///
    /// With two added qubits the circuit is V → UCR₀ → CNOT → W₀ → UCR₁ → CNOT → W₁ → UCR₂ → B
    /// (Fig. 5 of Rupprecht & Wölk, arXiv:2605.28489), with one it is V → UCR₀ → B (Eq. 6),
    /// and with none it is B. Each UCRⱼ is an Ry on an added qubit multiplexed by the input
    /// register, and B is block diagonal over the added qubits.
    struct DenseIsometrySynthesis {
        /// Ry angles of UCR₀, UCR₁ and UCR₂, one per input state; UCR₀ only for one added
        /// qubit and none for zero.
        rotationAngles : Double[][],
        /// W₀ and W₁ on the input register; empty unless two qubits are added.
        mixingGivens : GivensDecomposition[],
        /// B on the joint input and added register.
        blockGivens : GivensDecomposition,
        /// V on the input register, or empty when V is omitted.
        rightFactorGivens : GivensDecomposition[],
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

    /// Parameters of a composable sparse isometry operation.
    struct SparseIsometryParams {
        synthesis : SparseIsometrySynthesis,
        /// Qubits of the target register.
        numQubits : Int,
        rotationBitPrecision : Int,
    }

    /// Returns the bit-string tables of a permutation and of its inverse.
    function PermutationData(permTargets : Int[], numBits : Int) : (Bool[][], Bool[][]) {
        mutable data = Repeated(Repeated(false, numBits), Length(permTargets));
        mutable inverseData = data;
        for source in IndexRange(permTargets) {
            let target = permTargets[source];
            set data w/= source <- IntAsBoolArray(target, numBits);
            set inverseData w/= target <- IntAsBoolArray(source, numBits);
        }
        return (data, inverseData);
    }

    /// Applies |i⟩ → |P(i)⟩ with `permTargets[i]` = P(i) and `invPermTargets[j]` = P⁻¹(j) as
    /// little-endian bit strings.
    ///
    /// Looks up P(i) into a fresh register and swaps it with `target`. The old register then
    /// holds P⁻¹(target), which the measurement-based unlookup `Adjoint Select` erases for
    /// O(√N) Toffolis instead of the O(N) of a second coherent lookup.
    operation PermutationViaQROAM(
        permTargets : Bool[][],
        invPermTargets : Bool[][],
        target : Qubit[]
    ) : Unit is Adj {
        let n = Length(target);
        let N = Length(permTargets);
        let nRequired = Ceiling(Lg(IntAsDouble(N)));

        use loaded = Qubit[n];
        Select(permTargets, target[...nRequired - 1], loaded);
        for i in 0..n - 1 {
            SWAP(target[i], loaded[i]);
        }
        Adjoint Select(invPermTargets, target[...nRequired - 1], loaded);
    }

    /// Applies a dense isometry (Appendix B of Rupprecht & Wölk, arXiv:2605.28489) to the
    /// little-endian `input` register and the `added` qubits, which start in |0⟩ for the
    /// isometry's columns.
    ///
    /// `phaseGradient` is prepared by `PreparePhaseGradientState`, its length sets the bits of
    /// every quantized angle, and `angleReg` is a clean register of the same length.
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
        let rotations = Mapped(angles -> QuantizeRyAngles(angles, rotationBitPrecision), synthesis.rotationAngles);
        let mixingGivens = Mapped(givens -> QuantizeGivensDecomposition(givens, inputAddresses, rotationBitPrecision), synthesis.mixingGivens);
        let rightFactorGivens = Mapped(givens -> QuantizeGivensDecomposition(givens, inputAddresses, rotationBitPrecision), synthesis.rightFactorGivens);
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, blockAddresses, rotationBitPrecision);

        // The rotation tables are addressed by the little-endian input register; the Givens
        // operations take most-significant-first registers, hence `Reversed(input)`.
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
            ApplyMultiplexedRy(rotations[0], input, added[0], phaseGradient, angleReg);
        }
        if numAdded == 2 {
            let q0 = added[0];
            let q1 = added[1];
            CNOT(q1, q0);
            ApplyRealUnitaryViaGivens(mixingGivens[0], [q0], Reversed(input), phaseGradient, angleReg);
            ApplyControlledMultiplexedRy(rotations[1], input, [q0], q1, phaseGradient, angleReg);
            CNOT(q1, q0);
            ApplyRealUnitaryViaGivens(mixingGivens[1], [q1], Reversed(input), phaseGradient, angleReg);
            ApplyControlledMultiplexedRy(rotations[2], input, [q1], q0, phaseGradient, angleReg);
        }
        // The joint register Reversed(input + added) selects block q, or block 2·q1 + q0, of B.
        ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(input + added), phaseGradient, angleReg);
    }

    /// Applies a sparse isometry U = P_r · B · P_c to the little-endian register
    /// `input + added`, whose value is the row index of the isometry.
    ///
    /// The split into `input` and `added` only matches the signature of `ApplyDenseIsometry`;
    /// see there for `phaseGradient` and `angleReg`.
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
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, 1 <<< (totalBits - 1), Length(phaseGradient));

        PermutationViaQROAM(colPermData, colInvPermData, target);
        // The Givens layers act on the most-significant-first view of the register.
        ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(target), phaseGradient, angleReg);
        PermutationViaQROAM(rowPermData, rowInvPermData, target);
    }

    /// Applies `isometry(input, added, phaseGradient, angleReg)` to `qs` =
    /// `[input | added | phaseGradient]`, whose last `rotationBitPrecision` qubits are a
    /// caller-owned register prepared by `PreparePhaseGradientState`. The angle register is
    /// allocated here and returned clean.
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

    /// `ApplyIsometryOnRegister` on the target qubits alone, with a phase gradient register
    /// allocated, prepared, and released here.
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

    /// Returns `ApplyDenseIsometry` as an operation on the target register
    /// `[input | added]` that allocates its own phase gradient.
    function MakeDenseIsometryOp(params : DenseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            params.numInputQubits,
            params.numAddedQubits,
            params.rotationBitPrecision,
            ApplyDenseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplyDenseIsometry` as an operation on `[input | added | phaseGradient]`,
    /// whose last `rotationBitPrecision` qubits are a caller-owned register prepared by
    /// `PreparePhaseGradientState` and left in that state.
    function MakeDenseIsometryOpWithPhaseGradient(params : DenseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            params.numInputQubits,
            params.numAddedQubits,
            params.rotationBitPrecision,
            ApplyDenseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a dense isometry (allocates qubits).
    ///
    /// The target register keeps the synthesized output, as in the other `Make*Circuit`
    /// wrappers; only the phase gradient and angle registers are returned to |0⟩.
    operation MakeDenseIsometryCircuit(
        synthesis : DenseIsometrySynthesis,
        numInputQubits : Int,
        numAddedQubits : Int,
        rotationBitPrecision : Int
    ) : Unit {
        use target = Qubit[numInputQubits + numAddedQubits];
        ApplyIsometryOnTarget(numInputQubits, numAddedQubits, rotationBitPrecision, ApplyDenseIsometry(synthesis, _, _, _, _), target);
    }

    /// Returns `ApplySparseIsometry` as an operation on the target register that allocates
    /// its own phase gradient.
    function MakeSparseIsometryOp(params : SparseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnTarget(
            params.numQubits,
            0,
            params.rotationBitPrecision,
            ApplySparseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Returns `ApplySparseIsometry` as an operation on `[target | phaseGradient]`; see
    /// `MakeDenseIsometryOpWithPhaseGradient`.
    function MakeSparseIsometryOpWithPhaseGradient(params : SparseIsometryParams) : Qubit[] => Unit is Adj {
        ApplyIsometryOnRegister(
            params.numQubits,
            0,
            params.rotationBitPrecision,
            ApplySparseIsometry(params.synthesis, _, _, _, _),
            _
        )
    }

    /// Circuit entry point for a sparse isometry (allocates qubits); see
    /// `MakeDenseIsometryCircuit`.
    operation MakeSparseIsometryCircuit(
        synthesis : SparseIsometrySynthesis,
        numQubits : Int,
        rotationBitPrecision : Int
    ) : Unit {
        use target = Qubit[numQubits];
        ApplyIsometryOnTarget(numQubits, 0, rotationBitPrecision, ApplySparseIsometry(synthesis, _, _, _, _), target);
    }
}
