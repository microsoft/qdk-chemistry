// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

namespace QDKChemistry.Utils.MPSSparse {

    import Std.Math.*;
    import Std.Convert.*;
    import Std.Arrays.*;
    import Std.Canon.*;
    import Std.Diagnostics.*;
    import Std.ResourceEstimation.*;
    import Std.Measurement.*;
    import Std.TableLookup.Select;
    import PhaseGradient.RyViaPhaseGradient;
    import PhaseGradient.PreparePhaseGradientState;
    import QroamStatePrep.QroamStatePrep;
    import GivensDecomposition.*;

    export MPSSparse, MakeMPSSparseCircuit, SparseSiteUnitary, PermutationViaQROAM, SparseUnitaryDecomposition;

    /// # Summary
    /// Decomposition data for a single sparse MPS site unitary
    /// U = P_row · V_blockdiag · P_col.
    ///
    /// # Input
    /// ## colPermTargets
    /// Int[N]: column permutation targets.
    /// ## rowPermTargets
    /// Int[N]: row permutation targets.
    /// ## blockLayerAngles
    /// Double[numLayers][numAngles]: Givens angles for block-diagonal V.
    /// ## blockLayerShifted
    /// Bool[numLayers]: whether each Givens layer is shifted.
    /// ## blockPhases
    /// Bool[dim]: phase corrections for block-diagonal V.
    struct SparseUnitaryDecomposition {
        colPermTargets : Int[],
        rowPermTargets : Int[],
        blockLayerAngles : Double[][],
        blockLayerShifted : Bool[],
        blockPhases : Bool[],
    }

    // =============================================================================
    // Permutation via QROAM
    // =============================================================================

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

    /// # Summary
    /// Applies a permutation |i> -> |P(i)> using coherent table lookup and SWAP.
    ///
    /// # Description
    /// Implements the permutation by:
    ///   1. Loading P(address) into a fresh register via table lookup
    ///   2. SWAPping the target register with the loaded register
    ///   3. Uncomputing the old register with the inverse permutation
    ///
    /// # Input
    /// ## permTargets
    /// Bool[N][m]: The permutation targets encoded as bit strings.
    ///   permTargets[i] = binary encoding of P(i).
    /// ## invPermTargets
    /// Bool[N][m]: The inverse permutation targets encoded as bit strings.
    ///   invPermTargets[j] = binary encoding of P^{-1}(j).
    /// ## target
    /// The target register to be permuted.
    operation PermutationViaQROAM(
        permTargets : Bool[][],
        invPermTargets : Bool[][],
        target : Qubit[]
    ) : Unit {
        let n = Length(target);
        let N = Length(permTargets);
        let nRequired = Ceiling(Lg(IntAsDouble(N)));

        // Step 1: Load P(address) into a fresh register.
        use loaded = Qubit[n];
        Select(permTargets, target[...nRequired - 1], loaded);

        // Step 2: SWAP target <-> loaded
        for i in 0..n - 1 {
            SWAP(target[i], loaded[i]);
        }

        // Step 3: Uncompute loaded via XOR with inverse permutation.
        // After SWAP: target = P(i), loaded = i = invPermTargets[P(i)].
        // XOR invPermTargets[target] into loaded: loaded = i ⊕ i = 0.
        Select(invPermTargets, target[...nRequired - 1], loaded);
    }

    // =============================================================================
    // Sparse Site Unitary
    // =============================================================================

    /// # Summary
    /// Applies one sparse site unitary: P_col -> V_blockdiag -> P_row.
    ///
    /// # Input
    /// ## decomp
    /// The SparseUnitaryDecomposition for this site.
    /// ## newSite
    /// The 2-qubit new site register.
    /// ## ancilla
    /// The ancilla register.
    /// ## phaseGradient
    /// Phase gradient register.
    /// ## angleReg
    /// Reusable angle register for QROAM rotations.
    operation SparseSiteUnitary(
        decomp : SparseUnitaryDecomposition,
        newSite : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        // Merge site + ancilla into single target register
        let target = newSite + ancilla;
        let totalBits = Length(target);
        let numAddresses = 1 <<< (totalBits - 1);
        let (colPermData, colInvPermData) = PermutationData(decomp.colPermTargets, totalBits);
        let (rowPermData, rowInvPermData) = PermutationData(decomp.rowPermTargets, totalBits);

        // Quantize Givens data
        let rotationBits = Length(phaseGradient);
        let blockData = Mapped(
            layer -> QuantizeGivensAngles(layer, numAddresses, rotationBits),
            decomp.blockLayerAngles
        );
        let blockPhaseData = PhaseFlipsAsSelectData(decomp.blockPhases);

        // Step 1: Apply column permutation
        PermutationViaQROAM(colPermData, colInvPermData, target);

        // Step 2: Apply block-diagonal unitary via Givens layers
        // Use Reversed(newSite) + Reversed(ancilla) to get MSB-first ordering
        // that matches the target matrix row convention: row = physical * ancilla_dim + ancilla.
        // Note: Reversed(target) would give [anc_msb, ..., site_lsb] = ancilla*d + physical (wrong).
        ApplyRealUnitaryViaGivens(
            blockData,
            decomp.blockLayerShifted,
            blockPhaseData,
            Reversed(newSite) + Reversed(ancilla),
            phaseGradient,
            angleReg
        );

        // Step 3: Apply row permutation
        PermutationViaQROAM(rowPermData, rowInvPermData, target);
    }

    // =============================================================================
    // Full MPS Sparse preparation
    // =============================================================================

    /// # Summary
    /// MPS state preparation exploiting block sparsity.
    ///
    /// Each site unitary is decomposed as U = P_row · V_blockdiag · P_col
    /// where P_row, P_col are permutations (via QROAM + SWAP + X-measure)
    /// and V_blockdiag is block-diagonal (via Givens rotation layers).
    ///
    /// # Description
    /// Prepares an MPS by:
    ///   1. Preparing the initial state (first site) via QROAM state prep
    ///   2. Applying sparse site unitaries for sites 1..N-1
    ///
    /// References:
    ///   Rupprecht & Woelk (2026). Faster matrix product state preparation by
    ///   exploiting symmetry-induced block-sparsity. arXiv:2605.28489.
    operation MPSSparse(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        siteDecompositions : SparseUnitaryDecomposition[],
        state : Qubit[],
        ancilla : Qubit[]
    ) : Unit {
        // Initialize phase gradient register
        use phaseGradient = Qubit[rotationBits];
        PreparePhaseGradientState(phaseGradient);

        // Single shared angle register
        use angleReg = Qubit[rotationBits];

        // Prepare initial state
        let firstOrbital = siteToOrbitalOrder[0];
        let initReg = ancilla + state[2 * firstOrbital..2 * firstOrbital + 1];
        QroamStatePrep(initialStateVec, Reversed(initReg), phaseGradient, angleReg);

        // Apply sparse site unitaries
        for siteIdx in 0..numSites - 2 {
            let orbital = siteToOrbitalOrder[siteIdx + 1];
            let newSite = state[2 * orbital..2 * orbital + 1];
            SparseSiteUnitary(
                siteDecompositions[siteIdx],
                newSite,
                ancilla,
                phaseGradient,
                angleReg
            );
        }

        // Undo phase gradient state
        Adjoint PreparePhaseGradientState(phaseGradient);
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation MakeMPSSparseCircuit(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseUnitaryDecomposition[]
    ) : Unit {
        use state = Qubit[2 * numSites];
        use ancilla = Qubit[numAncillaQubits];
        MPSSparse(
            initialStateVec,
            numSites,
            siteToOrbitalOrder,
            rotationBits,
            siteDecompositions,
            state,
            ancilla
        );
        ResetAll(state + ancilla);
    }

}
