// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

/// MPS state preparation exploiting block sparsity.
///
/// Each site unitary is decomposed as U = P_row · V_blockdiag · P_col
/// where P_row, P_col are permutations (via QROAM + SWAP + X-measure)
/// and V_blockdiag is block-diagonal (via Givens rotation layers).
///
/// References:
///   Rupprecht & Woelk (2026). Faster matrix product state preparation by
///   exploiting symmetry-induced block-sparsity. arXiv:2605.28489.

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Canon.*;
import Std.Diagnostics.*;
import Std.ResourceEstimation.*;
import Std.Measurement.*;
import Std.TableLookup.Select;
import QDKChemistry.Utils.SelectSwap.SelectSwap;
import PhaseGradient.RyViaPhaseGradient;
import PhaseGradient.PreparePhaseGradientState;
import QroamStatePrep.QroamStatePrep;
import GivensDecomposition.*;

export MPSSparse, MakeMPSSparseCircuit, SparseSiteUnitary, PermutationViaQROAM;

// =============================================================================
// Permutation via QROAM
// =============================================================================

/// # Summary
/// Applies a permutation |i> -> |P(i)> using QROAM + SWAP + X-measurement.
///
/// # Description
/// Implements the permutation by:
///   1. Loading P(address) into a fresh register via QROAM (SelectSwap)
///   2. SWAPping the target register with the loaded register
///   3. X-basis measuring the old register (measurement-based uncomputation)
///   4. Applying precomputed sign fix-ups via controlled-Z
///
/// The X-measurement uncomputes the loaded register but may introduce known
/// sign flips. These are corrected by the signFixes array.
///
/// # Input
/// ## permTargets
/// Bool[N][m]: The permutation targets encoded as bit strings.
///   permTargets[i] = binary encoding of P(i).
/// ## signFixes
/// Bool[N][1]: sign fix-up data for each basis state.
/// ## target
/// The target register to be permuted.
operation PermutationViaQROAM(
    permTargets : Bool[][],
    signFixes : Bool[][],
    target : Qubit[]
) : Unit {
    let n = Length(target);
    let N = Length(permTargets);
    let nRequired = Ceiling(Lg(IntAsDouble(N)));

    // Step 1: Load P(address) via QROAM into fresh register
    use loaded = Qubit[n];
    SelectSwap(-1, permTargets, target[...nRequired - 1], loaded);

    // Step 2: SWAP target <-> loaded
    for i in 0..n - 1 {
        SWAP(target[i], loaded[i]);
    }

    // Step 3: X-basis measurement of old register (uncomputation)
    ApplyToEachA(H, loaded);
    let _ = MResetEachZ(loaded);

    // Step 4: Sign fix-up
    // Apply Z corrections based on precomputed sign pattern.
    // Model as Select lookup for resource estimation cost.
    if Length(signFixes) > 0 and Length(signFixes[0]) > 0 {
        use signReg = Qubit[1];
        Select(signFixes, target[...nRequired - 1], signReg);
        CZ(signReg[0], target[0]);
        Adjoint Select(signFixes, target[...nRequired - 1], signReg);
    }
}

// =============================================================================
// Sparse Site Unitary
// =============================================================================

/// # Summary
/// Applies one sparse site unitary: P_col -> V_blockdiag -> P_row.
///
/// # Input
/// ## colPermTargets
/// Bool[N][m]: column permutation targets as bit strings.
/// ## rowPermTargets
/// Bool[N][m]: row permutation targets as bit strings.
/// ## blockLayerAngles
/// Double[numLayers][numAngles]: Givens angles for block-diagonal V.
/// ## blockLayerShifted
/// Bool[numLayers]: whether each Givens layer is shifted.
/// ## blockPhases
/// Bool[dim]: phase corrections for block-diagonal V.
/// ## signFixes
/// Bool[N][1]: sign fix-ups for permutations.
/// ## newSite
/// The 2-qubit new site register.
/// ## ancilla
/// The ancilla register.
/// ## phaseGradient
/// Phase gradient register.
/// ## angleReg
/// Reusable angle register for QROAM rotations.
operation SparseSiteUnitary(
    colPermTargets : Bool[][],
    rowPermTargets : Bool[][],
    blockLayerAngles : Double[][],
    blockLayerShifted : Bool[],
    blockPhases : Bool[],
    signFixes : Bool[][],
    newSite : Qubit[],
    ancilla : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    // Merge site + ancilla into single target register
    let target = newSite + ancilla;
    let totalBits = Length(target);
    let numAddresses = 1 <<< (totalBits - 1);

    // Quantize Givens data
    let rotationBits = Length(phaseGradient);
    let blockData = Mapped(
        layer -> QuantizeGivensAngles(layer, numAddresses, rotationBits),
        blockLayerAngles
    );
    let blockPhaseData = PhaseFlipsAsSelectData(blockPhases);

    // Step 1: Apply column permutation
    PermutationViaQROAM(colPermTargets, signFixes, target);

    // Step 2: Apply block-diagonal unitary via Givens layers
    ApplyRealUnitaryViaGivens(
        blockData,
        blockLayerShifted,
        blockPhaseData,
        Reversed(target),
        phaseGradient,
        angleReg
    );

    // Step 3: Apply row permutation
    PermutationViaQROAM(rowPermTargets, signFixes, target);
}

// =============================================================================
// Full MPS Sparse preparation
// =============================================================================

/// # Summary
/// MPS state preparation via block-sparsity exploitation.
///
/// # Description
/// Prepares an MPS by:
///   1. Preparing the initial state (first site) via QROAM state prep
///   2. Applying sparse site unitaries for sites 1..N-1
///
/// Each sparse site unitary is decomposed as P_col -> V_blockdiag -> P_row,
/// exploiting U(1) symmetries of the MPS tensors.
operation MPSSparse(
    initialStateVec : Double[],
    numSites : Int,
    rotationBits : Int,
    siteColPermTargets : Bool[][][],
    siteRowPermTargets : Bool[][][],
    siteBlockLayerAngles : Double[][][],
    siteBlockLayerShifted : Bool[][],
    siteBlockPhases : Bool[][],
    siteSignFixes : Bool[][][],
    state : Qubit[],
    ancilla : Qubit[]
) : Unit {
    // Initialize phase gradient register
    use phaseGradient = Qubit[rotationBits];
    PreparePhaseGradientState(phaseGradient);

    // Single shared angle register
    use angleReg = Qubit[rotationBits];

    // Prepare initial state
    let initReg = ancilla + state[0..1];
    QroamStatePrep(initialStateVec, Reversed(initReg), phaseGradient, angleReg);

    // Apply sparse site unitaries
    for siteIdx in 0..numSites - 2 {
        let newSite = state[2 * (siteIdx + 1)..2 * (siteIdx + 1) + 1];
        if BeginEstimateCaching("SparseSiteUnitary", SingleVariant()) {
            SparseSiteUnitary(
                siteColPermTargets[siteIdx],
                siteRowPermTargets[siteIdx],
                siteBlockLayerAngles[siteIdx],
                siteBlockLayerShifted[siteIdx],
                siteBlockPhases[siteIdx],
                siteSignFixes[siteIdx],
                newSite,
                ancilla,
                phaseGradient,
                angleReg
            );
            EndEstimateCaching();
        }
    }

    // Undo phase gradient state
    Adjoint PreparePhaseGradientState(phaseGradient);
}

/// Circuit wrapper for resource estimation - allocates qubits internally.
operation MakeMPSSparseCircuit(
    initialStateVec : Double[],
    numSites : Int,
    rotationBits : Int,
    numAncillaQubits : Int,
    siteColPermTargets : Bool[][][],
    siteRowPermTargets : Bool[][][],
    siteBlockLayerAngles : Double[][][],
    siteBlockLayerShifted : Bool[][],
    siteBlockPhases : Bool[][],
    siteSignFixes : Bool[][][]
) : Unit {
    use state = Qubit[2 * numSites];
    use ancilla = Qubit[numAncillaQubits];
    MPSSparse(
        initialStateVec,
        numSites,
        rotationBits,
        siteColPermTargets,
        siteRowPermTargets,
        siteBlockLayerAngles,
        siteBlockLayerShifted,
        siteBlockPhases,
        siteSignFixes,
        state,
        ancilla
    );
}
