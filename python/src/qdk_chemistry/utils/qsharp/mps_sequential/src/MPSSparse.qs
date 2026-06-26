// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

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

export MPSSparse, MakeMPSSparseCircuit, MakeMPSSparseOp, ApplyMPSSparse, MPSSparseParams, SparseSiteUnitary, PermutationViaQROAM;

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

    // Step 1: Load P(address) via QROAM into fresh register
    use loaded = Qubit[n];
    SelectSwap(-1, permTargets, target[...nRequired - 1], loaded);

    // Step 2: SWAP target <-> loaded
    for i in 0..n - 1 {
        SWAP(target[i], loaded[i]);
    }

    // Step 3: Uncompute loaded via XOR with inverse permutation.
    // After SWAP: target = P(i), loaded = i = invPermTargets[P(i)].
    // XOR invPermTargets[target] into loaded: loaded = i ⊕ i = 0.
    // SelectSwap internally uses within/apply with measurement-based cleanup.
    SelectSwap(-1, invPermTargets, target[...nRequired - 1], loaded);
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
/// ## colInvPermTargets
/// Bool[N][m]: inverse column permutation targets as bit strings.
/// ## rowPermTargets
/// Bool[N][m]: row permutation targets as bit strings.
/// ## rowInvPermTargets
/// Bool[N][m]: inverse row permutation targets as bit strings.
/// ## blockLayerAngles
/// Double[numLayers][numAngles]: Givens angles for block-diagonal V.
/// ## blockLayerShifted
/// Bool[numLayers]: whether each Givens layer is shifted.
/// ## blockPhases
/// Bool[dim]: phase corrections for block-diagonal V.
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
    colInvPermTargets : Bool[][],
    rowPermTargets : Bool[][],
    rowInvPermTargets : Bool[][],
    blockLayerAngles : Double[][],
    blockLayerShifted : Bool[],
    blockPhases : Bool[],
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
    PermutationViaQROAM(colPermTargets, colInvPermTargets, target);

    // Step 2: Apply block-diagonal unitary via Givens layers
    // Use Reversed(newSite) + Reversed(ancilla) to get MSB-first ordering
    // that matches the target matrix row convention: row = physical * ancilla_dim + ancilla.
    // Note: Reversed(target) would give [anc_msb, ..., site_lsb] = ancilla*d + physical (wrong).
    ApplyRealUnitaryViaGivens(
        blockData,
        blockLayerShifted,
        blockPhaseData,
        Reversed(newSite) + Reversed(ancilla),
        phaseGradient,
        angleReg
    );

    // Step 3: Apply row permutation
    PermutationViaQROAM(rowPermTargets, rowInvPermTargets, target);
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
    rotationBits : Int,
    siteColPermTargets : Bool[][][],
    siteColInvPermTargets : Bool[][][],
    siteRowPermTargets : Bool[][][],
    siteRowInvPermTargets : Bool[][][],
    siteBlockLayerAngles : Double[][][],
    siteBlockLayerShifted : Bool[][],
    siteBlockPhases : Bool[][],
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
                siteColInvPermTargets[siteIdx],
                siteRowPermTargets[siteIdx],
                siteRowInvPermTargets[siteIdx],
                siteBlockLayerAngles[siteIdx],
                siteBlockLayerShifted[siteIdx],
                siteBlockPhases[siteIdx],
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
    siteColInvPermTargets : Bool[][][],
    siteRowPermTargets : Bool[][][],
    siteRowInvPermTargets : Bool[][][],
    siteBlockLayerAngles : Double[][][],
    siteBlockLayerShifted : Bool[][],
    siteBlockPhases : Bool[][]
) : Unit {
    use state = Qubit[2 * numSites];
    use ancilla = Qubit[numAncillaQubits];
    MPSSparse(
        initialStateVec,
        numSites,
        rotationBits,
        siteColPermTargets,
        siteColInvPermTargets,
        siteRowPermTargets,
        siteRowInvPermTargets,
        siteBlockLayerAngles,
        siteBlockLayerShifted,
        siteBlockPhases,
        state,
        ancilla
    );
}

/// Parameters struct for MPS sparse state preparation.
struct MPSSparseParams {
    initialStateVec : Double[],
    numSites : Int,
    rotationBits : Int,
    numAncillaQubits : Int,
    siteColPermTargets : Bool[][][],
    siteColInvPermTargets : Bool[][][],
    siteRowPermTargets : Bool[][][],
    siteRowInvPermTargets : Bool[][][],
    siteBlockLayerAngles : Double[][][],
    siteBlockLayerShifted : Bool[][],
    siteBlockPhases : Bool[][],
}

/// Applies MPS sparse state preparation on the system qubit array.
/// Ancilla qubits are allocated internally (they start and end in |0⟩).
operation ApplyMPSSparse(
    params : MPSSparseParams,
    qubits : Qubit[]
) : Unit {
    use ancilla = Qubit[params.numAncillaQubits];
    MPSSparse(
        params.initialStateVec,
        params.numSites,
        params.rotationBits,
        params.siteColPermTargets,
        params.siteColInvPermTargets,
        params.siteRowPermTargets,
        params.siteRowInvPermTargets,
        params.siteBlockLayerAngles,
        params.siteBlockLayerShifted,
        params.siteBlockPhases,
        qubits,
        ancilla
    );
}

/// Returns a Qubit[] => Unit callable for MPS sparse state preparation.
function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit {
    ApplyMPSSparse(params, _)
}
