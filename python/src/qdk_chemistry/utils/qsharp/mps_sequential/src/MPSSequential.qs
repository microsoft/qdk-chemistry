// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

import Std.Math.*;
import Std.Convert.*;
import Std.Arrays.*;
import Std.Canon.*;
import Std.Diagnostics.*;
import Std.ResourceEstimation.*;
import Std.TableLookup.Select;
import PhaseGradient.RyViaPhaseGradient;
import PhaseGradient.PreparePhaseGradientState;
import QroamStatePrep.QroamStatePrep;
import GivensDecomposition.*;

export MPSSequential, SiteUnitary, ApplyUCR, ApplyControlledUCR;

// =============================================================================
// Helper operations for the decomposition
// =============================================================================

/// Helper: Apply uniformly controlled Y-rotation (UCR).
///
/// For each ancilla LE value k, applies Ry(angles[k]) to targetQubit.
operation ApplyUCR(
    angles : Double[],
    ancilla : Qubit[],
    targetQubit : Qubit
) : Unit {
    for k in 0..Length(angles) - 1 {
        if AbsD(angles[k]) > 1e-15 {
            within {
                for i in 0..Length(ancilla) - 1 {
                    if ((k >>> i) &&& 1) == 0 {
                        X(ancilla[i]);
                    }
                }
            } apply {
                Controlled Ry(ancilla, (angles[k], targetQubit));
            }
        }
    }
}

/// Helper: Apply uniformly controlled Y-rotation, additionally controlled by controlQubit.
operation ApplyControlledUCR(
    controlQubit : Qubit,
    angles : Double[],
    ancilla : Qubit[],
    targetQubit : Qubit
) : Unit {
    for k in 0..Length(angles) - 1 {
        if AbsD(angles[k]) > 1e-15 {
            within {
                for i in 0..Length(ancilla) - 1 {
                    if ((k >>> i) &&& 1) == 0 {
                        X(ancilla[i]);
                    }
                }
            } apply {
                Controlled Ry([controlQubit] + ancilla, (angles[k], targetQubit));
            }
        }
    }
}

// =============================================================================
// CSD decomposition (Givens + QROAM + phase gradient)
// =============================================================================

/// # Summary
/// Applies a single site unitary using Givens decomposition.
///
/// # Description
/// Uses QROAM + phase gradient rotations. Each matrix (V, W₀, W₁, U) is pre-decomposed
/// into Givens rotation layers. Angle quantization is done internally in Q#.
///
/// The 9-step structure is preserved:
///   1. V on ancilla (via Givens layers)
///   2. UCR Ry on q0 (via QROAM + phase gradient)
///   3. CNOT(q1, q0)
///   4. W₀ on ancilla, controlled by q0 (via controlled Givens layers)
///   5. Controlled UCR on q1, ctrl by q0 (via controlled QROAM + phase gradient)
///   6. CNOT(q1, q0)
///   7. W₁ on ancilla, controlled by q1 (via controlled Givens layers)
///   8. Controlled UCR on q0, ctrl by q1 (via controlled QROAM + phase gradient)
///   9. Multiplexed U on ancilla+site (via block-diagonal Givens layers)
///
/// # Input
/// ## vLayerAngles
/// Double[numLayers][numAngles]: raw Givens rotation angles for V (radians).
/// ## vLayerShifted
/// Bool[numLayers]: whether each V layer is shifted.
/// ## vPhases
/// Bool[dim]: V phase correction (true = Z flip on that basis state).
/// ## rot0Angles
/// Double[dim]: UCR Ry angles for step 2 (standard Ry convention).
/// ## rot1Angles
/// Double[dim]: controlled UCR Ry angles for step 5.
/// ## rot2Angles
/// Double[dim]: controlled UCR Ry angles for step 8.
/// ## w0LayerAngles
/// Double[numLayers][numAngles]: raw Givens angles for W₀.
/// ## w0LayerShifted
/// Bool[numLayers]: whether each W₀ layer is shifted.
/// ## w0Phases
/// Bool[dim]: W₀ phase correction.
/// ## w1LayerAngles
/// Double[numLayers][numAngles]: raw Givens angles for W₁.
/// ## w1LayerShifted
/// Bool[numLayers]: whether each W₁ layer is shifted.
/// ## w1Phases
/// Bool[dim]: W₁ phase correction.
/// ## uLayerAngles
/// Double[numLayers][numAngles]: raw Givens angles for U (block-diagonal).
/// ## uLayerShifted
/// Bool[numLayers]: whether each U layer is shifted.
/// ## uPhases
/// Bool[4*dim]: U phase correction.
/// ## newSite
/// The 2-qubit site register [q0, q1].
/// ## ancilla
/// The ancilla register.
/// ## phaseGradient
/// Phase gradient register (pre-initialized).
operation SiteUnitary(
    vLayerAngles : Double[][],
    vLayerShifted : Bool[],
    vPhases : Bool[],
    rot0Angles : Double[],
    rot1Angles : Double[],
    rot2Angles : Double[],
    w0LayerAngles : Double[][],
    w0LayerShifted : Bool[],
    w0Phases : Bool[],
    w1LayerAngles : Double[][],
    w1LayerShifted : Bool[],
    w1Phases : Bool[],
    uLayerAngles : Double[][],
    uLayerShifted : Bool[],
    uPhases : Bool[],
    newSite : Qubit[],
    ancilla : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    let q0 = newSite[0];
    let q1 = newSite[1];
    let rotationBits = Length(phaseGradient);
    let ancillaDim = 1 <<< Length(ancilla);
    let numAddresses = ancillaDim / 2;

    // Quantize angles (fast classical computation done in Q#)
    let vData = Mapped(layer -> QuantizeGivensAngles(layer, numAddresses, rotationBits), vLayerAngles);
    let vPhaseData = PhaseFlipsAsSelectData(vPhases);
    let rot0Data = QuantizeRyAngles(rot0Angles, rotationBits);
    let rot1Data = QuantizeRyAngles(rot1Angles, rotationBits);
    let rot2Data = QuantizeRyAngles(rot2Angles, rotationBits);
    let w0Data = Mapped(layer -> QuantizeGivensAngles(layer, numAddresses, rotationBits), w0LayerAngles);
    let w0PhaseData = PhaseFlipsAsSelectData(w0Phases);
    let w1Data = Mapped(layer -> QuantizeGivensAngles(layer, numAddresses, rotationBits), w1LayerAngles);
    let w1PhaseData = PhaseFlipsAsSelectData(w1Phases);
    // U address space is (4*dim)/2 = 2*dim
    let uNumAddresses = 2 * ancillaDim;
    let uData = Mapped(layer -> QuantizeGivensAngles(layer, uNumAddresses, rotationBits), uLayerAngles);
    let uPhaseData = PhaseFlipsAsSelectData(uPhases);

    // Single shared angle register for all UCR steps (reused sequentially)
    // (angleReg is now passed in from the caller)

    // Step 1: V on ancilla (gate-based via Givens)
    // Reversed(ancilla) maps ancilla register to target with LSB=ancilla[0] as active qubit
    ApplyRealUnitaryViaGivens(vData, vLayerShifted, vPhaseData, Reversed(ancilla), phaseGradient, angleReg);

    // Step 2: UCR Ry on q0, addressed by ancilla
    within {
        Select(rot0Data, ancilla, angleReg);
    } apply {
        RyViaPhaseGradient(q0, angleReg, phaseGradient);
    }

    // Step 3: CNOT(q1, q0)
    CNOT(q1, q0);

    // Step 4: W₀ on ancilla, controlled by q0
    ApplyControlledRealUnitaryViaGivens(
        w0Data,
        w0LayerShifted,
        w0PhaseData,
        Reversed(ancilla),
        phaseGradient,
        q0,
        angleReg
    );

    // Step 5: Controlled UCR on q1, ctrl by q0
    within {
        Controlled Select([q0], (rot1Data, ancilla, angleReg));
    } apply {
        RyViaPhaseGradient(q1, angleReg, phaseGradient);
    }

    // Step 6: CNOT(q1, q0) — undo
    CNOT(q1, q0);

    // Step 7: W₁ on ancilla, controlled by q1
    ApplyControlledRealUnitaryViaGivens(
        w1Data,
        w1LayerShifted,
        w1PhaseData,
        Reversed(ancilla),
        phaseGradient,
        q1,
        angleReg
    );

    // Step 8: Controlled UCR on q0, ctrl by q1
    within {
        Controlled Select([q1], (rot2Data, ancilla, angleReg));
    } apply {
        RyViaPhaseGradient(q0, angleReg, phaseGradient);
    }

    // Step 9: Multiplexed U on ancilla, selected by (q0, q1)
    // Joint register = [q1, q0] + Reversed(ancilla) so block index = q1*2+q0 (MSB)
    ApplyBlockDiagUnitaryViaGivens(
        uData,
        uLayerShifted,
        uPhaseData,
        Reversed(ancilla),
        [q1, q0],
        phaseGradient,
        angleReg
    );
}

/// # Summary
/// MPS state preparation via CSD decomposition (Appendix B).
///
/// # Description
/// Implements the site unitary decomposition from Berry et al.
/// (PRX Quantum 6, 020327, https://doi.org/10.1103/PRXQuantum.6.020327,
/// Figure 5) using structured quantum gates:
///   V → UCR(rot2) → W₁ → UCR(rot1) → W₀ → UCR(rot0) → U
///
/// # Description
/// Calls SiteUnitary for each site. All matrices are pre-decomposed
/// into Givens rotation layers. Angle quantization is handled internally by Q#.
/// Requires a phase gradient register which is initialized and freed by this operation.
///
// Based on code originally published by Felix Rupprecht (DLR) on Zenodo:
///   https://zenodo.org/records/20393500
/// Rewritten and adapted for integration into the QDK Chemistry library.
///
/// # Input
/// ## initialStateVec
/// Real amplitudes of the initial state.
/// ## numSites
/// Number of MPS sites.
/// ## rotationBits
/// Phase gradient precision (number of bits).
/// ## siteVLayerAngles, siteVLayerShifted, siteVPhases
/// Per-site V Givens decomposition: Double[numSites-1][numLayers][numAngles],
/// Bool[numSites-1][numLayers], Bool[numSites-1][dim].
/// ## siteRot0Angles, siteRot1Angles, siteRot2Angles
/// Per-site UCR angles: Double[numSites-1][dim].
/// ## siteW0LayerAngles, siteW0LayerShifted, siteW0Phases
/// Per-site W₀ Givens decomposition.
/// ## siteW1LayerAngles, siteW1LayerShifted, siteW1Phases
/// Per-site W₁ Givens decomposition.
/// ## siteULayerAngles, siteULayerShifted, siteUPhases
/// Per-site U Givens decomposition.
/// ## state
/// State register.
/// ## ancilla
/// Ancilla register.
operation MPSSequential(
    initialStateVec : Double[],
    numSites : Int,
    rotationBits : Int,
    siteVLayerAngles : Double[][][],
    siteVLayerShifted : Bool[][],
    siteVPhases : Bool[][],
    siteRot0Angles : Double[][],
    siteRot1Angles : Double[][],
    siteRot2Angles : Double[][],
    siteW0LayerAngles : Double[][][],
    siteW0LayerShifted : Bool[][],
    siteW0Phases : Bool[][],
    siteW1LayerAngles : Double[][][],
    siteW1LayerShifted : Bool[][],
    siteW1Phases : Bool[][],
    siteULayerAngles : Double[][][],
    siteULayerShifted : Bool[][],
    siteUPhases : Bool[][],
    state : Qubit[],
    ancilla : Qubit[]
) : Unit {
    // Initialize phase gradient register
    use phaseGradient = Qubit[rotationBits];
    PreparePhaseGradientState(phaseGradient);

    // Single shared angle register for QROM-loaded angles (reused by all operations)
    use angleReg = Qubit[rotationBits];

    // Prepare initial state
    let initReg = ancilla + state[0..1];
    QroamStatePrep(initialStateVec, Reversed(initReg), phaseGradient, angleReg);

    // Apply site unitaries
    // All sites share the same circuit structure (same bond dimension / qubit counts),
    // so we cache the resource estimate from the first site and reuse it for the rest.
    for siteIdx in 0..numSites - 2 {
        let newSite = state[2 * (siteIdx + 1)..2 * (siteIdx + 1) + 1];
        if BeginEstimateCaching("SiteUnitary", SingleVariant()) {
            SiteUnitary(
                siteVLayerAngles[siteIdx],
                siteVLayerShifted[siteIdx],
                siteVPhases[siteIdx],
                siteRot0Angles[siteIdx],
                siteRot1Angles[siteIdx],
                siteRot2Angles[siteIdx],
                siteW0LayerAngles[siteIdx],
                siteW0LayerShifted[siteIdx],
                siteW0Phases[siteIdx],
                siteW1LayerAngles[siteIdx],
                siteW1LayerShifted[siteIdx],
                siteW1Phases[siteIdx],
                siteULayerAngles[siteIdx],
                siteULayerShifted[siteIdx],
                siteUPhases[siteIdx],
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
