// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// --------------------------------------------------------------------------------------------

// Shape-only logical resource estimation for dense sequential MPS state preparation
// (Berry et al., PRX Quantum 6, 020327; site-unitary CSD of Rupprecht & Woelk, App. B).
//
// The right bond lives on an ancilla register of `numAncillaQubits` qubits; each site s >= 1
// applies a site unitary on (new physical site, ancilla). For d = 4:
//   UCR(rot0) -> CNOT -> ctrl-W0 -> ctrl-UCR(rot1) -> CNOT -> ctrl-W1 -> ctrl-UCR(rot2) -> blockdiag(U0..U3)
// and for d = 2:
//   UCR(rot) -> blockdiag(U0, U1)
// with every orthogonal factor applied as Clements Givens layers (one QROAM-loaded rotation
// table per layer). Non-Clifford cost is data independent up to the number of Givens layers,
// so the angle tables are pseudo-random placeholders shared between layers.

import Std.Arrays.*;
import Std.ResourceEstimation.*;
import QDKChemistry.Utils.SelectSwap.ControlledQroamCleanRotation;
import QDKChemistry.Utils.SelectSwap.QroamCleanRotation;
import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
import QroamStatePrep.QroamStatePrep;
import GivensDecomposition.ApplyControlledRealUnitaryViaGivens;
import GivensDecomposition.ApplyBlockDiagUnitaryViaGivens;

/// Deterministic pseudo-random bit pattern used as placeholder rotation data.
function PseudoRandomBits(seed : Int, n : Int) : Bool[] {
    mutable x = (seed * 1103515245 + 12345) % 2147483648;
    mutable out : Bool[] = [];
    for _ in 1..n {
        set x = (x * 1103515245 + 12345) % 2147483648;
        set out += [((x >>> 16) &&& 1) == 1];
    }
    out
}

/// Placeholder QROAM rotation table with `numEntries` rows of `rotationBits` bits.
function DummyRotData(numEntries : Int, rotationBits : Int, seed : Int) : Bool[][] {
    let row = PseudoRandomBits(seed + 7, rotationBits);
    let rowAlt = PseudoRandomBits(seed + 11, rotationBits);
    MappedOverRange(k -> k % 2 == 0 ? row | rowAlt, 0..numEntries - 1)
}

/// Clements schedule of `numLayers` Givens layers (alternating unshifted / shifted), each
/// loading a full padded table of `numAddresses` angles. Layer data is shared.
function DummyGivensLayers(numLayers : Int, numAddresses : Int, rotationBits : Int) : (Bool[][][], Bool[]) {
    let even = DummyRotData(numAddresses, rotationBits, 1);
    let odd = DummyRotData(numAddresses, rotationBits, 2);
    let data = MappedOverRange(i -> i % 2 == 0 ? even | odd, 0..numLayers - 1);
    let shifted = MappedOverRange(i -> i % 2 == 1, 0..numLayers - 1);
    (data, shifted)
}

/// One sequential-MPS site unitary acting on `ancilla` (2^Length(ancilla) bond states) and
/// `newSite` (1 qubit for d = 2, 2 qubits for d = 4), with `layersW` Givens layers per W
/// factor (d = 4 only) and `layersU` Givens layers for the block-diagonal U factor.
/// Diagonal +-1 fix-ups after the Givens networks are omitted (gauge-absorbable).
operation SiteUnitary(
    physDim : Int,
    rotationBits : Int,
    layersW : Int,
    layersU : Int,
    newSite : Qubit[],
    ancilla : Qubit[],
    phaseGradient : Qubit[],
    angleReg : Qubit[]
) : Unit {
    let dim = 1 <<< Length(ancilla);
    let rot = DummyRotData(dim, rotationBits, 3);
    let (uData, uShifted) = DummyGivensLayers(layersU, (physDim * dim) / 2, rotationBits);
    if physDim == 2 {
        QroamCleanRotation(rot, ancilla, newSite[0], phaseGradient);
        ApplyBlockDiagUnitaryViaGivens(uData, uShifted, [], Reversed(ancilla), [newSite[0]], phaseGradient, angleReg);
    } else {
        let (wData, wShifted) = DummyGivensLayers(layersW, dim / 2, rotationBits);
        let q0 = newSite[0];
        let q1 = newSite[1];
        QroamCleanRotation(rot, ancilla, q0, phaseGradient);
        CNOT(q1, q0);
        ApplyControlledRealUnitaryViaGivens(wData, wShifted, [], Reversed(ancilla), phaseGradient, q0, angleReg);
        ControlledQroamCleanRotation(rot, ancilla, q0, q1, phaseGradient);
        CNOT(q1, q0);
        ApplyControlledRealUnitaryViaGivens(wData, wShifted, [], Reversed(ancilla), phaseGradient, q1, angleReg);
        ControlledQroamCleanRotation(rot, ancilla, q1, q0, phaseGradient);
        ApplyBlockDiagUnitaryViaGivens(uData, uShifted, [], Reversed(ancilla), [q1, q0], phaseGradient, angleReg);
    }
}

/// Sequential MPS state preparation (shape only) on `numSites` sites. Site s + 1 addresses
/// `siteAncillaBits[s]` ancilla qubits with `siteLayersW[s]` / `siteLayersU[s]` Givens layers;
/// pass `numSites - 1` entries for the full program. Identical sites are traced once.
operation MPSSequentialStatePrep(
    physDim : Int,
    numSites : Int,
    rotationBits : Int,
    numAncillaQubits : Int,
    initialStateVec : Double[],
    siteAncillaBits : Int[],
    siteLayersW : Int[],
    siteLayersU : Int[]
) : Unit {
    let qPerSite = physDim == 4 ? 2 | 1;
    use state = Qubit[qPerSite * numSites];
    use ancilla = Qubit[numAncillaQubits];
    use phaseGradient = Qubit[rotationBits];
    PreparePhaseGradientState(phaseGradient);
    use angleReg = Qubit[rotationBits];

    QroamStatePrep(initialStateVec, Reversed(ancilla + state[0..qPerSite - 1]), phaseGradient, angleReg);
    for s in IndexRange(siteAncillaBits) {
        let b = siteAncillaBits[s];
        let newSite = state[qPerSite * (s + 1)..qPerSite * (s + 2) - 1];
        let variant = ((physDim * 64 + b) * 1000000 + siteLayersU[s]) * 1000000 + siteLayersW[s];
        if BeginEstimateCaching("SiteUnitary", variant) {
            SiteUnitary(physDim, rotationBits, siteLayersW[s], siteLayersU[s], newSite, ancilla[0..b - 1], phaseGradient, angleReg);
            EndEstimateCaching();
        }
    }
    Adjoint PreparePhaseGradientState(phaseGradient);
}

/// A single site segment of `MPSSequentialStatePrep` with all program registers allocated.
operation SiteSegment(
    physDim : Int,
    numSites : Int,
    rotationBits : Int,
    numAncillaQubits : Int,
    siteAncillaBits : Int,
    layersW : Int,
    layersU : Int
) : Unit {
    let qPerSite = physDim == 4 ? 2 | 1;
    use state = Qubit[qPerSite * numSites];
    use ancilla = Qubit[numAncillaQubits];
    use phaseGradient = Qubit[rotationBits];
    use angleReg = Qubit[rotationBits];
    SiteUnitary(physDim, rotationBits, layersW, layersU, state[qPerSite..2 * qPerSite - 1], ancilla[0..siteAncillaBits - 1], phaseGradient, angleReg);
}
