// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

namespace QDKChemistry.Utils.MPSSequential {

    import Std.Arrays.*;
    import Std.Diagnostics.Fact;
    import Std.ResourceEstimation.*;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;
    import GivensDecomposition.*;

    export SequentialSiteDecomposition, MPSSequentialParams, MPSSequential, MakeMPSSequentialOp, MakeMPSSequentialCircuit, MakeMPSSequentialPlaceholderCircuit, MPSSiteQubits, ApplyMPSFermionicOrderSigns;

    /// # Summary
    /// Returns the qubit indices of one orbital in the blocked Jordan-Wigner layout.
    ///
    /// # Description
    /// A register of `numOrbitals` qubits holds one spinless mode per orbital, which is the
    /// ('0', '1') physical basis. A register of 2·numOrbitals qubits holds the α modes of
    /// all orbitals and then their β modes, so the α mode of `orbital` is qubit `orbital` and
    /// its β mode is qubit `numOrbitals + orbital`. The returned modes are little-endian in
    /// the physical index of the ('0', 'u', 'd', '2') basis.
    function MPSSiteQubitIndices(numQubits : Int, numOrbitals : Int, orbital : Int) : Int[] {
        MappedOverRange(channel -> channel * numOrbitals + orbital, 0..numQubits / numOrbitals - 1)
    }

    /// # Summary
    /// Returns the qubits of one orbital in the blocked Jordan-Wigner layout.
    ///
    /// # Description
    /// See `MPSSiteQubitIndices`.
    function MPSSiteQubits(state : Qubit[], numOrbitals : Int, orbital : Int) : Qubit[] {
        Subarray(MPSSiteQubitIndices(Length(state), numOrbitals, orbital), state)
    }

    /// # Summary
    /// Converts MPS chain-order fermionic signs to the blocked Jordan-Wigner convention.
    ///
    /// # Description
    /// An MPS basis state creates the modes of each site in chain order, the α mode before
    /// the β mode for spatial orbitals. A blocked Jordan-Wigner basis state creates its modes
    /// in qubit order. Reordering the creation operators contributes −1 for every occupied
    /// pair of modes whose relative order differs, which one CZ gate per such pair applies.
    ///
    /// # Input
    /// ## siteToOrbitalOrder
    /// Orbital that holds each chain site.
    /// ## state
    /// Blocked Jordan-Wigner register with one or two modes per orbital.
    operation ApplyMPSFermionicOrderSigns(siteToOrbitalOrder : Int[], state : Qubit[]) : Unit {
        let numSites = Length(siteToOrbitalOrder);
        let numQubits = Length(state);
        for first in 0..numSites - 1 {
            let firstModes = MPSSiteQubitIndices(numQubits, numSites, siteToOrbitalOrder[first]);
            for second in first + 1..numSites - 1 {
                // The modes of an earlier site are created before those of a later site.
                for firstMode in firstModes {
                    for secondMode in MPSSiteQubitIndices(numQubits, numSites, siteToOrbitalOrder[second]) {
                        if firstMode > secondMode {
                            CZ(state[firstMode], state[secondMode]);
                        }
                    }
                }
            }
        }
    }

    /// # Summary
    /// Cosine-sine decomposition of one dense MPS site unitary.
    ///
    /// # Description
    /// A site with the ('0', 'u', 'd', '2') physical basis is applied as
    ///   UCR(rot0) → CNOT → W₀ → UCR(rot1) → CNOT → W₁ → UCR(rot2) → U
    /// (Fig. 5 of Rupprecht & Wölk, arXiv:2605.28489). A site with the ('0', '1') physical
    /// basis is applied as UCR(rot0) → U, and its rot1, rot2, W₀ and W₁ fields are empty.
    /// Each orthogonal factor is stored as Givens rotation layers. The right factor V of the
    /// decomposition is absorbed into the preceding site or the initial state, so it never
    /// appears in the circuit.
    ///
    /// # Input
    /// ## rot0Angles, rot1Angles, rot2Angles
    /// Double[ancillaDim]: Ry angles of the uniformly controlled rotations.
    /// ## w0LayerAngles, w0LayerShifted, w0Phases
    /// Givens layers and sign corrections of W₀ (ancillaDim × ancillaDim).
    /// ## w1LayerAngles, w1LayerShifted, w1Phases
    /// Givens layers and sign corrections of W₁ (ancillaDim × ancillaDim).
    /// ## uLayerAngles, uLayerShifted, uPhases
    /// Merged Givens layers and sign corrections of the block-diagonal U (d·ancillaDim for
    /// d physical states).
    struct SequentialSiteDecomposition {
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
    }

    /// # Summary
    /// Parameters of a composable MPS sequential state preparation.
    struct MPSSequentialParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        numAncillaQubits : Int,
        siteDecompositions : SequentialSiteDecomposition[],
    }

    /// # Summary
    /// Quantized table data for one site unitary, as consumed by `PrepareSequentialMPS`.
    struct QuantizedSiteUnitary {
        rot0 : Bool[][],
        rot1 : Bool[][],
        rot2 : Bool[][],
        w0Layers : Bool[][][],
        w0Shifted : Bool[],
        w0Phases : Bool[][],
        w1Layers : Bool[][][],
        w1Shifted : Bool[],
        w1Phases : Bool[][],
        uLayers : Bool[][][],
        uShifted : Bool[],
        uPhases : Bool[][],
    }

    /// # Summary
    /// Quantizes the rotation angles of one site decomposition for table lookup.
    function QuantizeSiteUnitary(
        decomp : SequentialSiteDecomposition,
        numQubitsPerSite : Int,
        ancillaBits : Int,
        rotationBits : Int
    ) : QuantizedSiteUnitary {
        let ancillaDim = 1 <<< ancillaBits;
        // A Givens layer on an n-qubit register is addressed by its n - 1 upper qubits.
        let wAddresses = ancillaDim / 2;
        let uAddresses = (ancillaDim <<< numQubitsPerSite) / 2;
        new QuantizedSiteUnitary {
            rot0 = QuantizeRyAngles(decomp.rot0Angles, rotationBits),
            rot1 = QuantizeRyAngles(decomp.rot1Angles, rotationBits),
            rot2 = QuantizeRyAngles(decomp.rot2Angles, rotationBits),
            w0Layers = Mapped(layer -> QuantizeGivensAngles(layer, wAddresses, rotationBits), decomp.w0LayerAngles),
            w0Shifted = decomp.w0LayerShifted,
            w0Phases = PhaseFlipsAsSelectData(decomp.w0Phases),
            w1Layers = Mapped(layer -> QuantizeGivensAngles(layer, wAddresses, rotationBits), decomp.w1LayerAngles),
            w1Shifted = decomp.w1LayerShifted,
            w1Phases = PhaseFlipsAsSelectData(decomp.w1Phases),
            uLayers = Mapped(layer -> QuantizeGivensAngles(layer, uAddresses, rotationBits), decomp.uLayerAngles),
            uShifted = decomp.uLayerShifted,
            uPhases = PhaseFlipsAsSelectData(decomp.uPhases),
        }
    }

    /// # Summary
    /// Prepares the first site with QROM state preparation, then applies every site unitary.
    ///
    /// # Input
    /// ## siteData
    /// Returns the quantized data of the unitary for site `siteIdx + 1`. It is evaluated
    /// lazily, so cached site unitaries are never quantized during resource estimation.
    /// ## cacheSites
    /// Whether every site shares one circuit structure, so the resource estimator may trace
    /// only the first site unitary.
    operation PrepareSequentialMPS(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        siteData : Int -> QuantizedSiteUnitary,
        cacheSites : Bool,
        state : Qubit[],
        ancilla : Qubit[]
    ) : Unit {
        use phaseGradient = Qubit[rotationBits];
        PreparePhaseGradientState(phaseGradient);
        use angleReg = Qubit[rotationBits];

        // `initReg` is little-endian with the bond in its low qubits, so amplitude index
        // physical · ancillaDim + bond is exactly the register value it prepares.
        let initReg = ancilla + MPSSiteQubits(state, numSites, siteToOrbitalOrder[0]);
        QROMStatePrepare(
            new QROMStatePrepParams {
                amplitudes = initialStateVec,
                rotationBitPrecision = rotationBits,
                numStateQubits = Length(initReg),
            },
            initReg,
            phaseGradient
        );

        for siteIdx in 0..numSites - 2 {
            let newSite = MPSSiteQubits(state, numSites, siteToOrbitalOrder[siteIdx + 1]);
            if not cacheSites or BeginEstimateCaching("MPSSequential.SiteUnitary", SingleVariant()) {
                let site = siteData(siteIdx);
                // `newSite` is [q] for the ('0', '1') physical basis or [q0, q1] for the
                // ('0', 'u', 'd', '2') basis. The rotation tables are addressed by the
                // little-endian bond register; the Givens operations take
                // most-significant-first registers, hence `Reversed(ancilla)`.
                if Length(newSite) == 1 {
                    ApplyMultiplexedRy(site.rot0, ancilla, newSite[0], phaseGradient, angleReg);
                    // The joint register newSite + Reversed(ancilla) selects block q of U.
                    ApplyRealUnitaryViaGivens(site.uLayers, site.uShifted, site.uPhases, newSite + Reversed(ancilla), phaseGradient, angleReg);
                } else {
                    let q0 = newSite[0];
                    let q1 = newSite[1];
                    ApplyMultiplexedRy(site.rot0, ancilla, q0, phaseGradient, angleReg);
                    CNOT(q1, q0);
                    ApplyControlledRealUnitaryViaGivens(site.w0Layers, site.w0Shifted, site.w0Phases, Reversed(ancilla), phaseGradient, q0, angleReg);
                    ApplyControlledMultiplexedRy(site.rot1, ancilla, q0, q1, phaseGradient, angleReg);
                    CNOT(q1, q0);
                    ApplyControlledRealUnitaryViaGivens(site.w1Layers, site.w1Shifted, site.w1Phases, Reversed(ancilla), phaseGradient, q1, angleReg);
                    ApplyControlledMultiplexedRy(site.rot2, ancilla, q1, q0, phaseGradient, angleReg);
                    // The joint register [q1, q0] + Reversed(ancilla) selects block 2·q1 + q0 of U.
                    ApplyRealUnitaryViaGivens(site.uLayers, site.uShifted, site.uPhases, [q1, q0] + Reversed(ancilla), phaseGradient, angleReg);
                }
                if cacheSites {
                    EndEstimateCaching();
                }
            }
        }
        ApplyMPSFermionicOrderSigns(siteToOrbitalOrder, state);

        Adjoint PreparePhaseGradientState(phaseGradient);
    }

    /// # Summary
    /// MPS state preparation with dense sequential site unitaries.
    ///
    /// # Description
    /// Implements the sequential preparation of Berry et al. (PRX Quantum 6, 020327,
    /// https://doi.org/10.1103/PRXQuantum.6.020327) with the site unitary decomposition of
    /// Appendix B of Rupprecht & Wölk (arXiv:2605.28489). Every orthogonal factor is
    /// synthesized from Givens rotation layers with QROM-loaded angles and phase-gradient
    /// rotations.
    ///
    /// Based on code originally published by Felix Rupprecht (DLR) on Zenodo:
    ///   https://zenodo.org/records/20393500
    /// Rewritten and adapted for integration into the QDK Chemistry library.
    ///
    /// # Input
    /// ## initialStateVec
    /// Real amplitudes of the first site and its right bond, indexed by
    /// physical · ancillaDim + bond.
    /// ## numSites
    /// Number of MPS sites.
    /// ## siteToOrbitalOrder
    /// Orbital that holds each chain site.
    /// ## rotationBits
    /// Phase gradient precision (number of bits).
    /// ## siteDecompositions
    /// Decompositions of the site unitaries for sites 1..numSites-1.
    /// ## state
    /// Jordan-Wigner register with one mode per orbital for the ('0', '1') physical basis,
    /// or the blocked layout of the α modes of all orbitals followed by their β modes for
    /// the ('0', 'u', 'd', '2') basis. MPS basis states create the modes of each site in
    /// chain order, α before β; the circuit applies the fermionic signs of reordering them
    /// into qubit order.
    /// ## ancilla
    /// Bond register, returned to |0⟩ up to rotation quantization error.
    operation MPSSequential(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        siteDecompositions : SequentialSiteDecomposition[],
        state : Qubit[],
        ancilla : Qubit[]
    ) : Unit {
        Fact(Length(siteDecompositions) == numSites - 1, "MPS sequential preparation needs one decomposition per site after the first.");
        let ancillaBits = Length(ancilla);
        // Sites with the ('0', '1') physical basis use one qubit each, and sites with the
        // ('0', 'u', 'd', '2') physical basis use two.
        Fact(
            Length(state) == numSites or Length(state) == 2 * numSites,
            "The state register must hold one or two qubits per MPS site."
        );
        let numQubitsPerSite = Length(state) / numSites;
        PrepareSequentialMPS(
            initialStateVec,
            numSites,
            siteToOrbitalOrder,
            rotationBits,
            siteIdx -> QuantizeSiteUnitary(siteDecompositions[siteIdx], numQubitsPerSite, ancillaBits, rotationBits),
            false,
            state,
            ancilla
        );
    }

    operation ApplyMPSSequential(params : MPSSequentialParams, state : Qubit[]) : Unit {
        Fact(
            Length(state) == params.numQubitsPerSite * params.numSites,
            "State register size must equal the number of qubits per MPS site times the number of sites."
        );
        use ancilla = Qubit[params.numAncillaQubits];
        MPSSequential(
            params.initialStateVec,
            params.numSites,
            params.siteToOrbitalOrder,
            params.rotationBits,
            params.siteDecompositions,
            state,
            ancilla
        );
    }

    /// # Summary
    /// Returns a composable operation that prepares the MPS on a numQubitsPerSite·numSites-qubit
    /// register.
    function MakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit {
        ApplyMPSSequential(params, _)
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation MakeMPSSequentialCircuit(
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        numAncillaQubits : Int,
        siteDecompositions : SequentialSiteDecomposition[]
    ) : Unit {
        use state = Qubit[numQubitsPerSite * numSites];
        use ancilla = Qubit[numAncillaQubits];
        MPSSequential(
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

    /// # Summary
    /// Resource-estimation circuit with placeholder site data.
    ///
    /// # Description
    /// Every site unitary of the sequential preparation acts on the full ancilla register,
    /// so its cost depends only on `numAncillaQubits` and `rotationBits`. This circuit
    /// replaces the classical decompositions by placeholder data of the same structure and
    /// traces one site unitary, which makes estimates feasible for large bond dimensions.
    /// It does not prepare the target state.
    ///
    /// Table lookups and phase-gradient rotations cost the same for any angle values, so the
    /// placeholder reproduces the Givens layer and rotation structure of a full Clements
    /// decomposition without computing one. Sign corrections are omitted, which slightly
    /// underestimates the cost of real data. With one qubit per site, block unitaries of
    /// sites whose bonds are smaller than the ancilla register need fewer layers, so the
    /// placeholder overestimates them.
    operation MakeMPSSequentialPlaceholderCircuit(
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        numAncillaQubits : Int
    ) : Unit {
        use state = Qubit[numQubitsPerSite * numSites];
        use ancilla = Qubit[numAncillaQubits];
        let ancillaDim = 1 <<< numAncillaQubits;
        let numLayers = ancillaDim == 2 ? 1 | ancillaDim;
        let shifted = MappedOverRange(layer -> layer % 2 == 1, 0..numLayers - 1);
        let angle = 0.1;
        let rot = QuantizeRyAngles(Repeated(angle, ancillaDim), rotationBits);
        let hasMixing = numQubitsPerSite == 2;
        let wLayers = hasMixing ? Repeated(QuantizeGivensAngles([angle], ancillaDim / 2, rotationBits), numLayers) | [];
        let wShifted = hasMixing ? shifted | [];
        let placeholder = new QuantizedSiteUnitary {
            rot0 = rot,
            rot1 = hasMixing ? rot | [],
            rot2 = hasMixing ? rot | [],
            w0Layers = wLayers,
            w0Shifted = wShifted,
            w0Phases = [],
            w1Layers = wLayers,
            w1Shifted = wShifted,
            w1Phases = [],
            uLayers = Repeated(QuantizeGivensAngles([angle], (ancillaDim <<< numQubitsPerSite) / 2, rotationBits), numLayers),
            uShifted = shifted,
            uPhases = [],
        };
        PrepareSequentialMPS(
            initialStateVec,
            numSites,
            siteToOrbitalOrder,
            rotationBits,
            siteIdx -> placeholder,
            true,
            state,
            ancilla
        );
        ResetAll(state + ancilla);
    }
}
