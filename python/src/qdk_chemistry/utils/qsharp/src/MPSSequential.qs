// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center (DLR),
// licensed under the Apache License, Version 2.0, and modified for QDK Chemistry.

namespace QDKChemistry.Utils.MPSSequential {

    import Std.Arrays.*;
    import Std.Diagnostics.Fact;
    import QDKChemistry.Utils.PhaseGradient.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.PhaseGradient.ApplyMultiplexedRy;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;
    import QDKChemistry.Utils.UnitarySynthesis.*;

    export DenseSiteSynthesis, MPSSequentialParams, MPSSequential, MakeMPSSequentialOp, MakeMPSSequentialOpWithPhaseGradient, MakeMPSSequentialCircuit, PrepareMPS, MakeMPSOp, MakeMPSOpWithPhaseGradient, PrepareMPSCircuit;

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
    /// Prepares an MPS one site at a time, applying each site unitary with `applySite`.
    ///
    /// # Description
    /// Implements the sequential preparation of Berry et al. (PRX Quantum 6, 020327,
    /// https://doi.org/10.1103/PRXQuantum.6.020327): prepares the first site and its right
    /// bond with `QROMStatePrepare`, applies the site unitaries of sites 1..numSites-1, and
    /// converts the chain-order fermionic signs with `ApplyMPSFermionicOrderSigns`.
    ///
    /// # Input
    /// ## initialStateVec
    /// Real amplitudes of the first site and its right bond, indexed by
    /// physical · ancillaDim + bond.
    /// ## numSites
    /// Number of MPS sites.
    /// ## siteToOrbitalOrder
    /// Orbital that holds each chain site.
    /// ## siteDecompositions
    /// Decompositions of the site unitaries for sites 1..numSites-1.
    /// ## applySite
    /// Applies one site unitary given its decomposition, the little-endian bond register, the
    /// qubits of the new site, the phase gradient register, and a clean angle register.
    /// ## state
    /// Jordan-Wigner register with one mode per orbital for the ('0', '1') physical basis,
    /// or the blocked layout of the α modes of all orbitals followed by their β modes for
    /// the ('0', 'u', 'd', '2') basis. MPS basis states create the modes of each site in
    /// chain order, α before β; the circuit applies the fermionic signs of reordering them
    /// into qubit order.
    /// ## ancilla
    /// Bond register, returned to |0⟩ up to rotation quantization error.
    /// ## phaseGradient
    /// Register prepared by `PreparePhaseGradientState` and left in that state. Its length
    /// is the number of bits of every quantized rotation angle.
    operation PrepareMPS<'Site>(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : 'Site[],
        applySite : ('Site, Qubit[], Qubit[], Qubit[], Qubit[]) => Unit,
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit {
        Fact(Length(siteDecompositions) == numSites - 1, "MPS preparation needs one decomposition per site after the first.");
        // Sites with the ('0', '1') physical basis use one qubit each, and sites with the
        // ('0', 'u', 'd', '2') physical basis use two.
        Fact(
            Length(state) == numSites or Length(state) == 2 * numSites,
            "The state register must hold one or two qubits per MPS site."
        );
        let rotationBitPrecision = Length(phaseGradient);
        use angleReg = Qubit[rotationBitPrecision];

        // `initReg` is little-endian with the bond in its low qubits, so amplitude index
        // physical · ancillaDim + bond is exactly the register value it prepares.
        let initReg = ancilla + MPSSiteQubits(state, numSites, siteToOrbitalOrder[0]);
        QROMStatePrepare(
            new QROMStatePrepParams {
                amplitudes = initialStateVec,
                rotationBitPrecision = rotationBitPrecision,
                numStateQubits = Length(initReg),
            },
            initReg,
            phaseGradient
        );

        for siteIdx in 0..numSites - 2 {
            let newSite = MPSSiteQubits(state, numSites, siteToOrbitalOrder[siteIdx + 1]);
            applySite(siteDecompositions[siteIdx], ancilla, newSite, phaseGradient, angleReg);
        }
        ApplyMPSFermionicOrderSigns(siteToOrbitalOrder, state);
    }

    /// # Summary
    /// Returns an operation on a numStateQubits-qubit state register that allocates the bond
    /// register and its own phase gradient register for `prepare(state, ancilla, phaseGradient)`.
    function MakeMPSOp(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit
    ) : Qubit[] => Unit {
        (state) => {
            Fact(
                Length(state) == numStateQubits,
                "State register size must equal the number of qubits per MPS site times the number of sites."
            );
            use ancilla = Qubit[numAncillaQubits];
            use phaseGradient = Qubit[rotationBitPrecision];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                prepare(state, ancilla, phaseGradient);
            }
        }
    }

    /// # Summary
    /// Returns an operation on `[state | phaseGradient]` whose last rotationBitPrecision qubits are a
    /// caller-owned register prepared by `PreparePhaseGradientState`.
    function MakeMPSOpWithPhaseGradient(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit
    ) : Qubit[] => Unit {
        (qs) => {
            Fact(
                Length(qs) == numStateQubits + rotationBitPrecision,
                "The register must hold the MPS state followed by the phase gradient."
            );
            use ancilla = Qubit[numAncillaQubits];
            prepare(qs[0..numStateQubits - 1], ancilla, qs[numStateQubits...]);
        }
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation PrepareMPSCircuit(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit
    ) : Unit {
        use state = Qubit[numStateQubits];
        use ancilla = Qubit[numAncillaQubits];
        use phaseGradient = Qubit[rotationBitPrecision];
        within {
            PreparePhaseGradientState(phaseGradient);
        } apply {
            prepare(state, ancilla, phaseGradient);
        }
        ResetAll(state + ancilla);
    }

    /// # Summary
    /// Circuit data for one dense MPS site unitary, mirroring the C++
    /// `qdk::chemistry::utils::detail::DenseSiteSynthesis`.
    ///
    /// # Description
    /// A site with the ('0', 'u', 'd', '2') physical basis is applied as
    ///   UCR₀ → CNOT → W₀ → UCR₁ → CNOT → W₁ → UCR₂ → U
    /// (Fig. 5 of Rupprecht & Wölk, arXiv:2605.28489). A site with the ('0', '1') physical
    /// basis is applied as UCR₀ → U (Eq. 6 of the same reference). Each UCRₖ is a uniformly
    /// controlled Ry rotation addressed by the bond register, and U is block diagonal. The
    /// right factor V of the decomposition is absorbed into the preceding site or the initial
    /// state, so the C++ `right_factor` has no counterpart here.
    ///
    /// # Input
    /// ## rotationAngles
    /// Double[numRotations][ancillaDim]: Ry angles of UCR₀, UCR₁ and UCR₂ for four physical
    /// states, or of UCR₀ for two.
    /// ## mixingGivens
    /// Givens data of W₀ and W₁ (ancillaDim × ancillaDim); empty for two physical states.
    /// ## blockGivens
    /// Merged Givens data of the block-diagonal U (d·ancillaDim for d physical states).
    struct DenseSiteSynthesis {
        rotationAngles : Double[][],
        mixingGivens : GivensDecomposition[],
        blockGivens : GivensDecomposition,
    }

    /// # Summary
    /// Parameters of a composable MPS sequential state preparation.
    struct MPSSequentialParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : DenseSiteSynthesis[],
    }

    /// # Summary
    /// Applies one dense site unitary.
    ///
    /// # Description
    /// Applies the site unitary decomposition of Appendix B of Rupprecht & Wölk
    /// (arXiv:2605.28489), see `DenseSiteSynthesis`. Every orthogonal factor is synthesized
    /// from Givens rotation layers with QROM-loaded angles and phase-gradient rotations.
    ///
    /// # Input
    /// ## synthesis
    /// Decomposition of the site unitary.
    /// ## ancilla
    /// Little-endian bond register.
    /// ## newSite
    /// [q] for the ('0', '1') physical basis or [q0, q1] for the ('0', 'u', 'd', '2') basis.
    /// ## phaseGradient
    /// Phase gradient register.
    /// ## angleReg
    /// Clean register that receives each loaded angle.
    operation ApplyDenseSite(
        synthesis : DenseSiteSynthesis,
        ancilla : Qubit[],
        newSite : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        let rotationBitPrecision = Length(phaseGradient);
        let ancillaDim = 1 <<< Length(ancilla);
        // A Givens layer on an n-qubit register is addressed by its n - 1 upper qubits.
        let wAddresses = ancillaDim / 2;
        let uAddresses = (ancillaDim <<< Length(newSite)) / 2;
        let rotations = Mapped(angles -> QuantizeRyAngles(angles, rotationBitPrecision), synthesis.rotationAngles);
        let mixingGivens = Mapped(givens -> QuantizeGivensDecomposition(givens, wAddresses, rotationBitPrecision), synthesis.mixingGivens);
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, uAddresses, rotationBitPrecision);
        // The rotation tables are addressed by the little-endian bond register; the Givens
        // operations take most-significant-first registers, hence `Reversed(ancilla)`.
        ApplyMultiplexedRy(rotations[0], ancilla, newSite[0], phaseGradient, angleReg);
        if Length(newSite) == 2 {
            let q0 = newSite[0];
            let q1 = newSite[1];
            CNOT(q1, q0);
            ApplyRealUnitaryViaGivens(mixingGivens[0], [q0], Reversed(ancilla), phaseGradient, angleReg);
            ApplyControlledMultiplexedRy(rotations[1], ancilla, [q0], q1, phaseGradient, angleReg);
            CNOT(q1, q0);
            ApplyRealUnitaryViaGivens(mixingGivens[1], [q1], Reversed(ancilla), phaseGradient, angleReg);
            ApplyControlledMultiplexedRy(rotations[2], ancilla, [q1], q0, phaseGradient, angleReg);
        }
        // The joint register Reversed(ancilla + newSite) selects block q, or block
        // 2·q1 + q0, of U.
        ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(ancilla + newSite), phaseGradient, angleReg);
    }

    /// # Summary
    /// MPS state preparation with dense sequential site unitaries.
    ///
    /// # Description
    /// `PrepareMPS` with the site unitaries applied by `ApplyDenseSite`.
    operation MPSSequential(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : DenseSiteSynthesis[],
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit {
        PrepareMPS(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, ApplyDenseSite, state, ancilla, phaseGradient);
    }

    /// # Summary
    /// Returns a composable operation that prepares the MPS on a numQubitsPerSite·numSites-qubit
    /// register, allocating and preparing its own phase gradient register.
    function MakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit {
        MakeMPSOp(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSequential(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// # Summary
    /// Returns a composable operation on `[state | phaseGradient]` whose last rotationBitPrecision
    /// qubits are a caller-owned register prepared by `PreparePhaseGradientState`.
    function MakeMPSSequentialOpWithPhaseGradient(params : MPSSequentialParams) : Qubit[] => Unit {
        MakeMPSOpWithPhaseGradient(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSequential(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation MakeMPSSequentialCircuit(
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : DenseSiteSynthesis[]
    ) : Unit {
        PrepareMPSCircuit(
            numQubitsPerSite * numSites,
            numAncillaQubits,
            rotationBitPrecision,
            MPSSequential(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, _, _, _)
        );
    }

}
