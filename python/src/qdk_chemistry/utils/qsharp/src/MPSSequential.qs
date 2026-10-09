// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center (DLR),
// licensed under the Apache License, Version 2.0, and modified for QDK Chemistry.

/// Sequential MPS state preparation with dense site unitaries
/// (Berry et al., PRX Quantum 6, 020327; Rupprecht & Wölk, arXiv:2605.28489).
namespace QDKChemistry.Utils.MPSSequential {

    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Diagnostics.Fact;
    import QDKChemistry.Utils.PhaseGradient.ApplyControlledMultiplexedRy;
    import QDKChemistry.Utils.PhaseGradient.ApplyMultiplexedRy;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.QuantizeRyAngles;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;

    /// Returns the qubit indices of `orbital` in the blocked Jordan-Wigner layout.
    ///
    /// A register of `numOrbitals` qubits holds one spinless mode per orbital (the ('0', '1')
    /// basis). A register of 2·numOrbitals qubits holds all α modes and then all β modes, so
    /// the result [α, β] is little-endian in the ('0', 'u', 'd', '2') physical index.
    function MPSSiteQubitIndices(numQubits : Int, numOrbitals : Int, orbital : Int) : Int[] {
        MappedOverRange(channel -> channel * numOrbitals + orbital, 0..numQubits / numOrbitals - 1)
    }

    /// Returns the qubits of `orbital`; see `MPSSiteQubitIndices`.
    function MPSSiteQubits(state : Qubit[], numOrbitals : Int, orbital : Int) : Qubit[] {
        Subarray(MPSSiteQubitIndices(Length(state), numOrbitals, orbital), state)
    }

    /// Converts MPS chain-order fermionic signs to the blocked Jordan-Wigner convention.
    ///
    /// MPS basis states create the modes of each site in chain order, α before β, while
    /// Jordan-Wigner basis states create them in qubit order. Reordering gives −1 for every
    /// occupied pair of modes whose relative order differs, applied by one CZ per pair.
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

    /// Prepares an MPS one site at a time (Berry et al., PRX Quantum 6, 020327,
    /// https://doi.org/10.1103/PRXQuantum.6.020327).
    ///
    /// `initialStateVec` holds the real amplitudes of the first site and its right bond,
    /// indexed by physical · ancillaDim + bond. `applySite(site, ancilla, newSite,
    /// phaseGradient, angleReg)` applies the unitary of each later site. `state` uses the
    /// layout of `MPSSiteQubitIndices`, and `ancilla` is returned to |0⟩ up to rotation
    /// quantization error. `phaseGradient` is prepared by `PreparePhaseGradientState`, left
    /// in that state, and its length sets the bits of every quantized angle.
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

    /// Returns an operation on the state register that allocates the bond and phase gradient
    /// registers for `prepare(state, ancilla, phaseGradient)`.
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

    /// Returns an operation on `[state | phaseGradient]` whose last rotationBitPrecision qubits
    /// are a caller-owned register prepared by `PreparePhaseGradientState`.
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

    /// Circuit data for one dense MPS site unitary, mirroring the C++
    /// `qdk::chemistry::utils::detail::DenseSiteSynthesis`.
    ///
    /// A ('0', 'u', 'd', '2') site is applied as UCR₀ → CNOT → W₀ → UCR₁ → CNOT → W₁ → UCR₂ → U
    /// (Fig. 5 of Rupprecht & Wölk, arXiv:2605.28489) and a ('0', '1') site as UCR₀ → U
    /// (Eq. 6). Each UCRₖ is an Ry multiplexed by the bond register and U is block diagonal.
    /// The right factor V is absorbed into the preceding site, so the C++ `right_factor` has
    /// no counterpart here.
    struct DenseSiteSynthesis {
        /// Ry angles of UCR₀, UCR₁ and UCR₂, one per bond state; only UCR₀ for ('0', '1').
        rotationAngles : Double[][],
        /// W₀ and W₁ on the bond register; empty for ('0', '1').
        mixingGivens : GivensDecomposition[],
        /// U on the joint bond and site register.
        blockGivens : GivensDecomposition,
    }

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

    /// Applies one dense site unitary (Appendix B of Rupprecht & Wölk, arXiv:2605.28489) to
    /// the little-endian bond register `ancilla` and the site qubits `newSite`.
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

    /// Returns a composable `MPSSequential` operation; see `MakeMPSOp`.
    function MakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit {
        MakeMPSOp(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSequential(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// Returns a composable `MPSSequential` operation; see `MakeMPSOpWithPhaseGradient`.
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
