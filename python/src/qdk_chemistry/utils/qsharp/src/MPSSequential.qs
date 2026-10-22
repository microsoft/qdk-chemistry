// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center (DLR),
// licensed under the Apache License, Version 2.0, and modified for QDK Chemistry.

/// Sequential MPS state preparation with dense site isometries
/// (Berry et al., PRX Quantum 6, 020327; Rupprecht & Wölk, arXiv:2605.28489).
namespace QDKChemistry.Utils.MPSSequential {

    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Subarray;
    import Std.Diagnostics.Fact;
    import QDKChemistry.Utils.IsometrySynthesis.ApplyDenseIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.DenseIsometrySynthesis;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;

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
    operation ApplyMPSFermionicOrderSigns(siteToOrbitalOrder : Int[], state : Qubit[]) : Unit is Adj {
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
    /// phaseGradient, angleReg)` applies the isometry of each later site with the bond
    /// register as its input and the site qubits as its added qubits. `state` uses the
    /// layout of `MPSSiteQubitIndices`, and `ancilla` is returned to |0⟩ up to rotation
    /// quantization error. `phaseGradient` is prepared by `PreparePhaseGradientState`, left
    /// in that state, and its length sets the bits of every quantized angle.
    operation PrepareMPS<'Site>(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : 'Site[],
        applySite : ('Site, Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit is Adj {
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
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj
    ) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(numStateQubits, numAncillaQubits, rotationBitPrecision, prepare, _)
    }

    /// Returns an operation on `[state | phaseGradient]` whose last rotationBitPrecision qubits
    /// are a caller-owned register prepared by `PreparePhaseGradientState`.
    function MakeMPSOpWithPhaseGradient(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj
    ) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(numStateQubits, numAncillaQubits, rotationBitPrecision, prepare, _)
    }

    /// Applies `prepare(state, ancilla, phaseGradient)` to `state` with phase gradient and bond
    /// registers allocated here; see `PrepareMPSOnRegister`.
    operation PrepareMPSOnState(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        state : Qubit[]
    ) : Unit is Adj {
        use phaseGradient = Qubit[rotationBitPrecision];
        within {
            PreparePhaseGradientState(phaseGradient);
        } apply {
            PrepareMPSOnRegister(numStateQubits, numAncillaQubits, rotationBitPrecision, prepare, state + phaseGradient);
        }
    }

    /// Applies `prepare(state, ancilla, phaseGradient)` to `qs` = `[state | phaseGradient]`
    /// with a bond register `ancilla` allocated here.
    ///
    /// Quantized rotations return `ancilla` to |0⟩ only up to rotation quantization error, so
    /// it is reset before release in both directions. The operation and its adjoint are
    /// therefore inverse only up to that error, which is of the same order as the preparation
    /// error in `state`.
    operation PrepareMPSOnRegister(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        qs : Qubit[]
    ) : Unit is Adj {
        body (...) {
            Fact(
                Length(qs) == numStateQubits + rotationBitPrecision,
                "The register must hold the MPS state followed by the phase gradient."
            );
            use ancilla = Qubit[numAncillaQubits];
            prepare(qs[...numStateQubits - 1], ancilla, qs[numStateQubits...]);
            ResetAll(ancilla);
        }
        adjoint (...) {
            Fact(
                Length(qs) == numStateQubits + rotationBitPrecision,
                "The register must hold the MPS state followed by the phase gradient."
            );
            use ancilla = Qubit[numAncillaQubits];
            Adjoint prepare(qs[...numStateQubits - 1], ancilla, qs[numStateQubits...]);
            ResetAll(ancilla);
        }
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation PrepareMPSCircuit(
        numStateQubits : Int,
        numAncillaQubits : Int,
        rotationBitPrecision : Int,
        prepare : (Qubit[], Qubit[], Qubit[]) => Unit is Adj
    ) : Unit {
        use state = Qubit[numStateQubits];
        PrepareMPSOnState(numStateQubits, numAncillaQubits, rotationBitPrecision, prepare, state);
        ResetAll(state);
    }

    /// Parameters of a composable MPS sequential state preparation with dense site isometries.
    struct MPSSequentialParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : DenseIsometrySynthesis[],
    }

    /// `PrepareMPS` with the site isometries applied by `ApplyDenseIsometry`.
    operation MPSSequential(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : DenseIsometrySynthesis[],
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit is Adj {
        PrepareMPS(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, ApplyDenseIsometry, state, ancilla, phaseGradient);
    }

    /// Returns a composable `MPSSequential` operation; see `MakeMPSOp`.
    function MakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit is Adj {
        MakeMPSOp(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSequential(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// Returns a composable `MPSSequential` operation; see `MakeMPSOpWithPhaseGradient`.
    function MakeMPSSequentialOpWithPhaseGradient(params : MPSSequentialParams) : Qubit[] => Unit is Adj {
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
        siteDecompositions : DenseIsometrySynthesis[]
    ) : Unit {
        PrepareMPSCircuit(
            numQubitsPerSite * numSites,
            numAncillaQubits,
            rotationBitPrecision,
            MPSSequential(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, _, _, _)
        );
    }

}
