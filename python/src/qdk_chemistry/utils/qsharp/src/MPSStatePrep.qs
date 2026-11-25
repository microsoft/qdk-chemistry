// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

/// MPS state preparation one site at a time, with dense site isometries (`MPSSequential`) or
/// block-sparse ones exploiting symmetry (`MPSSparse`)
/// (Berry et al., PRX Quantum 6, 020327; Rupprecht & Wölk, arXiv:2605.28489).
namespace QDKChemistry.Utils.MPSStatePrep {

    import Std.Arrays.All;
    import Std.Arrays.Enumerated;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Sorted;
    import Std.Arrays.Subarray;
    import Std.Diagnostics.Fact;
    import Std.Math.BitSizeI;
    import QDKChemistry.Utils.IsometrySynthesis.ApplyDenseIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.ApplySparseIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.DenseIsometrySynthesis;
    import QDKChemistry.Utils.IsometrySynthesis.SparseIsometrySynthesis;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;

    /// Returns the qubit indices of `orbital` in the blocked Jordan-Wigner layout.
    internal function MPSSiteQubitIndices(numQubits : Int, numOrbitals : Int, orbital : Int) : Int[] {
        MappedOverRange(channel -> channel * numOrbitals + orbital, 0..numQubits / numOrbitals - 1)
    }

    /// Returns the qubits of `orbital`; see `MPSSiteQubitIndices`.
    internal function MPSSiteQubits(state : Qubit[], numOrbitals : Int, orbital : Int) : Qubit[] {
        Subarray(MPSSiteQubitIndices(Length(state), numOrbitals, orbital), state)
    }

    /// Converts MPS chain-order fermionic signs to the blocked Jordan-Wigner convention.
    internal operation ApplyMPSFermionicOrderSigns(siteToOrbitalOrder : Int[], state : Qubit[]) : Unit is Adj {
        let numSites = Length(siteToOrbitalOrder);
        let numQubits = Length(state);
        // Element-wise checks, because QIR generation does not support whole-array comparison.
        if numQubits == 2 * numSites and numSites >= 6 and All((index, orbital) -> index == orbital, Enumerated(siteToOrbitalOrder)) {
            within {
                // α_j holds the parity of α_j, ..., α_{N-1}.
                for j in numSites - 1..-1..2 {
                    CNOT(state[j], state[j - 1]);
                }
            } apply {
                for j in 1..numSites - 1 {
                    CZ(state[numSites + j - 1], state[j]);
                }
            }
        } else {
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
    }

    /// Prepares an MPS one site at a time (Berry et al., PRX Quantum 6, 020327,
    /// https://doi.org/10.1103/PRXQuantum.6.020327).
    internal operation PrepareMPS<'Site>(
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
        Fact(
            Length(siteToOrbitalOrder) == numSites and All((index, orbital) -> index == orbital, Enumerated(Sorted((a, b) -> a <= b, siteToOrbitalOrder))),
            "siteToOrbitalOrder must be a permutation of the MPS sites."
        );
        let qubitsPerSite = Length(state) / numSites;
        let initialBondBits = BitSizeI(Length(initialStateVec) / (1 <<< qubitsPerSite) - 1);
        Fact(
            Length(initialStateVec) == 1 <<< (qubitsPerSite + initialBondBits) and initialBondBits <= Length(ancilla),
            "The initial state must cover the first site and a power-of-two part of the bond register."
        );
        let rotationBitPrecision = Length(phaseGradient);

        let initReg = ancilla[...initialBondBits - 1] + MPSSiteQubits(state, numSites, siteToOrbitalOrder[0]);
        QROMStatePrepare(
            new QROMStatePrepParams {
                amplitudes = initialStateVec,
                rotationBitPrecision = rotationBitPrecision,
                numStateQubits = Length(initReg),
            },
            initReg,
            phaseGradient
        );

        {
            use angleReg = Qubit[rotationBitPrecision];
            for siteIdx in 0..numSites - 2 {
                let newSite = MPSSiteQubits(state, numSites, siteToOrbitalOrder[siteIdx + 1]);
                applySite(siteDecompositions[siteIdx], ancilla, newSite, phaseGradient, angleReg);
            }
        }
        ApplyMPSFermionicOrderSigns(siteToOrbitalOrder, state);
    }

    /// Applies `prepare(state, ancilla, phaseGradient)` to the first `numStateQubits` qubits of
    /// `state`, leaving any later ones untouched, with phase gradient and bond registers allocated
    /// here; see `PrepareMPSOnRegister`.
    internal operation PrepareMPSOnState(
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
            PrepareMPSOnRegister(numStateQubits, numAncillaQubits, rotationBitPrecision, prepare, state[...numStateQubits - 1] + phaseGradient);
        }
    }

    /// Applies `prepare(state, ancilla, phaseGradient)` to `qs` = `[state | phaseGradient]`
    /// with a bond register `ancilla` allocated here.
    ///
    /// The last `rotationBitPrecision` qubits of `qs` are a caller-owned register prepared by
    /// `PreparePhaseGradientState`. Quantized rotations return `ancilla` to |0⟩ only up to
    /// rotation quantization error, so
    /// it is reset before release in both directions. The operation and its adjoint are
    /// therefore inverse only up to that error, which is of the same order as the preparation
    /// error in `state`.
    internal operation PrepareMPSOnRegister(
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
    operation MPSSequential(params : MPSSequentialParams, state : Qubit[], ancilla : Qubit[], phaseGradient : Qubit[]) : Unit is Adj {
        PrepareMPS(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, ApplyDenseIsometry, state, ancilla, phaseGradient);
    }

    /// Returns `MPSSequential` on the state register, with its own phase gradient; see `PrepareMPSOnState`.
    function MakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSequential(params, _, _, _), _)
    }

    /// Returns `MPSSequential` on `[state | phaseGradient]`; see `PrepareMPSOnRegister`.
    function MakeMPSSequentialOpWithPhaseGradient(params : MPSSequentialParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSequential(params, _, _, _), _)
    }

    /// Circuit entry point for `MPSSequential`, leaving the state on the allocated qubits.
    operation MakeMPSSequentialCircuit(params : MPSSequentialParams) : Unit {
        use state = Qubit[params.numQubitsPerSite * params.numSites];
        MakeMPSSequentialOp(params)(state);
    }

    /// Parameters of a composable MPS state preparation with block-sparse site isometries.
    struct MPSSparseParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseIsometrySynthesis[],
    }

    /// `PrepareMPS` with the site isometries applied by `ApplySparseIsometry`.
    operation MPSSparse(params : MPSSparseParams, state : Qubit[], ancilla : Qubit[], phaseGradient : Qubit[]) : Unit is Adj {
        PrepareMPS(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, ApplySparseIsometry, state, ancilla, phaseGradient);
    }

    /// Returns `MPSSparse` on the state register, with its own phase gradient; see `PrepareMPSOnState`.
    function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnState(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSparse(params, _, _, _), _)
    }

    /// Returns `MPSSparse` on `[state | phaseGradient]`; see `PrepareMPSOnRegister`.
    function MakeMPSSparseOpWithPhaseGradient(params : MPSSparseParams) : Qubit[] => Unit is Adj {
        PrepareMPSOnRegister(params.numQubitsPerSite * params.numSites, params.numAncillaQubits, params.rotationBitPrecision, MPSSparse(params, _, _, _), _)
    }

    /// Circuit entry point for `MPSSparse`, leaving the state on the allocated qubits.
    operation MakeMPSSparseCircuit(params : MPSSparseParams) : Unit {
        use state = Qubit[params.numQubitsPerSite * params.numSites];
        MakeMPSSparseOp(params)(state);
    }

}
