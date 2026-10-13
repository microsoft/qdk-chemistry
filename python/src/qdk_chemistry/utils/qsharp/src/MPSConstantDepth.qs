// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

/// Constant-depth MPS state preparation with measurements and feed-forward (Protocol 1 of
/// Smith et al., PRX Quantum 5, 030344, https://doi.org/10.1103/PRXQuantum.5.030344).
///
/// Fragments of consecutive sites are prepared in parallel with the sequential site unitaries
/// of `MPSSequential`, adjacent bond registers are fused by Bell measurements, and the
/// measured Pauli defects are corrected by one layer of fragment unitaries chosen from the
/// measurement record. The operations measure qubits and are therefore not adjointable.
namespace QDKChemistry.Utils.MPSConstantDepth {

    import Std.Arrays.Chunks;
    import Std.Arrays.Flattened;
    import Std.Arrays.Mapped;
    import Std.Arrays.Reversed;
    import Std.Arrays.Zipped;
    import Std.Convert.ResultArrayAsInt;
    import Std.Diagnostics.Fact;
    import Std.Measurement.MResetEachZ;
    import QDKChemistry.Utils.MPSSequential.ApplyDenseSite;
    import QDKChemistry.Utils.MPSSequential.ApplyMPSFermionicOrderSigns;
    import QDKChemistry.Utils.MPSSequential.DenseSiteSynthesis;
    import QDKChemistry.Utils.MPSSequential.MPSSiteQubits;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;

    /// Preparation and correction data of one fragment of consecutive MPS sites.
    ///
    /// A Pauli defect on the left bond is labelled x | (z <<< k) for X^x Z^z on the k fusion
    /// qubits, little-endian. The defect with label g is removed by `corrections[g]` on the
    /// fragment's physical qubits and leaves the defect `pushedLabels[g]` on its right bond.
    struct MPSFragment {
        /// Real amplitudes on `bond + fusion`, followed by the qubits of the first site for the
        /// first fragment, with the bond in the low qubits.
        initialStateVec : Double[],
        /// Chain sites of the fragment, in order.
        sites : Int[],
        /// Qubits of the left bond fused with the previous fragment; zero for the first one.
        numFusionQubits : Int,
        /// Site unitaries after the initial state: every site, or every site after the first
        /// for the first fragment.
        siteDecompositions : DenseSiteSynthesis[],
        /// Physical correction of each incoming defect; empty for the first fragment.
        corrections : GivensDecomposition[],
        /// Outgoing defect of each incoming defect; empty for the first fragment.
        pushedLabels : Int[],
    }

    /// Parameters of a composable constant-depth MPS preparation.
    struct MPSConstantDepthParams {
        fragments : MPSFragment[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numBondQubits : Int,
    }

    /// Returns the little-endian physical register of `sites`, first site least significant.
    function FragmentQubits(state : Qubit[], numSites : Int, siteToOrbitalOrder : Int[], sites : Int[]) : Qubit[] {
        Flattened(Mapped(site -> MPSSiteQubits(state, numSites, siteToOrbitalOrder[site]), sites))
    }

    /// Prepares one fragment: the initial bond state, which for a later fragment is a Bell pair
    /// of its left bond `fusion` and the bond register, followed by its site unitaries.
    ///
    /// The last site unitary keeps the raw right bond in `bond`, so it can be fused with the
    /// next fragment.
    operation PrepareFragment(
        fragment : MPSFragment,
        isFirst : Bool,
        numSites : Int,
        siteToOrbitalOrder : Int[],
        state : Qubit[],
        bond : Qubit[],
        fusion : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        let numInitialSites = isFirst ? 1 | 0;
        Fact(
            Length(fragment.siteDecompositions) == Length(fragment.sites) - numInitialSites,
            "A fragment needs one decomposition per site after its initial state."
        );
        let initialSites = fragment.sites[...numInitialSites - 1];
        let initReg = bond + fusion + FragmentQubits(state, numSites, siteToOrbitalOrder, initialSites);
        QROMStatePrepare(
            new QROMStatePrepParams {
                amplitudes = fragment.initialStateVec,
                rotationBitPrecision = Length(phaseGradient),
                numStateQubits = Length(initReg),
            },
            initReg,
            phaseGradient
        );
        for (site, decomposition) in Zipped(fragment.sites[numInitialSites...], fragment.siteDecompositions) {
            let siteQubits = MPSSiteQubits(state, numSites, siteToOrbitalOrder[site]);
            ApplyDenseSite(decomposition, bond, siteQubits, phaseGradient, angleReg);
        }
    }

    /// Measures each qubit pair (left[j], right[j]) in the Bell basis and returns the label
    /// x | (z <<< k) of the inserted defect X^x Z^z. Both registers are reset.
    operation FuseBonds(left : Qubit[], right : Qubit[]) : Int {
        let numQubits = Length(right);
        for (leftQubit, rightQubit) in Zipped(left, right) {
            CNOT(leftQubit, rightQubit);
            H(leftQubit);
        }
        let z = ResultArrayAsInt(MResetEachZ(left));
        let x = ResultArrayAsInt(MResetEachZ(right));
        x ||| (z <<< numQubits)
    }

    /// Removes the incoming defect `incoming` from a fragment with its physical correction and
    /// returns the outgoing defect on its right bond.
    ///
    /// The defect is only known at run time, so every correction is compiled into its own
    /// branch. A correction without Givens layers or signs applies no gates.
    operation CorrectFragment(
        fragment : MPSFragment,
        incoming : Int,
        physical : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Int {
        let numLabels = Length(fragment.pushedLabels);
        Fact(Length(fragment.corrections) == numLabels, "A fragment needs one correction per incoming defect.");
        let numAddresses = (1 <<< Length(physical)) / 2;
        let corrections = Mapped(
            givens -> QuantizeGivensDecomposition(givens, numAddresses, Length(phaseGradient)),
            fragment.corrections
        );
        mutable outgoing = 0;
        for label in 1..numLabels - 1 {
            if incoming == label {
                ApplyRealUnitaryViaGivens(corrections[label], [], Reversed(physical), phaseGradient, angleReg);
                set outgoing = fragment.pushedLabels[label];
            }
        }
        outgoing
    }

    /// Prepares an MPS in depth independent of the number of fragments (Protocol 1 of Smith et
    /// al., PRX Quantum 5, 030344, https://doi.org/10.1103/PRXQuantum.5.030344).
    ///
    /// Fragment f is prepared with `phaseGradients[f]`; distinct registers keep the fragments
    /// parallel, and a shared register gives the same state. Defects are pushed toward the last
    /// fragment, whose right bond is trivial and absorbs them. `state` uses the layout of
    /// `MPSSiteQubits`. Program order interleaves the fragments to bound simulation memory;
    /// each correction depends on earlier fragments only through classical measurement data.
    operation PrepareMPSConstantDepth(
        fragments : MPSFragment[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        numBondQubits : Int,
        state : Qubit[],
        phaseGradients : Qubit[][]
    ) : Unit {
        let numFragments = Length(fragments);
        Fact(numFragments > 0, "MPS preparation needs at least one fragment.");
        Fact(Length(phaseGradients) == numFragments, "Each fragment needs a phase gradient register.");
        Fact(
            Length(state) == numSites or Length(state) == 2 * numSites,
            "The state register must hold one or two qubits per MPS site."
        );
        let rotationBitPrecision = Length(phaseGradients[0]);
        // Every fragment has its own registers, so no qubit is shared between fragments.
        use bondQubits = Qubit[numFragments * numBondQubits];
        use fusionQubits = Qubit[(numFragments - 1) * numBondQubits];
        use angleQubits = Qubit[numFragments * rotationBitPrecision];
        let bonds = Chunks(numBondQubits, bondQubits);
        let fusions = Chunks(numBondQubits, fusionQubits);
        let angleRegs = Chunks(rotationBitPrecision, angleQubits);

        PrepareFragment(fragments[0], true, numSites, siteToOrbitalOrder, state, bonds[0], [], phaseGradients[0], angleRegs[0]);
        mutable carried = 0;
        for f in 1..numFragments - 1 {
            let fragment = fragments[f];
            let numFusionQubits = fragment.numFusionQubits;
            Fact(numFusionQubits <= numBondQubits, "A fused bond must fit in the bond register.");
            let fusion = fusions[f - 1][...numFusionQubits - 1];
            PrepareFragment(fragment, false, numSites, siteToOrbitalOrder, state, bonds[f], fusion, phaseGradients[f], angleRegs[f]);
            let measured = FuseBonds(bonds[f - 1][...numFusionQubits - 1], fusion);
            let physical = FragmentQubits(state, numSites, siteToOrbitalOrder, fragment.sites);
            set carried = CorrectFragment(fragment, carried ^^^ measured, physical, phaseGradients[f], angleRegs[f]);
        }
        // Unfused bond qubits return to |0⟩ up to rotation quantization error.
        ResetAll(bondQubits);
        ApplyMPSFermionicOrderSigns(siteToOrbitalOrder, state);
    }

    /// Prepares the MPS on `state` with one phase gradient register per fragment.
    operation PrepareMPSConstantDepthOnState(params : MPSConstantDepthParams, state : Qubit[]) : Unit {
        let numFragments = Length(params.fragments);
        use phaseGradientQubits = Qubit[numFragments * params.rotationBitPrecision];
        let phaseGradients = Chunks(params.rotationBitPrecision, phaseGradientQubits);
        within {
            for phaseGradient in phaseGradients {
                PreparePhaseGradientState(phaseGradient);
            }
        } apply {
            PrepareMPSConstantDepth(
                params.fragments,
                params.numSites,
                params.siteToOrbitalOrder,
                params.numBondQubits,
                state,
                phaseGradients
            );
        }
    }

    /// Returns a composable operation on the state register; see `PrepareMPSConstantDepthOnState`.
    function MakeMPSConstantDepthOp(params : MPSConstantDepthParams) : Qubit[] => Unit {
        PrepareMPSConstantDepthOnState(params, _)
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation MakeMPSConstantDepthCircuit(
        fragments : MPSFragment[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numBondQubits : Int
    ) : Unit {
        use state = Qubit[numQubitsPerSite * numSites];
        PrepareMPSConstantDepthOnState(
            new MPSConstantDepthParams {
                fragments = fragments,
                numSites = numSites,
                numQubitsPerSite = numQubitsPerSite,
                siteToOrbitalOrder = siteToOrbitalOrder,
                rotationBitPrecision = rotationBitPrecision,
                numBondQubits = numBondQubits,
            },
            state
        );
        ResetAll(state);
    }
}
