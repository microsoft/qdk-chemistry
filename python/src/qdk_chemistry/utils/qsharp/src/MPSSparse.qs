// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

namespace QDKChemistry.Utils.MPSSparse {

    import Std.Math.*;
    import Std.Convert.*;
    import Std.Arrays.*;
    import Std.Canon.*;
    import Std.Diagnostics.*;
    import Std.ResourceEstimation.*;
    import Std.Measurement.*;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepParams;
    import QDKChemistry.Utils.QROMStatePrep.QROMStatePrepare;
    import QDKChemistry.Utils.MPSSequential.MPSSiteQubits;
    import QDKChemistry.Utils.MPSSequential.ApplyMPSFermionicOrderSigns;
    import GivensDecomposition.*;

    export MPSSparse, MPSSparseParams, MakeMPSSparseOp, MakeMPSSparseCircuit, PermutationViaQROAM, SparseSiteSynthesis;

    /// # Summary
    /// Circuit data for one block-sparse MPS site unitary, mirroring the C++
    /// `qdk::chemistry::utils::detail::SparseSiteSynthesis`.
    ///
    /// # Description
    /// The site unitary is U = P_r · B · P_c, where P_c|c⟩ = |columnPermutation[c]⟩,
    /// P_r|v⟩ = |rowPermutation[v]⟩, and B is block diagonal. Basis state
    /// physical · ancillaDim + bond of the joint site and bond register has that index.
    ///
    /// # Input
    /// ## columnPermutation
    /// Int[N]: image of each basis state under P_c.
    /// ## rowPermutation
    /// Int[N]: image of each basis state under P_r.
    /// ## blockGivens
    /// Givens data of the block-diagonal B.
    struct SparseSiteSynthesis {
        columnPermutation : Int[],
        rowPermutation : Int[],
        blockGivens : GivensDecomposition,
    }

    /// # Summary
    /// Parameters of a composable MPS sparse state preparation.
    struct MPSSparseParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseSiteSynthesis[],
    }

    // =============================================================================
    // Permutation via QROAM
    // =============================================================================

    function PermutationData(permTargets : Int[], numBits : Int) : (Bool[][], Bool[][]) {
        mutable data = Repeated(Repeated(false, numBits), Length(permTargets));
        mutable inverseData = data;
        for source in IndexRange(permTargets) {
            let target = permTargets[source];
            set data w/= source <- IntAsBoolArray(target, numBits);
            set inverseData w/= target <- IntAsBoolArray(source, numBits);
        }
        return (data, inverseData);
    }

    /// # Summary
    /// Applies a permutation |i> -> |P(i)> using table lookup, SWAP, and measurement-based
    /// uncomputation.
    ///
    /// # Description
    /// Implements the permutation by:
    ///   1. Loading P(address) into a fresh register via table lookup
    ///   2. SWAPping the target register with the loaded register
    ///   3. Erasing the old register, which now holds P^{-1}(target), by measuring it in the
    ///      X basis and undoing the resulting sign flips with a phase lookup on the target
    ///
    /// Step 3 is `Adjoint Select`, the measurement-based unlookup of the inverse table, so
    /// it costs O(sqrt(N)) Toffolis instead of the O(N) of a second coherent lookup.
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

        // Step 1: Load P(address) into a fresh register.
        use loaded = Qubit[n];
        Select(permTargets, target[...nRequired - 1], loaded);

        // Step 2: SWAP target <-> loaded
        for i in 0..n - 1 {
            SWAP(target[i], loaded[i]);
        }

        // Step 3: After the SWAP, loaded = i = invPermTargets[target], so the measurement-based
        // unlookup measures it in the X basis and fixes the signs (-1)^(m · P^{-1}(target)).
        Adjoint Select(invPermTargets, target[...nRequired - 1], loaded);
    }

    // =============================================================================
    // Full MPS Sparse preparation
    // =============================================================================

    /// # Summary
    /// MPS state preparation exploiting block sparsity.
    ///
    /// Each site unitary is decomposed as U = P_row · V_blockdiag · P_col
    /// where P_row, P_col are permutations (via table lookup + SWAP + measurement-based
    /// uncomputation, see `PermutationViaQROAM`) and V_blockdiag is block-diagonal (via
    /// Givens rotation layers).
    ///
    /// # Description
    /// Prepares an MPS by:
    ///   1. Preparing the initial state (first site) via QROM state preparation
    ///   2. Applying sparse site unitaries for sites 1..N-1
    ///   3. Converting chain-order fermionic signs to the blocked Jordan-Wigner convention
    ///
    /// `state` is a Jordan-Wigner register with one mode per orbital for the ('0', '1')
    /// physical basis, or the blocked layout of the α modes of all orbitals followed by their
    /// β modes for the ('0', 'u', 'd', '2') basis. `siteToOrbitalOrder` gives the orbital that
    /// holds each chain site.
    ///
    /// References:
    ///   Rupprecht & Woelk (2026). Faster matrix product state preparation by
    ///   exploiting symmetry-induced block-sparsity. arXiv:2605.28489.
    operation MPSSparse(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        siteDecompositions : SparseSiteSynthesis[],
        state : Qubit[],
        ancilla : Qubit[]
    ) : Unit {
        Fact(Length(siteDecompositions) == numSites - 1, "MPS sparse preparation needs one decomposition per site after the first.");
        Fact(
            Length(state) == numSites or Length(state) == 2 * numSites,
            "The state register must hold one or two qubits per MPS site."
        );

        // Initialize phase gradient register
        use phaseGradient = Qubit[rotationBits];
        PreparePhaseGradientState(phaseGradient);

        // Single shared angle register
        use angleReg = Qubit[rotationBits];

        // Prepare the first site and its right bond. `initReg` is little-endian, so
        // amplitude index physical * ancillaDim + bond lands on register value j.
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

        // Apply the sparse site unitaries U = P_r · B · P_c.
        for siteIdx in 0..numSites - 2 {
            let synthesis = siteDecompositions[siteIdx];
            // The little-endian joint register holds the bond in its low qubits, so its value
            // physical * ancillaDim + bond is the row index of the site isometry.
            let target = ancilla + MPSSiteQubits(state, numSites, siteToOrbitalOrder[siteIdx + 1]);
            let totalBits = Length(target);
            let (colPermData, colInvPermData) = PermutationData(synthesis.columnPermutation, totalBits);
            let (rowPermData, rowInvPermData) = PermutationData(synthesis.rowPermutation, totalBits);
            let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, 1 <<< (totalBits - 1), rotationBits);

            PermutationViaQROAM(colPermData, colInvPermData, target);
            // The Givens layers act on the most-significant-first view of the joint register.
            ApplyRealUnitaryViaGivens(blockGivens, Reversed(target), phaseGradient, angleReg);
            PermutationViaQROAM(rowPermData, rowInvPermData, target);
        }
        ApplyMPSFermionicOrderSigns(siteToOrbitalOrder, state);

        // Undo phase gradient state
        Adjoint PreparePhaseGradientState(phaseGradient);
    }

    /// # Summary
    /// Returns a composable operation that prepares the MPS on a numQubitsPerSite·numSites-qubit
    /// register.
    function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit {
        (state) => {
            Fact(
                Length(state) == params.numQubitsPerSite * params.numSites,
                "State register size must equal the number of qubits per MPS site times the number of sites."
            );
            use ancilla = Qubit[params.numAncillaQubits];
            MPSSparse(
                params.initialStateVec,
                params.numSites,
                params.siteToOrbitalOrder,
                params.rotationBits,
                params.siteDecompositions,
                state,
                ancilla
            );
        }
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation MakeMPSSparseCircuit(
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBits : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseSiteSynthesis[]
    ) : Unit {
        use state = Qubit[numQubitsPerSite * numSites];
        use ancilla = Qubit[numAncillaQubits];
        MPSSparse(
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

}
