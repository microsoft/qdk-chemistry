// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.

namespace QDKChemistry.Utils.MPSSparse {

    import Std.Math.*;
    import Std.Convert.*;
    import Std.Arrays.*;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.MPSSequential.MakeMPSOp;
    import QDKChemistry.Utils.MPSSequential.MakeMPSOpWithPhaseGradient;
    import QDKChemistry.Utils.MPSSequential.PrepareMPS;
    import QDKChemistry.Utils.MPSSequential.PrepareMPSCircuit;
    import QDKChemistry.Utils.UnitarySynthesis.*;

    export MPSSparse, MPSSparseParams, MakeMPSSparseOp, MakeMPSSparseOpWithPhaseGradient, MakeMPSSparseCircuit, PermutationViaQROAM, SparseSiteSynthesis;

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
    /// Applies one block-sparse site unitary U = P_r · B · P_c.
    ///
    /// # Description
    /// The permutations use table lookup, SWAP, and measurement-based uncomputation (see
    /// `PermutationViaQROAM`), and the block-diagonal B uses Givens rotation layers.
    ///
    /// # Input
    /// ## synthesis
    /// Decomposition of the site unitary.
    /// ## ancilla
    /// Little-endian bond register.
    /// ## newSite
    /// Qubits of the new site, little-endian in its physical index.
    /// ## phaseGradient
    /// Phase gradient register.
    /// ## angleReg
    /// Clean register that receives each loaded angle.
    operation ApplySparseSite(
        synthesis : SparseSiteSynthesis,
        ancilla : Qubit[],
        newSite : Qubit[],
        phaseGradient : Qubit[],
        angleReg : Qubit[]
    ) : Unit {
        // The little-endian joint register holds the bond in its low qubits, so its value
        // physical * ancillaDim + bond is the row index of the site isometry.
        let target = ancilla + newSite;
        let totalBits = Length(target);
        let (colPermData, colInvPermData) = PermutationData(synthesis.columnPermutation, totalBits);
        let (rowPermData, rowInvPermData) = PermutationData(synthesis.rowPermutation, totalBits);
        let blockGivens = QuantizeGivensDecomposition(synthesis.blockGivens, 1 <<< (totalBits - 1), Length(phaseGradient));

        PermutationViaQROAM(colPermData, colInvPermData, target);
        // The Givens layers act on the most-significant-first view of the joint register.
        ApplyRealUnitaryViaGivens(blockGivens, [], Reversed(target), phaseGradient, angleReg);
        PermutationViaQROAM(rowPermData, rowInvPermData, target);
    }

    /// # Summary
    /// MPS state preparation exploiting block sparsity.
    ///
    /// # Description
    /// `PrepareMPS` with the site unitaries applied by `ApplySparseSite`.
    ///
    /// References:
    ///   Rupprecht & Woelk (2026). Faster matrix product state preparation by
    ///   exploiting symmetry-induced block-sparsity. arXiv:2605.28489.
    operation MPSSparse(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : SparseSiteSynthesis[],
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit {
        PrepareMPS(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, ApplySparseSite, state, ancilla, phaseGradient);
    }

    /// # Summary
    /// Returns a composable operation that prepares the MPS on a numQubitsPerSite·numSites-qubit
    /// register, allocating and preparing its own phase gradient register.
    function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit {
        MakeMPSOp(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBits,
            MPSSparse(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// # Summary
    /// Returns a composable operation on `[state | phaseGradient]` whose last rotationBits
    /// qubits are a caller-owned register prepared by `PreparePhaseGradientState`.
    function MakeMPSSparseOpWithPhaseGradient(params : MPSSparseParams) : Qubit[] => Unit {
        MakeMPSOpWithPhaseGradient(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBits,
            MPSSparse(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
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
        PrepareMPSCircuit(
            numQubitsPerSite * numSites,
            numAncillaQubits,
            rotationBits,
            MPSSparse(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, _, _, _)
        );
    }

}
