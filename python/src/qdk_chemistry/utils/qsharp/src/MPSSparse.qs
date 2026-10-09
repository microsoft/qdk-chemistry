// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center (DLR),
// licensed under the Apache License, Version 2.0, and modified for QDK Chemistry.

/// MPS state preparation exploiting symmetry-induced block sparsity
/// (Rupprecht & Wölk, arXiv:2605.28489).
namespace QDKChemistry.Utils.MPSSparse {

    import Std.Arrays.IndexRange;
    import Std.Arrays.Reversed;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.MPSSequential.MakeMPSOp;
    import QDKChemistry.Utils.MPSSequential.MakeMPSOpWithPhaseGradient;
    import QDKChemistry.Utils.MPSSequential.PrepareMPS;
    import QDKChemistry.Utils.MPSSequential.PrepareMPSCircuit;
    import QDKChemistry.Utils.UnitarySynthesis.ApplyRealUnitaryViaGivens;
    import QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition;
    import QDKChemistry.Utils.UnitarySynthesis.QuantizeGivensDecomposition;

    /// Circuit data for one block-sparse MPS site unitary U = P_r · B · P_c, mirroring the C++
    /// `qdk::chemistry::utils::detail::SparseSiteSynthesis`.
    ///
    /// Basis states are indexed by physical · ancillaDim + bond, and B is block diagonal.
    struct SparseSiteSynthesis {
        /// P_c|c⟩ = |columnPermutation[c]⟩.
        columnPermutation : Int[],
        /// P_r|v⟩ = |rowPermutation[v]⟩.
        rowPermutation : Int[],
        blockGivens : GivensDecomposition,
    }

    /// Parameters of a composable MPS sparse state preparation.
    struct MPSSparseParams {
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseSiteSynthesis[],
    }

    /// Returns the bit-string tables of a permutation and of its inverse.
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

    /// Applies |i⟩ → |P(i)⟩ with `permTargets[i]` = P(i) and `invPermTargets[j]` = P⁻¹(j) as
    /// little-endian bit strings.
    ///
    /// Looks up P(i) into a fresh register and swaps it with `target`. The old register then
    /// holds P⁻¹(target), which the measurement-based unlookup `Adjoint Select` erases for
    /// O(√N) Toffolis instead of the O(N) of a second coherent lookup.
    operation PermutationViaQROAM(
        permTargets : Bool[][],
        invPermTargets : Bool[][],
        target : Qubit[]
    ) : Unit {
        let n = Length(target);
        let N = Length(permTargets);
        let nRequired = Ceiling(Lg(IntAsDouble(N)));

        use loaded = Qubit[n];
        Select(permTargets, target[...nRequired - 1], loaded);
        for i in 0..n - 1 {
            SWAP(target[i], loaded[i]);
        }
        Adjoint Select(invPermTargets, target[...nRequired - 1], loaded);
    }

    /// Applies one block-sparse site unitary U = P_r · B · P_c to the little-endian bond
    /// register `ancilla` and the site qubits `newSite`.
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

    /// `PrepareMPS` with the site unitaries applied by `ApplySparseSite`.
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

    /// Returns a composable `MPSSparse` operation; see `MakeMPSOp`.
    function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit {
        MakeMPSOp(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSparse(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// Returns a composable `MPSSparse` operation; see `MakeMPSOpWithPhaseGradient`.
    function MakeMPSSparseOpWithPhaseGradient(params : MPSSparseParams) : Qubit[] => Unit {
        MakeMPSOpWithPhaseGradient(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSparse(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// Circuit wrapper for resource estimation - allocates qubits internally.
    operation MakeMPSSparseCircuit(
        initialStateVec : Double[],
        numSites : Int,
        numQubitsPerSite : Int,
        siteToOrbitalOrder : Int[],
        rotationBitPrecision : Int,
        numAncillaQubits : Int,
        siteDecompositions : SparseSiteSynthesis[]
    ) : Unit {
        PrepareMPSCircuit(
            numQubitsPerSite * numSites,
            numAncillaQubits,
            rotationBitPrecision,
            MPSSparse(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, _, _, _)
        );
    }

}
