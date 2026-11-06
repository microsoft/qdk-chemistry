// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//
// Portions of this file are adapted from code by Felix Rupprecht published at
// https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center (DLR),
// licensed under the Apache License, Version 2.0, and modified for QDK Chemistry.

/// MPS state preparation exploiting symmetry-induced block sparsity
/// (Rupprecht & Wölk, arXiv:2605.28489).
namespace QDKChemistry.Utils.MPSSparse {

    import QDKChemistry.Utils.IsometrySynthesis.ApplySparseIsometry;
    import QDKChemistry.Utils.IsometrySynthesis.SparseIsometrySynthesis;
    import QDKChemistry.Utils.MPSSequential.MakeMPSOp;
    import QDKChemistry.Utils.MPSSequential.MakeMPSOpWithPhaseGradient;
    import QDKChemistry.Utils.MPSSequential.PrepareMPS;
    import QDKChemistry.Utils.MPSSequential.PrepareMPSCircuit;

    /// Parameters of a composable MPS sparse state preparation.
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
    operation MPSSparse(
        initialStateVec : Double[],
        numSites : Int,
        siteToOrbitalOrder : Int[],
        siteDecompositions : SparseIsometrySynthesis[],
        state : Qubit[],
        ancilla : Qubit[],
        phaseGradient : Qubit[]
    ) : Unit is Adj {
        PrepareMPS(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, ApplySparseIsometry, state, ancilla, phaseGradient);
    }

    /// Returns a composable `MPSSparse` operation; see `MakeMPSOp`.
    function MakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit is Adj {
        MakeMPSOp(
            params.numQubitsPerSite * params.numSites,
            params.numAncillaQubits,
            params.rotationBitPrecision,
            MPSSparse(params.initialStateVec, params.numSites, params.siteToOrbitalOrder, params.siteDecompositions, _, _, _)
        )
    }

    /// Returns a composable `MPSSparse` operation; see `MakeMPSOpWithPhaseGradient`.
    function MakeMPSSparseOpWithPhaseGradient(params : MPSSparseParams) : Qubit[] => Unit is Adj {
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
        siteDecompositions : SparseIsometrySynthesis[]
    ) : Unit {
        PrepareMPSCircuit(
            numQubitsPerSite * numSites,
            numAncillaQubits,
            rotationBitPrecision,
            MPSSparse(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, _, _, _)
        );
    }

}
