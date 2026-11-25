// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.MPSPreparationTests {
    import QDKChemistry.Utils.MPSSequential.MPSSequential;
    import QDKChemistry.Utils.MPSSequential.MPSSequentialParams;
    import QDKChemistry.Utils.MPSSparse.MPSSparse;
    import QDKChemistry.Utils.MPSSparse.MPSSparseParams;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;

    /// Returns `MPSSequential` on [state | bond] with its own phase gradient, leaving the bond register exposed.
    function TestMakeMPSSequentialOp(params : MPSSequentialParams) : Qubit[] => Unit {
        (qs) => {
            let numState = params.numQubitsPerSite * params.numSites;
            use phaseGradient = Qubit[params.rotationBitPrecision];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                MPSSequential(
                    params.initialStateVec,
                    params.numSites,
                    params.siteToOrbitalOrder,
                    params.siteDecompositions,
                    qs[...numState - 1],
                    qs[numState...],
                    phaseGradient
                );
            }
        }
    }

    /// Returns `MPSSparse` on [state | bond] with its own phase gradient, leaving the bond register exposed.
    function TestMakeMPSSparseOp(params : MPSSparseParams) : Qubit[] => Unit {
        (qs) => {
            let numState = params.numQubitsPerSite * params.numSites;
            use phaseGradient = Qubit[params.rotationBitPrecision];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                MPSSparse(
                    params.initialStateVec,
                    params.numSites,
                    params.siteToOrbitalOrder,
                    params.siteDecompositions,
                    qs[...numState - 1],
                    qs[numState...],
                    phaseGradient
                );
            }
        }
    }

    /// Returns an operation that applies `op` and then its adjoint.
    function TestMakeRoundTripOp(op : Qubit[] => Unit is Adj) : Qubit[] => Unit {
        (qs) => {
            op(qs);
            Adjoint op(qs);
        }
    }
}
