// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.MPSConstantDepthTests {
    import QDKChemistry.Utils.MPSConstantDepth.MPSConstantDepthParams;
    import QDKChemistry.Utils.MPSConstantDepth.PrepareMPSConstantDepth;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;

    /// Returns the constant-depth MPS preparation with one phase gradient register shared by all
    /// fragments. The prepared state is the same as with one register per fragment, but the
    /// simulated state stays small.
    function TestMakeSharedGradientOp(params : MPSConstantDepthParams) : Qubit[] => Unit {
        qs => {
            use phaseGradient = Qubit[params.rotationBitPrecision];
            within {
                PreparePhaseGradientState(phaseGradient);
            } apply {
                PrepareMPSConstantDepth(
                    params.fragments,
                    params.numSites,
                    params.siteToOrbitalOrder,
                    params.numBondQubits,
                    qs,
                    [phaseGradient, size = Length(params.fragments)]
                );
            }
        }
    }
}
