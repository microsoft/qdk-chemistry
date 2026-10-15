// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Test-only drivers that force the resource-estimation legacy paths during simulation.
/// Evaluated by `qsharp_test_context` so the shipped project's internal operations are reachable.
namespace QDKChemistry.TestUtils.HubbardPlaquetteTests {

    import QDKChemistry.Utils.HubbardPlaquette.HoppingPhasesWithLegacyCosts;
    import QDKChemistry.Utils.HubbardPlaquette.HubbardPlaquetteParams;
    import QDKChemistry.Utils.HubbardPlaquette.RepPlaquetteExpWithLegacyCosts;

    operation TestHoppingPhasesWithForcedLegacyCosts(
        angle : Double,
        pairs : Qubit[][],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int
    ) : Unit is Adj + Ctl {
        HoppingPhasesWithLegacyCosts(angle, pairs, useHammingWeightPhasing, maxBatchSize, true);
    }

    operation TestRepPlaquetteExpWithForcedLegacyCosts(
        params : HubbardPlaquetteParams,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        RepPlaquetteExpWithLegacyCosts(params, systems, true);
    }
}
