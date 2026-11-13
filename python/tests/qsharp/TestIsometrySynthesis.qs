// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.IsometrySynthesisTests {
    import Std.Arrays.IndexRange;
    import QDKChemistry.Utils.IsometrySynthesis.PermutationData;
    import QDKChemistry.Utils.IsometrySynthesis.PermutationViaQROAM;

    /// Returns an operation that applies Ry(prepAngles[k]) to qubit k of a little-endian
    /// register and then the permutation |i⟩ → |permutation[i]⟩.
    function TestMakePermutationOp(permutation : Int[], prepAngles : Double[]) : Qubit[] => Unit {
        (qs) => {
            for k in IndexRange(prepAngles) {
                Ry(prepAngles[k], qs[k]);
            }
            let (forward, inverse) = PermutationData(permutation, Length(qs));
            PermutationViaQROAM(forward, inverse, qs);
        }
    }
}
