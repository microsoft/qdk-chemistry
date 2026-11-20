// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.IsometrySynthesisTests {
    import Std.Arrays.IndexRange;
    import QDKChemistry.Utils.IsometrySynthesis.PermutationData;
    import QDKChemistry.Utils.IsometrySynthesis.PermutationViaQROAM;

    /// Returns an operation that prepares Ry(prepAngles) and applies |i⟩ → |permutation[i]⟩, or its inverse.
    function TestMakePermutationOp(permutation : Int[], prepAngles : Double[], maxSwapBits : Int, inverse : Bool) : Qubit[] => Unit {
        (qs) => {
            for k in IndexRange(prepAngles) {
                Ry(prepAngles[k], qs[k]);
            }
            let (forwardData, inverseData) = PermutationData(permutation, Length(qs));
            if inverse {
                Adjoint PermutationViaQROAM(forwardData, inverseData, maxSwapBits, qs);
            } else {
                PermutationViaQROAM(forwardData, inverseData, maxSwapBits, qs);
            }
        }
    }
}
