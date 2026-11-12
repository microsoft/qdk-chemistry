// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.PhaseGradientTests {

    import Std.Canon.ApplyXorInPlace;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.RzViaPhaseGradient;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;

    /// Test wrapper: Ry via phase gradient on `[target | angle | gradient]`.
    function MakeTestRyOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
        (qs) => {
            let angle = qs[1..nBits];
            let pg = qs[nBits + 1..2 * nBits];
            ApplyXorInPlace(angleValue, angle);
            within {
                PreparePhaseGradientState(pg);
            } apply {
                RyViaPhaseGradient(qs[0], angle, pg);
            }
        }
    }

    /// Test wrapper: Rz via phase gradient on `[target | angle | gradient]`.
    function MakeTestRzOnPlusOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
        (qs) => {
            let angle = qs[1..nBits];
            let pg = qs[nBits + 1..2 * nBits];
            H(qs[0]);
            ApplyXorInPlace(angleValue, angle);
            within {
                PreparePhaseGradientState(pg);
            } apply {
                RzViaPhaseGradient(qs[0], angle, pg);
            }
        }
    }

    /// Test wrapper: Ry round-trip on `[target | angle | gradient]`.
    function MakeTestRyRoundtripOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
        (qs) => {
            let angle = qs[1..nBits];
            let pg = qs[nBits + 1..2 * nBits];
            H(qs[0]);
            ApplyXorInPlace(angleValue, angle);
            within {
                PreparePhaseGradientState(pg);
                RyViaPhaseGradient(qs[0], angle, pg);
            } apply {}
        }
    }
}
