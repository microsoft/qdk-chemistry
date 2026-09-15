namespace QDKChemistry.TestUtils.PhaseGradientTests {
    import Std.Canon.ApplyXorInPlace;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PhaseGradient.RzViaPhaseGradient;

    function TestMakeRyOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
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

    function TestMakeRzOnPlusOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
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

    function TestMakeRyRoundtripOp(angleValue : Int, nBits : Int) : Qubit[] => Unit {
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
