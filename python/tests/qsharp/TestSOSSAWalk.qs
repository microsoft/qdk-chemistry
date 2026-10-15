// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.SOSSAWalkTests {
    import Std.Arrays.All;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Core.Length;
    import Std.Diagnostics.DumpMachine;
    import Std.Math.MaxI;
    import Std.Measurement.MeasureEachZ;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.SOSSAWalk.ControlledSelectWithUnlookup;
    import QDKChemistry.Utils.SOSSAWalk.SelectImpl;
    import QDKChemistry.Utils.SOSSAWalk.SelectParams;
    import QDKChemistry.Utils.SOSSAWalk.ShouldLoadFreeRiderSeparately;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;

    function TestMakeOuterInnerPrepOp(
        outerOp : (Qubit[]) => Unit is Adj + Ctl,
        innerOp : (Qubit[], Qubit[]) => Unit is Adj,
        nOuter : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let outerReg = qs[0..nOuter - 1];
            outerOp(outerReg);
            innerOp(outerReg, qs[nOuter...]);
        }
    }

    operation TestSelectDQ(
        selectData : SelectParams,
        xoValue : Int,
        bValue : Int,
        usePhaseGradient : Bool,
    ) : Unit {
        let numOrbitals = selectData.numOrbitals;
        let numPositiveOneBody = selectData.numPositiveOneBody;
        let numSF = selectData.numRanks * selectData.numCopies;
        let numOuterValues = numOrbitals + numSF;
        let xoBits = MaxI(1, AddressQubits(numOuterValues));
        let numBp1 = selectData.numBases + 1;
        let bBits = MaxI(1, AddressQubits(numBp1));
        let nFR = selectData.numFreeRiderBits;

        let nOuter = xoBits;
        let nInner = bBits + nFR;
        let nSpin = 2;
        let nSystem = 2 * numOrbitals;
        // The gradient register is allocated only on the QROM path. It is conjugated back to
        // |0...0>, so the direct path's state is recovered from the QROM dump by restricting
        // to the gradient=|0...0> subspace.
        let nGradient = if usePhaseGradient { selectData.rotationBitPrecision } else { 0 };
        let total = nOuter + nInner + nSpin + nSystem + nGradient;
        use qs = Qubit[total];

        let outerReg = qs[0..nOuter - 1];
        let innerReg = qs[nOuter..nOuter + nInner - 1];
        let spinReg = qs[nOuter + nInner..nOuter + nInner + nSpin - 1];
        let systemReg = qs[nOuter + nInner + nSpin..nOuter + nInner + nSpin + nSystem - 1];
        let gradientReg = qs[total - nGradient...];

        let xoReg = outerReg[0..xoBits - 1];
        ApplyXorInPlace(xoValue, xoReg);
        H(spinReg[0]);

        ApplyXorInPlace(bValue, innerReg[0..bBits - 1]);

        let frStart = bBits;
        if nFR >= 2 {
            if xoValue >= numOrbitals { X(innerReg[frStart]); }
            if xoValue >= numPositiveOneBody { X(innerReg[frStart + 1]); }

            // Rank as the inner PREPARE's free-rider data would carry it: 0 for one-body.
            let rValue = if xoValue >= numOrbitals { (xoValue - numOrbitals) / selectData.numCopies } else { 0 };
            ApplyXorInPlace(rValue, innerReg[frStart + 2..frStart + nFR - 1]);
        }

        X(systemReg[0]);

        if usePhaseGradient {
            within {
                PreparePhaseGradientState(gradientReg);
            } apply {
                SelectImpl(selectData, true, outerReg, innerReg, spinReg, systemReg, gradientReg);
            }
        } else {
            SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg, []);
        }

        DumpMachine();
        ResetAll(qs);
    }

    function TestShouldLoadFreeRiderSeparately(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
    ) : Bool {
        ShouldLoadFreeRiderSeparately(innerCoefficients, freeRiderData, coefficientBitPrecision)
    }

    operation TestBranchedRotationWordRoundTrip(
        sfData : Bool[][],
        dqData : Bool[][],
        numSFAddressQubits : Int,
        numDQAddressQubits : Int,
    ) : Bool {
        use isSF = Qubit();
        use sfAddress = Qubit[numSFAddressQubits];
        use dqAddress = Qubit[numDQAddressQubits];
        use target = Qubit[Length(sfData[0])];
        let addressReg = [isSF] + sfAddress + dqAddress;

        ApplyToEachCA(H, addressReg);
        ControlledSelectWithUnlookup(sfData, sfAddress, dqData, dqAddress, isSF, target);
        Adjoint ControlledSelectWithUnlookup(sfData, sfAddress, dqData, dqAddress, isSF, target);
        ApplyToEachCA(H, addressReg);

        let results = MeasureEachZ(addressReg + target);
        ResetAll(addressReg + target);
        All(result -> result == Zero, results)
    }
}
