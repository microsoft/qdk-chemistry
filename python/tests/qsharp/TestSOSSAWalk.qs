// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.SOSSAWalkTests {
    import Std.Arrays.All;
    import Std.Arrays.Filtered;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.DumpMachine;
    import Std.Math.MaxI;
    import Std.Math.Sqrt;
    import Std.Measurement.MeasureEachZ;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirtyBorrowedQubits;
    import QDKChemistry.Utils.SOSSAWalk.AmortizedRotationWordSweep;
    import QDKChemistry.Utils.SOSSAWalk.ApplyGivensRotationWords;
    import QDKChemistry.Utils.SOSSAWalk.ComputeSpinParity;
    import QDKChemistry.Utils.SOSSAWalk.ControlledSelectWithUnlookup;
    import QDKChemistry.Utils.SOSSAWalk.LoadRotationWord;
    import QDKChemistry.Utils.SOSSAWalk.SelectImpl;
    import QDKChemistry.Utils.SOSSAWalk.SelectParams;
    import QDKChemistry.Utils.SOSSAWalk.ShouldLoadFreeRiderSeparately;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;

    /// Lends the inner PREPARE nothing, so this exercises the clean lookup.
    function TestMakeOuterInnerPrepOp(
        outerOp : (Qubit[]) => Unit is Adj + Ctl,
        innerOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        nOuter : Int,
    ) : Qubit[] => Unit {
        (qs) => {
            let outerReg = qs[0..nOuter - 1];
            outerOp(outerReg);
            innerOp(outerReg, qs[nOuter...], []);
        }
    }

    operation TestSelectDQ(
        selectData : SelectParams,
        xoValue : Int,
        bValue : Int,
        usePhaseGradient : Bool,
    ) : Unit {
        TestSelectOnSystemState(selectData, xoValue, bValue, usePhaseGradient, []);
    }

    /// Runs SELECT on |1,0,...> when `systemAngles` is empty, else on an entangled Ry/CNOT/Rz system state.
    operation TestSelectOnSystemState(
        selectData : SelectParams,
        xoValue : Int,
        bValue : Int,
        usePhaseGradient : Bool,
        systemAngles : Double[],
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

        if Length(systemAngles) == 0 {
            X(systemReg[0]);
        } else {
            for q in 0..nSystem - 1 {
                Ry(systemAngles[q], systemReg[q]);
            }
            for q in 0..nSystem - 2 {
                CNOT(systemReg[q], systemReg[q + 1]);
            }
            for q in 0..nSystem - 1 {
                Rz(systemAngles[nSystem + q], systemReg[q]);
            }
        }

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

    /// Applies SELECT once, forward only, to a label-entangled superposition over (x_o, b, spins, system, reference).
    /// Row l of `loadTable` holds the [x_o | b | free-rider] word that the label value l loads.
    /// The word is erased again before the dump, so only the label, spins, system, reference and gradient remain.
    operation TestSelectOnEntangledInput(
        selectData : SelectParams,
        loadTable : Bool[][],
        numLabelQubits : Int,
        labelAngles : Double[],
        spinAngles : Double[],
        systemAngles : Double[],
        numReference : Int,
        usePhaseGradient : Bool,
    ) : Unit {
        let N = selectData.numOrbitals;
        let bBits = MaxI(1, AddressQubits(selectData.numBases + 1));
        let nLoad = Length(loadTable[0]);
        let xoBits = nLoad - bBits - selectData.numFreeRiderBits;
        let nGradient = if usePhaseGradient { selectData.rotationBitPrecision } else { 0 };
        use label = Qubit[numLabelQubits];
        use spinReg = Qubit[2];
        use systemReg = Qubit[2 * N];
        use reference = Qubit[numReference];
        use gradientReg = Qubit[nGradient];
        use loaded = Qubit[nLoad];

        for i in 0..numLabelQubits - 1 {
            Ry(labelAngles[i], label[i]);
            T(label[i]);
        }
        for i in 0..1 {
            Ry(spinAngles[i], spinReg[i]);
            S(spinReg[i]);
        }
        for q in 0..numReference - 1 {
            H(reference[q]);
            CNOT(reference[q], systemReg[q]);
        }
        for q in 0..2 * N - 1 {
            Ry(systemAngles[q], systemReg[q]);
            T(systemReg[q]);
        }

        Select(loadTable, label, loaded);
        if usePhaseGradient {
            within {
                PreparePhaseGradientState(gradientReg);
            } apply {
                SelectImpl(selectData, true, loaded[0..xoBits - 1], loaded[xoBits...], spinReg, systemReg, gradientReg);
            }
        } else {
            SelectImpl(selectData, false, loaded[0..xoBits - 1], loaded[xoBits...], spinReg, systemReg, []);
        }
        Select(loadTable, label, loaded);

        DumpMachine();
        ResetAll(label + spinReg + systemReg + reference + gradientReg + loaded);
    }

    function TestShouldLoadFreeRiderSeparately(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
        maxSwapBits : Int,
        availableDirty : Int,
    ) : Bool {
        ShouldLoadFreeRiderSeparately(
            innerCoefficients,
            freeRiderData,
            coefficientBitPrecision,
            maxSwapBits,
            availableDirty
        )
    }

    /// The borrowed register is conjugated too, so a dirty pass that disturbs it fails here.
    operation TestBranchedRotationWordRoundTrip(
        sfData : Bool[][],
        dqData : Bool[][],
        numSFAddressQubits : Int,
        numDQAddressQubits : Int,
        borrowDirty : Bool,
        sfSwapBits : Int,
        dqSwapBits : Int,
        useSkewTree : Bool,
    ) : Bool {
        use isSF = Qubit();
        use sfAddress = Qubit[numSFAddressQubits];
        use dqAddress = Qubit[numDQAddressQubits];
        use target = Qubit[Length(sfData[0])];
        // Only a dirty load borrows; the clean paths allocate their own scratch internally.
        let numDirty = if borrowDirty {
            MaxI(
                SelectSwapDirtyBorrowedQubits(sfSwapBits, Length(sfData[0])),
                SelectSwapDirtyBorrowedQubits(dqSwapBits, Length(dqData[0]))
            )
        } else {
            0
        };
        use dirty = Qubit[numDirty];
        let addressReg = [isSF] + sfAddress + dqAddress;
        let conjugated = addressReg + dirty;

        ApplyToEachCA(H, conjugated);
        ControlledSelectWithUnlookup(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, [], sfAddress, dqAddress, isSF, dirty, target);
        Adjoint ControlledSelectWithUnlookup(sfData, dqData, sfSwapBits, dqSwapBits, useSkewTree, [], sfAddress, dqAddress, isSF, dirty, target);
        ApplyToEachCA(H, conjugated);

        let results = MeasureEachZ(conjugated + target);
        ResetAll(conjugated + target);
        All(result -> result == Zero, results)
    }

    /// Traces one clean `LoadRotationWord`, so `SelectSwapForwardCost` can be checked against it.
    operation TestLoadRotationWordResourceProbe(data : Bool[][], numSwapBits : Int) : Unit {
        use isSF = Qubit();
        use address = Qubit[AddressQubits(Length(data))];
        use target = Qubit[Length(data[0])];
        LoadRotationWord(data, numSwapBits, false, [], address, isSF, [], target);
        ResetAll(target);
    }

    /// Runs the amortized sweep with `loaded` tables on Bell-paired lenders, copying each batch
    /// word into its own record, then erases the records with `expected`; true when all returns to zero.
    operation TestAmortizedSweepRecordsEachBatch(
        loadedSF : Bool[][][],
        loadedDQ : Bool[][][],
        expectedSF : Bool[][][],
        expectedDQ : Bool[][][],
        order : Int[],
        sfSwapBits : Int,
        dqSwapBits : Int,
        numLenders : Int,
        measure : Bool,
        runAdjoint : Bool,
    ) : Bool {
        let width = Length(loadedSF[0][0]);
        let numBatches = Length(loadedSF);
        use isSF = Qubit();
        use sfAddress = Qubit[AddressQubits(Length(loadedSF[0]))];
        use dqAddress = Qubit[AddressQubits(Length(loadedDQ[0]))];
        use lenders = Qubit[numLenders];
        use references = Qubit[numLenders];
        use records = Qubit[numBatches * width];
        let conjugated = [isSF] + sfAddress + dqAddress + references;

        ApplyToEachCA(H, conjugated);
        // Bell pairs expose any lender left flipped or phased once the pairing is undone.
        ApplyToEachCA(CNOT, Zipped(references, lenders));
        let copy = (t, word) => ApplyToEachCA(CNOT, Zipped(word, records[t * width..(t + 1) * width - 1]));
        if runAdjoint {
            Adjoint AmortizedRotationWordSweep(loadedSF, loadedDQ, order, sfSwapBits, dqSwapBits, sfAddress, dqAddress, isSF, lenders, measure, copy);
        } else {
            AmortizedRotationWordSweep(loadedSF, loadedDQ, order, sfSwapBits, dqSwapBits, sfAddress, dqAddress, isSF, lenders, measure, copy);
        }
        for t in 0..numBatches - 1 {
            let record = records[t * width..(t + 1) * width - 1];
            Controlled Select([isSF], (expectedSF[t], sfAddress, record));
            within { X(isSF); } apply {
                Controlled Select([isSF], (expectedDQ[t], dqAddress, record));
            }
        }
        ApplyToEachCA(CNOT, Zipped(references, lenders));
        ApplyToEachCA(H, conjugated);

        let results = MeasureEachZ(conjugated + lenders + records);
        ResetAll(conjugated + lenders + records);
        All(result -> result == Zero, results)
    }

    /// Applies SELECT to unequal superpositions of every valid (x_o, b) and both spins, with the system Choi-entangled to a reference.
    operation TestSelectOnEntangledState(selectData : SelectParams) : Unit {
        let numOrbitals = selectData.numOrbitals;
        let numOuterValues = numOrbitals + selectData.numRanks * selectData.numCopies;
        let xoBits = MaxI(1, AddressQubits(numOuterValues));
        let bBits = MaxI(1, AddressQubits(selectData.numBases + 1));
        let nFR = selectData.numFreeRiderBits;
        use outerReg = Qubit[xoBits];
        use innerReg = Qubit[bBits + nFR];
        use spinReg = Qubit[2];
        use systemReg = Qubit[2 * numOrbitals];
        use reference = Qubit[2 * numOrbitals];
        use gradientReg = Qubit[selectData.rotationBitPrecision];

        PreparePureStateD(MappedOverRange(i -> i < numOuterValues ? Sqrt(IntAsDouble(i + 1)) | 0.0, 0..(1 <<< xoBits) - 1), Reversed(outerReg));
        PreparePureStateD(MappedOverRange(i -> i <= selectData.numBases ? Sqrt(IntAsDouble(i + 2)) | 0.0, 0..(1 <<< bBits) - 1), Reversed(innerReg[0..bBits - 1]));
        for xo in 0..numOuterValues - 1 {
            let rValue = xo >= numOrbitals ? (xo - numOrbitals) / selectData.numCopies | 0;
            let frValue = (xo >= numOrbitals ? 1 | 0) + (xo >= selectData.numPositiveOneBody ? 2 | 0) + 4 * rValue;
            ApplyControlledOnInt(xo, ApplyXorInPlace(frValue, _), outerReg, innerReg[bBits..bBits + nFR - 1]);
        }
        Ry(0.7, spinReg[0]);
        Ry(1.9, spinReg[1]);
        for (ref, sys) in Zipped(reference, systemReg) {
            H(ref);
            CNOT(ref, sys);
        }

        within {
            PreparePhaseGradientState(gradientReg);
        } apply {
            SelectImpl(selectData, true, outerReg, innerReg, spinReg, systemReg, gradientReg);
        }

        DumpMachine();
        ResetAll(outerReg + innerReg + spinReg + systemReg + reference + gradientReg);
    }

    /// Dumps an entangled input after the parity-phase spin swap over all pairs but `skipPair`, N controlled SWAPs and the inverse preparation.
    operation TestSpinParitySwapRoundTrip(numPairs : Int, skipPair : Int) : Unit {
        use spin = Qubit();
        use down = Qubit[numPairs];
        use up = Qubit[numPairs];
        use reference = Qubit[2 * numPairs];
        use parity = Qubit();
        let kept = Filtered(i -> i != skipPair, MappedOverRange(i -> i, 0..numPairs - 1));

        Ry(0.9, spin);
        for (ref, sys) in Zipped(reference, down + up) {
            H(ref);
            CNOT(ref, sys);
        }
        Ry(0.4, up[0]);
        within {
            ComputeSpinParity(Subarray(kept, down), Subarray(kept, up), parity);
        } apply {
            CZ(spin, parity);
        }
        for (d, u) in Zipped(down, up) {
            Controlled SWAP([spin], (d, u));
        }
        Adjoint Ry(0.4, up[0]);
        for (ref, sys) in Zipped(reference, down + up) {
            CNOT(ref, sys);
            H(ref);
        }
        Adjoint Ry(0.9, spin);

        DumpMachine();
        ResetAll([spin] + down + up + reference + [parity]);

    }

    /// Applies the controlled `op` with its control in |+⟩, after flipping the `flipped` qubits, Choi-entangling the
    /// `entangled` ones with a reference and preparing the trailing phase gradient, then dumps the state.
    operation TestApplyControlledOnPartialChoiState(
        op : (Qubit, Qubit[]) => Unit is Adj,
        numQubits : Int,
        numGradientQubits : Int,
        entangled : Int[],
        flipped : Int[],
    ) : Unit {
        use qs = Qubit[numQubits];
        use reference = Qubit[Length(entangled)];
        use control = Qubit();
        for index in flipped {
            X(qs[index]);
        }
        for (ref, index) in Zipped(reference, entangled) {
            H(ref);
            CNOT(ref, qs[index]);
        }
        H(control);
        within {
            PreparePhaseGradientState(qs[numQubits - numGradientQubits...]);
        } apply {
            op(control, qs);
        }
        DumpMachine();
        ResetAll(qs + reference + [control]);
    }
}
