// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.SelectSwapDirtyTests {
    import Std.Arrays.All;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Measurement.MResetEachZ;
    import Std.StatePreparation.PrepareUniformSuperposition;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirty;
    import QDKChemistry.Utils.SelectSwapDirty.SelectSwapDirtyBorrowedQubits;

    /// One forward `SelectSwapDirty` load matches a plain `Select` on every address and alone restores
    /// a seeded, non-trivial lender, since SOSSA erases the word by measurement and never reruns the load.
    operation TestSelectSwapDirtyCorrectness(
        data : Bool[][],
        numSwapBits : Int,
        dirtySeed : Int,
    ) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let numDirty = SelectSwapDirtyBorrowedQubits(numSwapBits, m);

        use address = Qubit[nAddr];
        use dirty = Qubit[MaxI(1, numDirty)];
        use output = Qubit[m];

        mutable allCorrect = true;

        for addr in 0..2^nAddr - 1 {
            ApplyXorInPlace(addr, address);
            if numDirty > 0 {
                ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
            }

            SelectSwapDirty(numSwapBits, data, address, dirty, output);

            // A forward `Select` XORs the expected word back out, so any bit left over is wrong.
            Select(data, address, output);
            let residue = MResetEachZ(output);
            if not All(r -> r == Zero, residue) {
                Message($"FAIL word: addr={addr}, differs from Select in bits {residue}");
                set allCorrect = false;
            }

            // The lender must come back exactly as it went in, after the forward load alone.
            if numDirty > 0 {
                ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
                let dirtyResidue = MResetEachZ(dirty);
                if not All(r -> r == Zero, dirtyResidue) {
                    Message($"FAIL dirty disturbed: addr={addr}, residue={dirtyResidue}");
                    set allCorrect = false;
                }
            }
            ApplyXorInPlace(addr, address);
        }

        allCorrect
    }

    /// One dirty load undone by the width-0 lookup is the identity over superposed address and lender,
    /// catching phases the basis-state check cannot see; a second dirty load could cancel them.
    operation TestSelectSwapDirtyPhaseAgreement(
        data : Bool[][],
        numSwapBits : Int
    ) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let numDirty = MaxI(1, SelectSwapDirtyBorrowedQubits(numSwapBits, m));

        use address = Qubit[nAddr];
        use dirty = Qubit[numDirty];
        use output = Qubit[m];

        ApplyToEachA(H, address);
        ApplyToEachA(H, dirty);

        SelectSwapDirty(numSwapBits, data, address, dirty, output);
        // Width 0 is a plain `Select`, which never touches the lender and XORs the word back out.
        SelectSwapDirty(0, data, address, dirty, output);

        Adjoint ApplyToEachA(H, dirty);
        Adjoint ApplyToEachA(H, address);

        All(r -> r == Zero, MResetEachZ(address + dirty + output))
    }

    /// A borrowed `SelectSwap2D` loads what the clean one loads on every address, including
    /// aliased surplus rows, and restores a seeded, non-trivial lender.
    operation TestSelectSwap2DDirtyMatchesClean(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        dirtySeed : Int,
    ) : Bool {
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));
        let numDirty = SelectSwapDirtyBorrowedQubits(numSwapBits, m);

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use dirty = Qubit[MaxI(1, numDirty)];
        use target = Qubit[m];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for i in 0..2^nOuterAddr - 1 {
            for j in 0..2^nInnerAddr - 1 {
                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);
                if numDirty > 0 {
                    ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
                }

                // The erasure's phase on a basis-state address is global, so it cannot affect the comparison.
                within {
                    SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, dirty, target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }
                within {
                    SelectSwap2D(data, 0, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }

                let residue = MResetEachZ(copy);
                if not All(r -> r == Zero, residue) {
                    Message($"FAIL word: outer={i}, inner={j}, differs from clean in {residue}");
                    set allCorrect = false;
                }

                // The lender must come back exactly as it went in.
                if numDirty > 0 {
                    ApplyXorInPlace(dirtySeed % 2^numDirty, dirty[...numDirty - 1]);
                    let dirtyResidue = MResetEachZ(dirty);
                    if not All(r -> r == Zero, dirtyResidue) {
                        Message($"FAIL dirty disturbed: outer={i}, inner={j}, residue={dirtyResidue}");
                        set allCorrect = false;
                    }
                }

                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);
            }
        }

        allCorrect
    }

    /// One borrowed `SelectSwap2D` load undone by the width-0 lookup is the identity over superposed
    /// addresses and lender, so a relative phase left on a borrowed qubit fails the Hadamard-basis check.
    operation TestSelectSwap2DDirtyPhaseAgreement(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
    ) : Bool {
        let m = Length(data[0][0]);
        let nOuter = Length(data);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));
        let numDirty = SelectSwapDirtyBorrowedQubits(numSwapBits, m);
        Fact(numDirty > 0, "a borrowed load needs numSwapBits > 0");

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use dirty = Qubit[numDirty];
        use target = Qubit[m];

        within {
            if outerAddressAlwaysValid {
                PrepareUniformSuperposition(nOuter, outerAddr);
            } else {
                ApplyToEachA(H, outerAddr);
            }
            ApplyToEachA(H, innerAddr);
            ApplyToEachA(H, dirty);
        } apply {
            SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, dirty, target);
            // Width 0 is a plain `Select` that never touches the lender and XORs the word back out.
            SelectSwap2D(data, 0, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
        }

        All(r -> r == Zero, MResetEachZ(outerAddr + innerAddr + dirty + target))
    }

    /// Traces one `SelectSwap2D` load and its erasure on `numDirty` lent qubits for a costing regression.
    operation TestSelectSwap2DDirtyResourceProbe(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        numDirty : Int,
    ) : Unit {
        use outerAddr = Qubit[Ceiling(Lg(IntAsDouble(Length(data))))];
        use innerAddr = Qubit[Ceiling(Lg(IntAsDouble(Length(data[0]))))];
        use dirty = Qubit[numDirty];
        use target = Qubit[Length(data[0][0])];

        SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, dirty, target);
        Adjoint SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, dirty, target);
    }
}
