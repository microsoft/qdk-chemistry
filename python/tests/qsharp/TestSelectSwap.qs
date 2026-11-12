// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.SelectSwapTests {
    import Std.Arrays.All;
    import Std.Arrays.Mapped;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Convert.ResultAsBool;
    import Std.Core.Length;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Measurement.MResetEachZ;
    import Std.StatePreparation.PrepareUniformSuperposition;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SelectSwap.PadToAddressSpace;
    import QDKChemistry.Utils.SelectSwap.SelectSwap;
    import QDKChemistry.Utils.SelectSwap.SelectSwap2D;
    import QDKChemistry.Utils.SelectSwap.SelectSwapAliased;
    import QDKChemistry.Utils.SelectSwap.SelectSwapClean;
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;

    /// Loads `data[address]` into `output`, copies it into `copy`, then erases `output`.
    function TestMakeSelectSwap1DCopyOp(
        data : Bool[][],
        numSwapBits : Int
    ) : (Qubit[], Qubit[], Qubit[]) => Unit {
        (address, output, copy) => {
            within {
                SelectSwap(numSwapBits, data, address, output);
            } apply {
                ApplyToEachCA(CNOT, Zipped(output, copy));
            }
        }
    }

    operation TestSelectSwap1DPhaseAgreement(data : Bool[][], numSwapBits : Int) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));

        use address = Qubit[nAddr];
        ApplyToEachA(H, address);

        {
            use output = Qubit[m];
            within {
                SelectSwap(numSwapBits, data, address, output);
            } apply {
                Z(output[0]);
            }
        }
        {
            use output = Qubit[m];
            within {
                SelectSwap(0, data, address, output);
            } apply {
                Z(output[0]);
            }
        }

        Adjoint ApplyToEachA(H, address);

        All(result -> result == Zero, MResetEachZ(address))
    }

    /// Checks `SelectSwapAliased` against a bare `Select` by value on every address, including
    /// the surplus ones where zero-padding and aliasing differ.
    operation TestSelectSwapAliasedMatchesSelect1D(data : Bool[][], numSwapBits : Int) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));

        mutable allCorrect = true;
        for addr in 0..2^nAddr - 1 {
            use address = Qubit[nAddr];
            use swapped = Qubit[m];
            use plain = Qubit[m];

            ApplyXorInPlace(addr, address);
            SelectSwapAliased(numSwapBits, data, address, swapped);
            Select(data, address, plain);
            ApplyXorInPlace(addr, address);

            let swappedWord = Mapped(ResultAsBool, MResetEachZ(swapped));
            let plainWord = Mapped(ResultAsBool, MResetEachZ(plain));
            if swappedWord != plainWord {
                Message($"FAIL: addr={addr}, select-swap={swappedWord}, select={plainWord}");
                set allCorrect = false;
            }
            if not All(r -> r == Zero, MResetEachZ(address)) {
                Message($"FAIL: addr={addr} left the address register disturbed");
                set allCorrect = false;
            }
        }

        allCorrect
    }

    /// `SelectSwapClean` must match a zero-padded `Select` as a phase oracle through its erasure.
    operation TestSelectSwapCleanPhaseAgreement(
        data : Bool[][],
        numSwapBits : Int,
        useControl : Bool
    ) : Bool {
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let padded = PadToAddressSpace(data, nAddr);

        use control = Qubit();
        use address = Qubit[nAddr];
        use cleanOutput = Qubit[m];
        use referenceOutput = Qubit[m];
        within {
            H(control);
            ApplyToEachA(H, address);
        } apply {
            within {
                if useControl {
                    Controlled SelectSwapClean([control], (numSwapBits, data, address, cleanOutput));
                } else {
                    SelectSwapClean(numSwapBits, data, address, cleanOutput);
                }
            } apply {
                ApplyToEachA(Z, cleanOutput);
            }
            within {
                if useControl {
                    Controlled Select([control], (padded, address, referenceOutput));
                } else {
                    Select(padded, address, referenceOutput);
                }
            } apply {
                ApplyToEachA(Z, referenceOutput);
            }
        }

        All(r -> r == Zero, MResetEachZ([control] + address + cleanOutput + referenceOutput))
    }

    /// Traces `SelectSwapClean` in one or both directions for a costing regression.
    operation TestSelectSwapCleanResourceProbe(
        data : Bool[][],
        numSwapBits : Int,
        applyForward : Bool,
        applyAdjoint : Bool
    ) : Unit {
        use address = Qubit[Ceiling(Lg(IntAsDouble(Length(data))))];
        use output = Qubit[Length(data[0])];

        if applyForward {
            SelectSwapClean(numSwapBits, data, address, output);
        }
        if applyAdjoint {
            Adjoint SelectSwapClean(numSwapBits, data, address, output);
        }
    }

    /// Loads `data[outer][inner]` into `target`, copies it into `copy`, then erases `target`.
    function TestMakeSelectSwap2DCopyOp(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool
    ) : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit {
        (outerAddr, innerAddr, target, copy) => {
            within {
                SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
            } apply {
                ApplyToEachCA(CNOT, Zipped(target, copy));
            }
        }
    }

    operation TestSelectSwap2DPhaseAgreement(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool
    ) : Bool {
        let m = Length(data[0][0]);
        let nOuter = Length(data);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        within {
            if outerAddressAlwaysValid {
                PrepareUniformSuperposition(nOuter, outerAddr);
            } else {
                ApplyToEachA(H, outerAddr);
            }
            ApplyToEachA(H, innerAddr);
        } apply {
            {
                use target = Qubit[m];
                within {
                    SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
                } apply {
                    Z(target[0]);
                }
            }
            {
                use target = Qubit[m];
                within {
                    UnaryIteration(outerAddr, Length(data), (index) => {
                        Select(PadToAddressSpace(data[index], nInnerAddr), innerAddr, target);
                    });
                } apply {
                    Z(target[0]);
                }
            }
        }

        All(result -> result == Zero, MResetEachZ(outerAddr + innerAddr))
    }

    operation TestSelectSwap2DResourceProbe(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool,
        applyForward : Bool,
        applyAdjoint : Bool
    ) : Unit {
        let nOuterAddr = Ceiling(Lg(IntAsDouble(Length(data))));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(Length(data[0]))));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[Length(data[0][0])];

        if applyForward {
            SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
        }
        if applyAdjoint {
            Adjoint SelectSwap2D(data, numSwapBits, outerAddressAlwaysValid, outerAddr, innerAddr, [], target);
        }
    }
}
