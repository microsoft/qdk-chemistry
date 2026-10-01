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
    import QDKChemistry.Utils.UnaryIteration.UnaryIteration;

    operation TestSelectSwap1DCorrectness(
        data : Bool[][],
        numSwapBits : Int
    ) : Bool {
        let nData = Length(data);
        let m = Length(data[0]);
        let nAddr = Ceiling(Lg(IntAsDouble(nData)));

        use address = Qubit[nAddr];
        use output = Qubit[m];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for addr in 0..nData - 1 {
            ApplyXorInPlace(addr, address);

            within {
                SelectSwap(numSwapBits, data, address, output);
            } apply {
                ApplyToEachCA(CNOT, Zipped(output, copy));
            }

            ApplyXorInPlace(addr, address);

            let actual = Mapped(ResultAsBool, MResetEachZ(copy));
            if actual != data[addr] {
                Message($"FAIL: addr={addr}, actual={actual}, expected={data[addr]}");
                set allCorrect = false;
            }
        }

        allCorrect
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

    operation TestSelectSwap2DCorrectness(
        data : Bool[][][],
        numSwapBits : Int,
        outerAddressAlwaysValid : Bool
    ) : Bool {
        let nOuter = Length(data);
        let nInner = Length(data[0]);
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(nInner)));

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[m];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for outerIndex in 0..nOuter - 1 {
            for innerIndex in 0..nInner - 1 {
                ApplyXorInPlace(outerIndex, outerAddr);
                ApplyXorInPlace(innerIndex, innerAddr);

                within {
                    SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target, copy));
                }

                ApplyXorInPlace(outerIndex, outerAddr);
                ApplyXorInPlace(innerIndex, innerAddr);

                let actual = Mapped(ResultAsBool, MResetEachZ(copy));
                if actual != data[outerIndex][innerIndex] {
                    Message($"FAIL: (i={outerIndex},j={innerIndex}), actual={actual}, expected={data[outerIndex][innerIndex]}");
                    set allCorrect = false;
                }
            }
        }

        allCorrect
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
                    SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
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
            SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
        }
        if applyAdjoint {
            Adjoint SelectSwap2D(data, outerAddr, innerAddr, numSwapBits, outerAddressAlwaysValid, target);
        }
    }
}
