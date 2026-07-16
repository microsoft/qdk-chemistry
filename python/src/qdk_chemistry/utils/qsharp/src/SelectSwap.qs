// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP network for efficient QROM data loading (1D and 2D).
///
/// Implements the SELECT-SWAP technique that trades ancilla qubits for
/// reduced T-gate count when loading classical data into quantum registers.
/// Uses measurement-based uncomputation for the adjoint.
///
/// 1D operations:
///   SelectSwap — loads data[address] into output.
///
/// 2D operations:
///   Select2DLoad — loads data[outer][inner] into target via unary iteration + SWAP
///   ComputeOptimalLambda2D — optimal SWAP bits for 2D case
///
/// References:
///   Low, Kliuchnikov, Schaeffer (arXiv:1812.00954)
namespace QDKChemistry.Utils.SelectSwap {

    import Std.Arrays.Chunks;
    import Std.Arrays.Enumerated;
    import Std.Arrays.Flattened;
    import Std.Arrays.Mapped;
    import Std.Arrays.IndexRange;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.MostAndTail;
    import Std.Arrays.Padded;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsDouble;
    import Std.Convert.ResultAsBool;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Floor;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.Measurement.MResetEachZ;
    import Std.TableLookup.Select;

    // ═══════════════════════════════════════════════════════════════════════════
    //  1D SELECT-SWAP
    // ═══════════════════════════════════════════════════════════════════════════

    operation SelectSwap(numSwapBits : Int, data : Bool[][], address : Qubit[], output : Qubit[]) : Unit is Adj + Ctl {
        let (n, nRequired) = DimensionsForSelect(data, address);
        let addressFitted = address[...nRequired - 1];

        let numSwapBits = numSwapBits == -1 ? ComputeOptimalLambda1D(Length(data), Length(data[0])) | numSwapBits;

        Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");

        if numSwapBits == 0 {
            Select(data, addressFitted, output);
        } else {
            WithSelectSwap(numSwapBits, data, address, intermediate => ApplyToEachCA(CNOT, Zipped(intermediate, output)));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  2D SELECT-SWAP (unary iteration over outer × SELECT-SWAP over inner)
    // ═══════════════════════════════════════════════════════════════════════════

    operation Select2DLoad(data : Bool[][][], outerAddress : Qubit[], innerAddress : Qubit[], numSwapBits : Int, target : Qubit[]) : Unit is Adj {
        body (...) {
            if numSwapBits == 0 {
                UnaryIteration(outerAddress, Length(data), (index) => {
                    Select(data[index], innerAddress, target);
                });
            } else {
                let (n, nRequired) = DimensionsForSelect(data[0], innerAddress);
                let innerAddressFitted = innerAddress[...nRequired - 1];

                let m = Length(data[0][0]);
                let l = numSwapBits;
                let k = nRequired - numSwapBits;

                let innerAddressParts = Partitioned([k, l], innerAddressFitted);
                let chunkedDataRegister = Chunks(m, target);

                UnaryIteration(outerAddress, Length(data), (index) => {
                    let dataArray = CreatePaddedData(data[index], nRequired, m, k);
                    Select(dataArray, innerAddressParts[0], target);
                });

                SwapDataOutputs(innerAddressParts[1], chunkedDataRegister);
            }
        }

        adjoint (...) {
            let (n, nRequired) = DimensionsForSelect(data[0], innerAddress);

            let mapOne : (Int -> Bool[][]) = (index) -> {
                CreatePaddedData(data[index], nRequired, Length(data[index][0]), nRequired - numSwapBits)
            };

            let flattenedData = Flattened(MappedOverRange(mapOne, IndexRange(data)));
            Adjoint Select(flattenedData, innerAddress + outerAddress, target);
        }
    }

    function ComputeOptimalLambda2D(numOuterData : Int, numInnerData : Int, numBits : Int) : Int {
        mutable best = 2^32;
        mutable bestLambda = 0;

        let addressBits = Ceiling(Lg(IntAsDouble(numInnerData)));
        for lambda in 0..addressBits - 1 {
            let cost = SelectSwapCost2D(lambda, numOuterData, numInnerData, numBits);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  Unary Iteration
    // ══════════════════════════════════════════════════════════════════════════

    operation UnaryIteration(
        address : Qubit[],
        numActions : Int,
        action : (Int => Unit is Adj + Ctl),
    ) : Unit is Adj {
        Fact(numActions > 0, "actions cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(numActions)));
        Fact(
            Length(address) >= n,
            $"address register is too small, requires at least {n} qubits",
        );

        if numActions == 1 {
            action(0);
        } else {
            let (most, tail) = MostAndTail(address[...n - 1]);

            within {
                X(tail);
            } apply {
                SinglyControlledUnaryIteration(tail, most, 2^(n - 1), 0, action);
            }

            SinglyControlledUnaryIteration(tail, most, numActions - 2^(n - 1), 2^(n - 1), action);
        }
    }

    internal operation SinglyControlledUnaryIteration(
        ctl : Qubit,
        address : Qubit[],
        numActions : Int,
        actionOffset : Int,
        action : (Int => Unit is Adj + Ctl),
    ) : Unit is Adj {
        Fact(numActions > 0, "actions cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(numActions)));
        Fact(
            Length(address) >= n,
            $"address register is too small, requires at least {n} qubits",
        );

        if numActions == 1 {
            Controlled action([ctl], actionOffset);
        } else {
            use helper = Qubit();

            let (most, tail) = MostAndTail(address[...n - 1]);

            within {
                X(tail);
            } apply {
                AND(ctl, tail, helper);
            }

            SinglyControlledUnaryIteration(helper, most, 2^(n - 1), actionOffset, action);

            CNOT(ctl, helper);

            SinglyControlledUnaryIteration(helper, most, numActions - 2^(n - 1), actionOffset + 2^(n - 1), action);

            Adjoint AND(ctl, tail, helper);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  1D helper functions
    // ═══════════════════════════════════════════════════════════════════════════

    internal operation WithSelectSwap(numSwapBits : Int, data : Bool[][], address : Qubit[], action : (Qubit[] => Unit is Adj + Ctl)) : Unit is Adj + Ctl {
        body (...) {
            let (n, nRequired) = DimensionsForSelect(data, address);
            let addressFitted = address[...nRequired - 1];

            Fact(numSwapBits <= nRequired, "Too many bits for SWAP network");
            Fact(not IsEmpty(data), "data cannot be empty");
            let m = Length(data[0]);

            if numSwapBits == 0 {
                use output = Qubit[m];
                within {
                    Select(data, address, output);
                } apply {
                    action(output);
                }
            } else {
                let numSelectBits = nRequired - numSwapBits;
                let addressParts = Partitioned([numSelectBits, numSwapBits], addressFitted);

                use dataRegister = Qubit[m * 2^numSwapBits];

                let dataArray = CreatePaddedData(data, nRequired, m, numSelectBits);
                let chunkedDataRegister = Chunks(m, dataRegister);

                within {
                    WithSelectSwapSelectPart(dataArray, addressParts, dataRegister, chunkedDataRegister);
                } apply {
                    action(chunkedDataRegister[0]);
                }
            }
        }

        adjoint self;
    }

    internal operation WithSelectSwapSelectPart(data : Bool[][], addressParts : Qubit[][], target : Qubit[], chunkedTarget : Qubit[][]) : Unit {
        body (...) {
            Select(data, addressParts[0], target);
            SwapDataOutputs(addressParts[1], chunkedTarget);
        }

        adjoint (...) {
            Adjoint Select(data, addressParts[0] + addressParts[1], target);
        }
    }

    internal function ComputeOptimalLambda1D(numData : Int, numBits : Int) : Int {
        mutable best = 2^32;
        mutable bestLambda = 0;

        let addressBits = Ceiling(Lg(IntAsDouble(numData)));
        for lambda in 0..addressBits - 1 {
            let cost = SelectSwapCost1D(lambda, numData, numBits);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
    }

    internal function SelectSwapCost1D(lambda : Int, numData : Int, numBits : Int) : Int {
        if lambda == 0 {
            return numData - 2;
        } else {
            let addressBits = Ceiling(Lg(IntAsDouble(numData)));
            let split = MinI(Floor(Lg(IntAsDouble(2^lambda * numBits))), addressBits - 1);

            let select_cost = 2^(addressBits - lambda) - 2;
            let unselect_cost = MaxI(0, 2^split - 2) + 2^(addressBits - split) - 2;
            let swap_cost = (2^lambda - 1) * numBits;

            return select_cost + unselect_cost + swap_cost;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  2D helper functions
    // ═══════════════════════════════════════════════════════════════════════════

    internal function SelectSwapCost2D(lambda : Int, numOuterData : Int, numInnerData : Int, numBits : Int) : Int {
        let outerAddressBits = Ceiling(Lg(IntAsDouble(numOuterData)));
        let innerAddressBits = Ceiling(Lg(IntAsDouble(numInnerData)));
        let split = MinI(Floor(Lg(IntAsDouble(2^lambda * numBits))), (outerAddressBits + innerAddressBits) - 1);

        let unselect_cost = MaxI(0, 2^split - 2) + 2^(outerAddressBits + innerAddressBits - split) - 2;

        if lambda == 0 {
            return (numOuterData - 2) + numOuterData * (numInnerData - 2) + unselect_cost;
        } else {
            let select_cost = 2^(innerAddressBits - lambda) - 2;
            let swap_cost = (2^lambda - 1) * numBits;

            return numOuterData * select_cost + swap_cost + unselect_cost;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  shared helpers
    // ═══════════════════════════════════════════════════════════════════════════

    internal function DimensionsForSelect(data : Bool[][], address : Qubit[]) : (Int, Int) {
        let N = Length(data);
        Fact(N > 0, "data cannot be empty");

        let n = Ceiling(Lg(IntAsDouble(N)));
        Fact(Length(address) >= n, $"address register is too small, requires at least {n} qubits");

        return (N, n);
    }

    internal function CreatePaddedData(data : Bool[][], nRequired : Int, m : Int, k : Int) : Bool[][] {
        let dataPadded = Padded(-2^nRequired, [false, size = m], data);

        MappedOverRange(i -> Flattened(dataPadded[i..2^k..2^nRequired - 1]), 0..2^k - 1)
    }

    internal operation SwapDataOutputs(address : Qubit[], outputs : Qubit[][]) : Unit is Adj {
        let l = Length(address);
        for (i, control) in Enumerated(address) {
            let innerStepSize = 2^i;
            let outerStepSize = 2^(i + 1);
            let numSwaps = 2^l / 2^(i + 1);
            for j in 0..numSwaps - 1 {
                let targets1 = outputs[j * outerStepSize];
                let targets2 = outputs[j * outerStepSize + innerStepSize];
                ApplyToEachA(ts => Controlled SWAP([control], ts), Zipped(targets1, targets2));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test & estimation wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// 1D SelectSwap correctness: set address to |addr⟩, apply SelectSwap in within/apply,
    /// CNOT result to persistent copy register, then verify copy matches expected data.
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

    /// 2D Select2DLoad correctness: for each (i,j), load data[i][j] into target,
    /// CNOT to copy, verify.
    operation TestSelect2DLoadCorrectness(
        data : Bool[][][],
        numSwapBits : Int
    ) : Bool {
        let nOuter = Length(data);
        let nInner = Length(data[0]);
        let m = Length(data[0][0]);
        let nOuterAddr = Ceiling(Lg(IntAsDouble(nOuter)));
        let nInnerAddr = Ceiling(Lg(IntAsDouble(nInner)));
        let nTarget = if numSwapBits > 0 { m * (1 <<< numSwapBits) } else { m };

        use outerAddr = Qubit[nOuterAddr];
        use innerAddr = Qubit[nInnerAddr];
        use target = Qubit[nTarget];
        use copy = Qubit[m];

        mutable allCorrect = true;

        for i in 0..nOuter - 1 {
            for j in 0..nInner - 1 {
                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);

                within {
                    Select2DLoad(
                        data, outerAddr, innerAddr, numSwapBits, target);
                } apply {
                    ApplyToEachCA(CNOT, Zipped(target[0..m - 1], copy));
                }

                ApplyXorInPlace(i, outerAddr);
                ApplyXorInPlace(j, innerAddr);

                let actual = Mapped(ResultAsBool, MResetEachZ(copy));
                if actual != data[i][j] {
                    Message($"FAIL: (i={i},j={j}), actual={actual}, expected={data[i][j]}");
                    set allCorrect = false;
                }
            }
        }

        allCorrect
    }
}
