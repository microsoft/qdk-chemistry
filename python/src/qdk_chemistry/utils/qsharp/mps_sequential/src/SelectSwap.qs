// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SELECT-SWAP network for efficient QROM data loading.
///
/// Implements the SELECT-SWAP technique that trades ancilla qubits for
/// reduced T-gate count when loading classical data into quantum registers.
/// Uses measurement-based uncomputation for the adjoint.
///
/// Operations:
///   SelectSwap — loads data[address] into output using SWAP network
///
/// References:
///   Low et al. arXiv:1805.03662
///   Berry et al. arXiv:1902.02134
namespace QDKChemistry.Utils.SelectSwap {

    import Std.Arrays.Chunks;
    import Std.Arrays.Enumerated;
    import Std.Arrays.Flattened;
    import Std.Arrays.IsEmpty;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Padded;
    import Std.Arrays.Partitioned;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Floor;
    import Std.Math.Lg;
    import Std.Math.MaxI;
    import Std.Math.MinI;
    import Std.TableLookup.Select;
    import PhaseGradient.RyViaPhaseGradient;

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
    //  1D CONTROLLED QROAM-CLEAN (forward-only Select+Swap with Unlookup)
    // ═══════════════════════════════════════════════════════════════════════════

    /// # Summary
    /// Controlled QROM load using forward-only SelectSwap with measurement-based
    /// uncomputation (QROAMClean pattern with blocking).
    ///
    /// # Description
    /// Loads `data[address]` into an internal register controlled on `control`,
    /// applies Ry rotation on `activeQubit` using the loaded angle, then
    /// uncomputes the loaded data using measurement-based Adjoint Select.
    ///
    /// When `control = |0⟩`: no data loaded, Ry(0) = identity.
    /// When `control = |1⟩`: loads data[address], applies Ry, uncomputes.
    ///
    /// Uses SelectSwap blocking (lambda > 0) for the forward load to reduce
    /// T-gate count compared to plain Controlled Select.
    ///
    /// # Input
    /// ## data
    /// Bool[N][m]: N angle entries of m bits each.
    /// ## address
    /// Address register (at least ceil(lg(N)) qubits).
    /// ## control
    /// Control qubit.
    /// ## activeQubit
    /// Target qubit for Ry rotation.
    /// ## phaseGradient
    /// Phase gradient register for Ry via phase gradient.
    operation ControlledQroamCleanRotation(
        data : Bool[][],
        address : Qubit[],
        control : Qubit,
        activeQubit : Qubit,
        phaseGradient : Qubit[]
    ) : Unit {
        let N = Length(data);
        Fact(N > 0, "data cannot be empty");
        let m = Length(data[0]);
        let nRequired = Ceiling(Lg(IntAsDouble(N)));
        let addressFitted = address[...nRequired - 1];
        let lambda = ComputeOptimalLambdaControlled1D(N, m);

        if lambda == 0 {
            // No blocking: Controlled Select + Ry + Adjoint Select (Unlookup)
            let zeros = Repeated(Repeated(false, m), N);
            let extendedData = zeros + data;
            use angleReg = Qubit[m];
            Controlled Select([control], (data, addressFitted, angleReg));
            RyViaPhaseGradient(activeQubit, angleReg, phaseGradient);
            Adjoint Select(extendedData, addressFitted + [control], angleReg);
        } else {
            // With blocking: Controlled Select on N/K entries + Swap + Ry + Unlookup
            let k = nRequired - lambda;
            let addressParts = Partitioned([k, lambda], addressFitted);

            let paddedData = CreatePaddedData(data, nRequired, m, k);
            let zeros = Repeated(Repeated(false, m), N);
            let extPaddedData = CreatePaddedData(zeros + data, nRequired + 1, m, k + 1);

            use dataReg = Qubit[m * (1 <<< lambda)];
            let chunks = Chunks(m, dataReg);

            // Forward: Controlled Select loads blocked data into dataReg
            Controlled Select([control], (paddedData, addressParts[0], dataReg));
            // Swap: move correct chunk to position 0
            SwapDataOutputs(addressParts[1], chunks);
            // Rotation: Ry on activeQubit using chunks[0] as angle
            RyViaPhaseGradient(activeQubit, chunks[0], phaseGradient);
            // Uncompute: measurement-based Unlookup on extended padded data
            Adjoint Select(extPaddedData, addressParts[0] + addressParts[1] + [control], dataReg);
        }
    }

    /// # Summary
    /// Computes optimal lambda (number of swap bits) for controlled forward-only
    /// QROAMClean pattern.
    ///
    /// # Description
    /// The cost model for controlled forward-only SelectSwap is:
    ///   Controlled Select(N/K) + SwapDataOutputs + PhaseLookup(Unlookup)
    /// This differs from the standard SelectSwapCost which models a full round trip.
    internal function ComputeOptimalLambdaControlled1D(numData : Int, numBits : Int) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));

        mutable best = ControlledQroamCleanCost(0, numData, numBits);
        mutable bestLambda = 0;

        for lambda in 1..addressBits - 1 {
            let cost = ControlledQroamCleanCost(lambda, numData, numBits);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
    }

    /// Cost model for controlled forward-only QROAMClean:
    ///   Controlled Select (N/K entries) + Swap + Unlookup (PhaseLookup)
    internal function ControlledQroamCleanCost(lambda : Int, numData : Int, numBits : Int) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));

        // Controlled Select on padded data: 2^(addressBits-lambda) entries, +1 for control
        let ctrlSelectCost = 2^(addressBits - lambda) - 2 + 1;

        // Swap cost: (K-1) * m controlled SWAPs
        let swapCost = (2^lambda - 1) * numBits;

        // Unlookup (PhaseLookup) cost: depends on number of entries in extPaddedData
        // extPaddedData has 2^(k+1) = 2^(addressBits-lambda+1) entries
        let unlookupAddrBits = addressBits - lambda + 1;
        let n1 = unlookupAddrBits / 2;
        let n2 = unlookupAddrBits - n1;
        let unlookupCost = MaxI(0, 2^n1 - n1 - 1) + MaxI(0, 2^n2 - n2 - 1);

        return ctrlSelectCost + swapCost + unlookupCost;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  1D UNCONTROLLED QROAM-CLEAN (forward-only Select+Swap with Unlookup)
    // ═══════════════════════════════════════════════════════════════════════════

    /// # Summary
    /// QROM-clean rotation: loads angle data, applies Ry rotation, then
    /// uncomputes using measurement-based Adjoint Select.
    ///
    /// # Description
    /// Replaces `within { SelectSwap } apply { Ry }` which costs 2× the
    /// SelectSwap body (forward + adjoint). QROAMClean does forward-only +
    /// measurement-based Unlookup for roughly half the cost.
    ///
    /// # Input
    /// ## data
    /// Bool[N][m]: N angle entries of m bits each.
    /// ## address
    /// Address register (at least ceil(lg(N)) qubits).
    /// ## activeQubit
    /// Target qubit for Ry rotation.
    /// ## phaseGradient
    /// Phase gradient register for Ry via phase gradient.
    operation QroamCleanRotation(
        data : Bool[][],
        address : Qubit[],
        activeQubit : Qubit,
        phaseGradient : Qubit[]
    ) : Unit {
        let N = Length(data);
        Fact(N > 0, "data cannot be empty");
        let m = Length(data[0]);
        let nRequired = Ceiling(Lg(IntAsDouble(N)));
        let addressFitted = address[...nRequired - 1];
        let lambda = ComputeOptimalLambdaQroamClean1D(N, m);

        if lambda == 0 {
            // No blocking: Select + Ry + Adjoint Select (Unlookup)
            use angleReg = Qubit[m];
            Select(data, addressFitted, angleReg);
            RyViaPhaseGradient(activeQubit, angleReg, phaseGradient);
            Adjoint Select(data, addressFitted, angleReg);
        } else {
            // With blocking: Select on N/K entries + Swap + Ry + Unlookup
            let k = nRequired - lambda;
            let addressParts = Partitioned([k, lambda], addressFitted);
            let paddedData = CreatePaddedData(data, nRequired, m, k);

            use dataReg = Qubit[m * (1 <<< lambda)];
            let chunks = Chunks(m, dataReg);

            // Forward: Select loads blocked data into dataReg
            Select(paddedData, addressParts[0], dataReg);
            // Swap: move correct chunk to position 0
            SwapDataOutputs(addressParts[1], chunks);
            // Rotation: Ry on activeQubit using chunks[0] as angle
            RyViaPhaseGradient(activeQubit, chunks[0], phaseGradient);
            // Uncompute: measurement-based Unlookup
            Adjoint Select(paddedData, addressParts[0] + addressParts[1], dataReg);
        }
    }

    /// Computes optimal lambda for uncontrolled forward-only QROAMClean pattern.
    internal function ComputeOptimalLambdaQroamClean1D(numData : Int, numBits : Int) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));

        mutable best = QroamCleanCost(0, numData, numBits);
        mutable bestLambda = 0;

        for lambda in 1..addressBits - 1 {
            let cost = QroamCleanCost(lambda, numData, numBits);
            if cost < best {
                set bestLambda = lambda;
                set best = cost;
            }
        }

        return bestLambda;
    }

    /// Cost model for uncontrolled forward-only QROAMClean:
    ///   Select(N/K entries) + Swap + Unlookup(PhaseLookup)
    internal function QroamCleanCost(lambda : Int, numData : Int, numBits : Int) : Int {
        let addressBits = Ceiling(Lg(IntAsDouble(numData)));

        // Select on padded data: 2^(addressBits-lambda) entries
        let selectCost = 2^(addressBits - lambda) - 2;

        // Swap cost: (K-1) * m controlled SWAPs
        let swapCost = (2^lambda - 1) * numBits;

        // Unlookup (PhaseLookup): paddedData has 2^(addressBits-lambda) entries
        let unlookupAddrBits = addressBits - lambda;
        let n1 = unlookupAddrBits / 2;
        let n2 = unlookupAddrBits - n1;
        let unlookupCost = MaxI(0, 2^n1 - n1 - 1) + MaxI(0, 2^n2 - n2 - 1);

        return selectCost + swapCost + unlookupCost;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Internal: 1D helpers
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
    //  Internal: shared helpers
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
}
