// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HammingWeightPhasing {

    import QDKChemistry.Utils.Arithmetic.RippleCarryCGIncByLEWithCarryAction;
    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Mapped;
    import Std.Arrays.Tail;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;

    /// Rotates one nonempty Pauli string onto a single Z representative.
    internal operation MapPauliTermToSingleZ(ops : Pauli[], targets : Qubit[]) : Unit is Adj + Ctl {
        Fact(Length(ops) == Length(targets), "MapPauliTermToSingleZ needs one Pauli axis per target.");
        Fact(Length(ops) > 0, "MapPauliTermToSingleZ needs a non-empty Pauli string.");
        for i in 0..Length(ops) - 1 {
            if ops[i] == PauliX {
                H(targets[i]);
            } elif ops[i] == PauliY {
                Adjoint S(targets[i]);
                H(targets[i]);
            } else {
                Fact(ops[i] == PauliZ, "MapPauliTermToSingleZ does not accept PauliI.");
            }
        }
        let last = Length(targets) - 1;
        for i in 0..last - 1 {
            CNOT(targets[i], targets[last]);
        }
    }

    /// Plans an optimal adder tree for a Hamming-weight computation.
    /// See `HammingWeightPhase` for the technique and its attribution.
    internal function HammingWeightSchedule(count : Int) : ((Int, Int, Int, Int)[], Int[], Int) {
        if count <= 0 {
            return ([], [], 0);
        }
        let levels = BitSizeI(count);
        mutable initial : Int[] = [];
        for i in 0..count - 1 {
            set initial += [i];
        }
        mutable buckets : Int[][] = [[], size = levels + 1];
        set buckets w/= 0 <- initial;
        mutable schedule : (Int, Int, Int, Int)[] = [];
        mutable next = count;

        for k in 0..levels - 1 {
            mutable current = buckets[k];
            mutable upper = buckets[k + 1];
            while Length(current) >= 3 {
                set schedule += [(current[0], current[1], current[2], next)];
                set current = current[3...] + [current[2]];
                set upper += [next];
                set next += 1;
            }
            if Length(current) == 2 {
                set schedule += [(current[0], current[1], -1, next)];
                set current = [current[1]];
                set upper += [next];
                set next += 1;
            }
            set buckets w/= k <- current;
            set buckets w/= k + 1 <- upper;
        }

        mutable finalBits : Int[] = [];
        for k in 0..levels - 1 {
            set finalBits += [Length(buckets[k]) == 1 ? buckets[k][0] | -1];
        }
        return (schedule, finalBits, next);
    }

    /// Compresses three equal-significance bits into a sum and carry.
    internal operation FullAdderStep(a : Qubit, b : Qubit, c : Qubit, carry : Qubit) : Unit is Adj {
        CNOT(a, b);
        CNOT(a, c);
        AND(b, c, carry);
        CNOT(a, carry);
        CNOT(b, c);
        CNOT(a, c);
    }

    /// Compresses two equal-significance bits into a sum and carry.
    internal operation HalfAdderStep(a : Qubit, b : Qubit, carry : Qubit) : Unit is Adj {
        AND(a, b, carry);
        CNOT(a, b);
    }

    /// Rotates every term onto a single Z on its last qubit and computes the Hamming weight of
    /// those qubits with the adder tree `schedule`, indexed into `work`.
    internal operation ComputeHammingWeight(
        pauliOps : Pauli[][],
        targets : Qubit[][],
        schedule : (Int, Int, Int, Int)[],
        work : Qubit[]
    ) : Unit is Adj {
        for t in 0..Length(targets) - 1 {
            MapPauliTermToSingleZ(pauliOps[t], targets[t]);
        }
        for (a, b, c, carry) in schedule {
            if c < 0 {
                HalfAdderStep(work[a], work[b], work[carry]);
            } else {
                FullAdderStep(work[a], work[b], work[c], work[carry]);
            }
        }
    }

    /// # Summary
    /// Applies the same-angle rotation exp(-i·theta·P_t) to every term P_t through their Hamming
    /// weight :cite:`Kan2025`.
    ///
    /// # Description
    /// Each term is mapped onto a single Z, and an adder tree sums those qubits into a Hamming
    /// weight w for each batch of at most `maxBatchSize` terms. If a catalyst prepared at `theta` is
    /// given, adding w into it applies e^{2i·theta·w}, and the constant phase e^{-i·theta·count} is
    /// left to the caller. Otherwise each place value of w gets its own `Rz`, and the constant phase
    /// is applied here.
    ///
    /// # Input
    /// ## theta
    /// The rotation angle shared by every term.
    /// ## pauliOps
    /// The Pauli string of each term.
    /// ## targets
    /// The qubits of each term, disjoint across terms.
    /// ## useHammingWeightPhasing
    /// Whether to phase the tower through a Hamming-weight register rather than term by term.
    /// ## maxBatchSize
    /// Largest tower phased through a single Hamming-weight register, or -1 for no cap.
    /// ## catalyst
    /// A register prepared by `PrepareCatalyst(theta, _)`, with at least one qubit per place value
    /// of the largest batch's weight, and returned unchanged; or empty for the `Rz` ladder. Unused
    /// when `useHammingWeightPhasing` is false.
    internal operation HammingWeightPhase(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
        let count = Length(targets);
        Fact(Length(pauliOps) == count, "HammingWeightPhase needs one axis list per term.");
        if not useHammingWeightPhasing {
            for t in 0..count - 1 {
                Exp(pauliOps[t], -theta, targets[t]);
            }
        } elif count > 0 {
            let batchSize = HammingWeightBatchSize(count, maxBatchSize);
            let opBatches = Chunks(batchSize, pauliOps);
            let targetBatches = Chunks(batchSize, targets);
            for index in 0..Length(targetBatches) - 1 {
                let batch = targetBatches[index];
                let size = Length(batch);
                let inputs = Mapped(term -> Tail(term), batch);
                let (schedule, finalBits, _) = HammingWeightSchedule(size);
                Fact(All(bit -> bit >= 0, finalBits), "Every place value of the Hamming weight must hold a bit.");
                let places = Length(finalBits);
                use scratch = Qubit[Length(schedule)];
                let work = inputs + scratch;
                within {
                    ComputeHammingWeight(opBatches[index], batch, schedule, work);
                } apply {
                    let weight = Mapped(bit -> work[bit], finalBits);
                    if Length(catalyst) == 0 {
                        for j in 0..places - 1 {
                            Rz(2.0 * theta * IntAsDouble(1 <<< j), weight[j]);
                        }
                        // The ladder's angles sum to 2·theta·(2^places - 1). Under control this phase is not global.
                        R(PauliI, 2.0 * theta * IntAsDouble(size - ((1 <<< places) - 1)), inputs[0]);
                    } else {
                        AddIntoCatalyst(theta, weight, catalyst);
                    }
                }
            }
        }
    }

    /// Adds `weight` into a catalyst prepared by `PrepareCatalyst(theta, _)`, phasing it by
    /// e^{2i·theta·weight} (Supplementary Fig. 3 of :cite:`Kan2025`).
    internal operation AddIntoCatalyst(theta : Double, weight : Qubit[], catalyst : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            let places = Length(weight);
            Fact(Length(catalyst) >= places, "The catalyst needs a qubit per place value of the weight.");
            // Adding the weight phases the catalyst except where the sum wraps, which the carry-out restores.
            RippleCarryCGIncByLEWithCarryAction(
                R1(2.0 * theta * IntAsDouble(1 <<< places), _),
                weight,
                catalyst[0..places - 1]
            );
        }
        controlled (controls, ...) {
            // With the controls clear the addend is zero, which carries nothing, so only the weight
            // is gated, by one AND per bit once several controls are ANDed onto one qubit.
            let joined = Length(controls) > 1;
            use joint = Qubit[joined ? 1 | 0];
            let gate = joined ? joint | controls;
            use gated = Qubit[Length(gate) > 0 ? Length(weight) | 0];
            within {
                if joined {
                    Controlled X(controls, joint[0]);
                }
                for j in 0..Length(gated) - 1 {
                    AND(gate[0], weight[j], gated[j]);
                }
            } apply {
                AddIntoCatalyst(theta, Length(gated) > 0 ? gated | weight, catalyst);
            }
        }
    }

    /// Prepares the catalyst Σ_k e^{-2i·theta·k}|k⟩ (little-endian) that `HammingWeightPhase` adds
    /// Hamming weights into.
    internal operation PrepareCatalyst(theta : Double, catalyst : Qubit[]) : Unit is Adj + Ctl {
        for j in 0..Length(catalyst) - 1 {
            H(catalyst[j]);
            R1(-2.0 * theta * IntAsDouble(1 <<< j), catalyst[j]);
        }
    }

    /// Returns `count` capped at `maxBatchSize`, or `count` itself when the cap is -1.
    internal function HammingWeightBatchSize(count : Int, maxBatchSize : Int) : Int {
        Fact(maxBatchSize == -1 or maxBatchSize > 0, "maxBatchSize must be -1 or positive.");
        return maxBatchSize == -1 or maxBatchSize > count ? count | maxBatchSize;
    }
}
