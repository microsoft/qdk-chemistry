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
    /// Applies exp(-i theta P_t) for every term P_t, phasing the tower through a Hamming-weight
    /// register. With a catalyst the constant phase e^{-i theta count} is left to the caller.
    ///
    /// # Description
    /// Each term is first rotated onto a single Z on its last qubit, so the batch becomes
    /// `count` equal-angle rotations exp(-i theta Z). Their product is
    /// e^{-i theta count} e^{2 i theta w}, where w is the Hamming weight of those qubits. An
    /// adder tree computes w into log2(count) + 1 bits, which are phased at a cost that grows with
    /// the bits rather than the terms. This is the construction of :cite:`Kan2025`,
    /// whose Methods diagonalize every
    /// plaquette and every on-site pair into a layer of same-angle R_z gates and synthesize that
    /// layer collectively with HWP. When `useHammingWeightPhasing` is false, each term is applied
    /// as its own rotation instead.
    ///
    /// Without a catalyst, e^{2i·theta·w} is one `Rz(2·theta·2^j)` per place value 2^j of w, the
    /// original rotation ladder. `Rz(a) = e^{-ia/2} R1(a)`, so a single `R(PauliI, g)` on the first
    /// input both cancels the ladder's extra phase and supplies the batch constant
    /// e^{-i·theta·count}. Under control that phase is not global, which is why it is applied
    /// rather than dropped.
    ///
    /// With a catalyst, w is added into it once instead, the catalyzed Hamming-weight phasing of
    /// :cite:`Kan2025`. The catalyst holds Σ_y e^{-2i·theta·y}|y⟩ (see `PrepareCatalyst`), so adding
    /// w multiplies it by e^{2i·theta·w} and leaves it unchanged, except that a sum that wraps
    /// modulo 2^places drops 2^places from y, which one `R1(2·theta·2^places)` on the carry-out
    /// restores. The ripple-carry adder takes one AND per place value and measures its carries
    /// away, so a batch costs `places` AND operations and one rotation, however many place values
    /// w has. Under control only the weight is gated, at one more AND per place value: with the
    /// control clear the addend is zero, which carries nothing, so neither the catalyst nor the
    /// carry rotation needs a control. This path leaves the batch constant e^{-i·theta·count} to
    /// the caller, which can apply the constants of every tower as one rotation.
    ///
    /// A tower longer than `maxBatchSize` is split into consecutive batches of at most that many
    /// terms, each phased through its own Hamming-weight register. The phases are additive over
    /// the split, so the result is unchanged; only the cost moves. See `HammingWeightBatchSize`.
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
    /// A register `PrepareCatalyst(theta, catalyst)` prepared, with a qubit per place value of the
    /// largest batch's weight, which is returned in that state; or empty to synthesize every place
    /// value as an `Rz`. Unused when `useHammingWeightPhasing` is false.
    internal operation HammingWeightPhase(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
        body (...) {
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
                            // The ladder's angles sum to 2·theta·(2^places - 1).
                            R(PauliI, 2.0 * theta * IntAsDouble(size - ((1 <<< places) - 1)), inputs[0]);
                        } else {
                            Fact(Length(catalyst) >= places, "The catalyst needs a qubit per place value of the weight.");
                            // Adding the weight phases the catalyst except where the sum wraps, which the carry-out restores.
                            RippleCarryCGIncByLEWithCarryAction(
                                R1(2.0 * theta * IntAsDouble(1 <<< places), _),
                                weight,
                                catalyst[0..places - 1]
                            );
                        }
                    }
                }
            }
        }
        controlled (controls, ...) {
            let count = Length(targets);
            Fact(Length(pauliOps) == count, "HammingWeightPhase needs one axis list per term.");
            if not useHammingWeightPhasing {
                for t in 0..count - 1 {
                    Controlled Exp(controls, (pauliOps[t], -theta, targets[t]));
                }
            } elif count > 0 {
                // Chunking in a function keeps the index arithmetic out of this adjointable body.
                let batchSize = HammingWeightBatchSize(count, maxBatchSize);
                let opBatches = Chunks(batchSize, pauliOps);
                let targetBatches = Chunks(batchSize, targets);
                // Several controls are first ANDed onto one qubit, so each weight bit is gated by one AND.
                let joined = Length(catalyst) > 0 and Length(controls) > 1;
                use joint = Qubit[joined ? 1 | 0];
                within {
                    if joined {
                        Controlled X(controls, joint[0]);
                    }
                } apply {
                    let gate = joined ? joint | controls;
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
                                    Controlled Rz(controls, (2.0 * theta * IntAsDouble(1 <<< j), weight[j]));
                                }
                                // The ladder's angles sum to 2·theta·(2^places - 1).
                                let constant = 2.0 * theta * IntAsDouble(size - ((1 <<< places) - 1));
                                Controlled R(controls, (PauliI, constant, inputs[0]));
                            } else {
                                Fact(Length(catalyst) >= places, "The catalyst needs a qubit per place value of the weight.");
                                use gated = Qubit[Length(gate) > 0 ? places | 0];
                                within {
                                    for j in 0..Length(gated) - 1 {
                                        AND(gate[0], weight[j], gated[j]);
                                    }
                                } apply {
                                    // Adding the weight phases the catalyst except where the sum wraps, which the carry-out restores.
                                    RippleCarryCGIncByLEWithCarryAction(
                                        R1(2.0 * theta * IntAsDouble(1 <<< places), _),
                                        Length(gated) > 0 ? gated | weight,
                                        catalyst[0..places - 1]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// # Summary
    /// Prepares the catalyst through which `HammingWeightPhase` phases Hamming weights at `theta`.
    ///
    /// # Description
    /// The catalyst is the generalized phase gradient 2^{-n/2} Σ_y e^{-2i·theta·y}|y⟩ on n
    /// little-endian qubits. It is the product of (|0⟩ + e^{-2i·theta·2^j}|1⟩)/√2 over the qubits j,
    /// so it takes one rotation per qubit, and every addition returns it unchanged, so one
    /// preparation serves every tower at that angle. Each prefix of it is the catalyst of the same
    /// angle on fewer qubits, and dropping its first qubit leaves the catalyst of twice the angle.
    ///
    /// # Input
    /// ## theta
    /// The angle `HammingWeightPhase` is given.
    /// ## catalyst
    /// Qubits in |0⟩, at least one per place value of the largest weight to phase.
    internal operation PrepareCatalyst(theta : Double, catalyst : Qubit[]) : Unit is Adj + Ctl {
        for j in 0..Length(catalyst) - 1 {
            H(catalyst[j]);
            R1(-2.0 * theta * IntAsDouble(1 <<< j), catalyst[j]);
        }
    }

    /// # Summary
    /// The tower length phased through one Hamming-weight register, given the cap.
    ///
    /// # Description
    /// A batch of `n` terms costs one adder tree, which holds `n - popcount(n)` scratch qubits
    /// for as long as the weight is needed. Splitting a tower into batches lets the scratch of
    /// one batch be released before the next allocates, so the peak ancilla count follows the
    /// batch rather than the whole tower. The price is the phasing of each weight, which is paid
    /// once per batch instead of once per tower: one catalyst addition and rotation with a
    /// catalyst, one synthesized `Rz` per place value without one.
    ///
    /// It is therefore a qubit-for-rotations knob, and the useful setting depends on the device:
    /// hardware demonstrations of Fermi-Hubbard dynamics run on a fixed and comparatively small
    /// register, where the
    /// adder tree of a full lattice-sized tower may simply not fit, while a fault-tolerant
    /// estimate is usually better off spending the qubits to save the rotations. The default is
    /// no cap, which reproduces the uncapped construction exactly.
    ///
    /// Every batch is phased as given, however short. Use a cap of at least 8, since only from 8
    /// terms are the saved rotations worth the extra Toffolis and scratch qubits.
    ///
    /// # Input
    /// ## count
    /// Number of equal-angle terms in the tower.
    /// ## maxBatchSize
    /// Largest tower phased through a single register, or -1 for no cap.
    internal function HammingWeightBatchSize(count : Int, maxBatchSize : Int) : Int {
        Fact(maxBatchSize == -1 or maxBatchSize > 0, "maxBatchSize must be -1 or positive.");
        return maxBatchSize == -1 or maxBatchSize > count ? count | maxBatchSize;
    }
}
