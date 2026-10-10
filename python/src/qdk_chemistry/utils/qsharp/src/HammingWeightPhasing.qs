// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HammingWeightPhasing {

    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Mapped;
    import Std.Arrays.Sorted;
    import Std.Arrays.Tail;
    import Std.Arrays.Where;
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

    /// Applies exp(-i theta P_t) for every term P_t, phasing the batch through a Hamming-weight
    /// register once it reaches the measured break-even size.
    ///
    /// Each term is first rotated onto a single Z on its last qubit, so the batch becomes
    /// `count` equal-angle rotations exp(-i theta Z). Their product is
    /// e^{-i theta count} e^{2 i theta w}, where w is the Hamming weight of those qubits. An
    /// adder tree computes w into log2(count) + 1 bits, which Hamming-weight phasing then phases
    /// with one rotation per place value instead of one per term; the constant becomes a phase on
    /// the control when the whole is controlled. This is the construction of :cite:`Kan2025`,
    /// whose Methods diagonalize every
    /// plaquette and every on-site pair into a layer of same-angle R_z gates and synthesize that
    /// layer collectively with HWP. Each place-value rotation is a synthesized `Rz`. Below the
    /// break-even size each term is applied as its own rotation instead.
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
    /// ## maxBatchSize
    /// Largest tower phased through a single Hamming-weight register, or -1 for no cap.
    internal operation HammingWeightPhase(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        maxBatchSize : Int
    ) : Unit is Adj + Ctl {
        HammingWeightPhaseWithLegacyCosts(theta, pauliOps, targets, maxBatchSize, false);
    }

    /// `HammingWeightPhase`, with `legacyCosts` selecting the legacy phase ladder.
    internal operation HammingWeightPhaseWithLegacyCosts(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        maxBatchSize : Int,
        legacyCosts : Bool
    ) : Unit is Adj + Ctl {
        let count = Length(targets);
        Fact(Length(pauliOps) == count, "HammingWeightPhase needs one axis list per term.");
        if count > 0 {
            // Chunking in a function keeps the index arithmetic out of this adjointable body.
            let batchSize = HammingWeightBatchSize(count, maxBatchSize);
            let opBatches = Chunks(batchSize, pauliOps);
            let targetBatches = Chunks(batchSize, targets);
            for index in 0..Length(targetBatches) - 1 {
                HammingWeightPhaseBatchWithLegacyCosts(theta, opBatches[index], targetBatches[index], legacyCosts);
            }
        }
    }

    /// # Summary
    /// The tower length phased through one Hamming-weight register, given the cap.
    ///
    /// # Description
    /// A batch of `n` terms costs one adder tree: roughly `n` Toffolis and, more importantly,
    /// `n - popcount(n)` scratch qubits held for as long as the weight is needed. Splitting a
    /// tower into batches lets the scratch of one batch be released before the next allocates,
    /// so the peak ancilla count follows the batch rather than the whole tower, while the
    /// Toffoli count stays essentially the same. The price is the place-value rotations, which
    /// are paid once per batch instead of once per tower.
    ///
    /// It is therefore a qubit-for-rotations knob, and the useful setting depends on the device:
    /// hardware demonstrations of Fermi-Hubbard dynamics run on a fixed and comparatively small
    /// register, where the
    /// adder tree of a full lattice-sized tower may simply not fit, while a fault-tolerant
    /// estimate is usually better off spending the qubits to save the rotations. The default is
    /// no cap, which reproduces the uncapped construction exactly.
    ///
    /// A cap below the break-even of `UsesHammingWeightPhasing` leaves every batch too short to
    /// phase through a register, so the whole tower falls back to one rotation per term.
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

    /// One batch of equal-angle terms, phased through a single Hamming-weight register.
    /// See `HammingWeightPhase`, which splits a tower into batches of this shape.
    internal operation HammingWeightPhaseBatchWithLegacyCosts(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        legacyCosts : Bool
    ) : Unit is Adj + Ctl {
        let count = Length(targets);
        if not UsesHammingWeightPhasing(count) {
            for t in 0..count - 1 {
                Exp(pauliOps[t], -theta, targets[t]);
            }
        } else {
            let inputs = Mapped(term -> Tail(term), targets);
            let (schedule, finalBits, _) = HammingWeightSchedule(count);
            Fact(All(bit -> bit >= 0, finalBits), "Every place value of the Hamming weight must hold a bit.");
            let places = Length(finalBits);
            use scratch = Qubit[Length(schedule)];
            let work = inputs + scratch;
            within {
                ComputeHammingWeight(pauliOps, targets, schedule, work);
            } apply {
                // w = Σ_j 2^j w_j, so e^{2 i theta w} is one rotation per place value, the bit of
                // place value 2^j taking the angle 2 theta 2^j.
                // TEMPORARY (legacy parity): the legacy ladder also phased every place value with an
                // R(PauliI) and applied the whole batch constant at the end.
                for j in 0..places - 1 {
                    let angle = 2.0 * theta * IntAsDouble(1 <<< j);
                    Rz(angle, work[finalBits[j]]);
                    if legacyCosts {
                        R(PauliI, -angle, work[finalBits[j]]);
                    }
                }
                // `Rz(a) = e^{-ia/2} R1(a)`, so the ladder carries an extra
                // Π_j e^{-i theta 2^j} = e^{-i theta (2^places - 1)} beyond the intended phase.
                // `R(PauliI, g)` is e^{-ig/2}, so this g both supplies the batch constant
                // e^{-i theta count} and undoes the ladder's. Under control it is not global.
                let constant = legacyCosts ? count | count - ((1 <<< places) - 1);
                R(PauliI, 2.0 * theta * IntAsDouble(constant), inputs[0]);
            }
        }
    }

    /// Whether a tower of `count` equal-angle rotations is phased through a Hamming-weight
    /// register: not below the measured break-even of 8, where the adder tree costs more than the
    /// rotations it saves.
    internal function UsesHammingWeightPhasing(count : Int) : Bool {
        return count >= 8;
    }

    /// # Summary
    /// How many of `count` equal-angle terms to phase through Hamming-weight registers.
    ///
    /// # Description
    /// `HammingWeightPhase` splits a tower into batches of `HammingWeightBatchSize` terms and gives
    /// any batch below the break-even its own rotations. This returns the length of the leading part
    /// of the tower whose every batch reaches the break-even, so a caller can rotate the short
    /// remainder together with its other plain terms. It is 0 when no batch reaches the break-even.
    ///
    /// # Input
    /// ## count
    /// Number of equal-angle terms.
    /// ## maxBatchSize
    /// Largest tower phased through a single register, or -1 for no cap.
    internal function HammingWeightPhasedCount(count : Int, maxBatchSize : Int) : Int {
        if count <= 0 {
            return 0;
        }
        let batchSize = HammingWeightBatchSize(count, maxBatchSize);
        if not UsesHammingWeightPhasing(batchSize) {
            return 0;
        }
        let remainder = count % batchSize;
        return remainder == 0 or UsesHammingWeightPhasing(remainder) ? count | count - remainder;
    }

    /// # Summary
    /// Groups terms with equal rotation angles into towers for `HammingWeightPhase`.
    ///
    /// # Description
    /// Angles are compared exactly, and zero angles are never grouped. Each group contributes the
    /// first `HammingWeightPhasedCount` of its terms as one tower, and the rest stay plain terms, so
    /// a group shorter than the break-even of `UsesHammingWeightPhasing` is left untouched. Sorting
    /// keeps the grouping at O(n log n) for wide layers.
    ///
    /// # Input
    /// ## angles
    /// The rotation angle of each term.
    /// ## maxBatchSize
    /// Largest tower phased through a single register, or -1 for no cap.
    ///
    /// # Output
    /// The positions of the plain terms in their input order, and the ascending positions of each tower.
    internal function EqualAngleTowers(angles : Double[], maxBatchSize : Int) : (Int[], Int[][]) {
        let count = Length(angles);
        // Sort (angle, position) pairs: a comparator capturing `angles` breaks Base-profile lowering.
        mutable keyed : (Double, Int)[] = [];
        for position in 0..count - 1 {
            set keyed += [(angles[position], position)];
        }
        mutable order = [];
        for (_, position) in Sorted(AngleThenPosition, keyed) {
            set order += [position];
        }
        mutable plain = [true, size = count];
        mutable towers : Int[][] = [];
        mutable start = 0;
        while start < count {
            let angle = angles[order[start]];
            mutable stop = start + 1;
            while stop < count and angles[order[stop]] == angle {
                set stop += 1;
            }
            let phased = angle == 0.0 ? 0 | HammingWeightPhasedCount(stop - start, maxBatchSize);
            if phased > 0 {
                let tower = order[start..start + phased - 1];
                set towers += [tower];
                for position in tower {
                    set plain w/= position <- false;
                }
            }
            set start = stop;
        }
        return (Where(isPlain -> isPlain, plain), towers);
    }

    /// Orders `(angle, position)` pairs by angle, then position.
    function AngleThenPosition(left : (Double, Int), right : (Double, Int)) : Bool {
        let (leftAngle, leftPosition) = left;
        let (rightAngle, rightPosition) = right;
        leftAngle < rightAngle or (leftAngle == rightAngle and leftPosition <= rightPosition)
    }
}
