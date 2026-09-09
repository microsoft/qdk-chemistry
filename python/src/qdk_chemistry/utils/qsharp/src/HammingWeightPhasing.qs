// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HammingWeightPhasing {

    import Std.Arrays.Subarray;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;

    /// Picks the qubit that carries each term's phase after `MapTermToSingleZ`.
    ///
    /// # Parameters
    /// - `targets`: The qubits each term acts on.
    function Representatives(targets : Qubit[][]) : Qubit[] {
        mutable reps : Qubit[] = [];
        for term in targets {
            set reps += [term[Length(term) - 1]];
        }
        return reps;
    }

    /// Rotates a Pauli string onto a single `Z` on the last qubit it acts on.
    ///
    /// Conjugating by this `U` sends `P` to `Z` on that qubit, so
    /// `exp(-i theta P) = U^dagger exp(-i theta Z) U`, which is what lets terms with
    /// different Pauli axes share one Hamming-weight register. `X` is rotated by `H`
    /// and `Y` by `H S^dagger`; a CNOT ladder then collects the remaining factors onto
    /// the last qubit, using `CNOT Z_a Z_b CNOT = Z_b`.
    ///
    /// # Parameters
    /// - `ops`: The Pauli axes of the term, none of which may be `PauliI`.
    /// - `targets`: The qubits the term acts on, one per entry of `ops`.
    operation MapTermToSingleZ(ops : Pauli[], targets : Qubit[]) : Unit is Adj + Ctl {
        Fact(Length(ops) == Length(targets), "MapTermToSingleZ needs one Pauli axis per target.");
        Fact(Length(ops) > 0, "MapTermToSingleZ needs a non-empty Pauli string.");
        for i in 0..Length(ops) - 1 {
            if ops[i] == PauliX {
                H(targets[i]);
            } elif ops[i] == PauliY {
                Adjoint S(targets[i]);
                H(targets[i]);
            } else {
                // PauliI would make the ladder below collect a factor that is not there.
                Fact(ops[i] == PauliZ, "MapTermToSingleZ does not accept PauliI.");
            }
        }
        let last = Length(targets) - 1;
        for i in 0..last - 1 {
            CNOT(targets[i], targets[last]);
        }
    }

    /// Applies `exp(-i theta P)` for every Pauli string in a batch that shares `theta`.
    ///
    /// Each string is rotated onto a single `Z`, after which the batch is exactly the
    /// equal-angle single-qubit case that `HammingWeightPhase` reduces to logarithmically
    /// many rotations. The strings must act on pairwise disjoint qubits, which also makes
    /// them commute, so applying them as one block is order-independent and exact.
    ///
    /// # Parameters
    /// - `theta`: The angle shared by every term in the batch.
    /// - `pauliOps`: The Pauli axes of each term.
    /// - `targets`: The qubits each term acts on.
    operation HammingWeightPhaseTerms(theta : Double, pauliOps : Pauli[][], targets : Qubit[][]) : Unit is Adj + Ctl {
        Fact(Length(pauliOps) == Length(targets), "HammingWeightPhaseTerms needs one axis list per term.");
        within {
            for t in 0..Length(targets) - 1 {
                MapTermToSingleZ(pauliOps[t], targets[t]);
            }
        } apply {
            HammingWeightPhase(theta, Representatives(targets));
        }
    }

    /// Number of scratch qubits `ComputeHammingWeight` needs for `count` inputs.
    ///
    /// One carry qubit per adder, and the adder tree uses `count - w(count)` adders.
    ///
    /// # Parameters
    /// - `count`: Number of qubits whose weight is to be counted.
    function HammingWeightScratchCount(count : Int) : Int {
        let (schedule, _, _) = HammingWeightSchedule(count);
        return Length(schedule);
    }

    /// Number of weight bits `ComputeHammingWeight` writes for `count` inputs.
    ///
    /// # Parameters
    /// - `count`: Number of qubits whose weight is to be counted.
    function HammingWeightBits(count : Int) : Int {
        return count <= 0 ? 0 | BitSizeI(count);
    }

    /// Plans the adder tree that reduces `count` weight-one bits to a binary weight.
    ///
    /// The plan is pure classical bookkeeping, kept in a function because Q# cannot
    /// generate an adjoint through the `mutable` updates it needs.
    ///
    /// Bits are tracked in buckets by significance. Three bits of the same
    /// significance are compressed by a full adder into one bit of that significance
    /// and one carry of the next; a leftover pair is compressed by a half adder the
    /// same way. Each adder therefore removes exactly one bit overall, so the tree
    /// uses `count - w(count)` adders and as many Toffolis, which is optimal
    /// (Gidney, arXiv:1709.06648).
    ///
    /// Returns the adder list as `(a, b, c, carry)` indices into a work register laid
    /// out as inputs followed by scratch, where `c < 0` marks a half adder; the index
    /// holding each weight bit, `-1` where that bit is identically zero; and the total
    /// work register size.
    ///
    /// # Parameters
    /// - `count`: Number of qubits whose weight is to be counted.
    function HammingWeightSchedule(count : Int) : ((Int, Int, Int, Int)[], Int[], Int) {
        if count <= 0 {
            return ([], [], 0);
        }
        let levels = BitSizeI(count);

        mutable initial : Int[] = [];
        for i in 0..count - 1 {
            set initial += [i];
        }
        // One extra bucket so writing a carry out of the top level is always in range;
        // it provably stays empty, since a carry there would mean a weight above count.
        mutable buckets : Int[][] = [[], size = levels + 1];
        set buckets w/= 0 <- initial;

        mutable schedule : (Int, Int, Int, Int)[] = [];
        mutable next = count;

        for k in 0..levels - 1 {
            mutable cur = buckets[k];
            mutable up = buckets[k + 1];
            while Length(cur) >= 3 {
                // Full adder: the sum stays in `cur[2]`, the carry goes up a level.
                set schedule += [(cur[0], cur[1], cur[2], next)];
                set cur = cur[3...] + [cur[2]];
                set up += [next];
                set next += 1;
            }
            if Length(cur) == 2 {
                // Half adder: the sum stays in `cur[1]`, the carry goes up a level.
                set schedule += [(cur[0], cur[1], -1, next)];
                set cur = [cur[1]];
                set up += [next];
                set next += 1;
            }
            set buckets w/= k <- cur;
            set buckets w/= k + 1 <- up;
        }

        mutable finals : Int[] = [];
        for k in 0..levels - 1 {
            set finals += [Length(buckets[k]) == 1 ? buckets[k][0] | -1];
        }
        return (schedule, finals, next);
    }

    /// Compresses three bits of equal significance into a sum and a carry.
    ///
    /// Maps `(a, b, c, 0)` to `(a, a xor b, a xor b xor c, MAJ(a, b, c))` using a
    /// single `AND`. The first two outputs are residual: three bits carry more
    /// information than the two-bit count, so reversibility forces something to
    /// survive. They are restored by the adjoint of the enclosing `within` block,
    /// which is also where the carry is uncomputed -- by measurement, for no Toffolis,
    /// which is why this uses `AND` rather than `CCNOT`.
    ///
    /// # Parameters
    /// - `a`: First summand; left unchanged, held as residual.
    /// - `b`: Second summand; becomes residual.
    /// - `c`: Third summand; receives the sum bit.
    /// - `carry`: Ancilla in state zero; receives the carry bit.
    operation FullAdderStep(a : Qubit, b : Qubit, c : Qubit, carry : Qubit) : Unit is Adj {
        CNOT(a, b);
        CNOT(a, c);
        AND(b, c, carry);
        // carry now holds a xor ((a xor b) and (a xor c)), which is MAJ(a, b, c).
        CNOT(a, carry);
        CNOT(b, c);
        CNOT(a, c);
    }

    /// Compresses two bits of equal significance into a sum and a carry.
    ///
    /// Maps `(a, b, 0)` to `(a, a xor b, a and b)` using a single `AND`.
    ///
    /// # Parameters
    /// - `a`: First summand; left unchanged, held as residual.
    /// - `b`: Second summand; receives the sum bit.
    /// - `carry`: Ancilla in state zero; receives the carry bit.
    operation HalfAdderStep(a : Qubit, b : Qubit, carry : Qubit) : Unit is Adj {
        AND(a, b, carry);
        CNOT(a, b);
    }

    /// Computes the Hamming weight of `inputs` into the little-endian register `weight`.
    ///
    /// `scratch` must hold `HammingWeightScratchCount(Length(inputs))` qubits in state
    /// zero, and `weight` at least `HammingWeightBits(Length(inputs))`. The scratch is
    /// left dirty on purpose: it carries the partial sums, so this must be run inside a
    /// `within` block that later applies the adjoint, which is also what returns those
    /// qubits to zero for release.
    ///
    /// # Parameters
    /// - `inputs`: Qubits whose weight is counted; left dirty, restored by the adjoint.
    /// - `scratch`: Zeroed workspace for the adder carries; left dirty.
    /// - `weight`: Little-endian output register receiving the weight.
    operation ComputeHammingWeight(inputs : Qubit[], scratch : Qubit[], weight : Qubit[]) : Unit is Adj {
        let count = Length(inputs);
        let (schedule, finals, _) = HammingWeightSchedule(count);
        Fact(
            Length(scratch) >= Length(schedule),
            $"ComputeHammingWeight needs {Length(schedule)} scratch qubits, got {Length(scratch)}."
        );
        Fact(
            Length(weight) >= Length(finals),
            $"ComputeHammingWeight needs {Length(finals)} weight qubits, got {Length(weight)}."
        );
        let work = inputs + scratch;
        for (a, b, c, carry) in schedule {
            if c < 0 {
                HalfAdderStep(work[a], work[b], work[carry]);
            } else {
                FullAdderStep(work[a], work[b], work[c], work[carry]);
            }
        }
        for k in 0..Length(finals) - 1 {
            if finals[k] >= 0 {
                CNOT(work[finals[k]], weight[k]);
            }
        }
    }

    /// Applies `exp(-i theta Z)` to every qubit of `inputs` using only
    /// `ceil(log2(m+1))` rotations instead of `m`.
    ///
    /// The rotations all share one angle, so the phase they impart depends on the
    /// computational basis state only through its Hamming weight: with `w` ones among
    /// `m` qubits, `sum_j Z_j` has eigenvalue `m - 2w`, so the phase is
    /// `exp(-i theta (m - 2w))`. Computing `w` into an ancilla register therefore lets
    /// one rotation per *weight bit* replace one rotation per *qubit*, which is the
    /// whole saving: the rotation count becomes logarithmic in the batch size while the
    /// batch itself grows with the lattice. The arithmetic costs Toffolis instead, but
    /// only `m - w(m)` of them, which is far cheaper than synthesising `m` rotations.
    ///
    /// The identity part is applied explicitly rather than dropped, because it becomes
    /// an observable relative phase as soon as this is controlled, which is how phase
    /// estimation uses it. Controlling is also why the weight is computed in a `within`
    /// block: `C(V D V^dagger) = V C(D) V^dagger`, so the control falls only on the
    /// logarithmically many rotations and never on the adder tree.
    ///
    /// # Parameters
    /// - `theta`: The shared rotation angle; the circuit applies `exp(-i theta Z)` per qubit.
    /// - `inputs`: The qubits sharing that angle.
    ///
    /// # References
    /// Kivlichan et al., arXiv:1902.10673 App. A names Hamming weight phasing and builds
    /// it on Gidney's adder tree (arXiv:1709.06648); Campbell, arXiv:2012.09238v4,
    /// App. E Thm. 2 tightens its cost to m - w(m) Toffolis and applies it in a
    /// Hubbard plaquette Trotter step.
    operation HammingWeightPhase(theta : Double, inputs : Qubit[]) : Unit is Adj + Ctl {
        let count = Length(inputs);
        // Structured as if/else rather than early returns: Q# cannot generate an
        // adjoint through a `return` in a block that requires one.
        if count == 1 {
            // exp(-i theta Z) is Rz(2 theta) exactly; unlike the batched branch below
            // there is no identity component to compensate for.
            Rz(2.0 * theta, inputs[0]);
        } elif count > 1 {
            let bits = HammingWeightBits(count);
            use scratch = Qubit[HammingWeightScratchCount(count)];
            use weight = Qubit[bits];
            within {
                ComputeHammingWeight(inputs, scratch, weight);
            } apply {
                // exp(-i theta (m - 2w)) = exp(-i theta m) prod_k exp(2 i theta 2^k w_k),
                // and w_k = (I - Z)/2 on weight bit k.
                for k in 0..bits - 1 {
                    let scale = IntAsDouble(1 <<< k);
                    Rz(2.0 * theta * scale, weight[k]);
                    R(PauliI, -2.0 * theta * scale, weight[k]);
                }
                // the remaining m-dependent phase
                R(PauliI, 2.0 * theta * IntAsDouble(count), inputs[0]);
            }
        }
    }

    /// Resolves each term's sparse positions to the qubits it acts on.
    ///
    /// A function rather than inline code because the caller must be adjointable, and
    /// Q# cannot generate an adjoint through the accumulation this needs.
    ///
    /// # Parameters
    /// - `pauliIndices`: For each term, the positions in `systems` it acts on.
    /// - `systems`: The system qubits.
    function TermTargets(pauliIndices : Int[][], systems : Qubit[]) : Qubit[][] {
        mutable targets : Qubit[][] = [];
        for indices in pauliIndices {
            set targets += [Subarray(indices, systems)];
        }
        return targets;
    }

    /// Splits a term list into the blocks a batched evolution applies as units.
    ///
    /// Consecutive terms sharing a non-zero batch identifier form one block, applied
    /// together by Hamming weight phasing; every unbatched term is its own block. Kept
    /// as a function because Q# cannot generate an adjoint through the `mutable`
    /// updates this needs, and the caller is adjointable.
    ///
    /// Returns `(start, count)` pairs covering the whole list in order.
    ///
    /// # Parameters
    /// - `batchIds`: The batch identifier of each term, `0` when unbatched.
    function BatchSegments(batchIds : Int[]) : (Int, Int)[] {
        mutable segments : (Int, Int)[] = [];
        mutable start = 0;
        while start < Length(batchIds) {
            mutable count = 1;
            if batchIds[start] != 0 {
                while start + count < Length(batchIds) and batchIds[start + count] == batchIds[start] {
                    set count += 1;
                }
            }
            set segments += [(start, count)];
            set start += count;
        }
        return segments;
    }
}
