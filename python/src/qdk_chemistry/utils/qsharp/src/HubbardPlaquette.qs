// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HubbardPlaquette {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Flattened;
    import Std.Arrays.Mapped;
    import Std.Arrays.Subarray;
    import Std.Arrays.Tail;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.AbsD;
    import Std.Math.BitSizeI;
    import Std.Math.PI;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// # Summary
    /// Parameters of a repeated plaquette evolution.
    struct HubbardPlaquetteParams {
        /// Number of lattice columns.
        width : Int,
        /// Number of lattice rows.
        height : Int,
        /// On-site pair angle for a full interaction layer.
        interactionAngle : Double,
        /// Twice the hopping amplitude times the step duration.
        hoppingAngle : Double,
        /// Number of repetitions of the body.
        repetitions : Int,
        /// Largest tower phased through a single Hamming-weight register, or -1 for no cap.
        /// See `HammingWeightBatchSize` for what the cap buys and what it costs.
        maxBatchSize : Int,
    }

    /// # Summary
    /// Campbell's pink and gold four-cycle tilings of a periodic square lattice.
    ///
    /// # Input
    /// ## width
    /// Number of lattice columns.
    /// ## height
    /// Number of lattice rows.
    /// ## pink
    /// True for the first tiling, false for the second.
    ///
    /// # Output
    /// A list of plaquettes, each one an array of exactly 4 fermionic mode indices
    function PlaquetteSection(width : Int, height : Int, pink : Bool) : Int[][] {
        if not pink and width == 2 and height == 2 {
            return [];
        }
        let sites = width * height;
        let shift = pink ? 0 | 1;
        mutable cycles = [];
        for spin in 0..1 {
            let offset = spin * sites;
            for row in 0..2..height - 1 {
                for col in 0..2..width - 1 {
                    let top = (row + shift) % height;
                    let bottom = (row + shift + 1) % height;
                    let left = (col + shift) % width;
                    let right = (col + shift + 1) % width;
                    set cycles += [[
                        offset + top * width + left,
                        offset + top * width + right,
                        offset + bottom * width + right,
                        offset + bottom * width + left
                    ]];
                }
            }
        }
        return cycles;
    }

    /// One radix-2 butterfly of the fermionic fast Fourier transform.
    ///
    /// # Description
    /// The two modes must be adjacent in the Jordan-Wigner ordering.
    operation TwoModeFFFT(a : Int, b : Int, systems : Qubit[]) : Unit is Adj + Ctl {
        let lo = a < b ? a | b;
        let hi = a < b ? b | a;
        Fact(hi - lo == 1, "TwoModeFFFT needs two modes that are adjacent after routing.");
        let half = (a < b ? 1.0 | -1.0) * PI() / 8.0;
        let qs = systems[lo..hi];
        Exp([PauliX, PauliY], half, qs);
        Exp([PauliY, PauliX], -half, qs);
    }

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
        let count = Length(targets);
        Fact(Length(pauliOps) == count, "HammingWeightPhase needs one axis list per term.");
        if count > 0 {
            // Chunking in a function keeps the index arithmetic out of this adjointable body.
            let batchSize = HammingWeightBatchSize(count, maxBatchSize);
            let opBatches = Chunks(batchSize, pauliOps);
            let targetBatches = Chunks(batchSize, targets);
            for index in 0..Length(targetBatches) - 1 {
                HammingWeightPhaseBatch(theta, opBatches[index], targetBatches[index]);
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
    /// register (:cite:`Granet2025` simulate the model on a trapped-ion processor), where the
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
    internal operation HammingWeightPhaseBatch(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][]
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
                for t in 0..count - 1 {
                    MapPauliTermToSingleZ(pauliOps[t], targets[t]);
                }
                for (a, b, c, carry) in schedule {
                    if c < 0 {
                        HalfAdderStep(work[a], work[b], work[carry]);
                    } else {
                        FullAdderStep(work[a], work[b], work[c], work[carry]);
                    }
                }
            } apply {
                // TEMPORARY (legacy parity): the legacy ladder also phased every place value with an
                // R(PauliI) and applied the whole batch constant at the end.
                let legacy = UsesLegacyCosts();
                // w = Σ_j 2^j w_j, so e^{2 i theta w} is one rotation per place value, the bit of
                // place value 2^j taking the angle 2 theta 2^j.
                for j in 0..places - 1 {
                    let angle = 2.0 * theta * IntAsDouble(1 <<< j);
                    Rz(angle, work[finalBits[j]]);
                    if legacy {
                        R(PauliI, -angle, work[finalBits[j]]);
                    }
                }
                // `Rz(a) = e^{-ia/2} R1(a)`, so the ladder carries an extra
                // Π_j e^{-i theta 2^j} = e^{-i theta (2^places - 1)} beyond the intended phase.
                // `R(PauliI, g)` is e^{-ig/2}, so this g both supplies the batch constant
                // e^{-i theta count} and undoes the ladder's. Under control it is not global.
                let constant = legacy ? count | count - ((1 <<< places) - 1);
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
    /// Groups of qubits at a fixed stride, one group per term of an equal-angle family.
    internal function StridedGroups(
        count : Int,
        width : Int,
        step : Int,
        stride : Int,
        systems : Qubit[]
    ) : Qubit[][] {
        mutable groups = [];
        for index in 0..count - 1 {
            mutable group = [];
            for offset in 0..width - 1 {
                set group += [systems[index * step + offset * stride]];
            }
            set groups += [group];
        }
        return groups;
    }

    /// The on-site layer, phased through a Hamming-weight register.
    internal operation InteractionLayer(
        angle : Double,
        sites : Int,
        systems : Qubit[],
        maxBatchSize : Int
    ) : Unit is Adj + Ctl {
        // A negligible angle skips the adder tree, which a hopping-only model would pay for nothing.
        if AbsD(angle) > 1e-12 {
            // TEMPORARY (legacy parity): the legacy layer also phased a single-mode Z tower over
            // every mode, the conventional model's n_up + n_down terms.
            if UsesLegacyCosts() {
                HammingWeightPhase(-angle, [[PauliZ], size = Length(systems)], Mapped(q -> [q], systems));
            }
            HammingWeightPhase(
                angle,
                [[PauliZ, PauliZ], size = sites],
                StridedGroups(sites, 2, 1, sites, systems),
                maxBatchSize
            );
        }
    }

    /// Routes every four-cycle of a tiling onto adjacent modes: the swap count and, if `listSwaps`, the adjacent swaps.
    internal function RoutingSwaps(blocks : Int[][], count : Int, listSwaps : Bool) : (Int, Int[]) {
        // Interleaving the diagonals embeds the FFFT's butterfly path (both diagonals, then the
        // surviving pair) in the line, so every butterfly acts on adjacent modes and no basis
        // change carries a Jordan-Wigner string. Unused modes follow in order.
        mutable target = [];
        for block in blocks {
            set target += [block[0], block[2], block[1], block[3]];
        }
        mutable placed = [false, size = count];
        for mode in target {
            set placed w/= mode <- true;
        }
        for mode in 0..count - 1 {
            if not placed[mode] {
                set target += [mode];
            }
        }

        // order[position] is where the mode currently at that position must end up.
        mutable order = [0, size = count];
        for position in 0..count - 1 {
            set order w/= target[position] <- position;
        }

        // Every adjacent swap removes one inversion, so the swap count is the inversion count.
        // A Fenwick tree finds it in O(count log count), keeping the O(count^2) sort out of
        // resource estimation. Walking right to left, each element meets the already-inserted
        // elements to its right; those smaller than it are the inversions it takes part in.
        mutable tree = [0, size = count + 1];
        mutable inversions = 0;
        for index in count - 1..-1..0 {
            mutable lower = order[index];
            while lower > 0 {
                set inversions += tree[lower];
                set lower -= (lower &&& -lower);
            }
            mutable node = order[index] + 1;
            while node <= count {
                set tree w/= node <- tree[node] + 1;
                set node += (node &&& -node);
            }
        }

        // Odd-even transposition sort: disjoint adjacent swaps in each round.
        mutable swaps = [];
        if listSwaps {
            for round in 0..count - 1 {
                for position in round % 2..2..count - 2 {
                    if order[position] > order[position + 1] {
                        let held = order[position];
                        set order w/= position <- order[position + 1];
                        set order w/= position + 1 <- held;
                        set swaps += [position];
                    }
                }
            }
            Fact(Length(swaps) == inversions, "The routing swap count must match the swaps.");
        }
        return (inversions, swaps);
    }

    /// # Summary
    /// One hopping tiling: every plaquette's fswap, ffft, then its momentum-basis phase.
    internal operation HoppingLayer(
        kappa : Double,
        blocks : Int[][],
        systems : Qubit[],
        maxBatchSize : Int
    ) : Unit is Adj + Ctl {
        within {
            let (swapCount, swaps) = RoutingSwaps(blocks, Length(systems), not IsResourceEstimating());
            if IsResourceEstimating() {
                // Every routing swap is the same Clifford pair on a different pair of
                // modes, so the estimator only needs how many there are. Emitting one
                // and repeating its cost keeps the O(sites^1.5) gates out of the trace.
                if swapCount > 0 {
                    within {
                        RepeatEstimates(swapCount);
                    } apply {
                        SWAP(systems[0], systems[1]);
                        CZ(systems[0], systems[1]);
                    }
                }
            } else {
                for position in swaps {
                    SWAP(systems[position], systems[position + 1]);
                    CZ(systems[position], systems[position + 1]);
                }
            }
            for index in 0..Length(blocks) - 1 {
                let base = 4 * index;
                TwoModeFFFT(base + 1, base + 0, systems);
                TwoModeFFFT(base + 2, base + 3, systems);
            }
        } apply {
            // The third butterfly is fused into the phases, so the equal-angle family is the
            // middle pair of every plaquette: exp(i angle XX) exp(i angle YY) on each pair.
            HoppingPhases(kappa / 2.0, StridedGroups(Length(blocks), 2, 4, 1, systems[1...]), maxBatchSize);
        }
    }

    /// # Summary
    /// The momentum-basis phases of one hopping tiling: exp(i angle XX) exp(i angle YY) on every pair.
    ///
    /// # Description
    /// XX and YY commute, so CNOT, H, CNOT sends XX to Z on the first qubit and YY to -Z on the
    /// second, and X removes that sign. The tiling then becomes one tower of twice as many
    /// equal-angle Z rotations, which with Hamming-weight phasing saves a place-value rotation
    /// over separate XX and YY towers. Below the break-even each pair takes its own two
    /// rotations, and a batch cap short enough to put every batch below the break-even has the
    /// same effect, so the basis change is skipped rather than paid for nothing.
    internal operation HoppingPhases(angle : Double, pairs : Qubit[][], maxBatchSize : Int) : Unit is Adj + Ctl {
        let count = 2 * Length(pairs);
        // Every batch is at most this long, so this decides the path for the whole tower.
        let phasesBatch = count > 0 and UsesHammingWeightPhasing(HammingWeightBatchSize(count, maxBatchSize));
        // TEMPORARY (legacy parity): the legacy layer phased XX and YY as two separate towers.
        if UsesLegacyCosts() {
            HammingWeightPhase(-angle, [[PauliX, PauliX], size = Length(pairs)], pairs, maxBatchSize);
            HammingWeightPhase(-angle, [[PauliY, PauliY], size = Length(pairs)], pairs, maxBatchSize);
        } elif not phasesBatch {
            for pair in pairs {
                Exp([PauliX, PauliX], angle, pair);
                Exp([PauliY, PauliY], angle, pair);
            }
        } else {
            within {
                for pair in pairs {
                    CNOT(pair[0], pair[1]);
                    H(pair[0]);
                    CNOT(pair[0], pair[1]);
                    X(pair[1]);
                }
            } apply {
                HammingWeightPhase(
                    -angle,
                    [[PauliZ], size = count],
                    Mapped(q -> [q], Flattened(pairs)),
                    maxBatchSize
                );
            }
        }
    }

    /// One second-order Trotter step, I^(1/2) G I^(1/2) P.
    internal operation PlaquetteStep(
        params : HubbardPlaquetteParams,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        let sites = params.width * params.height;
        let pink = PlaquetteSection(params.width, params.height, true);
        let gold = PlaquetteSection(params.width, params.height, false);
        InteractionLayer(params.interactionAngle / 2.0, sites, systems, params.maxBatchSize);
        HoppingLayer(params.hoppingAngle, gold, systems, params.maxBatchSize);
        InteractionLayer(params.interactionAngle / 2.0, sites, systems, params.maxBatchSize);
        // TEMPORARY (legacy parity): the legacy step also applied the conventional model's scalar,
        // a real rotation under control.
        if UsesLegacyCosts() and AbsD(params.interactionAngle) > 1e-12 {
            R(PauliI, 2.0 * params.interactionAngle * IntAsDouble(sites), systems[0]);
        }
        HoppingLayer(params.hoppingAngle, pink, systems, params.maxBatchSize);
    }

    /// TEMPORARY (legacy parity): resource estimates count the legacy circuit's extra work, which
    /// changes no simulated result. Remove this function and every branch on it to revert.
    internal function UsesLegacyCosts() : Bool {
        return IsResourceEstimating();
    }

    /// # Summary
    /// The whole evolution: the repeated body inside its one-time hopping boundary.
    ///
    /// # Description
    /// Each step is the symmetric product `pink(s/2) I(s/2) gold(s) I(s/2) pink(s/2)`, so
    /// the adjacent half-angle pink layers of neighbouring steps merge into one full-angle
    /// layer. Only a single half-angle pink boundary survives at each end, which is what
    /// the `within` block applies. This is the "PIG" ordering of Eqs. (16a)-(16b) in
    /// :cite:`Apel2026`, which merges the more expensive hopping layers; it deviates from
    /// Eq. (D2) of :cite:`Campbell2022`, which puts the interaction outermost instead.
    ///
    /// The catalysts are prepared by the caller and left prepared, so phase estimation can
    /// prepare them once for every query.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## systems
    /// The system register.
    operation RepPlaquetteExp(
        params : HubbardPlaquetteParams,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        within {
            HoppingLayer(
                params.hoppingAngle / 2.0,
                PlaquetteSection(params.width, params.height, true),
                systems,
                params.maxBatchSize
            );
        } apply {
            if IsResourceEstimating() {
                within {
                    RepeatEstimates(params.repetitions);
                } apply {
                    PlaquetteStep(params, systems);
                }
            } else {
                for _ in 1..params.repetitions {
                    PlaquetteStep(params, systems);
                }
            }
        }
    }

    /// Returns a callable applying the evolution.
    function MakeRepPlaquetteExpOp(params : HubbardPlaquetteParams) : (Qubit[] => Unit is Adj + Ctl) {
        systems => RepPlaquetteExp(params, systems)
    }

    /// Builds the controlled evolution on the given qubit indices.
    operation MakeRepControlledPlaquetteExpCircuit(
        params : HubbardPlaquetteParams,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        ControlledRepPlaquetteExp(params, qs[control], Subarray(systems, qs));
    }

    /// Applies the evolution controlled on `control`; `targets` holds the systems.
    operation ControlledRepPlaquetteExp(
        params : HubbardPlaquetteParams,
        control : Qubit,
        targets : Qubit[]
    ) : Unit is Adj + Ctl {
        let numSystems = 2 * params.width * params.height;
        Fact(
            Length(targets) >= numSystems,
            "The targets must hold the system register."
        );
        Controlled RepPlaquetteExp([control], (params, targets[0..numSystems - 1]));
    }

    /// Returns `ControlledRepPlaquetteExp` as a callable.
    function MakeRepControlledPlaquetteExpOp(
        params : HubbardPlaquetteParams
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        ControlledRepPlaquetteExp(params, _, _)
    }
}
