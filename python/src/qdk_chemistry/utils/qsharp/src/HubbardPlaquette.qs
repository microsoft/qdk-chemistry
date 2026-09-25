// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HubbardPlaquette {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import Std.Arrays.Mapped;
    import Std.Arrays.Subarray;
    import Std.Arrays.Tail;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
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
        /// Single-mode Z angle; zero under the particle-hole shift.
        onsiteAngle : Double,
        /// Scalar phase applied once per step.
        identityAngle : Double,
        /// Twice the hopping amplitude times the step duration.
        hoppingAngle : Double,
        /// Number of repetitions of the body.
        repetitions : Int,
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

    /// The Pauli word of a Jordan-Wigner two-mode operator.
    internal function JordanWignerWord(lo : Int, hi : Int, first : Pauli, last : Pauli) : Pauli[] {
        mutable word = [first];
        for _ in lo + 1..hi - 1 {
            set word += [PauliZ];
        }
        return word + [last];
    }

    /// One radix-2 butterfly of the fermionic fast Fourier transform.
    operation TwoModeFFFT(a : Int, b : Int, systems : Qubit[]) : Unit is Adj + Ctl {
        let lo = a < b ? a | b;
        let hi = a < b ? b | a;
        let half = (a < b ? 1.0 | -1.0) * PI() / 8.0;
        let qs = systems[lo..hi];
        Exp(JordanWignerWord(lo, hi, PauliX, PauliY), half, qs);
        Exp(JordanWignerWord(lo, hi, PauliY, PauliX), -half, qs);
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

    /// Computes a Hamming weight in place across `inputs + scratch`.
    /// Applies equal-angle Z phases using logarithmically many rotations.
    ///
    /// Hamming weight phasing replaces `count` equal-angle rotations with `O(log count)`
    /// of them: an adder tree computes the Hamming weight of the inputs, and each output
    /// bit is phased by its place-value-scaled angle.
    /// Sec. 4.4 of :cite:`Apel2026` uses it for this Hubbard plaquette circuit.
    internal operation HammingWeightPhase(theta : Double, inputs : Qubit[]) : Unit is Adj + Ctl {
        let count = Length(inputs);
        let (schedule, finalBits, _) = HammingWeightSchedule(count);
        use scratch = Qubit[Length(schedule)];
        let work = inputs + scratch;
        within {
            for (a, b, c, carry) in schedule {
                if c < 0 {
                    HalfAdderStep(work[a], work[b], work[carry]);
                } else {
                    FullAdderStep(work[a], work[b], work[c], work[carry]);
                }
            }
        } apply {
            for k in 0..Length(finalBits) - 1 {
                if finalBits[k] >= 0 {
                    let scale = IntAsDouble(1 <<< k);
                    Rz(2.0 * theta * scale, work[finalBits[k]]);
                    R(PauliI, -2.0 * theta * scale, work[finalBits[k]]);
                }
            }
            R(PauliI, 2.0 * theta * IntAsDouble(count), inputs[0]);
        }
    }

    /// Applies one equal-angle batch, selecting HWP only at its measured break-even size.
    internal operation HammingWeightPhaseTerms(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][]
    ) : Unit is Adj + Ctl {
        Fact(Length(pauliOps) == Length(targets), "HammingWeightPhaseTerms needs one axis list per term.");
        // For a controlled batch of m terms, HWP costs 3 ceil(log2(m + 1)) + 1
        // rotations and m - w(m) AND operations, versus 2m rotations term by term.
        if Length(targets) < 8 {
            for t in 0..Length(targets) - 1 {
                Exp(pauliOps[t], -theta, targets[t]);
            }
        } else {
            within {
                for t in 0..Length(targets) - 1 {
                    MapPauliTermToSingleZ(pauliOps[t], targets[t]);
                }
            } apply {
                HammingWeightPhase(theta, Mapped(term -> Tail(term), targets));
            }
        }
    }

    /// # Summary
    /// Groups of qubits at a fixed stride, one group per term of an equal-angle family.
    ///
    /// The on-site pair family takes stride `sites` (a site and its spin partner), the
    /// single-mode family stride 0 with width 1, and a routed hopping tiling stride 1
    /// with the groups spaced four modes apart.
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

    /// The on-site layer, phased through Hamming-weight registers.
    ///
    /// # Input
    /// ## angle
    /// Pair rotation angle.
    /// ## onsite
    /// Single-mode rotation angle.
    /// ## sites
    /// Number of lattice sites.
    /// ## systems
    /// The system register.
    operation InteractionLayer(
        angle : Double,
        onsite : Double,
        sites : Int,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        if onsite != 0.0 {
            HammingWeightPhaseTerms(onsite, [[PauliZ], size = Length(systems)], StridedGroups(Length(systems), 1, 1, 1, systems));
        }
        if angle != 0.0 {
            HammingWeightPhaseTerms(angle, [[PauliZ, PauliZ], size = sites], StridedGroups(sites, 2, 1, sites, systems));
        }
    }

    /// # Summary
    /// The routing permutation: where the mode at each position must end up.
    ///
    /// # Description
    /// Concatenating the tiling's four-cycles and appending the modes it leaves alone
    /// gives the frame the routing network has to reach, so sorting this permutation
    /// with adjacent transpositions is exactly the routing problem.
    internal function RoutingOrder(blocks : Int[][], count : Int) : Int[] {
        mutable target = [];
        for block in blocks {
            set target += block;
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
        return order;
    }

    /// # Summary
    /// Adjacent swaps routing each plaquette of a tiling onto contiguous modes.
    internal function RoutingSwaps(blocks : Int[][], count : Int) : Int[] {
        mutable order = RoutingOrder(blocks, count);
        mutable swaps = [];
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
        return swaps;
    }

    /// # Summary
    /// How many adjacent swaps the routing network performs.
    ///
    /// # Description
    /// Each adjacent transposition removes exactly one inversion, so the length of
    /// RoutingSwaps is the inversion count of the routing permutation. A Fenwick tree
    /// counts those in O(count log count) instead of running the O(count^2) sort that
    /// would otherwise produce them, which is what keeps large lattices tractable.
    internal function RoutingSwapCount(blocks : Int[][], count : Int) : Int {
        let order = RoutingOrder(blocks, count);
        mutable tree = [0, size = count + 1];
        mutable inversions = 0;
        // Walking right to left, each element meets the already-inserted elements to its
        // right; those smaller than it are precisely the inversions it takes part in.
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
        return inversions;
    }

    /// # Summary
    /// One hopping tiling: every plaquette's FFFT, then its momentum-basis phase.
    ///
    /// # Description
    /// The basis changes are hoisted across the whole tiling so the phases sit together
    /// in the middle, which is what lets equal-angle families share a register once the
    /// modes of a plaquette are routed adjacent.
    ///
    /// # Input
    /// ## kappa
    /// Twice the hopping amplitude times the step duration.
    /// ## blocks
    /// Four-cycles of the tiling, in cycle order.
    /// ## systems
    /// The system register.
    operation HoppingLayer(kappa : Double, blocks : Int[][], systems : Qubit[]) : Unit is Adj + Ctl {
        within {
            if IsResourceEstimating() {
                // Every routing swap is the same Clifford pair on a different pair of
                // modes, so the estimator only needs how many there are. Emitting one
                // and repeating its cost keeps the O(sites^1.5) gates out of the trace,
                // and the inversion count keeps the O(sites^2) sort out with them.
                let swapCount = RoutingSwapCount(blocks, Length(systems));
                if swapCount > 0 {
                    within {
                        RepeatEstimates(swapCount);
                    } apply {
                        SWAP(systems[0], systems[1]);
                        CZ(systems[0], systems[1]);
                    }
                }
            } else {
                for position in RoutingSwaps(blocks, Length(systems)) {
                    SWAP(systems[position], systems[position + 1]);
                    CZ(systems[position], systems[position + 1]);
                }
            }
            for index in 0..Length(blocks) - 1 {
                let base = 4 * index;
                TwoModeFFFT(base + 0, base + 2, systems);
                TwoModeFFFT(base + 1, base + 3, systems);
            }
        } apply {
            let pairs = StridedGroups(Length(blocks), 2, 4, 1, systems);
            HammingWeightPhaseTerms(-kappa / 2.0, [[PauliX, PauliX], size = Length(pairs)], pairs);
            HammingWeightPhaseTerms(-kappa / 2.0, [[PauliY, PauliY], size = Length(pairs)], pairs);
        }
    }

    /// # Summary
    /// One second-order plaquette Trotter step.
    ///
    /// # Description
    /// The body is I^(1/2) G I^(1/2) P; the caller supplies the one-time P^(1/2)
    /// boundary that merging across repetitions leaves outside. Merging the hopping
    /// layer rather than the interaction is what saves: the body then carries two
    /// hopping layers instead of three, and hopping dominates the layer cost.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## systems
    /// The system register.
    operation PlaquetteStep(params : HubbardPlaquetteParams, systems : Qubit[]) : Unit is Adj + Ctl {
        let sites = params.width * params.height;
        let pink = PlaquetteSection(params.width, params.height, true);
        let gold = PlaquetteSection(params.width, params.height, false);
        InteractionLayer(params.interactionAngle / 2.0, params.onsiteAngle / 2.0, sites, systems);
        HoppingLayer(params.hoppingAngle, gold, systems);
        InteractionLayer(params.interactionAngle / 2.0, params.onsiteAngle / 2.0, sites, systems);
        if params.identityAngle != 0.0 {
            // R(PauliI, theta) is the global phase exp(-i theta / 2), so theta =
            // 2 * identityAngle realizes the step's exp(-i * identityAngle) factor.
            R(PauliI, 2.0 * params.identityAngle, systems[0]);
        }
        HoppingLayer(params.hoppingAngle, pink, systems);
    }

    /// # Summary
    /// The whole evolution: the repeated body inside its one-time hopping boundary.
    ///
    /// Each step is the symmetric product `pink(s/2) I(s/2) gold(s) I(s/2) pink(s/2)`, so
    /// the adjacent half-angle pink layers of neighbouring steps merge into one full-angle
    /// layer. Only a single half-angle pink boundary survives at each end, which is what
    /// the `within` block applies. This is the "PIG" ordering of Eqs. (16a)-(16b) in
    /// :cite:`Apel2026`, which merges the more expensive hopping layers; it deviates from
    /// Eq. (D2) of :cite:`Campbell2022`, which puts the interaction outermost instead.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## systems
    /// The system register.
    operation RepPlaquetteExp(params : HubbardPlaquetteParams, systems : Qubit[]) : Unit is Adj + Ctl {
        within {
            HoppingLayer(
                params.hoppingAngle / 2.0,
                PlaquetteSection(params.width, params.height, true),
                systems
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

    /// # Summary
    /// Applies a repeated plaquette evolution controlled on a single qubit.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## control
    /// The control qubit.
    /// ## systems
    /// The system register.
    operation ControlledRepPlaquetteExp(
        params : HubbardPlaquetteParams,
        control : Qubit,
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        Controlled RepPlaquetteExp([control], (params, systems));
    }

    /// # Summary
    /// Returns a callable applying a repeated plaquette evolution.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function MakeRepPlaquetteExpOp(params : HubbardPlaquetteParams) : (Qubit[] => Unit is Adj + Ctl) {
        RepPlaquetteExp(params, _)
    }

    /// # Summary
    /// Builds the controlled circuit for a repeated plaquette evolution.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## control
    /// Index of the control qubit.
    /// ## systems
    /// Indices of the system qubits.
    operation MakeRepControlledPlaquetteExpCircuit(
        params : HubbardPlaquetteParams,
        control : Int,
        systems : Int[]
    ) : Unit {
        use qs = Qubit[MaxInt([control] + systems) + 1];
        ControlledRepPlaquetteExp(params, qs[control], Subarray(systems, qs));
    }

    /// # Summary
    /// Returns a single-register callable applying the controlled evolution.
    ///
    /// # Description
    /// The control occupies the first qubit and the system follows, which is the shape a
    /// state-vector simulation drives. Phase estimation instead uses the two-argument
    /// form, where it supplies the ancilla separately.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function MakeRepControlledPlaquetteExpOnRegisterOp(
        params : HubbardPlaquetteParams
    ) : (Qubit[] => Unit is Adj + Ctl) {
        register => ControlledRepPlaquetteExp(params, register[0], register[1...])
    }

    /// # Summary
    /// Returns a single-control callable for a repeated plaquette evolution.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function MakeRepControlledPlaquetteExpOp(
        params : HubbardPlaquetteParams
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        ControlledRepPlaquetteExp(params, _, _)
    }
}
