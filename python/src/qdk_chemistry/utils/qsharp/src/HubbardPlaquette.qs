// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HubbardPlaquette {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import Std.Arrays.SequenceI;
    import Std.Arrays.Subarray;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.Math.PI;
    import Std.ResourceEstimation.IsResourceEstimating;
    import Std.ResourceEstimation.RepeatEstimates;

    /// # Summary
    /// Parameters of a repeated plaquette evolution.
    ///
    /// # Description
    /// Only the lattice shape and the layer angles cross from Python. Which modes each
    /// layer acts on is a function of the shape, so the tilings and spin pairings are
    /// derived here rather than shipped as index lists that grow with the lattice.
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

    /// # Summary
    /// The Pauli word of a Jordan-Wigner two-mode operator.
    ///
    /// # Description
    /// The endpoints carry `first` and `last`; every mode between them carries the
    /// parity `Z`. A pure function, so adjointable operations may call it.
    ///
    /// # Input
    /// ## lo
    /// The lower mode index.
    /// ## hi
    /// The higher mode index.
    /// ## first
    /// Pauli on the lower endpoint.
    /// ## last
    /// Pauli on the higher endpoint.
    internal function JordanWignerWord(lo : Int, hi : Int, first : Pauli, last : Pauli) : Pauli[] {
        mutable word = [first];
        for _ in lo + 1..hi - 1 {
            set word += [PauliZ];
        }
        return word + [last];
    }

    /// # Summary
    /// The qubits a Jordan-Wigner two-mode operator acts on, endpoints included.
    ///
    /// # Input
    /// ## lo
    /// The lower mode index.
    /// ## hi
    /// The higher mode index.
    /// ## systems
    /// The system register.
    internal function JordanWignerQubits(lo : Int, hi : Int, systems : Qubit[]) : Qubit[] {
        mutable qs = [];
        for mode in lo..hi {
            set qs += [systems[mode]];
        }
        return qs;
    }

    /// # Summary
    /// Fermionic swap of two adjacent Jordan-Wigner modes.
    ///
    /// # Input
    /// ## a
    /// The first mode.
    /// ## b
    /// The second mode, adjacent to `a`.
    operation FSwap(a : Qubit, b : Qubit) : Unit is Adj + Ctl {
        SWAP(a, b);
        CZ(a, b);
    }

    /// # Summary
    /// One radix-2 butterfly of the fermionic fast Fourier transform.
    ///
    /// # Description
    /// The Givens rotation mixing two modes into their symmetric and antisymmetric
    /// combinations, carrying the parity string of the modes between them.
    ///
    /// # Input
    /// ## a
    /// The first mode index.
    /// ## b
    /// The second mode index.
    /// ## systems
    /// The system register.
    operation TwoModeFFFT(a : Int, b : Int, systems : Qubit[]) : Unit is Adj + Ctl {
        let lo = a < b ? a | b;
        let hi = a < b ? b | a;
        let half = (a < b ? 1.0 | -1.0) * PI() / 8.0;
        let qs = JordanWignerQubits(lo, hi, systems);
        Exp(JordanWignerWord(lo, hi, PauliX, PauliY), half, qs);
        Exp(JordanWignerWord(lo, hi, PauliY, PauliX), -half, qs);
    }

    /// # Summary
    /// The equal-angle hopping phase on one bond of a plaquette.
    ///
    /// # Input
    /// ## kappa
    /// Twice the hopping amplitude times the step duration.
    /// ## a
    /// The first mode index.
    /// ## b
    /// The second mode index.
    /// ## systems
    /// The system register.
    operation HoppingPhase(kappa : Double, a : Int, b : Int, systems : Qubit[]) : Unit is Adj + Ctl {
        let lo = a < b ? a | b;
        let hi = a < b ? b | a;
        let qs = JordanWignerQubits(lo, hi, systems);
        Exp(JordanWignerWord(lo, hi, PauliX, PauliX), kappa / 2.0, qs);
        Exp(JordanWignerWord(lo, hi, PauliY, PauliY), kappa / 2.0, qs);
    }


    /// Returns the qubit carrying each mapped term's phase.
    internal function HammingWeightRepresentatives(targets : Qubit[][]) : Qubit[] {
        mutable representatives : Qubit[] = [];
        for term in targets {
            set representatives += [term[Length(term) - 1]];
        }
        return representatives;
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
    internal operation ComputeHammingWeight(
        inputs : Qubit[],
        scratch : Qubit[]
    ) : Unit is Adj {
        let (schedule, _, _) = HammingWeightSchedule(Length(inputs));
        let work = inputs + scratch;
        for (a, b, c, carry) in schedule {
            if c < 0 {
                HalfAdderStep(work[a], work[b], work[carry]);
            } else {
                FullAdderStep(work[a], work[b], work[c], work[carry]);
            }
        }
    }

    /// Applies equal-angle Z phases using logarithmically many rotations.
    internal operation HammingWeightPhase(theta : Double, inputs : Qubit[]) : Unit is Adj + Ctl {
        let count = Length(inputs);
        let (schedule, finalBits, _) = HammingWeightSchedule(count);
        use scratch = Qubit[Length(schedule)];
        let work = inputs + scratch;
        within {
            ComputeHammingWeight(inputs, scratch);
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
                HammingWeightPhase(theta, HammingWeightRepresentatives(targets));
            }
        }
    }

    /// # Summary
    /// Each system qubit as its own single-mode group.
    ///
    /// # Input
    /// ## systems
    /// The system register.
    internal function SingleModes(systems : Qubit[]) : Qubit[][] {
        mutable groups = [];
        for q in systems {
            set groups += [[q]];
        }
        return groups;
    }

    /// # Summary
    /// The spin-up and spin-down qubit of every site.
    ///
    /// # Input
    /// ## sites
    /// Number of lattice sites.
    /// ## systems
    /// The system register.
    internal function SpinPairQubits(sites : Int, systems : Qubit[]) : Qubit[][] {
        mutable groups = [];
        for site in 0..sites - 1 {
            set groups += [[systems[site], systems[site + sites]]];
        }
        return groups;
    }

    /// # Summary
    /// The phase pair of every block of a routed tiling.
    ///
    /// # Description
    /// Routing leaves block `i` on positions `4i .. 4i+3`, so its momentum-basis phase
    /// acts on the adjacent pair `(4i, 4i+1)`. Adjacent modes carry no Jordan-Wigner
    /// string, and distinct blocks occupy disjoint positions, so the whole tiling is one
    /// equal-angle family that can share a single Hamming-weight register.
    ///
    /// # Input
    /// ## count
    /// Number of blocks in the tiling.
    /// ## systems
    /// The routed system register.
    internal function BlockPhaseQubits(count : Int, systems : Qubit[]) : Qubit[][] {
        mutable groups = [];
        for index in 0..count - 1 {
            set groups += [[systems[4 * index], systems[4 * index + 1]]];
        }
        return groups;
    }

    /// # Summary
    /// The on-site layer, phased through Hamming-weight registers.
    ///
    /// # Description
    /// After the particle-hole shift every on-site term is the same angle on a disjoint
    /// qubit pair, so the whole lattice shares one register. The single-mode family is
    /// applied only when the shift leaves it nonzero.
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
            HammingWeightPhaseTerms(onsite, [[PauliZ], size = Length(systems)], SingleModes(systems));
        }
        if angle != 0.0 {
            HammingWeightPhaseTerms(angle, [[PauliZ, PauliZ], size = sites], SpinPairQubits(sites, systems));
        }
    }

    /// # Summary
    /// The mode ordering that makes every plaquette of a tiling contiguous.
    ///
    /// # Description
    /// Concatenating the tiling's four-cycles gives a permutation of all modes in which
    /// plaquette k occupies positions 4k..4k+3 in cycle order. Routing into this frame
    /// is what makes each plaquette's operators act on adjacent modes, so their
    /// Jordan-Wigner strings collapse from O(L^2) to length one and every plaquette in
    /// the tiling becomes mutually disjoint (arXiv:2609.05316, Sec. 4.5.2).
    ///
    /// # Input
    /// ## blocks
    /// Four-cycles of the tiling, in cycle order.
    /// ## count
    /// Total number of modes.
    ///
    /// # Output
    /// `target[position]` is the mode that should end up at that position.
    internal function RoutedOrder(blocks : Int[][], count : Int) : Int[] {
        mutable target = [];
        for block in blocks {
            set target += block;
        }
        // Modes the tiling does not touch keep their relative order at the tail.
        mutable placed = [false, size = count];
        for mode in target {
            set placed w/= mode <- true;
        }
        for mode in 0..count - 1 {
            if not placed[mode] {
                set target += [mode];
            }
        }
        return target;
    }

    /// # Summary
    /// The adjacent transpositions sorting `current` into `target`, as position pairs.
    ///
    /// # Description
    /// An odd-even transposition sort. Only adjacent modes are exchanged, which is what
    /// keeps each swap a two-qubit fermionic operation rather than a long-range one.
    ///
    /// # Input
    /// ## current
    /// The present mode ordering.
    /// ## target
    /// The desired mode ordering.
    ///
    /// # Output
    /// Positions to exchange, in application order.
    internal function TranspositionNetwork(current : Int[], target : Int[]) : Int[] {
        let count = Length(current);
        // rank[mode] is where the mode must end up.
        mutable rank = [0, size = count];
        for position in 0..count - 1 {
            set rank w/= target[position] <- position;
        }
        mutable order = [];
        for mode in current {
            set order += [rank[mode]];
        }
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
    /// Applies a network of adjacent fermionic swaps.
    ///
    /// # Input
    /// ## swaps
    /// Positions to exchange with their right neighbour, in order.
    /// ## systems
    /// The system register.
    operation ApplySwapNetwork(swaps : Int[], systems : Qubit[]) : Unit is Adj + Ctl {
        for position in swaps {
            FSwap(systems[position], systems[position + 1]);
        }
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
        let count = Length(systems);
        let identity = SequenceI(0, count - 1);
        let swaps = TranspositionNetwork(identity, RoutedOrder(blocks, count));
        // An empty tiling yields no swaps and no blocks, so this reduces to identity.
        within {
            // Route the tiling into its contiguous frame, then change basis there.
            ApplySwapNetwork(swaps, systems);
            for index in 0..Length(blocks) - 1 {
                let base = 4 * index;
                TwoModeFFFT(base + 0, base + 2, systems);
                TwoModeFFFT(base + 1, base + 3, systems);
            }
        } apply {
            // Every block's phase is the same angle on a disjoint adjacent pair, so the
            // tiling is not a loop over blocks but two equal-angle families: XX and YY,
            // which commute on a pair. Each family shares one Hamming-weight register,
            // turning the tiling's 2m rotations into 2*ceil(log2(m+1)).
            let pairs = BlockPhaseQubits(Length(blocks), systems);
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
    /// Builds the circuit for a repeated plaquette evolution on a fresh register.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## systems
    /// Indices of the system qubits.
    operation MakeRepPlaquetteExpCircuit(params : HubbardPlaquetteParams, systems : Int[]) : Unit {
        use qs = Qubit[MaxInt(systems) + 1];
        RepPlaquetteExp(params, Subarray(systems, qs));
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
        ControlledRepPlaquetteExpOnRegister(params, _)
    }

    /// # Summary
    /// Applies the controlled evolution to a register whose first qubit is the control.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## register
    /// The control qubit followed by the system qubits.
    operation ControlledRepPlaquetteExpOnRegister(
        params : HubbardPlaquetteParams,
        register : Qubit[]
    ) : Unit is Adj + Ctl {
        ControlledRepPlaquetteExp(params, register[0], register[1...]);
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
