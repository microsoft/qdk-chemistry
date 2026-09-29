// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HubbardPlaquette {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.PhaseGradient.GeneralizedPhaseGradientAngles;
    import QDKChemistry.Utils.PhaseGradient.PhaseByGeneralizedGradient;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradients;
    import QDKChemistry.Utils.PhaseGradient.PrepareGeneralizedPhaseGradient;
    import Std.Arrays.All;
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
    /// adder tree computes w into log2(count) + 1 bits, the generalized phase-gradient
    /// addition applies e^{2 i theta w} with a single payload rotation, and the constant
    /// becomes a phase on the control when the whole is controlled. This is the catalyzed
    /// variant of Sec. 4.4 of :cite:`Apel2026`. Below the break-even size each term is
    /// applied as its own rotation instead.
    ///
    /// # Input
    /// ## theta
    /// The rotation angle shared by every term.
    /// ## pauliOps
    /// The Pauli string of each term.
    /// ## targets
    /// The qubits of each term, disjoint across terms.
    /// ## catalyst
    /// `TowerCatalystSize(Length(targets))` qubits prepared for the phase `2.0 * theta`;
    /// empty below the break-even size.
    internal operation HammingWeightPhase(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
        let count = Length(targets);
        Fact(Length(pauliOps) == count, "HammingWeightPhase needs one axis list per term.");
        Fact(Length(catalyst) == TowerCatalystSize(count), "HammingWeightPhase got a catalyst of the wrong size.");
        if count < HammingWeightBreakEven() {
            for t in 0..count - 1 {
                Exp(pauliOps[t], -theta, targets[t]);
            }
        } else {
            let inputs = Mapped(term -> Tail(term), targets);
            let (schedule, finalBits, _) = HammingWeightSchedule(count);
            Fact(All(bit -> bit >= 0, finalBits), "Every place value of the Hamming weight must hold a bit.");
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
                PhaseByGeneralizedGradient(2.0 * theta, Mapped(bit -> work[bit], finalBits), catalyst);
                R(PauliI, 2.0 * theta * IntAsDouble(count), inputs[0]);
            }
        }
    }

    /// The smallest equal-angle batch worth phasing through a Hamming-weight register.
    internal function HammingWeightBreakEven() : Int {
        return 8;
    }

    /// Catalyst qubits one equal-angle batch of `count` rotations consumes.
    internal function TowerCatalystSize(count : Int) : Int {
        return count < HammingWeightBreakEven() ? 0 | BitSizeI(count);
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

    /// # Summary
    /// The on-site layer, phased through a Hamming-weight register.
    ///
    /// # Input
    /// ## angle
    /// Pair rotation angle.
    /// ## sites
    /// Number of lattice sites.
    /// ## systems
    /// The system register.
    /// ## catalyst
    /// `TowerCatalystSize(sites)` qubits holding the phase gradient for `2.0 * angle`.
    internal operation InteractionLayer(
        angle : Double,
        sites : Int,
        systems : Qubit[],
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
        // A negligible angle skips the adder tree, which a hopping-only model would pay for nothing.
        if AbsD(angle) > 1e-12 {
            HammingWeightPhase(
                angle,
                [[PauliZ, PauliZ], size = sites],
                StridedGroups(sites, 2, 1, sites, systems),
                catalyst
            );
        }
    }

    /// # Summary
    /// The routing permutation: where the mode at each position must end up.
    ///
    /// # Description
    /// Concatenating the tiling's four-cycles and appending the modes it leaves alone
    /// gives the frame the routing network has to reach, so sorting this permutation
    /// with adjacent transpositions is exactly the routing problem.
    ///
    /// Each four-cycle is placed with its diagonals interleaved. The three butterflies of
    /// a four-mode FFFT pair the two diagonals and then the two surviving modes, which is
    /// a path on the four corners; interleaving the diagonals embeds that path in the
    /// line, so every butterfly acts on adjacent positions and no basis change carries a
    /// Jordan-Wigner string. Cycle order instead leaves both diagonal butterflies two apart.
    internal function RoutingOrder(blocks : Int[][], count : Int) : Int[] {
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
        return order;
    }

    /// # Summary
    /// Adjacent swaps routing each plaquette of a tiling onto contiguous modes.
    internal function RoutingSwaps(blocks : Int[][], count : Int) : Int[] {
        mutable order = RoutingSwaps(blocks, count);
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
    /// in the middle, which is what lets every XX and YY term of the tiling share one
    /// Hamming-weight tower once the modes of a plaquette are routed adjacent.
    ///
    /// # Input
    /// ## kappa
    /// Twice the hopping amplitude times the step duration.
    /// ## blocks
    /// Four-cycles of the tiling, in cycle order.
    /// ## systems
    /// The system register.
    /// ## catalyst
    /// `TowerCatalystSize(2 * Length(blocks))` qubits holding the phase gradient for `-kappa`.
    internal operation HoppingLayer(
        kappa : Double,
        blocks : Int[][],
        systems : Qubit[],
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
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
                TwoModeFFFT(base + 1, base + 0, systems);
                TwoModeFFFT(base + 2, base + 3, systems);
            }
        } apply {
            // The third butterfly is fused into the phases, so the equal-angle family
            // is the middle pair of every plaquette.
            HoppingPhases(kappa / 2.0, StridedGroups(Length(blocks), 2, 4, 1, systems[1...]), catalyst);
        }
    }

    /// # Summary
    /// exp(i angle XX) exp(i angle YY) on every pair, as one equal-angle tower.
    ///
    /// # Description
    /// Sec. 4.4 of :cite:`Apel2026` phases a hopping tiling as a single tower of L^2
    /// rotations. With the catalyzed Hamming-weight phasing every tower costs one payload
    /// rotation and one catalyst, so one tower instead of separate XX and YY towers saves
    /// both, for w(L^2 / 2) more AND operations.
    ///
    /// XX and YY commute, so one Clifford diagonalizes both: CNOT, H, CNOT sends XX to Z
    /// on the first qubit of a pair and YY to -Z on the second, and the trailing X removes
    /// that sign. Both qubits then carry a parity bit of the same angle, which is what
    /// turns the tiling into a single tower of twice as many equal-angle Z rotations.
    ///
    /// # Input
    /// ## angle
    /// The rotation angle shared by every XX and YY term.
    /// ## pairs
    /// Disjoint qubit pairs, one per hopping term.
    /// ## catalyst
    /// `TowerCatalystSize(2 * Length(pairs))` qubits prepared for the phase `-2.0 * angle`.
    internal operation HoppingPhases(angle : Double, pairs : Qubit[][], catalyst : Qubit[]) : Unit is Adj + Ctl {
        Fact(
            Length(catalyst) == TowerCatalystSize(2 * Length(pairs)),
            "HoppingPhases got a catalyst of the wrong size."
        );
        if 2 * Length(pairs) < HammingWeightBreakEven() {
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
                    [[PauliZ], size = 2 * Length(pairs)],
                    Mapped(q -> [q], Flattened(pairs)),
                    catalyst
                );
            }
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
    /// The catalysts are prepared by the caller, once for every repetition and, under phase
    /// estimation, once for every query.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## systems
    /// The system register.
    /// ## catalysts
    /// `PlaquetteCatalystSize(params)` qubits holding `PlaquetteCatalystGradients(params)`.
    internal operation PlaquetteStep(
        params : HubbardPlaquetteParams,
        systems : Qubit[],
        catalysts : Qubit[]
    ) : Unit is Adj + Ctl {
        let sites = params.width * params.height;
        let pink = PlaquetteSection(params.width, params.height, true);
        let gold = PlaquetteSection(params.width, params.height, false);
        let (interaction, _, bulk) = PlaquetteCatalystSlices(params, catalysts);
        InteractionLayer(params.interactionAngle / 2.0, sites, systems, interaction);
        HoppingLayer(params.hoppingAngle, gold, systems, bulk);
        InteractionLayer(params.interactionAngle / 2.0, sites, systems, interaction);
        HoppingLayer(params.hoppingAngle, pink, systems, bulk);
    }

    /// # Summary
    /// Number of catalyst qubits the plaquette evolution consumes.
    ///
    /// # Description
    /// Every tower has one rotation per lattice site, so each catalyst has
    /// `TowerCatalystSize(sites)` qubits. The interaction towers phase by
    /// `interactionAngle` and need their own. The hopping towers phase by `-hoppingAngle`
    /// in the body and by half that on the boundary, and a catalyst for twice an angle is
    /// the one for the angle without its lowest qubit, so both share one register with a
    /// single extra qubit. This is the 2 floor(log2 L^2) + 3 of Sec. 4.4 of :cite:`Apel2026`.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function PlaquetteCatalystSize(params : HubbardPlaquetteParams) : Int {
        let bits = TowerCatalystSize(params.width * params.height);
        return bits == 0 ? 0 | 2 * bits + 1;
    }

    /// # Summary
    /// The phase gradients the plaquette catalyst register holds, in order, as (phase, qubits).
    ///
    /// # Description
    /// Each is the state Σ_k e^{-i·phase·k}|k⟩ that `PrepareGeneralizedPhaseGradient` prepares
    /// from `GeneralizedPhaseGradientAngles(phase, qubits)`. See `PlaquetteCatalystSize`.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function PlaquetteCatalystGradients(params : HubbardPlaquetteParams) : (Double, Int)[] {
        let bits = TowerCatalystSize(params.width * params.height);
        if bits == 0 {
            return [];
        }
        return [(params.interactionAngle, bits), (-params.hoppingAngle / 2.0, bits + 1)];
    }

    /// The interaction, boundary hopping and body hopping catalysts inside the register.
    internal function PlaquetteCatalystSlices(
        params : HubbardPlaquetteParams,
        catalysts : Qubit[]
    ) : (Qubit[], Qubit[], Qubit[]) {
        Fact(Length(catalysts) == PlaquetteCatalystSize(params), "The plaquette catalyst register has the wrong size.");
        let bits = TowerCatalystSize(params.width * params.height);
        if bits == 0 {
            return ([], [], []);
        }
        return (catalysts[0..bits - 1], catalysts[bits..2 * bits - 1], catalysts[bits + 1..2 * bits]);
    }

    /// Prepares the catalyst register `PlaquetteCatalystGradients(params)` describes.
    internal operation PreparePlaquetteCatalysts(
        params : HubbardPlaquetteParams,
        catalysts : Qubit[]
    ) : Unit is Adj + Ctl {
        let gradients = Mapped((phase, size) -> (phase, size, false), PlaquetteCatalystGradients(params));
        PreparePhaseGradients(gradients, catalysts);
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
    /// ## catalysts
    /// `PlaquetteCatalystSize(params)` qubits holding `PlaquetteCatalystGradients(params)`.
    operation RepPlaquetteExp(
        params : HubbardPlaquetteParams,
        systems : Qubit[],
        catalysts : Qubit[]
    ) : Unit is Adj + Ctl {
        let (_, boundary, _) = PlaquetteCatalystSlices(params, catalysts);
        within {
            HoppingLayer(
                params.hoppingAngle / 2.0,
                PlaquetteSection(params.width, params.height, true),
                systems,
                boundary
            );
        } apply {
            if IsResourceEstimating() {
                within {
                    RepeatEstimates(params.repetitions);
                } apply {
                    PlaquetteStep(params, systems, catalysts);
                }
            } else {
                for _ in 1..params.repetitions {
                    PlaquetteStep(params, systems, catalysts);
                }
            }
        }
    }

    /// # Summary
    /// Returns a callable applying a repeated plaquette evolution on catalysts it prepares around itself.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function MakeRepPlaquetteExpOp(params : HubbardPlaquetteParams) : (Qubit[] => Unit is Adj + Ctl) {
        systems => {
            use catalysts = Qubit[PlaquetteCatalystSize(params)];
            within {
                PreparePlaquetteCatalysts(params, catalysts);
            } apply {
                RepPlaquetteExp(params, systems, catalysts);
            }
        }
    }

    /// # Summary
    /// Builds the controlled circuit for a repeated plaquette evolution, preparing its catalysts.
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
        use catalysts = Qubit[PlaquetteCatalystSize(params)];
        within {
            PreparePlaquetteCatalysts(params, catalysts);
        } apply {
            ControlledRepPlaquetteExp(params, qs[control], Subarray(systems, qs) + catalysts);
        }
    }

    /// # Summary
    /// Applies a controlled plaquette evolution whose targets end in its catalyst register.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    /// ## control
    /// The control qubit.
    /// ## targets
    /// The system register first and the last `PlaquetteCatalystSize(params)` qubits
    /// holding `PlaquetteCatalystGradients(params)`, which are left prepared.
    /// Any qubits in between are unused.
    operation ControlledRepPlaquetteExp(
        params : HubbardPlaquetteParams,
        control : Qubit,
        targets : Qubit[]
    ) : Unit is Adj + Ctl {
        let numSystems = 2 * params.width * params.height;
        let numCatalysts = PlaquetteCatalystSize(params);
        Fact(
            Length(targets) >= numSystems + numCatalysts,
            "The targets must hold the system and the catalyst register."
        );
        Controlled RepPlaquetteExp(
            [control],
            (params, targets[0..numSystems - 1], targets[Length(targets) - numCatalysts...])
        );
    }

    /// # Summary
    /// Returns a single-control callable that expects its catalysts prepared by the caller.
    ///
    /// # Description
    /// Phase estimation prepares the register once around every query, which is what the
    /// `phase_gradients` circuit metadata requests. See `ControlledRepPlaquetteExp` for
    /// the target layout.
    ///
    /// # Input
    /// ## params
    /// The lattice shape and the layer angles.
    function MakeRepControlledPlaquetteExpOp(
        params : HubbardPlaquetteParams
    ) : ((Qubit, Qubit[]) => Unit is Adj + Ctl) {
        ControlledRepPlaquetteExp(params, _, _)
    }

    /// Test helper: prepares the catalyst one tower of rotations by `phi / 2` consumes, so a
    /// single layer can be checked in isolation. The evolution prepares all of its catalysts
    /// at once with `PreparePlaquetteCatalysts`.
    internal operation PrepareTowerCatalyst(phi : Double, catalyst : Qubit[]) : Unit is Adj + Ctl {
        PrepareGeneralizedPhaseGradient(GeneralizedPhaseGradientAngles(phi, Length(catalyst)), catalyst);
    }
}
