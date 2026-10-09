// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HubbardPlaquette {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightBatchSize;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase;
    import QDKChemistry.Utils.HammingWeightPhasing.UsesHammingWeightPhasing;
    import Std.Arrays.Flattened;
    import Std.Arrays.Mapped;
    import Std.Arrays.Subarray;
    import Std.Diagnostics.Fact;
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
        // A zero angle skips the adder tree, which a hopping-only model would pay for nothing.
        if angle != 0.0 {
            HammingWeightPhase(
                angle,
                [[PauliZ, PauliZ], size = sites],
                StridedGroups(sites, 2, 1, sites, systems),
                maxBatchSize
            );
        }
    }

    /// The mode order, as the mode at each position, that puts every four-cycle of a tiling on adjacent positions.
    internal function TilingOrder(blocks : Int[][], count : Int) : Int[] {
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
        return target;
    }

    /// The Jordan-Wigner mode order, which is where every evolution starts and ends.
    internal function IdentityOrder(count : Int) : Int[] {
        mutable order = [];
        for mode in 0..count - 1 {
            set order += [mode];
        }
        return order;
    }

    /// Rearranges the modes from the `current` order to the `target` order, each one listing the mode
    /// at every position: the swap count and, if `listSwaps`, the adjacent swaps.
    internal function RoutingSwaps(current : Int[], target : Int[], listSwaps : Bool) : (Int, Int[]) {
        let count = Length(current);
        Fact(Length(target) == count, "RoutingSwaps needs two orders of the same modes.");
        mutable targetPosition = [0, size = count];
        for position in 0..count - 1 {
            set targetPosition w/= target[position] <- position;
        }

        // order[position] is where the mode currently at that position must end up.
        mutable order = [0, size = count];
        for position in 0..count - 1 {
            set order w/= position <- targetPosition[current[position]];
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

    /// Applies the fermionic swaps that rearrange the modes from the `current` order to the `target` order.
    internal operation RouteModes(current : Int[], target : Int[], systems : Qubit[]) : Unit is Adj {
        let (swapCount, swaps) = RoutingSwaps(current, target, not IsResourceEstimating());
        if IsResourceEstimating() {
            // Every routing swap is the same Clifford pair on a different pair of modes, so the
            // estimator only needs how many there are. Emitting one and repeating its cost keeps
            // the O(sites^1.5) gates out of the trace.
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
    }

    /// # Summary
    /// One hopping tiling: every plaquette's fswap, ffft, then its momentum-basis phase.
    ///
    /// # Description
    /// The modes arrive in `order`, listing the mode at each position, and leave in it.
    internal operation HoppingLayer(
        kappa : Double,
        blocks : Int[][],
        order : Int[],
        systems : Qubit[],
        maxBatchSize : Int
    ) : Unit is Adj + Ctl {
        // A zero angle or an empty tiling skips the routing, basis change and adders, which would
        // be paid for nothing.
        if kappa != 0.0 and Length(blocks) > 0 {
            within {
                RouteModes(order, TilingOrder(blocks, Length(systems)), systems);
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
        if not phasesBatch {
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

    /// One PIG body, I^(1/2) G I^(1/2), followed by the given pink layer, on modes in `order`.
    internal operation PlaquetteStepWithPinkAngle(
        params : HubbardPlaquetteParams,
        pinkAngle : Double,
        order : Int[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        let sites = params.width * params.height;
        let pink = PlaquetteSection(params.width, params.height, true);
        let gold = PlaquetteSection(params.width, params.height, false);
        InteractionLayer(params.interactionAngle / 2.0, sites, systems, params.maxBatchSize);
        HoppingLayer(params.hoppingAngle, gold, order, systems, params.maxBatchSize);
        InteractionLayer(params.interactionAngle / 2.0, sites, systems, params.maxBatchSize);
        HoppingLayer(pinkAngle, pink, order, systems, params.maxBatchSize);
    }

    /// One interior second-order Trotter body, I^(1/2) G I^(1/2) P, on modes in `order`.
    internal operation PlaquetteStep(
        params : HubbardPlaquetteParams,
        order : Int[],
        systems : Qubit[]
    ) : Unit is Adj + Ctl {
        PlaquetteStepWithPinkAngle(params, params.hoppingAngle, order, systems);
    }

    /// # Summary
    /// The whole evolution: the repeated body inside its one-time hopping boundary.
    ///
    /// # Description
    /// Each step is the symmetric product `pink(s/2) I(s/2) gold(s) I(s/2) pink(s/2)`, so
    /// the adjacent half-angle pink layers of neighbouring steps merge into one full-angle
    /// layer. Only a single half-angle pink boundary is emitted at each end. This is the
    /// "PIG" ordering of Eqs. (16a)-(16b) in
    /// :cite:`Apel2026`, which merges the more expensive hopping layers; it deviates from
    /// Eq. (D2) of :cite:`Campbell2022`, which puts the interaction outermost instead.
    ///
    /// The modes are routed into pink order once at the start and back once at the end, so
    /// pink plaquettes are always local and only the gold layers route, from pink order. Both
    /// orders keep the spin-up modes ahead of their spin-down partners, so the on-site pairs
    /// stay one register half apart.
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
        if params.repetitions > 0 {
            let pink = PlaquetteSection(params.width, params.height, true);
            let identity = IdentityOrder(Length(systems));
            // A hopping-free model has no plaquette to keep local, so it skips the routing.
            let order = params.hoppingAngle != 0.0 ? TilingOrder(pink, Length(systems)) | identity;
            within {
                RouteModes(identity, order, systems);
            } apply {
                HoppingLayer(params.hoppingAngle / 2.0, pink, order, systems, params.maxBatchSize);

                if params.repetitions > 1 {
                    if IsResourceEstimating() {
                        within {
                            RepeatEstimates(params.repetitions - 1);
                        } apply {
                            PlaquetteStep(params, order, systems);
                        }
                    } else {
                        for _ in 1..params.repetitions - 1 {
                            PlaquetteStep(params, order, systems);
                        }
                    }
                }

                PlaquetteStepWithPinkAngle(params, params.hoppingAngle / 2.0, order, systems);
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
