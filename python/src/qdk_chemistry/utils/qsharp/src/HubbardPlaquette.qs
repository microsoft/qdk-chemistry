// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HubbardPlaquette {

    import QDKChemistry.Utils.CircuitComposition.MaxInt;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightBatchSize;
    import QDKChemistry.Utils.HammingWeightPhasing.HammingWeightPhase;
    import QDKChemistry.Utils.HammingWeightPhasing.PrepareCatalyst;
    import Std.Arrays.Any;
    import Std.Arrays.Chunks;
    import Std.Arrays.Filtered;
    import Std.Arrays.Flattened;
    import Std.Arrays.Mapped;
    import Std.Arrays.MappedOverRange;
    import Std.Arrays.Subarray;
    import Std.Arrays.Zipped;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
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
        /// Trotter order of each step, a positive even integer.
        trotterOrder : Int,
        /// Number of repetitions of the body.
        repetitions : Int,
        /// Whether to phase each tower of equal-angle rotations through a Hamming-weight register.
        useHammingWeightPhasing : Bool,
        /// Largest tower phased through a single Hamming-weight register, or -1 for no cap.
        maxBatchSize : Int,
        /// Whether to add each Hamming weight into a catalyst instead of using the `Rz` ladder.
        useCatalyst : Bool,
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

    /// The on-site layer, phased through a Hamming-weight register and, if nonempty, a catalyst
    /// prepared at `angle`.
    internal operation InteractionLayer(
        angle : Double,
        sites : Int,
        systems : Qubit[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
        // A zero angle skips the adder tree, which a hopping-only model would pay for nothing.
        if angle != 0.0 {
            HammingWeightPhase(
                angle,
                [[PauliZ, PauliZ], size = sites],
                StridedGroups(sites, 2, 1, sites, systems),
                useHammingWeightPhasing,
                maxBatchSize,
                catalyst
            );
        }
    }

    /// # Summary
    /// The mode order that puts every four-cycle of a tiling on adjacent positions.
    ///
    /// # Input
    /// ## blocks
    /// The four-cycles of a tiling, as returned by `PlaquetteSection`.
    /// ## count
    /// Total number of fermionic modes.
    ///
    /// # Output
    /// The mode at each position: each cycle `[a, b, c, d]` as `a, c, b, d`, then the unused modes in order.
    function TilingOrder(blocks : Int[][], count : Int) : Int[] {
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

    /// # Summary
    /// The adjacent fermionic swaps that rearrange the modes from one order to another.
    ///
    /// # Input
    /// ## current
    /// The mode at each position before routing.
    /// ## target
    /// The mode at each position after routing.
    /// ## listSwaps
    /// Whether to list the swaps, or only count them.
    ///
    /// # Output
    /// The swap count and, if `listSwaps`, the swaps in order of application, each as the lower position
    /// `p` of the pair `(p, p + 1)`.
    function RoutingSwaps(current : Int[], target : Int[], listSwaps : Bool) : (Int, Int[]) {
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
    /// The modes arrive in `order`, listing the mode at each position, and leave in it. A nonempty
    /// catalyst must be prepared at `-kappa / 2`, the angle of the momentum-basis tower.
    internal operation HoppingLayer(
        kappa : Double,
        blocks : Int[][],
        order : Int[],
        systems : Qubit[],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        catalyst : Qubit[]
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
                HoppingPhases(
                    kappa / 2.0,
                    StridedGroups(Length(blocks), 2, 4, 1, systems[1...]),
                    useHammingWeightPhasing,
                    maxBatchSize,
                    catalyst
                );
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
    /// over separate XX and YY towers. Without Hamming-weight phasing each pair takes its own
    /// two rotations, so the basis change is skipped rather than paid for nothing.
    internal operation HoppingPhases(
        angle : Double,
        pairs : Qubit[][],
        useHammingWeightPhasing : Bool,
        maxBatchSize : Int,
        catalyst : Qubit[]
    ) : Unit is Adj + Ctl {
        let count = 2 * Length(pairs);
        if not useHammingWeightPhasing {
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
                    true,
                    maxBatchSize,
                    catalyst
                );
            }
        }
    }

    /// # Summary
    /// The fractions of a step taken by the second-order steps of Suzuki's formula
    /// S_{2k}(s) = S_{2k-2}(u s)^2 S_{2k-2}((1 - 4u) s) S_{2k-2}(u s)^2, u = 1 / (4 - 4^{1/(2k-1)}).
    ///
    /// # Input
    /// ## order
    /// The Trotter order, a positive even integer.
    ///
    /// # Output
    /// The fraction of each second-order step, in order of application.
    function SuzukiStageFractions(order : Int) : Double[] {
        Fact(order >= 2 and order % 2 == 0, "The Trotter order must be a positive even integer.");
        mutable fractions = [1.0];
        for k in 2..order / 2 {
            let u = 1.0 / (4.0 - 4.0 ^ (1.0 / IntAsDouble(2 * k - 1)));
            mutable composed = [];
            for factor in [u, u, 1.0 - 4.0 * u, u, u] {
                set composed += Mapped(fraction -> factor * fraction, fractions);
            }
            set fractions = composed;
        }
        return fractions;
    }

    /// The distinct entries of `fractions`, in order of first appearance.
    internal function DistinctFractions(fractions : Double[]) : Double[] {
        mutable distinct = [];
        for fraction in fractions {
            if not Any(seen -> seen == fraction, distinct) {
                set distinct += [fraction];
            }
        }
        return distinct;
    }

    /// The catalyst registered for `fraction`, or an empty register if there is none.
    internal function CatalystFor(fraction : Double, catalysts : (Double, Qubit[])[]) : Qubit[] {
        for (catalystFraction, catalyst) in catalysts {
            if catalystFraction == fraction {
                return catalyst;
            }
        }
        return [];
    }

    /// One Trotter step on modes in `order`: each stage I^(1/2) G I^(1/2) followed by its pink layer.
    internal operation PlaquetteStep(
        params : HubbardPlaquetteParams,
        stageFractions : Double[],
        pinkFractions : Double[],
        order : Int[],
        systems : Qubit[],
        interactionCatalysts : (Double, Qubit[])[],
        hoppingCatalysts : (Double, Qubit[])[]
    ) : Unit is Adj + Ctl {
        let sites = params.width * params.height;
        let pink = PlaquetteSection(params.width, params.height, true);
        let gold = PlaquetteSection(params.width, params.height, false);
        let useHwp = params.useHammingWeightPhasing;
        let cap = params.maxBatchSize;
        for stage in 0..Length(stageFractions) - 1 {
            let fraction = stageFractions[stage];
            let pinkFraction = pinkFractions[stage];
            let halfInteraction = fraction * params.interactionAngle / 2.0;
            let interaction = CatalystFor(fraction, interactionCatalysts);
            InteractionLayer(halfInteraction, sites, systems, useHwp, cap, interaction);
            HoppingLayer(
                fraction * params.hoppingAngle,
                gold,
                order,
                systems,
                useHwp,
                cap,
                CatalystFor(fraction, hoppingCatalysts)
            );
            InteractionLayer(halfInteraction, sites, systems, useHwp, cap, interaction);
            HoppingLayer(
                pinkFraction * params.hoppingAngle,
                pink,
                order,
                systems,
                useHwp,
                cap,
                CatalystFor(pinkFraction, hoppingCatalysts)
            );
        }
    }

    /// # Summary
    /// The whole evolution: the repeated body inside its one-time hopping boundary.
    ///
    /// # Description
    /// Each step is `pink(s/2) I(s/2) gold(s) I(s/2) pink(s/2)`, so the half-angle pink layers of
    /// neighbouring steps merge, leaving one half-angle pink boundary at each end. This is the "PIG"
    /// ordering of Eqs. (16a)-(16b) in :cite:`Apel2026`, which merges the more expensive hopping
    /// layers, not Eq. (D2) of :cite:`Campbell2022`, which puts the interaction outermost. Higher
    /// orders compose these steps by `SuzukiStageFractions`.
    ///
    /// The modes are routed into pink order at the start and back at the end, so only the gold
    /// layers route. Both orders keep the spin-up modes ahead of their spin-down partners, so the
    /// on-site pairs stay one register half apart.
    ///
    /// With a catalyst, one is prepared for each distinct interaction and hopping angle. Dropping the
    /// half-angle hopping catalyst's first qubit doubles its angle for the full-angle layers. The
    /// towers' constant phases are applied as one rotation at the end, since under control they
    /// would shift the energy.
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
            let sites = params.width * params.height;
            let pink = PlaquetteSection(params.width, params.height, true);
            let gold = PlaquetteSection(params.width, params.height, false);
            let cap = params.maxBatchSize;
            let catalyzed = params.useHammingWeightPhasing and params.useCatalyst;
            let stageFractions = SuzukiStageFractions(params.trotterOrder);
            let first = stageFractions[0];
            let half = 0.5 * first;
            let merged = MappedOverRange(
                stage -> 0.5 * (stageFractions[stage] + stageFractions[stage + 1]),
                0..Length(stageFractions) - 2
            );
            // Each catalyst holds a qubit per place value of its largest weight; the half-angle hopping
            // one holds one more if a full-angle layer runs, the qubit dropped to double its angle.
            let interactionPlaces = catalyzed and params.interactionAngle != 0.0
                ? BitSizeI(HammingWeightBatchSize(sites, cap)) | 0;
            let hoppingPlaces = catalyzed and params.hoppingAngle != 0.0
                ? BitSizeI(HammingWeightBatchSize(2 * MaxInt([Length(pink), Length(gold)]), cap)) | 0;
            let interactionFractions = interactionPlaces > 0 ? DistinctFractions(stageFractions) | [];
            let fullHoppingFractions = hoppingPlaces > 0
                ? DistinctFractions(
                    (Length(gold) > 0 ? stageFractions | []) + merged + (params.repetitions > 1 ? [first] | [])
                )
                | [];
            let otherHoppingFractions = Filtered(fraction -> fraction != first, fullHoppingFractions);
            let fullFirst = Length(otherHoppingFractions) < Length(fullHoppingFractions);
            use interaction = Qubit[Length(interactionFractions) * interactionPlaces];
            use firstHopping = Qubit[hoppingPlaces > 0 ? hoppingPlaces + (fullFirst ? 1 | 0) | 0];
            use otherHopping = Qubit[Length(otherHoppingFractions) * hoppingPlaces];
            let interactionCatalysts = interactionPlaces > 0
                ? Zipped(interactionFractions, Chunks(interactionPlaces, interaction)) | [];
            let otherHoppingCatalysts = hoppingPlaces > 0
                ? Zipped(otherHoppingFractions, Chunks(hoppingPlaces, otherHopping)) | [];
            let doubledFirst = fullFirst ? [(first, firstHopping[1...])] | [];
            let hoppingCatalysts = hoppingPlaces > 0
                ? [(half, firstHopping)] + doubledFirst + otherHoppingCatalysts
                | [];
            let identity = IdentityOrder(Length(systems));
            // A hopping-free model has no plaquette to keep local, so it skips the routing.
            let order = params.hoppingAngle != 0.0 ? TilingOrder(pink, Length(systems)) | identity;
            within {
                for (fraction, catalyst) in interactionCatalysts {
                    PrepareCatalyst(fraction * params.interactionAngle / 2.0, catalyst);
                }
                PrepareCatalyst(-half * params.hoppingAngle / 2.0, firstHopping);
                for (fraction, catalyst) in otherHoppingCatalysts {
                    PrepareCatalyst(-fraction * params.hoppingAngle / 2.0, catalyst);
                }
                RouteModes(identity, order, systems);
            } apply {
                HoppingLayer(
                    half * params.hoppingAngle,
                    pink,
                    order,
                    systems,
                    params.useHammingWeightPhasing,
                    cap,
                    CatalystFor(half, hoppingCatalysts)
                );

                if params.repetitions > 1 {
                    let interiorPink = merged + [first];
                    if IsResourceEstimating() {
                        within {
                            RepeatEstimates(params.repetitions - 1);
                        } apply {
                            PlaquetteStep(
                                params,
                                stageFractions,
                                interiorPink,
                                order,
                                systems,
                                interactionCatalysts,
                                hoppingCatalysts
                            );
                        }
                    } else {
                        for _ in 1..params.repetitions - 1 {
                            PlaquetteStep(
                                params,
                                stageFractions,
                                interiorPink,
                                order,
                                systems,
                                interactionCatalysts,
                                hoppingCatalysts
                            );
                        }
                    }
                }

                PlaquetteStep(
                    params,
                    stageFractions,
                    merged + [half],
                    order,
                    systems,
                    interactionCatalysts,
                    hoppingCatalysts
                );
            }
            if catalyzed {
                // The constant phases every tower left out sum to these per step; `R(PauliI, g)` is
                // e^{-ig/2}.
                let perStep = params.interactionAngle * IntAsDouble(sites)
                    - params.hoppingAngle * IntAsDouble(Length(pink) + Length(gold));
                if perStep != 0.0 {
                    R(PauliI, 2.0 * IntAsDouble(params.repetitions) * perStep, systems[0]);
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
