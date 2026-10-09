// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.HammingWeightPhasing {

    import QDKChemistry.Utils.PhaseGradient.RzViaPhaseGradient;
    import Std.Arrays.All;
    import Std.Arrays.Chunks;
    import Std.Arrays.Fold;
    import Std.Arrays.Mapped;
    import Std.Arrays.Tail;
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.BitSizeI;
    import Std.Math.PI;
    import Std.Math.Round;

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
    /// layer collectively with HWP. Given a phase gradient, each place-value rotation is applied
    /// through it rather than synthesized; given none, each is synthesized as its own `Rz`, the
    /// original rotation ladder. Below the break-even size each term is applied as its own
    /// rotation instead.
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
    /// ## gradient
    /// The shared binary phase gradient, prepared by `PreparePhaseGradientState`, or empty to
    /// synthesize every place-value rotation as an `Rz`. Unused when every batch is below the
    /// break-even size.
    internal operation HammingWeightPhase(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        maxBatchSize : Int,
        gradient : Qubit[]
    ) : Unit is Adj + Ctl {
        let count = Length(targets);
        Fact(Length(pauliOps) == count, "HammingWeightPhase needs one axis list per term.");
        if count > 0 {
            // Chunking in a function keeps the index arithmetic out of this adjointable body.
            let batchSize = HammingWeightBatchSize(count, maxBatchSize);
            let opBatches = Chunks(batchSize, pauliOps);
            let targetBatches = Chunks(batchSize, targets);
            for index in 0..Length(targetBatches) - 1 {
                HammingWeightPhaseBatch(theta, opBatches[index], targetBatches[index], gradient);
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
    /// so the peak ancilla count follows the batch rather than the whole tower, while the adder
    /// tree's Toffoli count stays essentially the same. The price is the place-value rotations,
    /// which are paid once per batch instead of once per tower: phase gradient additions
    /// (Toffolis) with a gradient, synthesized `Rz` rotations without one.
    ///
    /// It is therefore a knob that trades qubits for those place-value rotations, and the useful
    /// setting depends on the device: hardware demonstrations of Fermi-Hubbard dynamics run on a
    /// fixed and comparatively small register, where the adder tree of a full lattice-sized tower
    /// may simply not fit, while a fault-tolerant estimate is usually better off spending the
    /// qubits to save the rotations. The default is no cap, which reproduces the uncapped
    /// construction exactly.
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

    /// One batch of equal-angle terms, phased through a single Hamming-weight register and, when
    /// one is given, the shared phase gradient. See `HammingWeightPhase`, which splits a tower
    /// into batches of this shape.
    internal operation HammingWeightPhaseBatch(
        theta : Double,
        pauliOps : Pauli[][],
        targets : Qubit[][],
        gradient : Qubit[]
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
            use scratch = Qubit[Length(schedule)];
            let work = inputs + scratch;
            within {
                ComputeHammingWeight(pauliOps, targets, schedule, work);
            } apply {
                PhaseHammingWeight(theta, count, Mapped(bit -> work[bit], finalBits), inputs[0], gradient);
            }
        }
    }

    /// # Summary
    /// Applies e^{-i·theta·count} e^{2i·theta·w} to the little-endian Hamming weight w of a batch
    /// of `count` equal-angle terms.
    ///
    /// # Description
    /// w = Σ_j 2^j w_j, so e^{2i·theta·w} is one rotation per place value, the bit of place value
    /// 2^j taking the angle 2·theta·2^j. With a phase gradient those rotations are applied by
    /// `PhaseByBinaryGradient`; without one each is synthesized as its own `Rz`. Either way
    /// `Rz(a) = e^{-ia/2} R1(a)`, so the layer carries an extra Π_j e^{-i·a_j/2}, and a single
    /// `R(PauliI, g)` on `anchor` both cancels it and supplies the batch constant e^{-i·theta·count}.
    /// Under control that phase is not global, which is why it is applied rather than dropped.
    ///
    /// # Input
    /// ## theta
    /// The rotation angle shared by every term of the batch.
    /// ## count
    /// Number of terms in the batch.
    /// ## weight
    /// The Hamming weight, little-endian, one qubit per place value.
    /// ## anchor
    /// Any qubit to carry the `R(PauliI, _)` constant; it is left unchanged.
    /// ## gradient
    /// The shared binary phase gradient, or empty to synthesize each place-value rotation.
    internal operation PhaseHammingWeight(
        theta : Double,
        count : Int,
        weight : Qubit[],
        anchor : Qubit,
        gradient : Qubit[]
    ) : Unit is Adj + Ctl {
        let places = Length(weight);
        if Length(gradient) == 0 {
            for j in 0..places - 1 {
                Rz(2.0 * theta * IntAsDouble(1 <<< j), weight[j]);
            }
            // The ladder's angles sum to 2·theta·(2^places - 1).
            R(PauliI, 2.0 * theta * IntAsDouble(count - ((1 <<< places) - 1)), anchor);
        } else {
            let words = BinaryGradientWords(2.0 * theta, places, Length(gradient));
            PhaseByBinaryGradient(words, weight, gradient);
            // Word x stands for the angle 4π·x/2^bits, rounded from 2·theta·2^j.
            R(
                PauliI,
                2.0 * theta * IntAsDouble(count)
                    - 4.0 * PI() * Fold((total, word) -> total + IntAsDouble(word), 0.0, words)
                        / IntAsDouble(1 <<< Length(gradient)),
                anchor
            );
        }
    }

    /// # Summary
    /// Applies e^{i·phi·w} to the little-endian integer w held in `weight`, through a shared
    /// binary phase gradient.
    ///
    /// # Description
    /// Hamming-weight phasing (:cite:`Kan2025`): w = Σ_j 2^j w_j, so the phase
    /// is a layer of one rotation per place value, the bit of place value 2^j taking the angle
    /// phi·2^j. Every one of those angles is classical, so `RzViaPhaseGradient` applies it by
    /// adding its rounded word into the gradient register, which costs one addition instead of a
    /// synthesized rotation and leaves the register prepared for the next use.
    ///
    /// `Rz(a)` is `diag(e^{-ia/2}, e^{ia/2})`, so the layer also carries a constant phase. The
    /// caller must cancel it: under control it is not a global phase.
    ///
    /// Cost: `Length(words)` additions of `Length(gradient)` bits, so
    /// `Length(words) · (Length(gradient) - 1)` AND operations, plus Cliffords. Under control the
    /// word load is controlled and the additions are not, which adds one CNOT per set bit.
    ///
    /// # Input
    /// ## words
    /// The rotation words `BinaryGradientWords` returns, one per place value of w.
    /// ## weight
    /// The integer w, little-endian, one qubit per place value.
    /// ## gradient
    /// A register prepared by `PreparePhaseGradientState`, returned in that state.
    operation PhaseByBinaryGradient(words : Int[], weight : Qubit[], gradient : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            RzWordLayer(words, weight, gradient, []);
        }
        controlled (controls, ...) {
            if Length(controls) <= 1 {
                RzWordLayer(words, weight, gradient, controls);
            } else {
                use joint = Qubit();
                within {
                    Controlled X(controls, joint);
                } apply {
                    RzWordLayer(words, weight, gradient, [joint]);
                }
            }
        }
    }

    /// One `RzViaPhaseGradient` per word, on the matching target, from one reusable angle register.
    ///
    /// # Description
    /// A word is loaded with X gates, so controlling the load costs one CNOT per set bit and
    /// leaves both the addition and the gradient register uncontrolled: with the control clear the
    /// word is zero, and adding zero is the identity.
    internal operation RzWordLayer(
        words : Int[],
        targets : Qubit[],
        gradient : Qubit[],
        controls : Qubit[]
    ) : Unit is Adj {
        Fact(Length(words) == Length(targets), "RzWordLayer needs one rotation word per target.");
        Fact(Length(gradient) > 0, "RzWordLayer needs a non-empty phase gradient register.");
        use angle = Qubit[Length(gradient)];
        for index in 0..Length(targets) - 1 {
            within {
                Controlled ApplyXorInPlace(controls, (words[index], angle));
            } apply {
                RzViaPhaseGradient(targets[index], angle, gradient);
            }
        }
    }

    /// # Summary
    /// The rotation words `PhaseByBinaryGradient` applies for the phase `phi` on `n` place values.
    ///
    /// # Description
    /// Word j is the `bits`-bit integer x whose `Rz(4π·x/2^bits)` is closest to `Rz(phi·2^j)`, so
    /// each angle is exact up to the 2π/2^bits resolution of a `bits`-qubit gradient. `Rz` has
    /// period 4π and the word covers that period exactly, so no place value overflows however
    /// large phi·2^j grows.
    function BinaryGradientWords(phi : Double, n : Int, bits : Int) : Int[] {
        Fact(bits > 0, "BinaryGradientWords needs a non-empty gradient register.");
        let modulus = 1 <<< bits;
        let scale = IntAsDouble(modulus) / (4.0 * PI());
        mutable words = [];
        for j in 0..n - 1 {
            let raw = Round(phi * IntAsDouble(1 <<< j) * scale);
            set words += [((raw % modulus) + modulus) % modulus];
        }
        return words;
    }

    /// Whether a tower of `count` equal-angle rotations is phased through a Hamming-weight
    /// register: not below the measured break-even of 8, where the adder tree costs more than the
    /// rotations it saves.
    internal function UsesHammingWeightPhasing(count : Int) : Bool {
        return count >= 8;
    }

    /// Phase gradient qubits a tower of `count` equal-angle rotations consumes: none when every
    /// batch the cap leaves is below the break-even.
    internal function TowerGradientSize(count : Int, maxBatchSize : Int, rotationBitPrecision : Int) : Int {
        let phasesBatch = count > 0 and UsesHammingWeightPhasing(HammingWeightBatchSize(count, maxBatchSize));
        return phasesBatch ? rotationBitPrecision | 0;
    }
}
