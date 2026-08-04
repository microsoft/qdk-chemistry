// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// QFT phase estimation driven by unary iteration over a signed-power schedule.
///
/// Unlike standard QPE, which applies controlled U^(2^k) once per phase qubit and
/// therefore consumes a power-of-two number of queries, this variant applies a
/// single chain of `numQueries` self-inverse blocks and lets unary iteration over
/// the phase register select which reflection to omit. Branch t of the phase
/// register then sees W^(numQueries - 2t), so any positive `numQueries` is allowed.
///
/// The phase register is prepared by `phaseQubitPrep` (a cosine window state)
/// rather than by uniform Hadamards, which suppresses spectral leakage from the
/// truncated, non-power-of-two schedule.
///
/// The schedule itself is block-encoding agnostic. It only needs
///   * a self-inverse block encoding B acting on the target register, and
///   * the sub-register that the walk reflection R acts on,
/// from which it builds W = R·B. Any block encoding that fits that shape -
/// PREPARE-SELECT-PREPARE or anything else - can be scheduled by
/// `MakeSignedPowerScheduleOp` without this module knowing what it is.
namespace QDKChemistry.Utils.UnaryPhaseEstimation {

    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyQFT;
    import Std.Canon.ApplyToEach;
    import Std.Convert.IntAsDouble;
    import Std.Diagnostics.Fact;
    import Std.Math.Ceiling;
    import Std.Math.Lg;
    import Std.Math.AbsI;
    import Std.Math.PI;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.SingleVariant;
    import QDKChemistry.Utils.PrepSelPrep.PrepSelPrep;
    import QDKChemistry.Utils.PrepSelPrep.PSPWalk;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;
    import QDKChemistry.Utils.UnaryIteration.UnaryIterationWithControl;

    /// Number of phase qubits required to address `numQueries + 1` reflection slots.
    function PhaseRegisterSize(numQueries : Int) : Int {
        Fact(numQueries > 0, "numQueries must be positive");
        return Ceiling(Lg(IntAsDouble(numQueries + 1)));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Signed-power schedule (block-encoding agnostic)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Applies `numBlocks` self-inverse blocks, omitting one of `numBlocks + 1` reflections.
    ///
    /// With reflection A, block B, and walk W = A·B, the branch selected by address t
    /// applies W^(numBlocks - 2t): pairs before the omitted reflection compose as W†
    /// and pairs after it compose as W. Because the slot sweep and the address decode
    /// share one unary-iteration ladder, the whole schedule costs O(numBlocks) Toffolis
    /// rather than the O(numBlocks · log numBlocks) of `numBlocks` separately controlled
    /// walk steps.
    operation UnaryIterationPowerSchedule(
        address : Qubit[],
        numBlocks : Int,
        applyReflectionUnlessSelected : (Qubit => Unit is Adj),
        applyBlock : (Unit => Unit is Adj),
    ) : Unit is Adj {
        Fact(numBlocks > 0, "numBlocks must be positive");
        UnaryIterationWithControl(address, numBlocks + 1, (slot, selected) => {
            applyReflectionUnlessSelected(selected);
            if slot < numBlocks {
                applyBlock();
            }
        });
    }

    /// Signed-power schedule of the walk W = Reflect(reflectionRegister)·B, for any B.
    ///
    /// This is the generic engine behind unary-iteration QPE. The caller describes its
    /// block encoding with two callables over the flat target register:
    ///
    /// # Parameters
    /// - `applyBlockEncoding`: Applies one self-inverse block encoding B to the target
    ///   register. It must be its own inverse; the walk is formed here by pairing it with
    ///   the reflection.
    /// - `reflectionRegisterOf`: Selects the sub-register that the walk reflection acts on,
    ///   i.e. the block-encoding ancillas whose all-zero state flags success. It receives the
    ///   same flat target register as `applyBlockEncoding`.
    /// - `numQueries`: Number of blocks applied; need not be a power of two.
    /// - `phaseReg`: The phase register, little-endian, addressing which reflection to omit.
    /// - `allQubits`: The flat target register (system qubits followed by block-encoding
    ///   ancillas).
    operation SignedPowerSchedule(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        reflectionRegisterOf : (Qubit[] -> Qubit[]),
        numQueries : Int,
        phaseReg : Qubit[],
        allQubits : Qubit[],
    ) : Unit is Adj {
        let reflectionRegister = reflectionRegisterOf(allQubits);
        UnaryIterationPowerSchedule(phaseReg, numQueries, (skipControl) => {
            within {
                X(skipControl);
            } apply {
                Controlled Reflect([skipControl], reflectionRegister);
            }
        }, () => {
            applyBlockEncoding(allQubits);
        });
    }

    /// Bind a block encoding into the `(phaseRegister, targets) => Unit` callable QPE expects.
    ///
    /// The returned callable applies the ENTIRE schedule - all `numQueries` blocks - in one
    /// call, which is why it takes the whole phase register rather than a single control qubit.
    /// See `SignedPowerSchedule` for the meaning of the two describing callables.
    function MakeSignedPowerScheduleOp(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        reflectionRegisterOf : (Qubit[] -> Qubit[]),
        numQueries : Int,
    ) : (Qubit[], Qubit[]) => Unit {
        SignedPowerSchedule(applyBlockEncoding, reflectionRegisterOf, numQueries, _, _)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PREPARE-SELECT-PREPARE adapter
    // ═══════════════════════════════════════════════════════════════════════════

    /// Trailing sub-register of `allQubits`, i.e. the block-encoding ancillas.
    ///
    /// This is the reflection register of a PREPARE-SELECT-PREPARE walk: the ancillas are
    /// exactly the qubits whose all-zero state flags a successful block encoding.
    function TrailingAncillaRegister(numSystemQubits : Int, allQubits : Qubit[]) : Qubit[] {
        allQubits[numSystemQubits...]
    }

    /// Applies PREPARE†·SELECT·PREPARE to a flat `[systemReg | ancillaReg]` register.
    operation PSPBlockEncodingOnRegister(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        allQubits : Qubit[],
    ) : Unit is Adj {
        PrepSelPrep(prepareOp, selectOp, allQubits[0..numSystemQubits - 1], allQubits[numSystemQubits...]);
    }

    /// Schedule a PREPARE-SELECT-PREPARE block encoding for unary-iteration QPE.
    ///
    /// The walk W = Reflect(ancillas)·PREPARE†·SELECT·PREPARE is never materialized as a
    /// controlled operation: the reflection is applied by the schedule, which omits exactly
    /// the one selected by the phase register. Register layout is
    /// `[systemReg | ancillaReg]`, matching the QPE convention.
    function MakePSPSignedPowerScheduleOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numQueries : Int,
    ) : (Qubit[], Qubit[]) => Unit {
        MakeSignedPowerScheduleOp(
            PSPBlockEncodingOnRegister(prepareOp, selectOp, numSystemQubits, _),
            TrailingAncillaRegister(numSystemQubits, _),
            numQueries
        )
    }

    /// Repeat the controlled PSP walk `numQueries` times under a single control qubit.
    ///
    /// This is the textbook/iterative QPE schedule, i.e. c-W^numQueries. It is the
    /// non-unary counterpart of `MakePSPSignedPowerScheduleOp` and is exposed with the same
    /// `(controlRegister, targets)` shape so callers can switch between the two.
    operation RepeatedControlledPSPWalk(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numQueries : Int,
        controlReg : Qubit[],
        allQubits : Qubit[],
    ) : Unit {
        Fact(Length(controlReg) == 1, "the repeated controlled walk expects a single control qubit");
        let systems = allQubits[0..numSystemQubits - 1];
        let ancillas = allQubits[numSystemQubits...];
        for _ in 0..numQueries - 1 {
            if BeginEstimateCaching("RepeatedControlledPSPWalk", SingleVariant()) {
                Controlled PSPWalk(controlReg, (prepareOp, selectOp, systems, ancillas));
                EndEstimateCaching();
            }
        }
    }

    /// Creates the PSP walk callable consumed by phase estimation.
    ///
    /// Mirrors the shape phase estimation expects: the caller picks between the
    /// unary-iteration signed-power schedule (`useUnaryIteration = true`, `controlReg` is the
    /// whole phase register) and the repeated controlled walk (`useUnaryIteration = false`,
    /// `controlReg` holds one qubit). Register layout is `[systemReg | ancillaReg]`.
    function MakePSPWalkOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numQueries : Int,
        useUnaryIteration : Bool,
    ) : (Qubit[], Qubit[]) => Unit {
        if useUnaryIteration {
            MakePSPSignedPowerScheduleOp(prepareOp, selectOp, numSystemQubits, numQueries)
        } else {
            RepeatedControlledPSPWalk(prepareOp, selectOp, numSystemQubits, numQueries, _, _)
        }
    }

    /// Build a unary-iteration QPE circuit for an arbitrary (non-power-of-two) query count.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `signedPowerSchedule`: Applies the ENTIRE signed-power schedule on
    ///   (phase register, targets) in a single call. It is not one walk step: the caller
    ///   builds it already bound to `numQueries` - normally via
    ///   `MakeSignedPowerScheduleOp` - and it internally sweeps all `numQueries + 1`
    ///   reflection slots, applying `numQueries` blocks and skipping the one reflection
    ///   selected by the phase register. Fusing that sweep with the address decode is what
    ///   makes the schedule cost O(numQueries) Toffolis instead of
    ///   O(numQueries * log numQueries); do not lift the repetition to this call site.
    ///   The phase register is passed little-endian, matching the unary addressing convention.
    /// - `numQueries`: Total number of block applications; need not be a power of two.
    ///   Used here only to size the phase register - it must equal the query count that
    ///   `signedPowerSchedule` was built with, or the decoded phase will be wrong.
    /// - `ancillas`: An array of indices for the phase ancilla qubits.
    /// - `systems`: An array of indices for the system qubits (state prep target).
    /// - `phaseQubitPrep`: Prepares the window state on the phase register (big-endian).
    /// - `numAncillas`: Number of extra ancillas required by the block encoding.
    /// - `ancillaPrep`: A function to prepare persistent ancillas (e.g., phase gradient state).
    /// # Returns
    /// - `Result[]`: The phase register, LEAST-significant bit first. Circuit executors emit
    ///   the first measured `Result` as the right-most character of the bitstring (the Qiskit
    ///   convention), so this ordering is what makes `int(bitstring, 2)` recover the measured
    ///   value `y`, which satisfies `y / 2^numBits = 2 * phi` for a walk eigenphase `phi`.
    operation MakeUnaryQPECircuit(
        statePrep : Qubit[] => Unit,
        signedPowerSchedule : (Qubit[], Qubit[]) => Unit,
        numQueries : Int,
        ancillas : Int[],
        systems : Int[],
        phaseQubitPrep : Qubit[] => Unit,
        numAncillas : Int,
        ancillaPrep : Qubit[] => Unit is Adj,
    ) : Result[] {
        let numBits = PhaseRegisterSize(numQueries);
        Fact(
            Length(ancillas) == numBits,
            $"phase register must hold {numBits} qubits for {numQueries} queries",
        );

        let totalQubits = numBits + Length(systems) + numAncillas;
        use qs = Qubit[totalQubits];
        let phaseAncillas = Subarray(ancillas, qs);
        let systemQubits = Subarray(systems, qs);
        let beAncillas = if numAncillas == 0 {
            []
        } else {
            qs[numBits + Length(systems)..Length(qs) - 1]
        };
        let allTargets = systemQubits + beAncillas;

        statePrep(systemQubits);
        ancillaPrep(beAncillas);
        phaseQubitPrep(phaseAncillas);

        // ApplyQFT and the window state are big-endian; unary addressing is little-endian.
        // One call applies all `numQueries` blocks: the schedule owns the repetition so the
        // slot sweep and the phase-register decode share a single unary-iteration ladder.
        signedPowerSchedule(Reversed(phaseAncillas), allTargets);

        Adjoint ApplyQFT(phaseAncillas);

        ResetAll(allTargets);
        mutable results = [Zero, size = numBits];
        // `Std.Canon.ApplyQFT` maps a little-endian input to a big-endian output, so
        // `Adjoint ApplyQFT` leaves the phase little-endian in `phaseAncillas`.
        //
        // The register is returned LEAST-significant bit first because that is what the
        // circuit-executor bitstring convention requires: an executor emits the first
        // measured `Result` as the RIGHT-most character of the bitstring (matching Qiskit),
        // so returning `phaseAncillas` in its natural little-endian order is exactly what
        // makes `int(bitstring, 2)` recover the measured value on the Python side.
        for idx in 0..numBits - 1 {
            set results w/= idx <- MResetZ(phaseAncillas[idx]);
        }
        return results;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test wrapper
    // ═══════════════════════════════════════════════════════════════════════════

    internal operation NoAncillaPrep(qs : Qubit[]) : Unit is Adj {}

    /// Signed-power schedule with reflection A = Z and block B = X on one target
    /// prepared in Ry(0.7)|0>.
    ///
    /// Branch `addressValue` must apply exactly (Z·X)^(numBlocks - 2*addressValue),
    /// including the relative phase, which distinguishes every power in the schedule.
    operation TestUnaryIterationSignedPower(numBlocks : Int, addressValue : Int) : Unit {
        let numAddressQubits = QDKChemistry.Utils.UnaryIteration.AddressQubits(numBlocks + 1);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + 1);
        let address = qs[0..numAddressQubits - 1];
        let target = qs[numAddressQubits];
        ApplyXorInPlace(addressValue, address);
        Ry(0.7, target);
        UnaryIterationPowerSchedule(address, numBlocks, (selected) => {
            within {
                X(selected);
            } apply {
                Controlled Z([selected], target);
            }
        }, () => {
            X(target);
        });
        ApplyXorInPlace(addressValue, address);
    }

    /// Dump harness: applies a signed-power schedule with the phase register in superposition.
    ///
    /// The qubits are leaked so the caller can read the joint state and check, for every
    /// address branch, which walk power the schedule actually applied. This reproduces the
    /// exact register handoff `MakeUnaryQPECircuit` performs, including `Reversed`.
    operation TestSchedulePhaseRamp(
        signedPowerSchedule : (Qubit[], Qubit[]) => Unit,
        numQueries : Int,
        numTargets : Int,
        systemAngle : Double,
        applyInverseQft : Bool,
    ) : Unit {
        let numBits = PhaseRegisterSize(numQueries);
        let qs = QIR.Runtime.AllocateQubitArray(numBits + numTargets);
        let phaseReg = qs[0..numBits - 1];
        let targets = qs[numBits...];
        ApplyToEachA(H, phaseReg);
        Ry(systemAngle, targets[0]);
        signedPowerSchedule(Reversed(phaseReg), targets);
        if applyInverseQft {
            Adjoint ApplyQFT(phaseReg);
        }
    }

    /// PREPARE for the synthetic PSP block encoding: a single-ancilla Ry rotation.
    internal operation TestRyPrepare(theta : Double, ancilla : Qubit[]) : Unit is Adj + Ctl {
        Ry(theta, ancilla[0]);
    }

    /// SELECT for the synthetic PSP block encoding: a sign flip on the system qubit.
    internal operation TestSignSelect(ancilla : Qubit[], system : Qubit[]) : Unit is Adj + Ctl {
        Controlled Z(ancilla, system[0]);
    }

    /// Phase ramp for the synthetic PSP block encoding, with the system in `|1>`.
    ///
    /// `PREPARE = Ry(theta)` and `SELECT = c-Z` block-encode `diag(1, cos theta)`, so the
    /// walk phase seen by `|1>` is exactly `theta`. Choosing `theta = pi*j/N` therefore puts
    /// the answer exactly on bin `j`, which makes the post-QFT register contents exact and
    /// lets a test pin their endianness without sampling.
    operation TestSyntheticSchedulePhaseRamp(
        numQueries : Int,
        theta : Double,
        applyInverseQft : Bool,
    ) : Unit {
        TestSchedulePhaseRamp(
            MakePSPSignedPowerScheduleOp(TestRyPrepare(theta, _), TestSignSelect, 1, numQueries),
            numQueries,
            2,
            PI(),
            applyInverseQft
        );
    }

    /// Checks the generic schedule against the explicit walk power, on a PSP block encoding.
    ///
    /// `B = PREPARE†·SELECT·PREPARE` with `PREPARE = Ry(theta)` on one ancilla and
    /// `SELECT = c-Z` is Hermitian and therefore self-inverse, so `W = Reflect(ancilla)·B`
    /// is a genuine qubitization walk encoding `cos(theta)` on the system qubit.
    ///
    /// The schedule at address `t` is applied, then `W^(numQueries - 2t)` is explicitly
    /// undone. If the schedule realizes the documented signed power, the dumped state must
    /// be exactly the prepared input, with no residue on the address or ancilla registers.
    /// Nothing here is tied to a particular block encoding: it exercises the
    /// block-encoding-agnostic path through the generic schedule.
    operation TestPSPSignedPowerSchedule(numQueries : Int, addressValue : Int, theta : Double) : Unit {
        let numAddressQubits = QDKChemistry.Utils.UnaryIteration.AddressQubits(numQueries + 1);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + 2);
        let address = qs[0..numAddressQubits - 1];
        let targets = qs[numAddressQubits...];
        let system = targets[0..0];
        let ancilla = targets[1..1];

        ApplyXorInPlace(addressValue, address);
        Ry(0.9, system[0]);

        let schedule = MakePSPSignedPowerScheduleOp(TestRyPrepare(theta, _), TestSignSelect, 1, numQueries);
        schedule(address, targets);

        let power = numQueries - 2 * addressValue;
        for _ in 1..AbsI(power) {
            if power > 0 {
                Adjoint PSPWalk(TestRyPrepare(theta, _), TestSignSelect, system, ancilla);
            } else {
                PSPWalk(TestRyPrepare(theta, _), TestSignSelect, system, ancilla);
            }
        }

        ApplyXorInPlace(addressValue, address);
    }

    /// Runs `MakeUnaryQPECircuit` on a synthetic one-qubit walk with an exact eigenphase.
    ///
    /// The two self-inverse reflections are `R = X` and
    /// `B = Rz(theta) X Rz(-theta) = cos(theta) X + sin(theta) Y`, whose product is the
    /// walk `W = B·R = Rz(2*theta)`, with `W|0> = e^{-i*theta}|0>` and
    /// `W|1> = e^{+i*theta}|1>`. A uniform window is used, so `numQueries` must be
    /// `2^b - 1` for the window to exactly fill the phase register and the outcome to
    /// be deterministic.
    ///
    /// With `theta = -pi*k/(numQueries + 1)` the returned bits must read `k` for
    /// `systemState = 1` and `(-k) mod (numQueries + 1)` for `systemState = 0`, which
    /// pins the documented relation `y = -+2*phi mod 1` together with every endianness
    /// convention in the chain: big-endian window state, little-endian unary addressing,
    /// and the bit order of the measured phase register.
    operation TestUnaryQpeSyntheticWalk(numQueries : Int, theta : Double, systemState : Int) : Result[] {
        let numBits = PhaseRegisterSize(numQueries);
        Fact(2^numBits == numQueries + 1, "numQueries must be one less than a power of two");

        return MakeUnaryQPECircuit(
            (systems) => {
                if systemState == 1 {
                    X(systems[0]);
                }
            },
            (address, targets) => {
                UnaryIterationPowerSchedule(address, numQueries, (selected) => {
                    within {
                        X(selected);
                    } apply {
                        CNOT(selected, targets[0]);
                    }
                }, () => {
                    Rz(-theta, targets[0]);
                    X(targets[0]);
                    Rz(theta, targets[0]);
                });
            },
            numQueries,
            Std.Arrays.SequenceI(0, numBits - 1),
            [numBits],
            ApplyToEach(H, _),
            0,
            NoAncillaPrep
        );
    }
}
