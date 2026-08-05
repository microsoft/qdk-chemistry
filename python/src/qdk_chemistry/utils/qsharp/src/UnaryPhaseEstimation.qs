// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

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
    import QDKChemistry.Utils.PrepSelPrep.MakePrepSelPrepOp;
    import QDKChemistry.Utils.PrepSelPrep.MakeTrailingAncillaSelector;
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

    /// Applies `numQueries` self-inverse blocks, omitting the one reflection `phaseReg` selects.
    ///
    /// With reflection R = Reflect(reflectionRegisterOf(allQubits)), block B, and walk W = R·B,
    /// the branch selected by address t applies a signed power of W: pairs before the omitted
    /// reflection compose as W† and pairs after it compose as W, leaving W^(numQueries - 2t).
    /// Because the slot sweep and the address decode share one unary-iteration ladder, the
    /// whole schedule costs O(numQueries) Toffolis rather than the O(numQueries · log numQueries)
    /// of `numQueries` separately controlled walk steps.
    ///
    /// The block encoding is never controlled: only the reflections are, which is what keeps the
    /// cost linear and lets any self-inverse B drive the schedule.
    ///
    /// # Parameters
    /// - `applyBlockEncoding`: Applies one self-inverse block encoding B to the flat target
    ///   register. It must be its own inverse; the walk is formed here by pairing it with R.
    /// - `reflectionRegisterOf`: Selects the sub-register the reflection acts on, i.e. the
    ///   block-encoding ancillas whose all-zero state flags success.
    /// - `numQueries`: Number of blocks applied; need not be a power of two.
    /// - `phaseReg`: The phase register, little-endian, addressing which reflection to omit.
    /// - `allQubits`: The flat target register (system qubits followed by block-encoding ancillas).
    internal operation ApplySignedPowerSchedule(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        reflectionRegisterOf : (Qubit[] -> Qubit[]),
        numQueries : Int,
        phaseReg : Qubit[],
        allQubits : Qubit[],
    ) : Unit is Adj {
        Fact(numQueries > 0, "numQueries must be positive");
        let reflectionRegister = reflectionRegisterOf(allQubits);
        UnaryIterationWithControl(phaseReg, numQueries + 1, (slot, selected) => {
            within {
                X(selected);
            } apply {
                Controlled Reflect([selected], reflectionRegister);
            }
            if slot < numQueries {
                applyBlockEncoding(allQubits);
            }
        });
    }

    /// Build a unary-iteration QPE circuit for an arbitrary (non-power-of-two) query count.
    /// # Parameters
    /// - `statePrep`: A function to prepare the initial quantum state on system qubits.
    /// - `applyBlockEncoding`: Applies ONE self-inverse block encoding B to the flat target
    ///   register, uncontrolled. The schedule below owns the repetition and applies it
    ///   `numQueries` times; do not lift that repetition to this call site, because fusing the
    ///   slot sweep with the address decode is what makes the schedule cost O(numQueries)
    ///   Toffolis instead of O(numQueries * log numQueries).
    /// - `reflectionRegisterOf`: Selects the sub-register the walk reflection acts on, i.e. the
    ///   block-encoding ancillas whose all-zero state flags success.
    /// - `numQueries`: Total number of block applications; need not be a power of two.
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
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        reflectionRegisterOf : (Qubit[] -> Qubit[]),
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
        ApplySignedPowerSchedule(
            applyBlockEncoding,
            reflectionRegisterOf,
            numQueries,
            Reversed(phaseAncillas),
            allTargets
        );

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

    /// Block B = X for the synthetic single-target walk.
    internal operation TestXBlock(qubits : Qubit[]) : Unit is Adj {
        X(qubits[0]);
    }

    /// Signed-power schedule with reflection A = Z and block B = X on one target
    /// prepared in Ry(0.7)|0>.
    ///
    /// `Reflect` on a one-qubit register is `Z`, so passing the whole target register as the
    /// reflection register gives the A = Z, B = X walk.
    ///
    /// Branch `addressValue` must apply exactly (Z·X)^(numBlocks - 2*addressValue),
    /// including the relative phase, which distinguishes every power in the schedule.
    operation TestUnaryIterationSignedPower(numBlocks : Int, addressValue : Int) : Unit {
        let numAddressQubits = QDKChemistry.Utils.UnaryIteration.AddressQubits(numBlocks + 1);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + 1);
        let address = qs[0..numAddressQubits - 1];
        let target = qs[numAddressQubits...];
        ApplyXorInPlace(addressValue, address);
        Ry(0.7, target[0]);
        ApplySignedPowerSchedule(TestXBlock, MakeTrailingAncillaSelector(0), numBlocks, address, target);
        ApplyXorInPlace(addressValue, address);
    }

    /// Dump harness: applies a signed-power schedule with the phase register in superposition.
    ///
    /// The qubits are leaked so the caller can read the joint state and check, for every
    /// address branch, which walk power the schedule actually applied. This reproduces the
    /// exact register handoff `MakeUnaryQPECircuit` performs, including `Reversed`.
    operation TestSchedulePhaseRamp(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        reflectionRegisterOf : (Qubit[] -> Qubit[]),
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
        ApplySignedPowerSchedule(
            applyBlockEncoding,
            reflectionRegisterOf,
            numQueries,
            Reversed(phaseReg),
            targets
        );
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
            MakePrepSelPrepOp(TestRyPrepare(theta, _), TestSignSelect, 1, 1, 1),
            MakeTrailingAncillaSelector(1),
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

        ApplySignedPowerSchedule(
            MakePrepSelPrepOp(TestRyPrepare(theta, _), TestSignSelect, 1, 1, 1),
            MakeTrailingAncillaSelector(1),
            numQueries,
            address,
            targets
        );

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

    /// Block B = Rz(2*theta)·Z, which pairs with the one-qubit reflection to give W = Rz(2*theta).
    internal operation TestRzBlock(theta : Double, qubits : Qubit[]) : Unit is Adj {
        Z(qubits[0]);
        Rz(2.0 * theta, qubits[0]);
    }

    /// Runs `MakeUnaryQPECircuit` on a synthetic one-qubit walk with an exact eigenphase.
    ///
    /// `Reflect` on the one-qubit target register is `R = Z`, and the block is
    /// `B = Rz(2*theta) Z`, which is self-inverse because `Z Rz(a) Z = Rz(-a)`. Their product is
    /// the walk `W = B·R = Rz(2*theta)`, with `W|0> = e^{-i*theta}|0>` and
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
            TestRzBlock(theta, _),
            MakeTrailingAncillaSelector(0),
            numQueries,
            Std.Arrays.SequenceI(0, numBits - 1),
            [numBits],
            ApplyToEach(H, _),
            0,
            NoAncillaPrep
        );
    }
}
