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
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;
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
    /// With reflection R = `applyReflection`, block B = `applyBlockEncoding` and walk W = R·B,
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
    /// - `applyReflection`: Applies the reflection R the walk pairs B with, to the same flat
    ///   register. Taking it as a callable rather than a sub-register selector is what keeps
    ///   this module independent of how a block encoding lays out its ancillas.
    /// - `numQueries`: Number of blocks applied; need not be a power of two.
    /// - `phaseReg`: The phase register, little-endian, addressing which reflection to omit.
    /// - `allQubits`: The flat target register (system qubits followed by block-encoding ancillas).
    internal operation ApplySignedPowerSchedule(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        numQueries : Int,
        phaseReg : Qubit[],
        allQubits : Qubit[],
    ) : Unit is Adj {
        Fact(numQueries > 0, "numQueries must be positive");
        UnaryIterationWithControl(phaseReg, numQueries + 1, (slot, selected) => {
            within {
                X(selected);
            } apply {
                Controlled applyReflection([selected], allQubits);
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
    /// - `applyReflection`: Applies the reflection the walk pairs B with, on the same flat
    ///   register.
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
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
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
            applyReflection,
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
    //  Test wrappers
    //
    //  Each one takes the block encoding and its reflection as callables, so a caller
    //  supplies whichever block encoding it wants to exercise and nothing here is tied
    //  to a particular one.
    // ═══════════════════════════════════════════════════════════════════════════

    internal operation NoAncillaPrep(qs : Qubit[]) : Unit is Adj {}

    /// `X` on the first qubit: a self-inverse factor usable as a block or as a reflection.
    internal operation TestApplyX(qubits : Qubit[]) : Unit is Adj + Ctl {
        X(qubits[0]);
    }

    /// `Z` on the first qubit: a self-inverse factor usable as a block or as a reflection.
    internal operation TestApplyZ(qubits : Qubit[]) : Unit is Adj + Ctl {
        Z(qubits[0]);
    }

    /// Signed-power schedule with reflection R = Z and block B = X on one target
    /// prepared in Ry(0.7)|0>.
    ///
    /// Branch `addressValue` must apply exactly (Z·X)^(numBlocks - 2*addressValue),
    /// including the relative phase, which distinguishes every power in the schedule.
    operation TestUnaryIterationSignedPower(numBlocks : Int, addressValue : Int) : Unit {
        let numAddressQubits = AddressQubits(numBlocks + 1);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + 1);
        let address = qs[0..numAddressQubits - 1];
        let target = qs[numAddressQubits...];
        ApplyXorInPlace(addressValue, address);
        Ry(0.7, target[0]);
        ApplySignedPowerSchedule(TestApplyX, TestApplyZ, numBlocks, address, target);
        ApplyXorInPlace(addressValue, address);
    }

    /// Dump harness: applies a signed-power schedule with the phase register in superposition.
    ///
    /// The qubits are leaked so the caller can read the joint state and check, for every
    /// address branch, which walk power the schedule actually applied. This reproduces the
    /// exact register handoff `MakeUnaryQPECircuit` performs, including `Reversed`.
    operation TestSchedulePhaseRamp(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
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
            applyReflection,
            numQueries,
            Reversed(phaseReg),
            targets
        );
        if applyInverseQft {
            Adjoint ApplyQFT(phaseReg);
        }
    }

    /// Checks the generic schedule against the explicit walk power.
    ///
    /// The schedule at address `t` is applied, then `W^(numQueries - 2t)` is explicitly undone
    /// with the same two callables. If the schedule realizes the documented signed power, the
    /// dumped state must be exactly the prepared input, with no residue on the address or
    /// ancilla registers.
    ///
    /// `applyBlockEncoding` has to be self-inverse; that is what makes `W = R·B` a genuine
    /// qubitization walk.
    operation TestSignedPowerScheduleAgainstWalk(
        applyBlockEncoding : (Qubit[] => Unit is Adj),
        applyReflection : (Qubit[] => Unit is Adj + Ctl),
        numQueries : Int,
        addressValue : Int,
        numTargets : Int,
        systemAngle : Double,
    ) : Unit {
        let numAddressQubits = AddressQubits(numQueries + 1);
        let qs = QIR.Runtime.AllocateQubitArray(numAddressQubits + numTargets);
        let address = qs[0..numAddressQubits - 1];
        let targets = qs[numAddressQubits...];

        ApplyXorInPlace(addressValue, address);
        Ry(systemAngle, targets[0]);

        ApplySignedPowerSchedule(applyBlockEncoding, applyReflection, numQueries, address, targets);

        let walk = (register) => {
            applyBlockEncoding(register);
            applyReflection(register);
        };
        let power = numQueries - 2 * addressValue;
        for _ in 1..AbsI(power) {
            if power > 0 {
                Adjoint walk(targets);
            } else {
                walk(targets);
            }
        }

        ApplyXorInPlace(addressValue, address);
    }

    /// Block B = Rz(theta)·X·Rz(-theta), a reflection about an axis in the XY plane.
    ///
    /// It is self-inverse, and it does not commute with the reflection `X` it is paired with,
    /// so the walk W = B·X = Rz(2*theta) has genuinely distinct powers. A pair of commuting
    /// factors (two diagonal ones, say) would make every schedule branch collapse to the same
    /// operator and silently pass no matter what the address decode did.
    internal operation TestRzWalkBlock(theta : Double, qubits : Qubit[]) : Unit is Adj {
        Rz(-theta, qubits[0]);
        X(qubits[0]);
        Rz(theta, qubits[0]);
    }

    /// Runs `MakeUnaryQPECircuit` on a synthetic one-qubit walk with an exact eigenphase.
    ///
    /// The reflection on the one-qubit target register is `R = X` and the block is the
    /// self-inverse `B = Rz(theta) X Rz(-theta)`. Their product is the walk
    /// `W = B·R = Rz(2*theta)`, with `W|0> = e^{-i*theta}|0>` and
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
            TestRzWalkBlock(theta, _),
            TestApplyX,
            numQueries,
            Std.Arrays.SequenceI(0, numBits - 1),
            [numBits],
            ApplyToEach(H, _),
            0,
            NoAncillaPrep
        );
    }
}
