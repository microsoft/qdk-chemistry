// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Sum of Squares Spectral Amplification (SOSSA) walk operator.
///
/// Composable design: each sub-operation (OuterPrepare, InnerPrepare, Select)
/// is built independently as a Q# callable, and this module assembles them into
/// the block encoding B that unary-iteration phase estimation queries, and into the
/// controlled walk that standard and iterative phase estimation query.
///
/// Walk operator (Low et al., Phys. Rev. X 15 (2025), inline above Eq. (9); App. A 2):
///   W = Ref_{a,B} · B,  B = U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Mapped;
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.MaxI;
    import Std.Measurement.MResetX;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.SingleVariant;
    import Std.Arithmetic.ApplyIfLessOrEqualL;
    import Std.Arithmetic.IncByLUsingIncByLE;
    import Std.Arithmetic.RippleCarryCGIncByLE;
    import Std.Convert.IntAsBigInt;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;
    import QDKChemistry.Utils.UnaryIteration.AddressQubits;

    /// Parameters for SELECT factory functions.
    struct SelectParams {
        numOrbitals : Int,
        numRanks : Int,
        numBases : Int,
        numCopies : Int,
        numPositiveOneBody : Int,
        OneBodyRotationAngles : Double[][],
        TwoBodyRotationAngles : Double[][],
        rotationBitPrecision : Int,
        /// Whether every Givens angle is pre-reduced to [0, π), so its word drops the two always-zero top bits.
        compressedAngleWords : Bool,
        /// Givens angles held at once (λ, :cite:`Low2026` E3); 0 or ≥ N-1 keeps them all resident.
        rotationBatchSize : Int,
        /// Whether streamed batches and inner alias tables borrow idle system qubits for swaps.
        borrowDirty : Bool,
        /// Cap on any walk QROAM's swap width: -1 lets each selector choose, 0 forces plain `Select`.
        maxSwapBits : Int,
        /// Whether streamed batches share J + 1 telescoping lookups, borrowing `sysRegUp` when `borrowDirty` (:cite:`Low2026` E3).
        amortizeBatchLookups : Bool,
        /// Whether each streamed angle word is erased by measurement right after its rotation.
        measureAngleWords : Bool,
        /// Whether unswapped Givens-angle loads use a box-restricted skew tree instead of `Select`.
        useSkewTree : Bool,
        /// Whether x_o is ordered [SF at 2^nR·c + r | D1 | Q1] and SELECT recomputes the generator
        /// type, rank, spin and b == B from it instead of keeping them live (:cite:`Low2026` E2).
        recomputeOuterRegisters : Bool,
        /// Whether the N-1 Givens rotations run as the log-depth binary tree (:cite:`Low2026` Eq. D6)
        /// instead of the nearest-neighbour sweep; the rotation angles are always given as sweep angles.
        givensTree : Bool,
        /// Whether each phase-gradient Givens step is one controlled adder of a full-angle word
        /// (:cite:`Low2026` FIG. 39, Lemma 6) instead of two uncontrolled adders of a half-angle word.
        givensControlledAdder : Bool,
        /// Number of free-rider bits at the end of innerReg loaded by inner PREPARE QROM.
        /// Must be at least 2 unless `recomputeOuterRegisters`, which reads none.
        /// Layout: [sf_vs_dq(1), d_vs_q(1), r_bits(⌈log₂ R⌉)].
        /// SelectImpl reads isSF and dvsq from the first two bits.
        numFreeRiderBits : Int,
        /// Index into innerReg of the qubit the inner PREPARE loads the sign of the sampled
        /// (x_o, b) coefficient into, or -1 when the inner PREPARE supplies no sign bit.
        signQubitIndex : Int,
        /// Distinct nonzero Givens angle vectors that the one-hot SELECT (arXiv:2611.09822) loads one qubit each for.
        oneHotAngles : Double[][],
        /// `oneHotAngles` index of each DQ row x_o, or -1 for none; empty keeps the binary angle word.
        oneHotDQIndex : Int[],
        /// `oneHotAngles` index of each SF row b * R + r, or -1 for none.
        oneHotSFIndex : Int[],
        /// Whether spin swaps apply a Bell-basis parity phase (arXiv:2611.09822) instead of N controlled SWAPs.
        bellParitySpinSwap : Bool,
        /// Whether each one-hot SELECT qubit stands for two angle vectors, chosen by one more qubit (:cite:`Shirobokov2026` Eq. 23).
        oneHotPairedVectors : Bool,
        /// Whether SELECT leaves its trailing parity swap to the walk, which shares it with the following SELECT†.
        walkSharesSpinParity : Bool,
    }

    /// Sizes of the registers packed into the target register handed to a walk callable.
    ///
    /// Layout: allQubits = [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    ///
    /// `numInnerQubits` is the full width the inner PREPARE acts on. Only its first
    /// `numReflectInner` qubits — the index, uniform and inequality-flag registers that
    /// carry the success flag — sit in `allQubits`; the QROM output and free-rider bits
    /// behind them are scratch that `SOSSABlockEncodingOnRegister` allocates itself.
    ///
    /// `numOuterPrepareGradientQubits` is the prefix of the phase-gradient register the outer
    /// PREPARE reads; it is shared with SELECT rather than added to the total width.
    struct SOSSAWalkLayout {
        numSystemQubits : Int,
        numOuterQubits : Int,
        numOuterIndexQubits : Int,
        numInnerQubits : Int,
        numReflectInner : Int,
        numPhaseGradientQubits : Int,
        numOuterPrepareGradientQubits : Int,
        numFreeRiderQubits : Int,
        /// Whether one Bell-basis parity serves the spin swaps that end SELECT and begin SELECT†;
        /// the SELECT must then be built with `walkSharesSpinParity`.
        sharedSpinParity : Bool,
    }

    /// The individual registers sliced out of the target register.
    struct SOSSAWalkRegisters {
        systemReg : Qubit[],
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        phaseGradientReg : Qubit[],
    }

    /// Index at which each register starts, followed by the end of the last one.
    internal function SOSSAWalkRegisterBounds(layout : SOSSAWalkLayout) : Int[] {
        let outerStart = layout.numSystemQubits;
        let innerStart = outerStart + layout.numOuterQubits;
        let spinStart = innerStart + layout.numReflectInner;
        let gradientStart = spinStart + 2;
        [outerStart, innerStart, spinStart, gradientStart, gradientStart + layout.numPhaseGradientQubits]
    }

    /// Slice the flat target register into the registers `layout` describes.
    internal function SOSSAWalkRegistersOf(layout : SOSSAWalkLayout, allQubits : Qubit[]) : SOSSAWalkRegisters {
        let bounds = SOSSAWalkRegisterBounds(layout);
        let outerStart = bounds[0];
        let innerStart = bounds[1];
        let spinStart = bounds[2];
        let gradientStart = bounds[3];
        new SOSSAWalkRegisters {
            systemReg = allQubits[0..outerStart - 1],
            outerReg = allQubits[outerStart..innerStart - 1],
            innerReg = allQubits[innerStart..spinStart - 1],
            spinReg = allQubits[spinStart..gradientStart - 1],
            phaseGradientReg = if bounds[4] > gradientStart {
                allQubits[gradientStart..bounds[4] - 1]
            } else {
                []
            }
        }
    }

    /// SELECT implementation (arXiv:2502.15882v1, Appendix B.3, B.5-B.6).
    ///
    /// Implements: within{SelectSpins} apply{ within{GivensRotations} apply{MajoranaOp} }
    ///
    /// isSF and dvsq are read from the free-rider register at the end of innerReg,
    /// spinDQ and spinSF are initialized during outer and inner preparation steps.
    ///
    /// # Parameters
    /// ## usePhaseGradient
    /// When true, uses QROM + phase gradient for Givens rotations (production).
    /// When false, uses direct controlled-Ry gates (simulation/testing).
    ///
    /// Register layout:
    ///   outerReg:  [xoReg (xoBits)]
    ///   innerReg:  [bReg (bBits)] [alias garbage...] [freeRider: isSF(1) + dvsq(1) + rBits(...)]
    ///   spinReg:   [spinDQ (1)] [spinSF (1)]
    ///   systemReg: [sysDown (N)] [sysUp (N)]
    ///   phaseGradientReg: [bRot qubits] (prepared externally; empty for direct mode)
    operation SelectImpl(
        params : SelectParams,
        usePhaseGradient : Bool,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj + Ctl {
        let N = params.numOrbitals;
        let numSF = params.numRanks * params.numCopies;
        let recompute = params.recomputeOuterRegisters;
        // E2 orders x_o as [SF at 2^nR·c + r | D1 | Q1], so SF spans [0, sfSpan) and r is its low nR bits.
        let nR = if params.numRanks > 0 { AddressQubits(params.numRanks) } else { 0 };
        let sfSpan = (1 <<< nR) * params.numCopies;
        let Xo = if recompute { sfSpan + N } else { N + numSF };
        let xoBits = MaxI(1, AddressQubits(Xo));
        let numBp1 = params.numBases + 1;
        let bBits = MaxI(1, AddressQubits(numBp1));
        let numRotAngles = N - 1;
        let rotationParams = if params.givensTree {
            new SelectParams {
                ...params,
                OneBodyRotationAngles = Mapped(angles -> SweepToTreeAngles(N, angles), params.OneBodyRotationAngles),
                TwoBodyRotationAngles = Mapped(angles -> SweepToTreeAngles(N, angles), params.TwoBodyRotationAngles),
                oneHotAngles = Mapped(angles -> SweepToTreeAngles(N, angles), params.oneHotAngles)
            }
        } else {
            params
        };

        // Register slicing
        let xoReg = outerReg[0..xoBits - 1];
        let spinDQ = spinReg[0];
        let spinSF = spinReg[1];
        let bReg = innerReg[0..bBits - 1];
        let sysRegDown = systemReg[0..N - 1];
        let sysRegUp = systemReg[N..2 * N - 1];

        // Free-rider data from inner PREPARE QROM: [sf_vs_dq(1), d_vs_q(1), r_bits...].
        // The two leading bits are read unconditionally below, so they are required rather
        // than optional: with nFR = 0 the reads below would run off the end of innerReg.
        // Recomputing from x_o reads none of them.
        let nInner = Length(innerReg);
        let nFR = params.numFreeRiderBits;
        Fact(recompute or nFR >= 2, "SelectImpl requires at least two free-rider bits (sf_vs_dq and d_vs_q)");

        // `spin` by default; the recomputed [x_o < sfSpan] flag when recomputing from x_o.
        use flag = Qubit();
        use bEqBQubit = Qubit[if recompute { 0 } else { 1 }];
        let isSF = if recompute { flag } else { innerReg[nInner - nFR] };
        let rBits = if recompute { xoReg[0..nR - 1] } elif nFR > 2 { innerReg[nInner - nFR + 2..nInner - 1] } else { [] };

        // sign of the sampled (x_o, b) coefficient.
        if params.signQubitIndex >= 0 {
            Z(innerReg[params.signQubitIndex]);
        }
        // Q1's a^dagger sign on the Majorana selector, which nothing below touches (:cite:`Low2026` Eq. E12).
        if recompute and params.numPositiveOneBody < N {
            ApplyIfLessOrEqualL(Z, IntAsBigInt(sfSpan + params.numPositiveOneBody), xoReg, spinSF);
        }

        // The Majorana operator runs in the Givens-rotated basis, so it is passed into the
        // rotation step instead of being wrapped around it.
        let majoranaStep : (Unit => Unit is Adj + Ctl) = () => {
            if recompute {
                // SF: Z unless b == B, the identity term, applied as a multi-controlled Z.
                Controlled Z([isSF], sysRegDown[0]);
                ApplyControlledOnInt(params.numBases, Controlled Z([isSF], _), bReg, sysRegDown[0]);
                within { X(isSF); } apply {
                    CNOT(isSF, sysRegDown[0]);
                    Controlled Z([isSF, spinSF], sysRegDown[0]);
                }
            } else {
                MajoranaOp(isSF, innerReg[nInner - nFR + 1], bEqBQubit[0], flag, spinSF, sysRegDown[0]);
            }
        };

        // Givens rotations: basis change to localize amplitude on qubit 0, around the Majorana operator.
        let rotatedMajorana : (Unit => Unit is Adj + Ctl) = () => {
            if usePhaseGradient {
                WithGivensRotationsQROM(
                    rotationParams,
                    N,
                    numSF,
                    numBp1,
                    numRotAngles,
                    xoBits,
                    isSF,
                    xoReg,
                    bReg,
                    rBits,
                    sysRegDown,
                    sysRegUp,
                    phaseGradientReg,
                    bEqBQubit,
                    majoranaStep
                );
            } else {
                within {
                    ApplyMultiControlledRotations(
                        rotationParams,
                        N,
                        numSF,
                        numBp1,
                        numRotAngles,
                        xoBits,
                        isSF,
                        xoReg,
                        bReg,
                        rBits,
                        sysRegDown,
                        bEqBQubit
                    );
                } apply {
                    // Majorana operator (Fig. 4 / Appendix B.6)
                    majoranaStep();
                }
            }
        };

        // Adding 2^(xoBits+1) - sfSpan sets the carry to [x_o < sfSpan], leaves r in the low
        // nR bits of an SF index and turns a DQ index into its one-body index a; spinDQ then
        // holds the active spin, and spinSF stays the DQ Majorana selector.
        let recomputeSpins : (Unit => Unit is Adj + Ctl) = () => {
            IncByLUsingIncByLE(RippleCarryCGIncByLE, IntAsBigInt((1 <<< (xoBits + 1)) - sfSpan), xoReg + [isSF]);
            Controlled SWAP([isSF], (spinDQ, spinSF));
        };
        if params.bellParitySpinSwap {
            let activeSpin = if recompute { spinDQ } else { flag };
            within {
                if recompute {
                    recomputeSpins();
                } else {
                    SelectSpins(isSF, spinDQ, spinSF, flag, [], []);
                }
            } apply {
                use parity = Qubit();
                within {
                    ComputeSpinParity(sysRegDown, sysRegUp, parity);
                } apply {
                    CZ(activeSpin, parity);
                }
                rotatedMajorana();
                if not params.walkSharesSpinParity {
                    within {
                        ComputeSpinParity(sysRegDown, sysRegUp, parity);
                    } apply {
                        CZ(activeSpin, parity);
                    }
                }
            }
        } else {
            within {
                if recompute {
                    recomputeSpins();
                    Controlled ApplyToEachCA([spinDQ], (SWAP, Zipped(sysRegDown, sysRegUp)));
                } else {
                    SelectSpins(isSF, spinDQ, spinSF, flag, sysRegDown, sysRegUp);
                }
            } apply {
                rotatedMajorana();
            }
        }
    }

    /// Write into `parity` the number of singlet pairs (registerDown[i], registerUp[i]) mod 2 (arXiv:2611.09822).
    ///
    /// A pair SWAP is -1 on the singlet and +1 on the triplets, so CZ(spin, parity) between this
    /// and its adjoint is the controlled SWAP of the two registers for N Toffolis. The adjoint
    /// erases `parity` with an X measurement and a Clifford phase fix-up, so it costs no Toffolis.
    internal operation ComputeSpinParity(registerDown : Qubit[], registerUp : Qubit[], parity : Qubit) : Unit is Adj {
        body ... {
            within {
                ApplyToEachA(CNOT, Zipped(registerDown, registerUp));
                ApplyToEachA(H, registerDown);
            } apply {
                for (down, up) in Zipped(registerDown, registerUp) {
                    CCNOT(down, up, parity);
                }
            }
        }
        adjoint ... {
            within {
                ApplyToEachA(CNOT, Zipped(registerDown, registerUp));
                ApplyToEachA(H, registerDown);
            } apply {
                if MResetX(parity) == One {
                    ApplyToEachA(CZ, Zipped(registerDown, registerUp));
                }
            }
        }
    }

    /// Apply CZ(spin, parity) with spin = isSF ? spinSF : spinDQ, using one AND with measurement uncompute.
    internal operation ApplySpinParityPhase(isSF : Qubit, spinDQ : Qubit, spinSF : Qubit, parity : Qubit) : Unit is Adj {
        use differs = Qubit();
        within {
            CNOT(spinDQ, spinSF);
            AND(isSF, spinSF, differs);
        } apply {
            CZ(differs, parity);
        }
        CZ(spinDQ, parity);
    }

    /// Apply the self-inverse block B = U† · Ref_B · U used by the SOSSA walk.
    ///
    /// `numOuterPrepareGradientQubits` is the prefix of `phaseGradientReg` that the outer
    /// PREPARE reads (nonzero only for a QROM PREPARE); it is appended to `outerReg` when
    /// the PREPARE is applied. The gradient is a persistent resource that every consumer
    /// leaves unchanged, so PREPARE and SELECT may both address it.
    operation SOSSABlockEncoding(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        numOuterPrepareGradientQubits : Int,
        numFreeRiderQubits : Int,
        sharedSpinParity : Bool,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        ApplySOSSABlockEncoding(
            [],
            outerPrepareOp,
            freeRiderOp,
            innerPrepareOp,
            selectOp,
            numReflectInner,
            numOuterIndexQubits,
            numOuterPrepareGradientQubits,
            numFreeRiderQubits,
            sharedSpinParity,
            outerReg,
            innerReg,
            spinReg,
            systemReg,
            phaseGradientReg,
        );
    }

    /// Apply B = U† · Ref_B · U with Ref_B controlled on `controls`.
    ///
    /// U runs unconditionally: with the controls off U† · U = I, so this is controlled-B
    /// without a controlled PREPARE, which the inner alias-sampling PREPARE does not provide.
    /// With no controls it is the uncontrolled B.
    internal operation ApplySOSSABlockEncoding(
        controls : Qubit[],
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        numOuterPrepareGradientQubits : Int,
        numFreeRiderQubits : Int,
        sharedSpinParity : Bool,
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        systemReg : Qubit[],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        let outerIndexReg = outerReg[0..numOuterIndexQubits - 1];
        let outerPrepareReg = outerReg + phaseGradientReg[0..numOuterPrepareGradientQubits - 1];
        let freeRiderReg = if numFreeRiderQubits > 0 {
            innerReg[Length(innerReg) - numFreeRiderQubits...]
        } else {
            []
        };

        within {
            outerPrepareOp(outerPrepareReg);
            H(spinReg[0]);
        } apply {
            within {
                freeRiderOp(outerIndexReg, freeRiderReg);
            } apply {
                // With `sharedSpinParity`, SELECT omits its trailing spin swap and one parity serves it and
                // SELECT†'s leading swap: the walk leaves the system unchanged between them (arXiv:2611.09822).
                use parity = Qubit[sharedSpinParity ? 1 | 0];
                within {
                    within {
                        // Only SELECT touches `systemReg`, and it runs in `apply`, so it is idle and lendable here.
                        innerPrepareOp(outerIndexReg, innerReg, systemReg);
                        H(spinReg[1]);
                    } apply {
                        selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
                        if sharedSpinParity {
                            let N = Length(systemReg) / 2;
                            ComputeSpinParity(systemReg[0..N - 1], systemReg[N...], parity[0]);
                            ApplySpinParityPhase(freeRiderReg[0], spinReg[0], spinReg[1], parity[0]);
                        }
                    }
                } apply {
                    let reflected = innerReg[0..numReflectInner - 1] + [spinReg[1]];
                    if Length(controls) == 0 {
                        Reflect(reflected);
                    } else {
                        Controlled Reflect(controls, reflected);
                    }
                }
            }
        }
    }

    /// Apply the SOSSA block encoding to a flat target register.
    ///
    /// Adapts `SOSSABlockEncoding` to the `Qubit[] => Unit is Adj` shape that the generic
    /// signed-power schedule consumes, by slicing the flat register with `layout`.
    /// The inner PREPARE's QROM output and free-rider bits are allocated here rather than
    /// taken from `allQubits`. Both are written and exactly uncompute inside this operation.
    operation SOSSABlockEncodingOnRegister(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        allQubits : Qubit[],
    ) : Unit is Adj {
        let regs = SOSSAWalkRegistersOf(layout, allQubits);
        use innerScratch = Qubit[layout.numInnerQubits - layout.numReflectInner];
        SOSSABlockEncoding(
            outerPrepareOp,
            freeRiderOp,
            innerPrepareOp,
            selectOp,
            layout.numReflectInner,
            layout.numOuterIndexQubits,
            layout.numOuterPrepareGradientQubits,
            layout.numFreeRiderQubits,
            layout.sharedSpinParity,
            regs.outerReg,
            regs.innerReg + innerScratch,
            regs.spinReg,
            regs.systemReg,
            regs.phaseGradientReg,
        );
    }

    /// Apply controlled-W^power, W = Ref_a · B, to a flat target register.
    ///
    /// Ref_a reflects about the zero state of `[outerReg | innerReg | spinReg]`, the same
    /// register the unary-iteration QPE reflects; the phase-gradient tail is not reflected.
    /// Both reflections are controlled and U runs unconditionally, see `ApplySOSSABlockEncoding`.
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    /// `cacheName` keys the resource-estimation cache of one walk step. It must identify the walk,
    /// so that distinct walks in one program are estimated separately and the powers of one walk share it.
    /// W and W† cost the same, so both directions share the entry. The adjoint is written out because
    /// a generated one would run `EndEstimateCaching` before the step and cache an empty estimate.
    operation ControlledSOSSAWalkOnRegister(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        cacheName : String,
        power : Int,
        control : Qubit,
        allQubits : Qubit[],
    ) : Unit is Adj {
        body ... {
            for _ in 1..power {
                if BeginEstimateCaching(cacheName, SingleVariant()) {
                    ControlledSOSSAWalkStep(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, control, allQubits);
                    EndEstimateCaching();
                }
            }
        }
        adjoint ... {
            for _ in 1..power {
                if BeginEstimateCaching(cacheName, SingleVariant()) {
                    Adjoint ControlledSOSSAWalkStep(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, control, allQubits);
                    EndEstimateCaching();
                }
            }
        }
    }

    /// Apply one step of controlled-W, see `ControlledSOSSAWalkOnRegister`.
    operation ControlledSOSSAWalkStep(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        control : Qubit,
        allQubits : Qubit[],
    ) : Unit is Adj {
        let regs = SOSSAWalkRegistersOf(layout, allQubits);
        {
            // The block encoding returns the inner scratch to |0⟩, so release it before Ref_a.
            use innerScratch = Qubit[layout.numInnerQubits - layout.numReflectInner];
            ApplySOSSABlockEncoding(
                [control],
                outerPrepareOp,
                freeRiderOp,
                innerPrepareOp,
                selectOp,
                layout.numReflectInner,
                layout.numOuterIndexQubits,
                layout.numOuterPrepareGradientQubits,
                layout.numFreeRiderQubits,
                layout.sharedSpinParity,
                regs.outerReg,
                regs.innerReg + innerScratch,
                regs.spinReg,
                regs.systemReg,
                regs.phaseGradientReg,
            );
        }
        Controlled Reflect([control], regs.outerReg + regs.innerReg + regs.spinReg);
    }

    /// Select spin qubit and SWAP up/down registers (arXiv:2502.15882v1, Step 4).
    ///
    /// Coherently computes `spin` from (isSF, spinDQ, spinSF):
    ///   - DQ mode (isSF=0): spin ← spinDQ
    ///   - SF mode (isSF=1): spin ← spinSF
    /// Then SWAPs registerDown ↔ registerUp controlled on spin.
    operation SelectSpins(
        isSF : Qubit,
        spinDQ : Qubit,
        spinSF : Qubit,
        spin : Qubit,
        registerDown : Qubit[],
        registerUp : Qubit[]
    ) : Unit is Adj + Ctl {
        // DQ mode: copy spinDQ to spin (fires when isSF=0)
        within { X(isSF); } apply { CCNOT(isSF, spinDQ, spin); }
        // SF mode: copy spinSF to spin (fires when isSF=1)
        CCNOT(isSF, spinSF, spin);
        // SWAP up/down registers based on spin
        Controlled ApplyToEachCA([spin], (SWAP, Zipped(registerDown, registerUp)));
    }

    /// Controlled Majorana Operator on single qubit (arXiv:2502.15882v1, Fig. 4 / Appendix B.6).
    ///
    /// - `sf_vs_dq`: 1 if SF (two-body), 0 if DQ (one-body)
    /// - `d_vs_q`: 0 for D1 (annihilation), 1 for Q1 (creation)
    /// - `bEqB`: 1 if b==B (identity term for SF), 0 otherwise
    /// - `spin`: computed spin qubit controlling up/down
    /// - `system_reg_0`: target qubit (qubit 0 after Givens rotation)
    operation MajoranaOp(
        sf_vs_dq : Qubit,
        d_vs_q : Qubit,
        bEqB : Qubit,
        spin : Qubit,
        majoranaSel : Qubit,
        system_reg_0 : Qubit
    ) : Unit is Adj + Ctl {
        // SF two-body (b < B): Z on system_reg_0 when sf_vs_dq=1 AND bEqB=0
        within { X(bEqB); } apply {
            Controlled Z([sf_vs_dq, bEqB], system_reg_0);
        }
        // DQ: a_k = (X + iY)/2 as a two-term LCU on majoranaSel, which the inner
        // reflection projects. X on the |0> branch, ZX = iY on the |1> branch.
        within { X(sf_vs_dq); } apply {
            CNOT(sf_vs_dq, system_reg_0);
            Controlled Z([sf_vs_dq, majoranaSel], system_reg_0);
        }
        // Q1 uses a_k^dagger = (X - iY)/2, so the sign flip belongs on the branch that
        // carries the iY term -- the Majorana selector, not the spin selector.
        within { X(sf_vs_dq); } apply {
            Controlled Z([sf_vs_dq, d_vs_q], majoranaSel);
        }
    }

    /// Build a SELECT using QROM + phase gradient rotation.
    function MakeSelectPhaseGradient(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, true, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// Build a SELECT using direct rotation synthesis.
    function MakeSelectDirectRotation(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, false, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// The SOSSA block encoding B as the `Qubit[] => Unit is Adj` callable QPE consumes.
    ///
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    /// the gradient tail is only present when `layout.numPhaseGradientQubits > 0`.
    function MakeSOSSABlockEncodingOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : (Qubit[] => Unit is Adj) {
        SOSSABlockEncodingOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, _)
    }

    /// Circuit entry point: allocates the flat register and applies the block encoding once.
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    operation MakeSOSSABlockEncodingCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : Unit {
        let bounds = SOSSAWalkRegisterBounds(layout);
        use allQubits = Qubit[layout.numSystemQubits + bounds[4] - bounds[0]];
        if layout.numPhaseGradientQubits > 0 {
            let gradientStart = Length(allQubits) - layout.numPhaseGradientQubits;
            PreparePhaseGradientState(allQubits[gradientStart...]);
        }
        SOSSABlockEncodingOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, allQubits);
        ResetAll(allQubits);
    }

    /// Controlled-W^power as the `(control, targets)` callable standard and iterative QPE consume.
    ///
    /// Target layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    function MakeControlledSOSSAWalkOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        cacheName : String,
        power : Int,
    ) : ((Qubit, Qubit[]) => Unit is Adj) {
        ControlledSOSSAWalkOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, cacheName, power, _, _)
    }

    /// Circuit entry point: allocates the control and the flat target register and applies
    /// controlled-W^power once.
    operation MakeControlledSOSSAWalkCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        cacheName : String,
        power : Int,
    ) : Unit {
        let bounds = SOSSAWalkRegisterBounds(layout);
        use control = Qubit();
        use allQubits = Qubit[bounds[4]];
        if layout.numPhaseGradientQubits > 0 {
            PreparePhaseGradientState(allQubits[bounds[3]...]);
        }
        ControlledSOSSAWalkOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, cacheName, power, control, allQubits);
        ResetAll([control] + allQubits);
    }
}
