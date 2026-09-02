// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Sum of Squares Spectral Amplification (SOSSA) walk operator.
///
/// Composable design: each sub-operation (OuterPrepare, InnerPrepare, Select)
/// is built independently as a Q# callable, and this module assembles them into
/// the block encoding B that phase estimation queries.
///
/// Walk operator (Low et al., Phys. Rev. X 15 (2025), inline above Eq. (9); App. A 2):
///   W = Ref_{a,B} · B,  B = U† · Ref_B · U
/// where U = OuterPREP · within{InnerPREP} apply{SELECT}.
///
/// Only B lives here. `Ref_{a,B}` is a plain reflection about the all-zero state of the
/// leading `SOSSAWalkAncillaCount(layout) - layout.numPhaseGradientQubits` ancillas, so
/// callers pair `MakeSOSSABlockEncodingOp` with the generic
/// `QDKChemistry.Utils.PrepSelPrep.MakeAncillaReflectionOp`.
namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Padded;
    import Std.Arrays.Reversed;
    import Std.Arrays.Subarray;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyToEachCA;
    import Std.Convert.IntAsBoolArray;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;

    import Std.Math.AbsD;
    import Std.Math.BitSizeI;

    import Std.Math.PI;
    import Std.Math.Round;
    import Std.StatePreparation.PreparePureStateD;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.AliasSampling.ConditionalAliasSamplingPrepareWithFreeRider;
    import Std.Arithmetic.RippleCarryCGIncByLE;
    import QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState, QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;
    import QDKChemistry.Utils.PrepSelPrep.Reflect;


    // ═══════════════════════════════════════════════════════════════════════════
    // Inner PREPARE
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build an inner PREPARE using conditional alias sampling (2D QROM).
    ///
    /// Uses ConditionalAliasSamplingPrepareWithFreeRider to prepare:
    ///   |x_o⟩|0⟩ → |x_o⟩ Σ_b √(p̃_{x_o,b}) e^{iπ·sign} |b⟩|garbage⟩
    ///
    /// Pass `freeRiderData = []` to leave the free-rider word to `MakeFreeRiderLoadOp`. That
    /// pays off only when the lookup takes the select-swap path, where the word widens the
    /// QROAM output the swap network is charged for, four times per block encoding. On the
    /// unary-iteration path the cost does not depend on the output width, so carrying it here
    /// is free and a separate load would be pure overhead.
    ///
    /// The returned callable expects:
    ///   outerReg — conditional address register (x_o)
    ///   innerReg — target register layout: indexReg[nIdx] + uniformReg[μ]
    ///              + flagQubit[1] + qromOutput[μ + nIdx + 2] + freeRiderReg[nFR]
    function MakeInnerPrepareAliasSampling(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][],
        coefficientBitPrecision : Int,
    ) : (Qubit[], Qubit[]) => Unit is Adj {
        let nCoeffs = Length(innerCoefficients[0]);
        let nIndexBits = BitSizeI(nCoeffs - 1);
        let mu = coefficientBitPrecision;
        let nFreeRider = if Length(freeRiderData) > 0 { Length(freeRiderData[0]) } else { 0 };
        let qromEnd = 2 * nIndexBits + 2 * mu + 2;
        (outerReg, innerReg) => {
            let indexReg = innerReg[0..nIndexBits - 1];
            let uniformReg = innerReg[nIndexBits..nIndexBits + mu - 1];
            let flagQubit = innerReg[nIndexBits + mu];
            let qromOut = innerReg[nIndexBits + mu + 1..qromEnd];
            let freeRiderReg = if nFreeRider > 0 {
                innerReg[qromEnd + 1..qromEnd + nFreeRider]
            } else {
                []
            };
            ConditionalAliasSamplingPrepareWithFreeRider(
                innerCoefficients,
                freeRiderData,
                mu,
                outerReg,
                indexReg,
                uniformReg,
                flagQubit,
                qromOut,
                freeRiderReg, -1
            );
        }
    }

    /// Load the free-rider word (G, r) for the current x_o.
    ///
    /// It is a function of x_o alone, so the block encoding loads it once around both SELECT
    /// calls rather than letting each inner PREPARE carry it: one `Select` round trip against
    /// four widened QROAM round trips.
    function MakeFreeRiderLoadOp(freeRiderData : Bool[][]) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, freeRiderReg) => {
            if Length(freeRiderData) > 0 and Length(freeRiderReg) > 0 {
                Select(freeRiderData, outerReg, freeRiderReg);
            }
        }
    }

    /// Build an inner PREPARE using direct controlled preparation.
    /// Prepares the b superposition via controlled PreparePureStateD. The free-rider word is
    /// loaded by `MakeFreeRiderLoadOp`, hoisted out of the block encoding's inner loop.
    function MakeInnerPrepareDirect(
        innerCoefficients : Double[][],
        freeRiderData : Bool[][]
    ) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        let nCoeffs = Length(innerCoefficients[0]);
        let nIndexBits = BitSizeI((if nCoeffs > 1 { nCoeffs } else { 2 }) - 1);
        // innerReg layout: bReg[nIndexBits] + freeRiderReg[nFR]
        (outerReg, innerReg) => {
            let bReg = innerReg[0..nIndexBits - 1];

            let xo = Length(innerCoefficients);
            for i in 0..xo - 1 {
                let nPadded = 1 <<< nIndexBits;
                let paddedAmps = Padded(-nPadded, 0.0, innerCoefficients[i]);
                ApplyControlledOnInt(
                    i,
                    PreparePureStateD(paddedAmps, _),
                    outerReg,
                    Reversed(bReg),
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SELECT
    // ═══════════════════════════════════════════════════════════════════════════

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
        /// Number of free-rider bits at the end of innerReg loaded by inner PREPARE QROM.
        /// Layout: [sf_vs_dq(1), d_vs_q(1), r_bits(⌈log₂ R⌉)].
        /// When > 0, SelectImpl reads isSF and dvsq from innerReg instead of computing them.
        numFreeRiderBits : Int,
    }

    /// Build a SELECT using QROM + phase gradient rotation.
    ///
    /// Givens rotations are applied via split OneBody/TwoBody Select QROM + RyViaPhaseGradient.
    /// The phase gradient register is allocated and prepared externally by QPE.
    function MakeSelectPhaseGradient(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, true, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// Build a SELECT using direct rotation synthesis.
    ///
    /// Givens rotations are applied via multi-controlled Ry gates.
    /// Useful for simulation and testing (no ancilla overhead).
    /// The phaseGradientReg argument is accepted but ignored.
    function MakeSelectDirectRotation(
        params : SelectParams
    ) : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl {
        (outerReg, innerReg, spinReg, systemReg, phaseGradientReg) => {
            SelectImpl(params, false, outerReg, innerReg, spinReg, systemReg, phaseGradientReg);
        }
    }

    /// SELECT implementation (arXiv:2502.15882v1, Appendix B.3, B.5-B.6).
    ///
    /// Implements: within{SelectSpins} apply{ within{GivensRotations} apply{MajoranaOp} }
    ///
    /// isSF and dvsq are read from the free-rider register at the end of innerReg,
    /// loaded by inner PREPARE's conditional alias sampling QROM.
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
        let Xo = N + numSF;
        let xoBits = BitSizeI((if Xo > 1 { Xo } else { 2 }) - 1);
        let numBp1 = params.numBases + 1;
        let bBits = BitSizeI((if numBp1 > 1 { numBp1 } else { 2 }) - 1);
        let numRotAngles = N - 1;

        // Register slicing
        let xoReg = outerReg[0..xoBits - 1];
        let spinDQ = spinReg[0];
        let spinSF = spinReg[1];
        let bReg = innerReg[0..bBits - 1];
        let sysRegDown = systemReg[0..N - 1];
        let sysRegUp = systemReg[N..2 * N - 1];

        // Free-rider data from inner PREPARE QROM: [sf_vs_dq(1), d_vs_q(1), r_bits...]
        let nInner = Length(innerReg);
        let nFR = params.numFreeRiderBits;
        let isSF = innerReg[nInner - nFR];       // sf_vs_dq
        let dvsq = innerReg[nInner - nFR + 1];   // d_vs_q

        use spin = Qubit();
        use bEqBQubit = Qubit();

        // The Majorana operator runs in the Givens-rotated basis, so it is passed into the
        // rotation step instead of being wrapped around it. That lets the QROM path hold its
        // angle word live across forward chain -> Majorana -> inverse chain and pay for the
        // tables once per SELECT rather than twice. The direct-rotation path has no table to
        // amortize, so it keeps the plain within/apply shape.
        let majoranaStep : (Unit => Unit is Adj + Ctl) = () => {
            MajoranaOp(isSF, dvsq, bEqBQubit, spin, sysRegDown[0]);
            within { X(isSF); } apply {
                Controlled Z([isSF], spinSF);
            }
        };

        within {
            SelectSpins(isSF, spinDQ, spinSF, spin, sysRegDown, sysRegUp);
        } apply {
            // Givens rotations: basis change to localize amplitude on qubit 0
            if usePhaseGradient {
                let rBits = if nFR > 2 { innerReg[nInner - nFR + 2..nInner - 1] } else { [] };
                WithGivensRotationsQROM(
                    params,
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
                    phaseGradientReg,
                    bEqBQubit,
                    majoranaStep
                );
            } else {
                within {
                    ApplyMultiControlledRotations(
                        params,
                        N,
                        numSF,
                        numBp1,
                        numRotAngles,
                        xoBits,
                        isSF,
                        xoReg,
                        bReg,
                        sysRegDown,
                        bEqBQubit
                    );
                } apply {
                    // Majorana operator (Fig. 4 / Appendix B.6)
                    majoranaStep();
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Walk step
    // ═══════════════════════════════════════════════════════════════════════════

    /// Apply the self-inverse block B = U† · Ref_B · U used by the SOSSA walk.
    ///
    /// `numOuterPrepareGradientQubits` is the prefix of `phaseGradientReg` that the outer
    /// PREPARE reads (nonzero only for a QROM PREPARE); it is appended to `outerReg` when
    /// the PREPARE is applied. The gradient is a persistent resource that every consumer
    /// leaves unchanged, so PREPARE and SELECT may both address it.
    operation SOSSABlockEncoding(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numReflectInner : Int,
        numOuterIndexQubits : Int,
        numOuterPrepareGradientQubits : Int,
        numFreeRiderQubits : Int,
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

        outerPrepareOp(outerPrepareReg);
        H(spinReg[0]);
        // The free-rider word depends only on x_o, which is fixed for the whole block, so it
        // is loaded once around both SELECT calls. It sits past `numReflectInner`, so holding
        // it live across the reflection does not disturb it.
        within {
            freeRiderOp(outerIndexReg, freeRiderReg);
        } apply {
            within {
                innerPrepareOp(outerIndexReg, innerReg);
                H(spinReg[1]);
            } apply {
                selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
            }

            Reflect(innerReg[0..numReflectInner - 1] + [spinReg[1]]);

            within {
                innerPrepareOp(outerIndexReg, innerReg);
                H(spinReg[1]);
            } apply {
                Adjoint selectOp(outerIndexReg, innerReg, spinReg, systemReg, phaseGradientReg);
            }
        }
        H(spinReg[0]);
        Adjoint outerPrepareOp(outerPrepareReg);
    }

    /// Apply the SOSSA block encoding to a flat target register.
    ///
    /// Adapts `SOSSABlockEncoding` to the `Qubit[] => Unit is Adj` shape that the generic
    /// signed-power schedule consumes, by slicing the flat register with `layout`.
    ///
    /// The inner PREPARE's QROM output and free-rider bits are allocated here rather than
    /// taken from `allQubits`. Both are written and exactly uncompute inside this operation,
    /// because SELECT only reads them, so they never carry the success flag. Keeping them
    /// off the flat register is what makes the ancillas the caller reflects about a
    /// contiguous block, so a SOSSA walk pairs with the generic `MakeAncillaReflectionOp`.
    operation SOSSABlockEncodingOnRegister(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
        allQubits : Qubit[],
    ) : Unit is Adj {
        let regs = SplitSOSSAWalkRegisters(layout, allQubits);
        use innerScratch = Qubit[SOSSAInnerScratchCount(layout)];
        SOSSABlockEncoding(
            outerPrepareOp,
            freeRiderOp,
            innerPrepareOp,
            selectOp,
            layout.numReflectInner,
            layout.numOuterIndexQubits,
            layout.numOuterPrepareGradientQubits,
            layout.numFreeRiderQubits,
            regs.outerReg,
            regs.innerReg + innerScratch,
            regs.spinReg,
            regs.systemReg,
            regs.phaseGradientReg,
        );
    }

    /// The SOSSA block encoding B as the `Qubit[] => Unit is Adj` callable QPE consumes.
    ///
    /// The register it takes is `[systemReg | outerReg | innerReg | spinReg | phaseGradientReg]`;
    /// the gradient tail is only present when `layout.numPhaseGradientQubits > 0`.
    function MakeSOSSABlockEncodingOp(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : (Qubit[] => Unit is Adj) {
        SOSSABlockEncodingOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, _)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Register layout
    // ═══════════════════════════════════════════════════════════════════════════

    /// Number of spin qubits in the block encoding: spinReg = [spinDQ, spinSF].
    function NumSOSSASpinQubits() : Int {
        2
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
    }

    /// The individual registers sliced out of the target register.
    struct SOSSAWalkRegisters {
        systemReg : Qubit[],
        outerReg : Qubit[],
        innerReg : Qubit[],
        spinReg : Qubit[],
        phaseGradientReg : Qubit[],
    }

    /// Width of the inner-PREPARE scratch that is allocated rather than passed in.
    function SOSSAInnerScratchCount(layout : SOSSAWalkLayout) : Int {
        layout.numInnerQubits - layout.numReflectInner
    }

    /// Number of block-encoding ancillas (everything except the system register).
    ///
    /// The reflected ancillas come first and the persistent phase gradient last, so a caller
    /// reflects about `SOSSAWalkAncillaCount(layout) - layout.numPhaseGradientQubits` qubits
    /// starting at `layout.numSystemQubits`.
    function SOSSAWalkAncillaCount(layout : SOSSAWalkLayout) : Int {
        layout.numOuterQubits + layout.numReflectInner + NumSOSSASpinQubits() + layout.numPhaseGradientQubits
    }

    /// Slice a flat target register into the SOSSA block-encoding registers.
    ///
    /// `innerReg` is only the reflected prefix; `SOSSABlockEncodingOnRegister` appends the
    /// scratch it allocates before handing the register to the inner PREPARE and SELECT.
    function SplitSOSSAWalkRegisters(layout : SOSSAWalkLayout, allQubits : Qubit[]) : SOSSAWalkRegisters {
        let outerStart = layout.numSystemQubits;
        let innerStart = outerStart + layout.numOuterQubits;
        let spinStart = innerStart + layout.numReflectInner;
        let gradientStart = spinStart + NumSOSSASpinQubits();
        new SOSSAWalkRegisters {
            systemReg = allQubits[0..outerStart - 1],
            outerReg = allQubits[outerStart..innerStart - 1],
            innerReg = allQubits[innerStart..spinStart - 1],
            spinReg = allQubits[spinStart..gradientStart - 1],
            phaseGradientReg = if layout.numPhaseGradientQubits > 0 {
                allQubits[gradientStart..gradientStart + layout.numPhaseGradientQubits - 1]
            } else {
                []
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // QPE-facing entry point
    // ═══════════════════════════════════════════════════════════════════════════

    /// Circuit entry point: allocates the flat register and applies the block encoding once.
    /// Register layout: [systemReg | outerReg | innerReg | spinReg | phaseGradientReg].
    operation MakeSOSSABlockEncodingCircuit(
        outerPrepareOp : (Qubit[]) => Unit is Adj + Ctl,
        freeRiderOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        innerPrepareOp : (Qubit[], Qubit[]) => Unit is Adj,
        selectOp : (Qubit[], Qubit[], Qubit[], Qubit[], Qubit[]) => Unit is Adj + Ctl,
        layout : SOSSAWalkLayout,
    ) : Unit {
        use allQubits = Qubit[layout.numSystemQubits + SOSSAWalkAncillaCount(layout)];
        if layout.numPhaseGradientQubits > 0 {
            let gradientStart = Length(allQubits) - layout.numPhaseGradientQubits;
            PreparePhaseGradientState(allQubits[gradientStart...]);
        }
        SOSSABlockEncodingOnRegister(outerPrepareOp, freeRiderOp, innerPrepareOp, selectOp, layout, allQubits);
        ResetAll(allQubits);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════════

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
        for i in 0..Length(registerDown) - 1 {
            Controlled SWAP([spin], (registerDown[i], registerUp[i]));
        }
    }

    /// Givens rotation chain with CNOT sandwich (arXiv:2502.15882v1, Appendix B.5).
    ///
    /// Each step G_{j,j+1}(θ) = CX(j→j+1) · Ry(-2θ, j) · CX(j→j+1) acts as a
    /// 2×2 rotation in the single-excitation subspace {|01⟩,|10⟩} of (target[j], target[j+1]).
    /// The full chain maps orbital content to qubit 0 for MajoranaOp.
    ///
    /// NOTE: This is the direct rotation (simulation) version. The production
    /// implementation should use QROM to load rotation angles into an ancilla
    /// register, then apply phase-gradient rotation (Rz via addition to a
    /// phase-gradient register). See MakeSelectPhaseGradient.
    ///
    /// DQ rotations: controlled on xoReg ∈ [0, N), unconditional on b.
    /// SF rotations: controlled on (xoReg, bReg) jointly.
    operation ApplyMultiControlledRotations(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        isSF : Qubit,
        xoReg : Qubit[],
        bReg : Qubit[],
        sysRegDown : Qubit[],
        bEqBQubit : Qubit
    ) : Unit is Adj + Ctl {
        for j in 0..numRotAngles - 1 {
            CNOT(sysRegDown[j], sysRegDown[j + 1]);

            // DQ rotations: x_o in [0, N)
            // No extra control on sysRegDown[j+1]: the CNOT sandwich already
            // restricts the 1-excitation subspace, and the Givens rotation must
            // act on ALL sectors (0, 1, 2 excitations) so that its adjoint
            // properly uncomputes after MajoranaOp changes the particle number.
            for a in 0..N - 1 {
                let angle = params.OneBodyRotationAngles[a][j];
                ApplyControlledOnInt(a, Ry(-2.0 * angle, _), xoReg, sysRegDown[j]);
            }

            // SF rotations: x_o in [N, N+numSF), conditioned on b
            for xoIdx in 0..numSF - 1 {
                let xo = N + xoIdx;
                let r = xoIdx / params.numCopies;
                for b in 0..numBp1 - 1 {
                    let angleIdx = b * params.numRanks + r;
                    if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                        let angle = params.TwoBodyRotationAngles[angleIdx][j];
                        let condValue = xo + b * (1 <<< xoBits);
                        ApplyControlledOnInt(condValue, Ry(-2.0 * angle, _), xoReg + bReg, sysRegDown[j]);
                    }
                }
            }

            CNOT(sysRegDown[j], sysRegDown[j + 1]);
        }
        // Set bEqB flag: 1 when (isSF AND b == B)
        ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, bEqBQubit);
    }

    /// Applies the Givens basis change from a QROM-loaded angle word, runs `action` in that
    /// rotated basis, then undoes the chain and releases the angle word.
    ///
    /// Loads ALL (N-1) rotation angles at once using two Select calls:
    ///   - SF: Select over min(R*2^bBits, (B+1)*2^rankBits) entries, uncontrolled
    ///   - DQ: Select(N entries) addressed by xoReg[0..⌈log₂N⌉-1], fires when isSF=0
    ///
    /// `action` runs inside the table load rather than around this whole operation, so the
    /// angle word stays live across the forward chain, `action`, and the inverse chain, and
    /// the tables are paid for once per SELECT instead of twice. That is sound only because
    /// `action` leaves the QROM address (`isSF`, `xoReg`, `bReg`, `rBits`) untouched, so the
    /// unlookup addresses the same entry that was loaded.
    ///
    /// Cost: (L_SF - 2) + unlookup(L_SF) for SF, 2*(N-1) for DQ, and 2 × (N-1) × Adder(bRot)
    /// for the rotations. The SF unlookup is measurement-based and costs O(sqrt(L)), which is
    /// the paper's R + B phase fixup: at FeMoco-54 (N=54, R=10, B=27) it is 37 Toffolis
    /// against the paper's R + B = 37.
    ///
    /// L_SF is above the paper's R*B because `Select` pads whichever register addresses the
    /// low bits out to a power of two; `SFTableRankAddressedFirst` picks the cheaper of the
    /// two orderings, which is all that can be done without a non-power-of-two stride.
    ///
    /// Reference: arXiv:2502.15882v1, Appendix B.5 and B step 7; Babbush et al. (arXiv:1805.03662).
    operation WithGivensRotationsQROM(
        params : SelectParams,
        N : Int,
        numSF : Int,
        numBp1 : Int,
        numRotAngles : Int,
        xoBits : Int,
        isSF : Qubit,
        xoReg : Qubit[],
        bReg : Qubit[],
        rBits : Qubit[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        bEqBQubit : Qubit,
        action : (Unit => Unit is Adj + Ctl),
    ) : Unit is Adj + Ctl {
        let bRot = params.rotationBitPrecision;
        let bBits = Length(bReg);
        let R = params.numRanks;
        let nRotBits = numRotAngles * bRot;
        let nDQBits = BitSizeI((if N > 1 { N } else { 2 }) - 1);

        // DQ table: N entries × (N-1)*bRot bits, addressed by xoReg[0..nDQBits-1]
        let dqData = BuildDQBulkRotationData(params, N, numRotAngles, bRot);

        // SF table: addressed by (bReg ++ rBits) or (rBits ++ bReg), whichever is smaller.
        // Each entry is (N-1)*bRot rotation bits plus the bEqB flag (1 when b == B).
        let rankFirst = SFTableRankAddressedFirst(R, params.numBases, bBits, Length(rBits));
        let sfData = BuildSFBulkRotationData(params, R, numRotAngles, bRot, bBits, Length(rBits), rankFirst);
        let sfAddress = if rankFirst { rBits + bReg } else { bReg + rBits };

        // Allocate rotation target register: (N-1)*bRot rotation bits + 1 bEqB flag bit.
        use rotTarget = Qubit[nRotBits + 1];

        // The two table loads are conjugated around their consumers so `rotTarget` is
        // uncomputed and released in |0⟩, matching how every other QROM read in this
        // project is written (QROMStatePrep.qs, SelectSwap.qs).
        //
        // Order matters: the SF load is outermost so that its uncompute runs last, when
        // `rotTarget` again holds exactly `sfData[address]` and nothing else. Reordering it
        // inside the DQ load leaves the DQ word on the register at that point, and the
        // measurement-based unlookup would then apply the phase fixup for a word the target
        // does not hold.
        within {
            // SF load: uncontrolled, addressed by `sfAddress`. Dropping the control is
            // what makes the uncompute `Adjoint Select`, a measurement-based unlookup costing
            // O(sqrt(L)) rather than the full L of a controlled adjoint -- this is the phase
            // fixup of arXiv:2502.15882v1, Appendix B step 7.
            // SF entries include the bEqB flag at position nRotBits.
            Select(sfData, sfAddress, rotTarget);

            // Uncontrolled means DQ entries also read the table. They address row 0, because
            // the inner PREPARE gives every one-body generator b = 0 and r = 0 (see
            // `_inner_conditional_coefficients` and `_compute_free_rider_data` in
            // block_encoding/sossa.py). That row is classical, so removing it again costs
            // CNOTs and no Toffolis.
            within { X(isSF); } apply {
                for index in 0..Length(sfData[0]) - 1 {
                    if sfData[0][index] {
                        CNOT(isSF, rotTarget[index]);
                    }
                }
            }

            // DQ load: fires when isSF=0, addressed by first ⌈log₂N⌉ bits of xoReg.
            // DQ entries have only rotation bits (no bEqB), so target excludes last qubit.
            // This one stays controlled: SF values of x_o alias onto real DQ rows rather than
            // onto a single constant, so the trick above does not apply, and giving the table
            // its own isSF address bit doubles it -- which costs more than the cheaper
            // unlookup saves at every size measured.
            within { X(isSF); } apply {
                Controlled Select([isSF], (dqData, xoReg[0..nDQBits - 1], rotTarget[0..nRotBits - 1]));
            }
        } apply {
            within {
                // Copy bEqB flag from QROM output to persistent qubit.
                // Cost: 1 CNOT (vs ⌈log₂(B+1)⌉ Toffoli for ApplyControlledOnInt).
                CNOT(rotTarget[nRotBits], bEqBQubit);

                // Apply all Givens rotations from the loaded register.
                // Uses DFTHC-style conjugation: CNOT(j→j+1) + S†H converts Rz→Ry
                // and conditions on particle-number subspace, all with an uncontrolled
                // adder (n Toffoli) instead of a controlled adder (2n Toffoli).
                // Reference: Sanders et al. (arXiv:2007.07391, §IIA1, Figure 4a).
                //
                // The CNOT control must be sysRegDown[j], matching
                // ApplyMultiControlledRotations: it maps the one-excitation states
                // |1_j 0_{j+1}⟩ and |0_j 1_{j+1}⟩ onto |11⟩ and |01⟩, which differ only in
                // qubit j, so the Ry on qubit j rotates between them. Reversing the control
                // maps them onto |10⟩ and |11⟩, which differ only in qubit j+1, and the Ry
                // then mixes the one-excitation sector into the zero-excitation sector
                // instead of performing the Givens rotation.
                for j in 0..numRotAngles - 1 {
                    within {
                        CNOT(sysRegDown[j], sysRegDown[j + 1]);
                    } apply {
                        RyViaPhaseGradient(sysRegDown[j], rotTarget[j * bRot..(j + 1) * bRot - 1], phaseGradientReg);
                    }
                }
            } apply {
                action();
            }
        }
    }

    /// Build DQ bulk rotation data: N entries, each containing all (N-1) quantized angles.
    /// Addressed by xoReg[0..⌈log₂N⌉-1] (the orbital index for one-body terms).
    internal function BuildDQBulkRotationData(
        params : SelectParams,
        N : Int,
        numRotAngles : Int,
        bRot : Int,
    ) : Bool[][] {
        mutable table : Bool[][] = [];
        for xo in 0..N - 1 {
            mutable bits : Bool[] = [];
            for j in 0..numRotAngles - 1 {
                let angle = if j < Length(params.OneBodyRotationAngles[xo]) {
                    params.OneBodyRotationAngles[xo][j]
                } else {
                    0.0
                };
                set bits += IntAsBoolArray(QuantizeGivensAngle(angle, bRot), bRot);
            }
            set table += [bits];
        }
        return table;
    }

    /// Whether the SF rotation table is cheaper addressed `rBits ++ bReg` than `bReg ++ rBits`.
    ///
    /// `Select`'s address is the integer value of the register, so the *low* register is padded
    /// out to its full power of two while the high one is not: `b` low costs `R * 2^bBits`
    /// entries and `r` low costs `(B+1) * 2^rankBits`. Neither dominates -- which is smaller
    /// depends only on how close `R` and `B+1` sit to a power of two -- so take the smaller.
    /// Both orderings put (b=0, r=0) at address 0, which the one-body cancellation relies on.
    ///
    /// Reaching the paper's `R*B` needs an address with a non-power-of-two stride, which
    /// `Select` cannot express; it would take a nested unary iteration over b then r, whose
    /// uncompute would no longer be the measurement-based unlookup.
    internal function SFTableRankAddressedFirst(
        numRanks : Int,
        numBases : Int,
        bBits : Int,
        rankBits : Int,
    ) : Bool {
        (numBases + 1) * (1 <<< rankBits) < numRanks * (1 <<< bBits)
    }

    /// Build SF bulk rotation data: all (N-1) quantized angles per entry, plus a 1-bit bEqB
    /// flag indicating b == numBases.
    ///
    /// `rankFirst` selects the address layout, and must match the register order the caller
    /// hands to `Select`: `addr = r + b * 2^rankBits` when true, `addr = b + r * 2^bBits`
    /// otherwise. The table is sized to exactly cover the reachable addresses of that layout.
    internal function BuildSFBulkRotationData(
        params : SelectParams,
        R : Int,
        numRotAngles : Int,
        bRot : Int,
        bBits : Int,
        rankBits : Int,
        rankFirst : Bool,
    ) : Bool[][] {
        let bSlots = 1 <<< bBits;
        let rSlots = 1 <<< rankBits;
        let tableSize = if rankFirst { (params.numBases + 1) * rSlots } else { R * bSlots };

        mutable table : Bool[][] = [];
        for idx in 0..tableSize - 1 {
            let b = if rankFirst { idx / rSlots } else { idx % bSlots };
            let r = if rankFirst { idx % rSlots } else { idx / bSlots };

            mutable bits : Bool[] = [];
            for j in 0..numRotAngles - 1 {
                let angleIdx = b * params.numRanks + r;
                let angle = if r < R and angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                    params.TwoBodyRotationAngles[angleIdx][j]
                } else {
                    0.0
                };
                set bits += IntAsBoolArray(QuantizeGivensAngle(angle, bRot), bRot);
            }
            // Append bEqB flag: true when b == numBases (the identity term)
            set bits += [b == params.numBases];
            set table += [bits];
        }
        return table;
    }

    /// Quantize a Givens rotation angle for phase gradient application.
    ///
    /// RyViaPhaseGradient applies Ry(4π·x/2^b). To achieve Ry(-2θ):
    ///   4π·x/2^b = -2θ  →  x = -2^b · θ / (2π)  (mod 2^b)
    internal function QuantizeGivensAngle(angle : Double, bRot : Int) : Int {
        // Rejects NaN and ±∞ as well as absurd magnitudes: the comparison is false for
        // NaN, so this fails loudly instead of silently folding a non-finite angle into
        // an in-range bit pattern (NaN and +∞ both quantize to 2^bRot-1, -∞ to 0).
        // Givens angles come from Atan2/hypot and are in [-π, π].
        Fact(AbsD(angle) <= 4.0 * PI(), "QuantizeGivensAngle: angle must be finite and within [-4π, 4π]");
        let scale = IntAsDouble(1 <<< bRot);
        let raw = Round(-scale * angle / (2.0 * PI()));
        ((raw % (1 <<< bRot)) + (1 <<< bRot)) % (1 <<< bRot)
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
        system_reg_0 : Qubit
    ) : Unit is Adj + Ctl {
        // SF two-body (b < B): Z on system_reg_0 when sf_vs_dq=1 AND bEqB=0
        within { X(bEqB); } apply {
            Controlled Z([sf_vs_dq, bEqB], system_reg_0);
        }
        // DQ: X on system_reg_0 with spin-dependent Z
        within { X(sf_vs_dq); } apply {
            CNOT(sf_vs_dq, system_reg_0);
            Controlled Z([sf_vs_dq, spin], system_reg_0);
        }
        // DQ Q1 sign flip: Z(spin) when sf_vs_dq=0 AND d_vs_q=1
        within { X(sf_vs_dq); } apply {
            Controlled Z([sf_vs_dq, d_vs_q], spin);
        }
    }


    // ═══════════════════════════════════════════════════════════════════════════
    // Test wrappers
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generic wrapper: applies an operation to a freshly allocated register.
    operation TestApplyOuterPrep(op : (Qubit[]) => Unit is Adj + Ctl, n : Int) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(n);
        op(qs);
    }

    /// Wrapper: applies outer prep then inner prep on separate registers.
    operation TestApplyOuterInnerPrep(
        outerOp : (Qubit[]) => Unit is Adj + Ctl,
        innerOp : (Qubit[], Qubit[]) => Unit is Adj,
        nOuter : Int,
        nInner : Int,
    ) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(nOuter + nInner);
        let outerReg = qs[0..nOuter - 1];
        let innerReg = qs[nOuter..nOuter + nInner - 1];
        outerOp(outerReg);
        innerOp(outerReg, innerReg);
    }


    /// Test the full SELECT on an entry with known angles.
    ///
    /// `usePhaseGradient` selects between the two rotation backends, which must agree: the
    /// direct multi-controlled rotations and the QROM-plus-phase-gradient chain implement the
    /// same Givens basis change, the latter to `rotationBitPrecision` accuracy. Running both
    /// and comparing the resulting states is what makes the QROM path testable, since it has
    /// no independent reference to be checked against.
    ///
    /// `bValue` addresses the SF angle table, and the rank is derived from `xoValue` exactly
    /// as the inner PREPARE would emit it, so the two backends see a consistent (b, r). An
    /// `xoValue >= numOrbitals` with a nonzero `bValue` is what exercises the SF table: at
    /// b = r = 0 both address layouts of `BuildSFBulkRotationData` agree, so a mismatch
    /// between the table build and the address concatenation would go unseen.
    operation TestSelectDQ(
        selectData : SelectParams,
        xoValue : Int,
        bValue : Int,
        usePhaseGradient : Bool,
    ) : Unit {
        let N = selectData.numOrbitals;
        let numPositiveOneBody = selectData.numPositiveOneBody;
        let numSF = selectData.numRanks * selectData.numCopies;
        let Xo = N + numSF;
        let xoBits = BitSizeI((if Xo > 1 { Xo } else { 2 }) - 1);
        let numBp1 = selectData.numBases + 1;
        let bBits = BitSizeI((if numBp1 > 1 { numBp1 } else { 2 }) - 1);
        let nFR = selectData.numFreeRiderBits;

        let nOuter = xoBits;
        let nInner = bBits + nFR;
        let nSpin = 2;
        let nSystem = 2 * N;
        // The gradient register is allocated only on the QROM path. It is conjugated back to
        // |0...0>, so the direct path's state is recovered from the QROM dump by restricting
        // to the gradient=|0...0> subspace.
        let nGradient = if usePhaseGradient { selectData.rotationBitPrecision } else { 0 };
        let total = nOuter + nInner + nSpin + nSystem + nGradient;
        let qs = QIR.Runtime.AllocateQubitArray(total);

        let outerReg = qs[0..nOuter - 1];
        let innerReg = qs[nOuter..nOuter + nInner - 1];
        let spinReg = qs[nOuter + nInner..nOuter + nInner + nSpin - 1];
        let systemReg = qs[nOuter + nInner + nSpin..nOuter + nInner + nSpin + nSystem - 1];
        let gradientReg = qs[total - nGradient...];

        let xoReg = outerReg[0..xoBits - 1];
        for bit in 0..xoBits - 1 {
            if (xoValue >>> bit) &&& 1 == 1 {
                X(xoReg[bit]);
            }
        }
        H(spinReg[0]); // spinDQ

        for bit in 0..bBits - 1 {
            if (bValue >>> bit) &&& 1 == 1 {
                X(innerReg[bit]);
            }
        }

        let frStart = bBits;
        if xoValue >= N { X(innerReg[frStart]); }
        if xoValue >= numPositiveOneBody { X(innerReg[frStart + 1]); }

        // Rank as the inner PREPARE's free-rider data would carry it: 0 for one-body.
        let rValue = if xoValue >= N { (xoValue - N) / selectData.numCopies } else { 0 };
        for bit in 0..nFR - 3 {
            if (rValue >>> bit) &&& 1 == 1 {
                X(innerReg[frStart + 2 + bit]);
            }
        }

        X(systemReg[0]);

        if usePhaseGradient {
            // Conjugated so the gradient register returns to |0...0> and does not contribute
            // its own amplitudes to the comparison against the direct path.
            within {
                PreparePhaseGradientState(gradientReg);
            } apply {
                SelectImpl(selectData, true, outerReg, innerReg, spinReg, systemReg, gradientReg);
            }
        } else {
            SelectImpl(selectData, false, outerReg, innerReg, spinReg, systemReg, []);
        }
    }


}
