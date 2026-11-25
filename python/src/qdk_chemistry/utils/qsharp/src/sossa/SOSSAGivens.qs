// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// SOSSA rotation-table selection and paired one-hot rotations.

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.Any;
    import Std.Arrays.IndexRange;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Core.Length;
    import Std.Intrinsic.AND;
    import Std.Math.MaxI;

    /// Givens rotation chain with CNOT sandwich (arXiv:2502.15882v1, Appendix B.5).
    ///
    /// Each step G_{j,j+1}(θ) = CX(j→j+1) · Ctrl_{j+1}[Ry(2θ, j)] · CX(j→j+1) acts as a
    /// 2×2 rotation in the single-excitation subspace {|01⟩,|10⟩} of (target[j], target[j+1]).
    /// The control on target[j+1] keeps the rotation out of the |00⟩/|11⟩ sector, so the chain
    /// preserves particle number; it maps orbital content to qubit 0 for MajoranaOp.
    ///
    /// NOTE: This is the direct rotation (simulation) version. The production
    /// implementation should use QROM to load rotation angles into an ancilla
    /// register, then apply phase-gradient rotation (Rz via addition to a
    /// phase-gradient register). See MakeSelectPhaseGradient.
    ///
    /// DQ rotations: controlled on xoReg ∈ [0, N) and the neighbor target[j+1], unconditional on b.
    /// SF rotations: controlled on (xoReg, bReg) and the neighbor target[j+1] jointly.
    /// When recomputing from x_o, xoReg holds a (DQ) or r in its low bits (SF) beside `isSF`,
    /// and `bEqBQubit` is empty.
    /// With `params.givensTree` the pair (j, j+1) becomes the tree edge `GivensRotationPairs` gives.
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
        rBits : Qubit[],
        sysRegDown : Qubit[],
        bEqBQubit : Qubit[]
    ) : Unit is Adj + Ctl {
        let recompute = params.recomputeOuterRegisters;
        let nR = Length(rBits);
        let bBits = Length(bReg);
        let pairs = GivensRotationPairs(N, params.givensTree);
        for j in numRotAngles - 1..-1..0 {
            let (p, c) = pairs[j];
            // Tree pairs need not be adjacent; the CZ string keeps each gate a fermionic Givens rotation.
            for s in p + 1..c - 1 {
                CZ(sysRegDown[s], sysRegDown[c]);
            }
            CNOT(sysRegDown[p], sysRegDown[c]);

            // DQ rotations: x_o in [0, N)
            for a in 0..N - 1 {
                let angle = params.OneBodyRotationAngles[a][j];
                if recompute {
                    ApplyControlledOnInt(
                        a + (1 <<< (xoBits + 1)),
                        Ry(2.0 * angle, _),
                        xoReg + [isSF, sysRegDown[c]],
                        sysRegDown[p]
                    );
                } else {
                    ApplyControlledOnInt(
                        a + (1 <<< xoBits),
                        Ry(2.0 * angle, _),
                        xoReg + [sysRegDown[c]],
                        sysRegDown[p]
                    );
                }
            }

            if recompute {
                // SF rotations depend on (r, b) only, so the copies c share one rotation.
                for r in 0..params.numRanks - 1 {
                    for b in 0..numBp1 - 1 {
                        let angleIdx = b * params.numRanks + r;
                        if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                            let condValue = r + (b <<< nR) + (1 <<< (nR + bBits)) + (1 <<< (nR + bBits + 1));
                            ApplyControlledOnInt(
                                condValue,
                                Ry(2.0 * params.TwoBodyRotationAngles[angleIdx][j], _),
                                rBits + bReg + [isSF, sysRegDown[c]],
                                sysRegDown[p]
                            );
                        }
                    }
                }
            } else {
                // SF rotations: x_o in [N, N+numSF), conditioned on b
                for xoIdx in 0..numSF - 1 {
                    let xo = N + xoIdx;
                    let r = xoIdx / params.numCopies;
                    for b in 0..numBp1 - 1 {
                        let angleIdx = b * params.numRanks + r;
                        if angleIdx < Length(params.TwoBodyRotationAngles) and j < Length(params.TwoBodyRotationAngles[angleIdx]) {
                            let angle = params.TwoBodyRotationAngles[angleIdx][j];
                            let condValue = xo + b * (1 <<< xoBits) + (1 <<< (xoBits + bBits));
                            ApplyControlledOnInt(
                                condValue,
                                Ry(2.0 * angle, _),
                                xoReg + bReg + [sysRegDown[c]],
                                sysRegDown[p]
                            );
                        }
                    }
                }
            }

            CNOT(sysRegDown[p], sysRegDown[c]);
            for s in p + 1..c - 1 {
                CZ(sysRegDown[s], sysRegDown[c]);
            }
        }
        // Set bEqB flag: 1 when (isSF AND b == B)
        for flag in bEqBQubit {
            ApplyControlledOnInt(params.numBases, q => Controlled X([isSF], q), bReg, flag);
        }
    }

    /// One controlled-adder Givens step whose full-angle word comes from a paired one-hot register: bit k
    /// enters the gradient as (c ⊕ u ⊕ x1)(u ⊕ x0) ⊕ x01 with u = c·s (:cite:`Shirobokov2026` Eq. 23), where
    /// x0, x1 and x01 are the parities of the `oneHotPairs` qubits whose first, second or both words set it,
    /// then one uncontrolled adder costs b - 2 Toffolis, for 2b - 1 per step or one less per always-zero bit.
    internal operation ApplyPairedOneHotGivensStep(
        p : Int,
        c : Int,
        sysRegDown : Qubit[],
        oneHotPairs : Qubit[],
        second : Qubit,
        firstWords : Bool[][],
        secondWords : Bool[][],
        phaseGradientReg : Qubit[],
    ) : Unit is Adj {
        let n = Length(phaseGradientReg);
        let ctrl = sysRegDown[c];
        let g = phaseGradientReg;
        within {
            GivensPhaseGradientFrame(p, c, sysRegDown, g);
        } apply {
            use u = Qubit();
            use t = Qubit[n];
            within {
                AND(ctrl, second, u);
                for k in 0..n - 1 {
                    if Any(word -> k < Length(word) and word[k], firstWords + secondWords) {
                        within {
                            CNOT(u, ctrl);
                            for q in IndexRange(oneHotPairs) {
                                if secondWords[q][k] {
                                    CNOT(oneHotPairs[q], ctrl);
                                }
                                if firstWords[q][k] {
                                    CNOT(oneHotPairs[q], u);
                                }
                            }
                        } apply {
                            AND(u, ctrl, t[k]);
                        }
                        for q in IndexRange(oneHotPairs) {
                            if firstWords[q][k] and secondWords[q][k] {
                                CNOT(oneHotPairs[q], t[k]);
                            }
                        }
                    }
                }
            } apply {
                // Gidney's adder of t into g; carry[i] holds c_{i+1} and g[i], t[i] hold g ⊕ c_i, t ⊕ c_i while it is live.
                use carry = Qubit[MaxI(0, n - 2)];
                for i in 0..n - 2 {
                    if i > 0 {
                        CNOT(carry[i - 1], t[i]);
                        CNOT(carry[i - 1], g[i]);
                    }
                    if i < n - 2 {
                        AND(t[i], g[i], carry[i]);
                        if i > 0 {
                            CNOT(carry[i - 1], carry[i]);
                        }
                    }
                }
                // The gradient's top qubit is |−⟩, so flipping it on t ⊕ c_{n-1} is a phase, and
                // c_{n-1} = c_{n-2} ⊕ (t ⊕ c_{n-2})(g ⊕ c_{n-2}) needs no Toffoli.
                Z(t[n - 1]);
                if n > 1 {
                    CZ(t[n - 2], g[n - 2]);
                }
                if n > 2 {
                    Z(carry[n - 3]);
                }
                for i in n - 2..-1..0 {
                    if i < n - 2 {
                        if i > 0 {
                            CNOT(carry[i - 1], carry[i]);
                        }
                        Adjoint AND(t[i], g[i], carry[i]);
                    }
                    CNOT(t[i], g[i]);
                    if i > 0 {
                        CNOT(carry[i - 1], t[i]);
                        CNOT(carry[i - 1], g[i]);
                    }
                }
            }
        }
    }
}
