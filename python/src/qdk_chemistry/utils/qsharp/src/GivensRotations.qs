// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Fermionic Givens rotation kernels.

namespace QDKChemistry.Utils.SOSSAWalk {

    import Std.Arrays.MappedOverRange;
    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Diagnostics.Fact;
    import Std.Intrinsic.AND;
    import Std.Math.AbsD;
    import Std.Math.ArcTan2;
    import Std.Math.Cos;
    import Std.Math.PI;
    import Std.Math.Round;
    import Std.Math.Sin;
    import Std.Math.Sqrt;
    import QDKChemistry.Utils.PhaseGradient.RyViaPhaseGradient;

    /// Quantize a Givens rotation angle for phase gradient application.
    ///
    /// RyViaPhaseGradient applies Ry(4π·x/2^b). The gated Givens rotation is built as the
    /// controlled CRy(2θ) = Ry(θ)·CNOT·Ry(-θ)·CNOT from two uncontrolled Ry(θ), so each word
    /// encodes the half-angle θ:
    ///   4π·x/2^b = θ  →  x = 2^b · θ / (4π)  (mod 2^b)
    /// With `fullAngle` the word drives one controlled Ry(2θ) instead, so x = 2^b · 2θ / (4π), which is
    /// the half-angle word at b + 1 bits with its top bit dropped (that bit only flips both Ry signs).
    internal function QuantizeGivensAngle(angle : Double, bRot : Int, fullAngle : Bool) : Int {
        Fact(AbsD(angle) <= 4.0 * PI(), "QuantizeGivensAngle: angle must be finite and within [-4π, 4π]");
        let scale = IntAsDouble(if fullAngle { 2 <<< bRot } else { 1 <<< bRot });
        let raw = Round(scale * angle / (4.0 * PI()));
        ((raw % (1 <<< bRot)) + (1 <<< bRot)) % (1 <<< bRot)
    }

    /// Mode pairs `(p, c)` of the N-1 Givens rotations indexed by angle: `(j, j+1)` for the sweep,
    /// or the binary tree of :cite:`Low2026` Eq. D6 with contiguous-block labels, `(p, p + s)` for
    /// p a multiple of 2s, largest stride s first; each layer's Jordan-Wigner spans are disjoint,
    /// giving rotation depth ⌈log₂ N⌉.
    function GivensRotationPairs(N : Int, tree : Bool) : (Int, Int)[] {
        if not tree {
            return MappedOverRange(j -> (j, j + 1), 0..N - 2);
        }
        mutable stride = 1;
        while 2 * stride < N {
            set stride *= 2;
        }
        mutable pairs = [];
        while stride >= 1 {
            for p in 0..2 * stride..N - 1 - stride {
                set pairs += [(p, p + stride)];
            }
            set stride /= 2;
        }
        pairs
    }

    /// Tree angles that rotate the same unit vector onto mode 0 as the sweep angles `sweepAngles`.
    ///
    /// The vector is rebuilt from the sweep (u_j = cos θ_j Π_{k<j} sin θ_k), then zeroed edge by
    /// edge, deepest first, with the sweep's rule θ = atan2(u_c, u_p) merging child c into parent p.
    function SweepToTreeAngles(N : Int, sweepAngles : Double[]) : Double[] {
        mutable u = [0.0, size = N];
        mutable r = 1.0;
        for j in 0..N - 2 {
            let theta = if j < Length(sweepAngles) { sweepAngles[j] } else { 0.0 };
            set u w/= j <- r * Cos(theta);
            set r *= Sin(theta);
        }
        set u w/= N - 1 <- r;
        let pairs = GivensRotationPairs(N, true);
        mutable treeAngles = [0.0, size = N - 1];
        for e in N - 2..-1..0 {
            let (p, c) = pairs[e];
            set treeAngles w/= e <- ArcTan2(u[c], u[p]);
            set u w/= p <- Sqrt(u[p] * u[p] + u[c] * u[c]);
            set u w/= c <- 0.0;
        }
        treeAngles
    }

    /// Applies the Givens rotations for angles `first..last` on their mode `pairs`, reading the
    /// `wordBits`-bit angle words out of `rotTarget` (which holds exactly that window, low index
    /// first); a word shorter than the gradient is zero-extended by the adder. With `controlledAdder`
    /// each step is one controlled adder on fresh clean ancillas, as resident words cannot be erased early.
    internal operation ApplyGivensRotationWords(
        rotTarget : Qubit[],
        first : Int,
        last : Int,
        wordBits : Int,
        pairs : (Int, Int)[],
        sysRegDown : Qubit[],
        phaseGradientReg : Qubit[],
        controlledAdder : Bool,
    ) : Unit is Adj {
        for j in last..-1..first {
            let word = rotTarget[(j - first) * wordBits..(j - first + 1) * wordBits - 1];
            let (p, c) = pairs[j];
            if controlledAdder {
                use clean = Qubit[Length(phaseGradientReg) - 1];
                ApplyControlledAdderGivensStep(p, c, sysRegDown, word, phaseGradientReg, clean);
            } else {
                within {
                    // Tree pairs need not be adjacent; the CZ string keeps each gate a fermionic Givens rotation.
                    for s in p + 1..c - 1 {
                        CZ(sysRegDown[s], sysRegDown[c]);
                    }
                    CNOT(sysRegDown[p], sysRegDown[c]);
                } apply {
                    // :cite:`Low2026` FIG. 40: controlled RZ(2θ) from two parallel uncontrolled RZ(θ) gates.
                    within {
                        CNOT(sysRegDown[c], sysRegDown[p]);
                    } apply {
                        Adjoint RyViaPhaseGradient(sysRegDown[p], word, phaseGradientReg);
                    }
                    RyViaPhaseGradient(sysRegDown[p], word, phaseGradientReg);
                }
            }
        }
    }

    /// One fermionic Givens step on modes (p, c): Ry(4π·x/2^b) on p controlled on c, made from a single
    /// adder of `word` into the phase gradient controlled on c (:cite:`Low2026` FIG. 39, Lemma 6), costing
    /// 2b - 2 Toffolis with b - 1 `clean` ancillas (Gidney) or 3b - 3 without (Muñoz-Coreas-Thapliyal), one below
    /// Lemma 7 because the gradient's |−⟩ top qubit makes its controlled flip a CZ.
    internal operation ApplyControlledAdderGivensStep(
        p : Int,
        c : Int,
        sysRegDown : Qubit[],
        word : Qubit[],
        phaseGradientReg : Qubit[],
        clean : Qubit[],
    ) : Unit is Adj {
        let n = Length(phaseGradientReg);
        Fact(Length(word) <= n, "ApplyControlledAdderGivensStep: the word must not be wider than the phase gradient");
        // A compressed word is zero-extended at its top to the gradient's width.
        use pad = Qubit[n - Length(word)];
        let ctrl = sysRegDown[c];
        let a = word + pad;
        let g = phaseGradientReg;
        within {
            GivensPhaseGradientFrame(p, c, sysRegDown, g);
        } apply {
            if Length(clean) >= n - 1 {
                // Carries c_{i+1} = maj(a_i, g_i, c_i) are ANDed into `clean`, then g_i ^= ctrl·(a_i ⊕ c_i) unwinds them.
                for i in 0..n - 2 {
                    if i > 0 {
                        CNOT(clean[i - 1], a[i]);
                        CNOT(clean[i - 1], g[i]);
                    }
                    AND(a[i], g[i], clean[i]);
                    if i > 0 {
                        CNOT(clean[i - 1], clean[i]);
                    }
                }
                // The gradient's top qubit is |−⟩, so flipping it on ctrl·(a ⊕ c) is the phase CZ(ctrl, a ⊕ c).
                CZ(ctrl, a[n - 1]);
                if n > 1 {
                    CZ(ctrl, clean[n - 2]);
                }
                for i in n - 2..-1..0 {
                    if i > 0 {
                        CNOT(clean[i - 1], clean[i]);
                    }
                    Adjoint AND(a[i], g[i], clean[i]);
                    CCNOT(ctrl, a[i], g[i]);
                    if i > 0 {
                        CNOT(clean[i - 1], a[i]);
                        CNOT(clean[i - 1], g[i]);
                    }
                }
            } else {
                for i in 1..n - 1 {
                    CNOT(a[i], g[i]);
                }
                for i in n - 2..-1..1 {
                    CNOT(a[i], a[i + 1]);
                }
                for i in 0..n - 2 {
                    CCNOT(g[i], a[i], a[i + 1]);
                }
                for i in n - 1..-1..1 {
                    if i == n - 1 {
                        CZ(ctrl, a[i]);
                    } else {
                        CCNOT(ctrl, a[i], g[i]);
                    }
                    CCNOT(g[i - 1], a[i - 1], a[i]);
                }
                for i in 1..n - 2 {
                    CNOT(a[i], a[i + 1]);
                }
                for i in 1..n - 1 {
                    CNOT(a[i], g[i]);
                }
                if n == 1 {
                    CZ(ctrl, a[0]);
                } else {
                    CCNOT(ctrl, a[0], g[0]);
                }
            }
        }
    }

    /// Basis change around a phase-gradient Givens step on modes (p, c), after which adding x into the
    /// gradient while c is set applies Ry(4π·x/2^b) on p (the RyViaPhaseGradient frame with a CZ string).
    internal operation GivensPhaseGradientFrame(p : Int, c : Int, sysRegDown : Qubit[], phaseGradientReg : Qubit[]) : Unit is Adj {
        for s in p + 1..c - 1 {
            CZ(sysRegDown[s], sysRegDown[c]);
        }
        CNOT(sysRegDown[p], sysRegDown[c]);
        Adjoint S(sysRegDown[p]);
        H(sysRegDown[p]);
        X(sysRegDown[p]);
        for g in phaseGradientReg {
            CNOT(sysRegDown[p], g);
        }
    }
}
