/// # Summary
/// FOQCS-LCU (Fast One-Qubit-Controlled Select - Linear Combination of Unitaries)
/// block encoding of the 1D Heisenberg Hamiltonian, vendored from
/// [QuantumComputingLab/foqcs-lcu](https://github.com/QuantumComputingLab/foqcs-lcu).
///
/// The Heisenberg Hamiltonian is:
/// ```text
/// H = Σ_{ℓ=0}^{n-1} (g^x X_ℓ + g^z Z_ℓ + g^y Y_ℓ)
///   + Σ_{ℓ=0}^{n-2} (J^x X_ℓ X_{ℓ+1} + J^z Z_ℓ Z_{ℓ+1} + J^y Y_ℓ Y_{ℓ+1})
/// ```
///
/// # References
/// - F. Della Chiara, M. Nibbi, Y. Shen, R. Van Beeumen, Efficient LCU block
///  encodings through Dicke states preparation, Daan Camps, Roel Van Beeumen,
///  2025, arXiv:2507.20887.

import Std.Canon.ApplyToEachCA;
import Std.Convert.IntAsDouble;
import Std.Core.Length;
import Std.Math.AbsD;
import Std.Math.ArcCos;
import Std.Math.Ceiling;
import Std.Math.Lg;
import Std.Math.PI;
import Std.Math.Sqrt;
import Reflect.ReflectAboutZero;

// ===================================================================
// 1. Dicke state building blocks
// ===================================================================

/// Gamma gate Γ – the elementary rotation used inside Dicke state preparation.
/// Maps |10⟩ → cos(θ/2)|10⟩ + sin(θ/2)|01⟩.
operation Gamma(theta : Double, q0 : Qubit, q1 : Qubit) : Unit is Adj + Ctl {
    Controlled Ry([q1], (theta, q0));
    CNOT(q0, q1);
}

/// Balanced Dicke state with 1 excitation: (1/√L) Σᵢ |one-hot_i⟩ on L qubits.
/// `fromZero`: if true, starts by flipping the last qubit (input is |0…0⟩).
/// `offsetIter`: shifts the angle denominator (used when chaining iterations).
///
/// ## Controlled specialization
/// Only the first X gate are controlled.
operation BalancedDicke1Excitation(
    qubits : Qubit[],
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    body ... {
        let L = Length(qubits);
        if fromZero {
            X(qubits[L - 1]);
        }
        for idx in 0..L - 2 {
            let i = L - 1 - idx; // i goes L-1, L-2, …, 1
            let theta = 2.0 * ArcCos(Sqrt(1.0 / IntAsDouble(i + 1 + offsetIter)));
            Gamma(theta, qubits[i - 1], qubits[i]);
        }
    }
    adjoint auto;
    controlled (ctls, ...) {
        let L = Length(qubits);
        if fromZero {
            Controlled X(ctls, qubits[L - 1]);
        }
        for idx in 0..L - 2 {
            let i = L - 1 - idx;
            let theta = 2.0 * ArcCos(Sqrt(1.0 / IntAsDouble(i + 1 + offsetIter)));
            Gamma(theta, qubits[i - 1], qubits[i]);
        }
    }
    controlled adjoint auto;
}

/// Element-wise CNOT from reg1 to reg2 (same length).
/// All CNOTs can be done in parallel.
operation ElementWiseCnot(reg1 : Qubit[], reg2 : Qubit[]) : Unit is Adj + Ctl {
    for i in 0..Length(reg1) - 1 {
        CNOT(reg1[i], reg2[i]);
    }
}

/// Double Dicke state with 1 register for X and 1 for Z, same excitation pattern
/// on both: (1/√L) Σᵢ |one-hot_i⟩_X |one-hot_i⟩_Z
operation BalancedDoubleDicke1Excitation(
    xReg : Qubit[],
    zReg : Qubit[],
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    BalancedDicke1Excitation(xReg, fromZero, offsetIter);
    ElementWiseCnot(xReg, zReg);
}

/// CNOT ladder: converts a Dicke₁ excitation into a Dicke₂ k-NN pattern.
///
/// For k > 0: CNOT(i, i+k) for i = L-k-1 down to 0  (top-down sweep).
/// For k < 0: CNOT(i, i+k) for i = |k| up to L-1    (bottom-up sweep).
operation CnotLadder(qubits : Qubit[], k : Int) : Unit is Adj + Ctl {
    let L = Length(qubits);
    if k > 0 {
        for idx in 0..L - k - 1 {
            let i = L - k - 1 - idx;
            CNOT(qubits[i], qubits[i + k]);
        }
    } elif k < 0 {
        let absK = -k;
        for i in absK..L - 1 {
            CNOT(qubits[i], qubits[i + k]);
        }
    }
}

/// Balanced Dicke₂ state with 2 k-nearest-neighbour excitation:
///   (1/√(L-k)) Σᵢ |one-hot_i ⊕ one-hot_{i+k}⟩
operation BalancedDicke2KNNExcitation(
    qubits : Qubit[],
    k : Int,
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    let L = Length(qubits);
    BalancedDicke1Excitation(qubits[0..L - k - 1], fromZero, offsetIter);
    CnotLadder(qubits, k);
}

/// Double version of the balanced Dicke₂ k-NN state, with the same
/// excitation pattern on both registers.
operation BalancedDoubleDicke2KNNExcitation(
    xReg : Qubit[],
    zReg : Qubit[],
    k : Int,
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    BalancedDicke2KNNExcitation(xReg, k, fromZero, offsetIter);
    ElementWiseCnot(xReg, zReg);
}

// ===================================================================
// 1b. Non-contiguous Dicke state building blocks
// ===================================================================
//
// When Pauli terms of the same type (e.g. XX with offset 1) appear at
// non-contiguous starting positions (e.g. {0, 2, 5}), we cannot use
// a single BalancedDicke2KNNExcitation on the full range because the
// CNOT ladder would create spurious pairs at intermediate positions.
//
// Instead we:
//   1. Apply BalancedDicke1Excitation on the *extracted* start qubits
//      (passed as a temporary array), creating a one-hot superposition.
//   2. Apply individual CNOTs from each start to its k-neighbor.
//      Since Dicke₁ has exactly one excitation, only one CNOT fires
//      and there is no cascading — pairs must be non-overlapping.
// ===================================================================

/// Balanced Dicke₁ on non-contiguous qubit positions.
/// Creates (1/√m) Σᵢ |one-hot_{positions[i]}⟩ on the register.
///
/// ## Controlled specialization
/// Only the Dicke₁ starter is controlled (via BalancedDicke1Excitation).
operation NonContiguousBalancedDicke1Excitation(
    register : Qubit[],
    positions : Int[],
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    body ... {
        mutable selected : Qubit[] = [];
        for pos in positions {
            set selected += [register[pos]];
        }
        BalancedDicke1Excitation(selected, fromZero, offsetIter);
    }
    adjoint ... {
        mutable selected : Qubit[] = [];
        for pos in positions {
            set selected += [register[pos]];
        }
        Adjoint BalancedDicke1Excitation(selected, fromZero, offsetIter);
    }
    controlled (ctls, ...) {
        mutable selected : Qubit[] = [];
        for pos in positions {
            set selected += [register[pos]];
        }
        Controlled BalancedDicke1Excitation(ctls, (selected, fromZero, offsetIter));
    }
    controlled adjoint (ctls, ...) {
        mutable selected : Qubit[] = [];
        for pos in positions {
            set selected += [register[pos]];
        }
        Controlled Adjoint BalancedDicke1Excitation(ctls, (selected, fromZero, offsetIter));
    }
}

/// Double Dicke₁ on non-contiguous positions (for Y-type single-body terms).
/// Same excitation pattern on both xReg and zReg.
operation NonContiguousBalancedDoubleDicke1Excitation(
    xReg : Qubit[],
    zReg : Qubit[],
    positions : Int[],
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    NonContiguousBalancedDicke1Excitation(xReg, positions, fromZero, offsetIter);
    for pos in positions {
        CNOT(xReg[pos], zReg[pos]);
    }
}

/// Non-contiguous Balanced Dicke₂ k-NN excitation.
/// Creates (1/√m) Σᵢ |one-hot_{positions[i]} ⊕ one-hot_{positions[i]+k}⟩
///
/// REQUIREMENT: pairs must be non-overlapping, i.e., for sorted positions,
/// positions[i] + k < positions[i+1] for all consecutive entries.
///
/// ## Controlled specialization
/// Only the Dicke₁ starter is controlled; the per-pair CNOTs run
/// unconditionally (they are no-ops when no excitation is present).
operation NonContiguousBalancedDicke2KNNExcitation(
    register : Qubit[],
    positions : Int[],
    k : Int,
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    body ... {
        mutable startQubits : Qubit[] = [];
        for pos in positions {
            set startQubits += [register[pos]];
        }
        BalancedDicke1Excitation(startQubits, fromZero, offsetIter);
        for pos in positions {
            CNOT(register[pos], register[pos + k]);
        }
    }
    adjoint ... {
        for pos in positions {
            CNOT(register[pos], register[pos + k]);
        }
        mutable startQubits : Qubit[] = [];
        for pos in positions {
            set startQubits += [register[pos]];
        }
        Adjoint BalancedDicke1Excitation(startQubits, fromZero, offsetIter);
    }
    controlled (ctls, ...) {
        mutable startQubits : Qubit[] = [];
        for pos in positions {
            set startQubits += [register[pos]];
        }
        Controlled BalancedDicke1Excitation(ctls, (startQubits, fromZero, offsetIter));
        for pos in positions {
            CNOT(register[pos], register[pos + k]);
        }
    }
    controlled adjoint (ctls, ...) {
        for pos in positions {
            CNOT(register[pos], register[pos + k]);
        }
        mutable startQubits : Qubit[] = [];
        for pos in positions {
            set startQubits += [register[pos]];
        }
        Controlled Adjoint BalancedDicke1Excitation(ctls, (startQubits, fromZero, offsetIter));
    }
}

/// Double non-contiguous Dicke₂ k-NN (for Y-type 2-body terms).
/// Same excitation pattern on both xReg and zReg.
operation NonContiguousBalancedDoubleDicke2KNNExcitation(
    xReg : Qubit[],
    zReg : Qubit[],
    positions : Int[],
    k : Int,
    fromZero : Bool,
    offsetIter : Int
) : Unit is Adj + Ctl {
    NonContiguousBalancedDicke2KNNExcitation(xReg, positions, k, fromZero, offsetIter);
    for pos in positions {
        CNOT(xReg[pos], zReg[pos]);
        CNOT(xReg[pos + k], zReg[pos + k]);
    }
}

// ===================================================================
// 2. Unbalanced Dicke₁ state (for sub-PREP with non-uniform coefficients)
// ===================================================================

/// Compute the sequence of Gamma rotation angles for an unbalanced Dicke₁
/// state from a vector of non-negative real amplitudes (must be normalized).
///
/// The rotation applied at step `ℓ` is:
/// ```text
/// θ̂_ℓ = 2 arccos(|α_{n-ℓ-1}| / √(1 - Σ_{j=0}^{n-ℓ-2} |α_j|²))
/// ```
///
/// Returns (angles[], cutoff).
function ComputeAngles(absCoeffs : Double[]) : (Double[], Int) {
    let L = Length(absCoeffs);
    mutable angles : Double[] = [];
    mutable sumSq = 0.0;
    mutable cutoff = L - 1;
    mutable found = false;
    for idx in 0..L - 1 {
        let i = L - 1 - idx;
        if not found {
            if sumSq >= 1.0 - 1e-15 {
                set cutoff = idx;
                set found = true;
            } else {
                let denom = Sqrt(1.0 - sumSq);
                mutable cosArg = absCoeffs[i] / denom;
                if cosArg > 1.0 {
                    set cosArg = 1.0;
                }
                if cosArg < -1.0 {
                    set cosArg = -1.0;
                }
                set angles += [2.0 * ArcCos(cosArg)];
                set sumSq += absCoeffs[i] * absCoeffs[i];
            }
        }
    }
    (angles, cutoff)
}

/// Unbalanced Dicke₁ state preparation:
///   Σᵢ |cᵢ| e^{i φᵢ} |one-hot_i⟩
/// `absCoeffs` must be normalized (Σ |cᵢ|² = 1).
/// `phases` are per-qubit R1 angles applied after amplitude preparation.
///
/// ## Controlled specialization
/// Only the first X gate are controlled.
operation UnbalancedDicke1Excitation(
    absCoeffs : Double[],
    phases : Double[],
    qubits : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let L = Length(qubits);
        let (angles, cutoff) = ComputeAngles(absCoeffs);

        X(qubits[L - 1]);
        for idx in 0..cutoff - 1 {
            Gamma(angles[idx], qubits[L - idx - 2], qubits[L - idx - 1]);
        }
        // Per-qubit phase correction (R1(0) = identity, so safe to apply to all).
        for i in 0..L - 1 {
            if AbsD(phases[i]) > 1e-15 {
                R1(phases[i], qubits[i]);
            }
        }
    }
    adjoint auto;
    controlled (ctls, ...) {
        let L = Length(qubits);
        let (angles, cutoff) = ComputeAngles(absCoeffs);

        Controlled X(ctls, qubits[L - 1]);
        for idx in 0..cutoff - 1 {
            Gamma(angles[idx], qubits[L - idx - 2], qubits[L - idx - 1]);
        }
        for i in 0..L - 1 {
            if AbsD(phases[i]) > 1e-15 {
                R1(phases[i], qubits[i]);
            }
        }
    }
    controlled adjoint auto;
}

// ===================================================================
// 3. FOQCS-LCU SELECT oracle
// ===================================================================

/// SELECT for FOQCS-LCU on the Heisenberg model.
///
/// Given an ancilla register laid out as [subPrepReg | xReg | zReg]
/// and a system register:
///   • CX from xReg[i] to system[i]  →  applies X when xReg[i] = 1
///   • CZ from zReg[i] to system[i]  →  applies Z when zReg[i] = 1
///
/// When both xReg[i] = zReg[i] = 1 the combined effect is XZ = −iY.
/// The −i phase is compensated inside the PREP coefficients.
///
/// Partially applied as `SelectFoqcsLcu(numSubPrep, _, _)` this
/// matches the `select : ((Qubit[], Qubit[]) => Unit is Adj + Ctl)`
/// signature expected by `ApplyFOQCSPower`.
operation SelectFoqcsLcu(
    numSubPrep : Int,
    ancilla : Qubit[],
    systemReg : Qubit[]
) : Unit is Adj + Ctl {
    let L = Length(systemReg);
    let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
    let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];
    for i in 0..L - 1 {
        CNOT(xReg[i], systemReg[i]);
    }
    for i in 0..L - 1 {
        Controlled Z([zReg[i]], systemReg[i]);
    }
}

// ===================================================================
// 4. Coefficient computation for the Heisenberg model
// ===================================================================

/// Compute the six sub-PREP amplitudes (magnitude + phase) and the
/// normalization factor λ for the Heisenberg FOQCS-LCU block encoding.
///
/// Coefficient ordering: [gx, gy, gz, Jx, Jy, Jz].
///
/// Phase corrections:
///   • gy picks up a (1−i)/√2 = e^{−iπ/4} factor (Y = −i XZ)
///   • Jy picks up a   i     = e^{+iπ/2} factor
///
/// Returns (absCoeffs[6], phases[6], lambda).
function ComputeHeisenbergCoeffs(
    L : Int,
    J : Double[],
    g : Double[]
) : (Double[], Double[], Double) {
    let Ld = IntAsDouble(L);
    let Lm1 = IntAsDouble(L - 1);

    mutable mags = [0.0, size = 6];
    mags w/= 0 <- Sqrt(g[0] * Ld);       // gx
    mags w/= 1 <- Sqrt(g[1] * Ld);       // gy
    mags w/= 2 <- Sqrt(g[2] * Ld);       // gz
    mags w/= 3 <- Sqrt(J[0] * Lm1);      // Jx
    mags w/= 4 <- Sqrt(J[1] * Lm1);      // Jy
    mags w/= 5 <- Sqrt(J[2] * Lm1);      // Jz

    mutable normSq = 0.0;
    for m in mags {
        set normSq += m * m;
    }
    let norm = Sqrt(normSq);

    mutable absCoeffs = [0.0, size = 6];
    for i in 0..5 {
        absCoeffs w/= i <- mags[i] / norm;
    }

    // Phases (from −iY = XZ correction).
    let phases = [0.0, -PI() / 4.0, 0.0, 0.0, PI() / 2.0, 0.0];

    // λ = norm² is the 1-norm of the LCU  (Σ |αᵢ|).
    let lambda = normSq;

    (absCoeffs, phases, lambda)
}

/// Negate all phase angles (used for conjugated PREP in the block encoding).
function NegatePhases(phases : Double[]) : Double[] {
    mutable result = [0.0, size = Length(phases)];
    for i in 0..Length(phases) - 1 {
        result w/= i <- -phases[i];
    }
    result
}

// ===================================================================
// 5. PREP oracles for the Heisenberg model
// ===================================================================

/// Full PREP for the Heisenberg FOQCS-LCU block encoding.
///
/// Register layout (left to right):
///   subPrepReg[0..5] | xReg[0..L-1] | zReg[0..L-1]
///
/// Each sub-PREP qubit controls a different Dicke state preparation:
///   0  →  gx : Dicke₁ on xReg
///   1  →  gy : Dicke₁-double on (xReg, zReg)
///   2  →  gz : Dicke₁ on zReg
///   3  →  Jx : Dicke₂-1NN on xReg
///   4  →  Jy : Dicke₂-1NN-double on (xReg, zReg)
///   5  →  Jz : Dicke₂-1NN on zReg
///
/// ## Controlled specialization
/// Only the `UnbalancedDicke1Excitation` sub-PREP step is controlled.
/// The subsequent controlled-Dicke preparations on xReg/zReg are
/// already gated by the sub-PREP one-hot pattern and run unconditionally.
operation PrepFoqcsLcuHeisenberg(
    absCoeffs : Double[],
    phases : Double[],
    ancilla : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0..numSubPrep - 1];
        let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];

        // 1. Sub-PREP: create one-hot superposition on 6 ancilla qubits.
        UnbalancedDicke1Excitation(absCoeffs, phases, subPrepReg);

        // 2. Controlled Dicke state preparations (single-qubit control each).
        //    gx: balanced Dicke₁ on xReg
        Controlled BalancedDicke1Excitation([subPrepReg[0]], (xReg, true, 0));
        //    gz: balanced Dicke₁ on zReg
        Controlled BalancedDicke1Excitation([subPrepReg[2]], (zReg, true, 0));
        //    gy: balanced Dicke₁-double on (xReg, zReg)
        Controlled BalancedDoubleDicke1Excitation([subPrepReg[1]], (xReg, zReg, true, 0));
        //    Jx: balanced Dicke₂-1NN on xReg
        Controlled BalancedDicke2KNNExcitation([subPrepReg[3]], (xReg, 1, true, 0));
        //    Jz: balanced Dicke₂-1NN on zReg
        Controlled BalancedDicke2KNNExcitation([subPrepReg[5]], (zReg, 1, true, 0));
        //    Jy: balanced Dicke₂-1NN-double on (xReg, zReg)
        Controlled BalancedDoubleDicke2KNNExcitation([subPrepReg[4]], (xReg, zReg, 1, true, 0));
    }
    adjoint auto;
    controlled (ctls, ...) {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0..numSubPrep - 1];
        let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];

        // Only the sub-PREP is controlled; the Dicke preparations are
        // self-gated by the one-hot ancilla pattern and left unconditional.
        Controlled UnbalancedDicke1Excitation(ctls, (absCoeffs, phases, subPrepReg));

        Controlled BalancedDicke1Excitation([subPrepReg[0]], (xReg, true, 0));
        Controlled BalancedDicke1Excitation([subPrepReg[2]], (zReg, true, 0));
        Controlled BalancedDoubleDicke1Excitation([subPrepReg[1]], (xReg, zReg, true, 0));
        Controlled BalancedDicke2KNNExcitation([subPrepReg[3]], (xReg, 1, true, 0));
        Controlled BalancedDicke2KNNExcitation([subPrepReg[5]], (zReg, 1, true, 0));
        Controlled BalancedDoubleDicke2KNNExcitation([subPrepReg[4]], (xReg, zReg, 1, true, 0));
    }
    controlled adjoint auto;
}

/// The shared Dicke-state routing that follows the sub-PREP in the optimal
/// Heisenberg PREP.  All gates here are self-gated by the sub-PREP one-hot
/// pattern, so this operation runs unconditionally even in the controlled
/// version of `PrepFoqcsLcuHeisenbergOptimal`.
operation PrepFoqcsLcuHeisenbergOptimalRouting(
    subPrepReg : Qubit[],
    xReg : Qubit[],
    zReg : Qubit[]
) : Unit is Adj + Ctl {
    let L = Length(xReg);

    // CNOT starters for gx, gz, and gy.
    CNOT(subPrepReg[0], xReg[L - 1]);
    CNOT(subPrepReg[2], zReg[L - 1]);
    CNOT(subPrepReg[1], xReg[L - 1]);

    if L > 1 {
        // First Dicke_1 iteration on the tail pair, reusing the starters above.
        BalancedDicke1Excitation(xReg[L - 2..L - 1], false, L - 2);
        BalancedDicke1Excitation(zReg[L - 2..L - 1], false, L - 2);

        // Starters for the Jx, Jz, and Jy Dicke_2,1NN branches.
        CNOT(subPrepReg[3], xReg[L - 2]);
        CNOT(subPrepReg[5], zReg[L - 2]);
        CNOT(subPrepReg[4], xReg[L - 2]);

        // Complete the remaining shared Dicke_1 preparation on each prefix.
        BalancedDicke1Excitation(xReg[0..L - 2], false, 0);
        BalancedDicke1Excitation(zReg[0..L - 2], false, 0);

        // Activate Jx and Jy on xReg at the same time, while Jz acts on zReg.
        CNOT(subPrepReg[3], subPrepReg[4]);
        Controlled CnotLadder([subPrepReg[5]], (zReg, 1));
        Controlled CnotLadder([subPrepReg[4]], (xReg, 1));
        CNOT(subPrepReg[3], subPrepReg[4]);
    }

    // Finish the doubled gy / Jy branches by copying xReg into zReg.
    CNOT(subPrepReg[1], subPrepReg[4]);
    Controlled ElementWiseCnot([subPrepReg[4]], (xReg, zReg));
    CNOT(subPrepReg[1], subPrepReg[4]);
}

/// Optimal PREP for the Heisenberg FOQCS-LCU block encoding.
///
/// This is the compact gate-count realization from the FOQCS reference
/// implementation: it keeps the 6-qubit one-hot sub-PREP, but shares the
/// Dicke-state work across the gx/gy/Jx/Jy and gz/Jz branches.
///
/// ## Controlled specialization
/// Only `UnbalancedDicke1Excitation` (the sub-PREP step) is controlled.
/// The subsequent routing (`PrepFoqcsLcuHeisenbergOptimalRouting`) is
/// already self-gated by the sub-PREP one-hot pattern and runs
/// unconditionally.  When the outer control is |0⟩ the sub-PREP qubits
/// stay |0…0⟩, so every downstream CNOT/Dicke gate acts as identity.
operation PrepFoqcsLcuHeisenbergOptimal(
    absCoeffs : Double[],
    phases : Double[],
    ancilla : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0..numSubPrep - 1];
        let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];
        if numSubPrep != 6 {
            fail $"optimal subPrepReg must have 6 qubits, got {numSubPrep}.";
        }
        if L == 0 {
            fail "xReg and zReg must be non-empty.";
        }

        // 1. Sub-PREP on the six Heisenberg coefficients.
        UnbalancedDicke1Excitation(absCoeffs, phases, subPrepReg);

        // 2–3. Routing: CNOT starters, shared Dicke preps, CNOT ladders, copy.
        PrepFoqcsLcuHeisenbergOptimalRouting(subPrepReg, xReg, zReg);
    }
    adjoint auto;
    controlled (ctls, ...) {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0..numSubPrep - 1];
        let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];
        if numSubPrep != 6 {
            fail $"optimal subPrepReg must have 6 qubits, got {numSubPrep}.";
        }
        if L == 0 {
            fail "xReg and zReg must be non-empty.";
        }

        // Only the sub-PREP is controlled; routing is self-gated.
        Controlled UnbalancedDicke1Excitation(ctls, (absCoeffs, phases, subPrepReg));
        PrepFoqcsLcuHeisenbergOptimalRouting(subPrepReg, xReg, zReg);
    }
    controlled adjoint auto;
}

// ===================================================================
// 6. Common FOQCS-LCU block encoding framework
// ===================================================================

/// Common FOQCS-LCU block encoding parameterized by a PREP operation.
///
/// Implements B[H] = PREP(c*)† · SELECT · PREP(c) using the caller-supplied
/// `prep` operation.  The register `targetRegister` is sliced as:
///   [subPrepReg (numSubPrep)] [xReg (L)] [zReg (L)]
///
/// ## Controlled specialization
/// Only the PREP / PREP† pair is controlled.
/// `SelectFoqcsLcu` runs unconditionally because it is sandwiched
/// between PREP/PREP†: when the outer control is |0⟩, the ancilla
/// stays |0…0⟩ so SELECT acts as identity.
operation FoqcsBlockEncoding(
    prep : (Double[], Double[], Qubit[]) => Unit is Adj + Ctl,
    absCoeffs : Double[],
    phases : Double[],
    numSubPrep : Int,
    systemReg : Qubit[],
    targetRegister : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let conjPhases = NegatePhases(phases);

        prep(absCoeffs, phases, targetRegister);
        SelectFoqcsLcu(numSubPrep, targetRegister, systemReg);
        Adjoint prep(absCoeffs, conjPhases, targetRegister);
    }
    adjoint auto;
    controlled (ctls, ...) {
        let conjPhases = NegatePhases(phases);

        Controlled prep(ctls, (absCoeffs, phases, targetRegister));
        SelectFoqcsLcu(numSubPrep, targetRegister, systemReg);
        Controlled Adjoint prep(ctls, (absCoeffs, conjPhases, targetRegister));
    }
    controlled adjoint auto;
}

// ===================================================================
// 7. Full FOQCS-LCU block encodings
// ===================================================================

/// Full FOQCS-LCU block encoding of the 1D Heisenberg Hamiltonian.
///
/// B[H] = PREP(c*)† · SELECT · PREP(c)
///
/// encodes H/λ in the ⟨0|_anc · B · |0⟩_anc subspace.
///
/// NOTE: The unpreparation uses conjugated phase angles (negated) so that
/// the block encoding yields  Σ cₗ² · Uₗ  instead of  Σ |cₗ|² · Uₗ.
/// This is essential for the Y-term phase correction: the SELECT oracle
/// produces ZX = iY, and the squared phase factor  e^{−iπ/2}  (from gy)
/// cancels the extra i to recover the real Hamiltonian coefficient.
///
/// # Register layout
/// `targetRegister` is sliced as:
///   [subPrepReg (6)] [xReg (L)] [zReg (L)]
/// Total: 2L + 6 qubits.
///
/// # Input
/// ## L         – system size (number of spin-½ sites)
/// ## J         – coupling constants [Jx, Jy, Jz]  (length 3, all ≥ 0)
/// ## g         – local fields [gx, gy, gz]         (length 3, all ≥ 0)
/// ## systemReg – L qubits (spin chain)
/// ## targetRegister – 2L + 6 qubits (block-encoding ancilla)
/// ## optimal – if true, uses `PrepFoqcsLcuHeisenbergOptimal` (shared Dicke
///              routing); otherwise uses the straightforward per-term PREP.
///
/// ## Controlled specialization
/// Only the PREP / PREP† pair is controlled.
/// `SelectFoqcsLcu` runs unconditionally because it is sandwiched
/// between PREP/PREP†: when the outer control is |0⟩, the ancilla
/// stays |0…0⟩ so SELECT acts as identity.
operation HeisenbergFoqcsBlockEncoding(
    L : Int,
    J : Double[],
    g : Double[],
    optimal : Bool,
    systemReg : Qubit[],
    targetRegister : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let numSubPrep = 6;
        let expectedLen = 2 * L + numSubPrep;
        if Length(targetRegister) != expectedLen {
            fail $"targetRegister must have {expectedLen} qubits (2L+6), got {Length(targetRegister)}.";
        }

        let (absCoeffs, phases, _lambda) = ComputeHeisenbergCoeffs(L, J, g);

        if optimal {
            FoqcsBlockEncoding(PrepFoqcsLcuHeisenbergOptimal, absCoeffs, phases, numSubPrep, systemReg, targetRegister);
        } else {
            FoqcsBlockEncoding(PrepFoqcsLcuHeisenberg, absCoeffs, phases, numSubPrep, systemReg, targetRegister);
        }
    }
    adjoint auto;
    controlled (ctls, ...) {
        let numSubPrep = 6;
        let expectedLen = 2 * L + numSubPrep;
        if Length(targetRegister) != expectedLen {
            fail $"targetRegister must have {expectedLen} qubits (2L+6), got {Length(targetRegister)}.";
        }

        let (absCoeffs, phases, _lambda) = ComputeHeisenbergCoeffs(L, J, g);

        // Only PREP and Adjoint PREP are controlled; SELECT is
        // self-gated by the ancilla pattern.
        if optimal {
            Controlled FoqcsBlockEncoding(ctls, (PrepFoqcsLcuHeisenbergOptimal, absCoeffs, phases, numSubPrep, systemReg, targetRegister));
        } else {
            Controlled FoqcsBlockEncoding(ctls, (PrepFoqcsLcuHeisenberg, absCoeffs, phases, numSubPrep, systemReg, targetRegister));
        }
    }
    controlled adjoint auto;
}

/// Convenience function: compute λ for a given Heisenberg model.
/// λ = L·(gx+gy+gz) + (L-1)·(Jx+Jy+Jz).
function ComputeHeisenbergLambda(L : Int, J : Double[], g : Double[]) : Double {
    let (_, _, lambda) = ComputeHeisenbergCoeffs(L, J, g);
    lambda
}
