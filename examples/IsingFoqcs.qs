import Std.Convert.IntAsDouble;
import Std.Math.Sqrt;
import HeisenbergFoqcs.UnbalancedDicke1Excitation, HeisenbergFoqcs.BalancedDicke1Excitation, HeisenbergFoqcs.BalancedDicke2KNNExcitation, HeisenbergFoqcs.SelectFoqcsLcu, HeisenbergFoqcs.NegatePhases, HeisenbergFoqcs.CnotLadder, HeisenbergFoqcs.FoqcsBlockEncoding;
import Reflect.ReflectAboutZero;
import Std.Math.Ceiling, Length;

/// Full FOQCS-LCU block encoding of the 1D Ising Hamiltonian.
///
/// B[H] = PREP(c*)† · SELECT · PREP(c)
///
/// encodes H/λ in the ⟨0|_anc · B · |0⟩_anc subspace.
///
/// # Register layout
/// `targetRegister` is sliced as:
///   [subPrepReg (3)] [xReg (L)] [zReg (L)]
/// Total: 2L + 3 qubits.
///
/// ## Controlled specialization
/// Only the PREP / PREP† pair is controlled.
/// `SelectFoqcsLcu` runs unconditionally because it is sandwiched
/// between PREP/PREP†: when the outer control is |0⟩, the ancilla
/// stays |0…0⟩ so SELECT acts as identity.
operation IsingFoqcsBlockEncoding(
    L : Int,
    J : Double,
    g : Double[],
    systemReg: Qubit[],
    targetRegister : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let hx = g[0];
        let hz = g[1];
        let (absCoeffs, phases, _lambda) = ComputeIsingCoeffs(L, J, hx, hz);
        let numSubPrep = Length(absCoeffs);

        FoqcsBlockEncoding(PrepFoqcsLcuIsingOptimal, absCoeffs, phases, numSubPrep, systemReg, targetRegister);
    }
    adjoint auto;
    controlled (ctls, ...) {
        let hx = g[0];
        let hz = g[1];
        let (absCoeffs, phases, _lambda) = ComputeIsingCoeffs(L, J, hx, hz);
        let numSubPrep = Length(absCoeffs);

        Controlled FoqcsBlockEncoding(ctls, (PrepFoqcsLcuIsingOptimal, absCoeffs, phases, numSubPrep, systemReg, targetRegister));
    }
    controlled adjoint auto;
}


/// Full PREP for the Ising FOQCS-LCU block encoding.
///
/// Register layout (left to right):
///   subPrepReg[0..2] | xReg[0..L-1] | zReg[0..L-1]
///
/// Each sub-PREP qubit controls a different Dicke state preparation:
///   0  →  hx : Dicke₁ on xReg
///   1  →  hz : Dicke₁ on zReg
///   2  →  J : Dicke₂-1NN on zReg
///
/// ## Controlled specialization
/// Only the `UnbalancedDicke1Excitation` sub-PREP step is controlled.
/// The subsequent controlled-Dicke preparations on xReg/zReg are
/// already gated by the sub-PREP one-hot pattern and run unconditionally.
operation PrepFoqcsLcuIsing(
    absCoeffs : Double[],
    phases : Double[],
    ancilla : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0 .. numSubPrep - 1];
        let xReg = ancilla[numSubPrep .. numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L .. numSubPrep + 2 * L - 1];

        // 1. Sub-PREP: create one-hot superposition on ancilla qubits.
        UnbalancedDicke1Excitation(absCoeffs, phases, subPrepReg);

        // 2. Controlled Dicke state preparations (single-qubit control each).
        //    hx: balanced Dicke₁ on xReg
        Controlled BalancedDicke1Excitation([subPrepReg[0]], (xReg, true, 0));
        //    hz: balanced Dicke₁ on zReg
        Controlled BalancedDicke1Excitation([subPrepReg[1]], (zReg, true, 0));
        //    J: balanced Dicke₂-1NN on zReg
        Controlled BalancedDicke2KNNExcitation([subPrepReg[2]], (zReg, 1, true, 0));
    }
    adjoint auto;
    controlled (ctls, ...) {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0 .. numSubPrep - 1];
        let xReg = ancilla[numSubPrep .. numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L .. numSubPrep + 2 * L - 1];

        // Only the sub-PREP is controlled; the Dicke preparations are
        // self-gated by the one-hot ancilla pattern and left unconditional.
        Controlled UnbalancedDicke1Excitation(ctls, (absCoeffs, phases, subPrepReg));

        Controlled BalancedDicke1Excitation([subPrepReg[0]], (xReg, true, 0));
        Controlled BalancedDicke1Excitation([subPrepReg[1]], (zReg, true, 0));
        Controlled BalancedDicke2KNNExcitation([subPrepReg[2]], (zReg, 1, true, 0));
    }
    controlled adjoint auto;
}

/// Optimal PREP for the Ising FOQCS-LCU block encoding.
///
/// This realization replaces the controlled `BalancedDicke2KNNExcitation`
/// with a controlled `BalancedDicke1Excitation` followed by a shared
/// `CnotLadder`, reducing the controlled gate count.
///
/// ## Controlled specialization
/// Only `UnbalancedDicke1Excitation` (the sub-PREP step) is controlled.
/// The subsequent routing is already self-gated by the sub-PREP one-hot
/// pattern and runs unconditionally.  When the outer control is |0⟩ the
/// sub-PREP qubits stay |0…0⟩, so every downstream gate acts as identity.
operation PrepFoqcsLcuIsingOptimal(
    absCoeffs : Double[],
    phases : Double[],
    ancilla : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0 .. numSubPrep - 1];
        let xReg = ancilla[numSubPrep .. numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L .. numSubPrep + 2 * L - 1];

        // 1. Sub-PREP: create one-hot superposition on ancilla qubits.
        UnbalancedDicke1Excitation(absCoeffs, phases, subPrepReg);

        //    J: balanced Dicke₂-1NN on zReg (optimized decomposition)
        //    Only L-1 sites have a nearest neighbour, so the Dicke₁
        //    superposition must span the first L-1 qubits.
        Controlled BalancedDicke1Excitation([subPrepReg[2]], (zReg[0 .. Length(zReg) - 2], true, 0));
        CnotLadder(zReg, 1);

        // 2. Controlled Dicke state preparations (single-qubit control each).
        //    hx: balanced Dicke₁ on xReg
        Controlled BalancedDicke1Excitation([subPrepReg[0]], (xReg, true, 0));
        //    hz: balanced Dicke₁ on zReg
        Controlled BalancedDicke1Excitation([subPrepReg[1]], (zReg, true, 0));
    }
    adjoint auto;
    controlled (ctls, ...) {
        let numSubPrep = Length(absCoeffs);
        let L = (Length(ancilla) - numSubPrep) / 2;
        let subPrepReg = ancilla[0 .. numSubPrep - 1];
        let xReg = ancilla[numSubPrep .. numSubPrep + L - 1];
        let zReg = ancilla[numSubPrep + L .. numSubPrep + 2 * L - 1];

        // Only the sub-PREP is controlled; routing is self-gated.
        Controlled UnbalancedDicke1Excitation(ctls, (absCoeffs, phases, subPrepReg));
        Controlled BalancedDicke1Excitation([subPrepReg[0]], (xReg, true, 0));
        Controlled BalancedDicke1Excitation([subPrepReg[1]], (zReg, true, 0));

        Controlled BalancedDicke1Excitation([subPrepReg[2]], (zReg[0 .. Length(zReg) - 2], true, 0));
        Controlled CnotLadder([subPrepReg[2]], (zReg, 1));

    }
    controlled adjoint auto;
}

/// Compute the three sub-PREP amplitudes (magnitude + phase) and the
/// normalization factor λ for the Ising FOQCS-LCU block encoding.
///
/// The 1D Ising Hamiltonian is:
/// ```text
/// H = Σ_{ℓ=0}^{n-1} (hx X_ℓ + hz Z_ℓ) + Σ_{ℓ=0}^{n-2} J Z_ℓ Z_{ℓ+1}
/// ```
///
/// Coefficient ordering: [hx, hz, Jzz].
///
/// Returns (absCoeffs[3], phases[3], lambda).
function ComputeIsingCoeffs(
    L : Int,
    J : Double,
    hx : Double,
    hz : Double
) : (Double[], Double[], Double) {
    let Ld = IntAsDouble(L);
    let Lm1 = IntAsDouble(L - 1);

    mutable mags = [0.0, size = 3];
    mags w/= 0 <- Sqrt(hx * Ld);    // transverse field X
    mags w/= 1 <- Sqrt(hz * Ld);    // longitudinal field Z
    mags w/= 2 <- Sqrt(J * Lm1);    // nearest-neighbor ZZ coupling

    mutable normSq = 0.0;
    for m in mags {
        set normSq += m * m;
    }
    let norm = Sqrt(normSq);

    mutable absCoeffs = [0.0, size = 3];
    for i in 0 .. 2 {
        absCoeffs w/= i <- mags[i] / norm;
    }

    // No Y terms, so all phases are zero.
    let phases = [0.0, 0.0, 0.0];

    // λ = norm² = hx·L + hz·L + J·(L-1).
    let lambda = normSq;

    (absCoeffs, phases, lambda)
}

/// Compute the two sub-PREP amplitudes (magnitude + phase) and the
/// normalization factor λ for the Ising 1D (Eq. C13) block encoding.
///
/// The 1D Ising Hamiltonian is:
/// ```text
/// H = g Σ_{i=0}^{n-1} Z_i + J Σ_{i=0}^{n-2} X_i X_{i+1}
/// ```
///
/// Coefficient ordering: [g, J].
///
/// Returns (absCoeffs[2], phases[2], lambda).
function ComputeIsing1DCoeffs(
    L : Int,
    g : Double,
    J : Double
) : (Double[], Double[], Double) {
    let Ld = IntAsDouble(L);
    let Lm1 = IntAsDouble(L - 1);

    mutable mags = [0.0, size = 2];
    mags w/= 0 <- Sqrt(g * Ld);     // field Z
    mags w/= 1 <- Sqrt(J * Lm1);    // coupling XX

    mutable normSq = 0.0;
    for m in mags {
        set normSq += m * m;
    }
    let norm = Sqrt(normSq);

    mutable absCoeffs = [0.0, size = 2];
    for i in 0 .. 1 {
        absCoeffs w/= i <- mags[i] / norm;
    }

    let phases = [0.0, 0.0];
    let lambda = normSq;

    (absCoeffs, phases, lambda)
}

/// Full FOQCS-LCU block encoding of the 1D Ising model (Eq. C13).
///
/// H = g Σ_{i=0}^{n-1} Z_i + J Σ_{i=0}^{n-2} X_i X_{i+1}
///
/// B[H] = PREP(c*)† · SELECT · PREP(c)
///
/// # Register layout
/// `targetRegister` is sliced as:
///   [xReg (L)] [zReg (L)]
/// Total: 2L qubits.  The sub-PREP is embedded in zReg[L-1] and xReg[L-1].
operation Ising1DFoqcsBlockEncoding(
    L : Int,
    g : Double,
    J : Double,
    systemReg : Qubit[],
    targetRegister : Qubit[]
) : Unit is Adj + Ctl {
    body ... {
        let (absCoeffs, phases, _lambda) = ComputeIsing1DCoeffs(L, g, J);
        let conjPhases = NegatePhases(phases);

        PrepFoqcsLcuIsing1D(absCoeffs, phases, targetRegister);
        SelectFoqcsLcu(0, targetRegister, systemReg);
        Adjoint PrepFoqcsLcuIsing1D(absCoeffs, conjPhases, targetRegister);
    }
    adjoint auto;
    controlled (ctls, ...) {
        let (absCoeffs, phases, _lambda) = ComputeIsing1DCoeffs(L, g, J);
        let conjPhases = NegatePhases(phases);

        Controlled PrepFoqcsLcuIsing1D(ctls, (absCoeffs, phases, targetRegister));
        SelectFoqcsLcu(0, targetRegister, systemReg);
        Controlled Adjoint PrepFoqcsLcuIsing1D(ctls, (absCoeffs, conjPhases, targetRegister));
    }
    controlled adjoint auto;
}


/// # Summary
/// Implements the P_R state preparation circuit for the 1D Ising model (Eq. C13).
/// https://arxiv.org/pdf/2601.18767v1
/// H = g Σ_{i=0}^{n-1} Z_i + J Σ_{i=0}^{n-2} X_i X_{i+1}
///
/// ## Sub-PREP embedding
/// The two sub-PREP amplitudes are embedded at zReg[L-1] (g) and xReg[L-1] (J).
/// Subsequent BalancedDicke₁ calls use `fromZero = false` to redistribute
/// the sub-PREP excitation across the target register.  When a branch
/// is not selected, that register stays |0…0⟩ and all Gamma / CNOT gates
/// act as identity (zero-subspace preservation).
///
/// ## Controlled specialization
/// Only `UnbalancedDicke1Excitation` (the sub-PREP step) is controlled.
/// The rest is self-gated by the zero-subspace.
operation PrepFoqcsLcuIsing1D(
    absCoeffs : Double[],
    phases : Double[],
    ancilla : Qubit[],
) : Unit is Adj + Ctl {
    body ... {
        let L = Length(ancilla) / 2;
        let xReg = ancilla[0 .. L - 1];
        let zReg = ancilla[L .. 2 * L - 1];

        // Sub-PREP: absCoeffs[0]=c_g → zReg[L-1], absCoeffs[1]=c_J → xReg[L-1]
        UnbalancedDicke1Excitation(absCoeffs, phases, [zReg[L - 1], xReg[L - 1]]);

        //    g: balanced Dicke₁ over all of zReg (redistribute from zReg[L-1])
        BalancedDicke1Excitation(zReg, false, 0);

        //    J: balanced Dicke₁ on xReg[1..L-1], then CnotLadder(k=-1)
        //    to create L-1 nearest-neighbour XX pair excitations on xReg.
        BalancedDicke1Excitation(xReg[1 .. L - 1], false, 0);
        CnotLadder(xReg, -1);
    }
    adjoint auto;
    controlled (ctls, ...) {
        let L = Length(ancilla) / 2;
        let xReg = ancilla[0 .. L - 1];
        let zReg = ancilla[L .. 2 * L - 1];
        Controlled UnbalancedDicke1Excitation(ctls, (absCoeffs, phases, [zReg[L - 1], xReg[L - 1]]));
        BalancedDicke1Excitation(zReg, false, 0);
        BalancedDicke1Excitation(xReg[1 .. L - 1], false, 0);
        CnotLadder(xReg, -1);
    }
    controlled adjoint auto;
}
