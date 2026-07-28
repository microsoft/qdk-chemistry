// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// FOQCS-LCU (Fast One-Qubit Control Select – Linear Combination of Unitaries)
/// block encoding of translationally-structured spin-model Hamiltonians.
///
/// A spin Hamiltonian is described as a sum of homogeneous Pauli-term
/// *families*.  Each family is a single Pauli letter (``X``/``Y``/``Z``) acting
/// either on every site (1-body) or on every nearest-neighbour pair at a fixed
/// offset ``k`` (2-body).  The block encoding is
///
/// $$
///     B[H] = \mathrm{PREP}(c^*)^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREP}(c)
/// $$
///
/// which encodes ``H / lambda`` in the ``<0|_anc B |0>_anc`` subspace.  The
/// ancilla register is laid out as ``[subPrepReg | xReg | zReg]`` with
/// ``subPrepReg`` of length ``numFamilies`` and ``xReg``/``zReg`` each of length
/// ``numSites``.
///
/// # References
/// - F. Della Chiara, M. Nibbi, Y. Shen, D. Camps, R. Van Beeumen, "Efficient
///   LCU block encodings through Dicke states preparation", 2025,
///   arXiv:2507.20887.
namespace QDKChemistry.Utils.Foqcs {

    import Std.Convert.IntAsDouble;
    import Std.Core.Length;
    import Std.Math.AbsD;
    import Std.Math.ArcCos;
    import Std.Math.Sqrt;

    // ===================================================================
    // 1. Parameter struct
    // ===================================================================

    /// Fully-resolved description of a FOQCS-LCU block encoding.
    ///
    /// Each family ``f`` contributes one sub-PREP qubit.  ``paulisPerFamily[f]``
    /// is the homogeneous Pauli pattern (length 1 for a field term, length 2 for
    /// a coupling term); ``offsets[f]`` is the nearest-neighbour separation ``k``
    /// for 2-body families (ignored for 1-body).  ``absCoeffs`` and ``phases``
    /// are the normalized sub-PREP amplitudes and phase corrections.
    struct FoqcsParams {
        paulisPerFamily : Pauli[][],
        offsets : Int[],
        absCoeffs : Double[],
        phases : Double[],
        numSites : Int,
    }

    // ===================================================================
    // 2. Dicke-state building blocks
    // ===================================================================

    /// Gamma gate – the elementary rotation used inside Dicke state preparation.
    /// Maps |10> -> cos(theta/2)|10> + sin(theta/2)|01>.
    operation Gamma(theta : Double, q0 : Qubit, q1 : Qubit) : Unit is Adj + Ctl {
        Controlled Ry([q1], (theta, q0));
        CNOT(q0, q1);
    }

    /// Balanced Dicke state with 1 excitation: (1/sqrt(L)) sum_i |one-hot_i> on L qubits.
    /// `fromZero`: if true, starts by flipping the last qubit (input is |0...0>).
    /// `offsetIter`: shifts the angle denominator (used when chaining iterations).
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
                let i = L - 1 - idx;
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
    operation ElementWiseCnot(reg1 : Qubit[], reg2 : Qubit[]) : Unit is Adj + Ctl {
        for i in 0..Length(reg1) - 1 {
            CNOT(reg1[i], reg2[i]);
        }
    }

    /// Double Dicke state (X and Z registers share the same excitation pattern):
    /// (1/sqrt(L)) sum_i |one-hot_i>_X |one-hot_i>_Z.
    operation BalancedDoubleDicke1Excitation(
        xReg : Qubit[],
        zReg : Qubit[],
        fromZero : Bool,
        offsetIter : Int
    ) : Unit is Adj + Ctl {
        BalancedDicke1Excitation(xReg, fromZero, offsetIter);
        ElementWiseCnot(xReg, zReg);
    }

    /// CNOT ladder: converts a Dicke_1 excitation into a Dicke_2 k-NN pattern.
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

    /// Balanced Dicke_2 state with 2 k-nearest-neighbour excitations:
    ///   (1/sqrt(L-k)) sum_i |one-hot_i (+) one-hot_{i+k}>
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

    /// Double version of the balanced Dicke_2 k-NN state, same pattern on both registers.
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

    /// Compute the Gamma rotation angles for an unbalanced Dicke_1 state from a
    /// vector of non-negative real amplitudes (must be normalized).
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

    /// Unbalanced Dicke_1 state preparation:
    ///   sum_i |c_i| e^{i phi_i} |one-hot_i>
    /// `absCoeffs` must be normalized (sum |c_i|^2 = 1).
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

    /// Negate all phase angles (used for the conjugated un-PREP).
    function NegatePhases(phases : Double[]) : Double[] {
        mutable result = [0.0, size = Length(phases)];
        for i in 0..Length(phases) - 1 {
            result w/= i <- -phases[i];
        }
        result
    }

    // ===================================================================
    // 3. Generic SELECT oracle
    // ===================================================================

    /// FOQCS-LCU SELECT for a spin model.
    ///
    /// Ancilla layout ``[subPrepReg | xReg | zReg]``, system register ``systemReg``:
    ///   * CX from xReg[i] to system[i] applies X when xReg[i] = 1
    ///   * CZ from zReg[i] to system[i] applies Z when zReg[i] = 1
    /// When both are set the combined effect is XZ = -iY; the -i phase is
    /// compensated inside the PREP coefficients.
    operation SelectFoqcs(
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
    // 4. Generic PREP oracle
    // ===================================================================

    /// Dispatch one homogeneous Pauli-term family to its Dicke preparation.
    ///
    /// `ctl` is the single sub-PREP control qubit for this family.  The routing
    /// is self-gated by the one-hot sub-PREP pattern, so callers apply it
    /// unconditionally even inside a controlled PREP.
    operation ApplyFamilyDicke(
        paulis : Pauli[],
        k : Int,
        ctl : Qubit[],
        xReg : Qubit[],
        zReg : Qubit[]
    ) : Unit is Adj + Ctl {
        let width = Length(paulis);
        if width == 1 {
            let p = paulis[0];
            if p == PauliX {
                Controlled BalancedDicke1Excitation(ctl, (xReg, true, 0));
            } elif p == PauliZ {
                Controlled BalancedDicke1Excitation(ctl, (zReg, true, 0));
            } elif p == PauliY {
                Controlled BalancedDoubleDicke1Excitation(ctl, (xReg, zReg, true, 0));
            } else {
                fail "FOQCS: unsupported 1-body Pauli (must be X, Y, or Z).";
            }
        } elif width == 2 {
            let p = paulis[0];
            if p == PauliX {
                Controlled BalancedDicke2KNNExcitation(ctl, (xReg, k, true, 0));
            } elif p == PauliZ {
                Controlled BalancedDicke2KNNExcitation(ctl, (zReg, k, true, 0));
            } elif p == PauliY {
                Controlled BalancedDoubleDicke2KNNExcitation(ctl, (xReg, zReg, k, true, 0));
            } else {
                fail "FOQCS: unsupported 2-body Pauli (must be XX, YY, or ZZ).";
            }
        } else {
            fail "FOQCS: only 1-body and 2-body homogeneous families are supported.";
        }
    }

    /// Generic FOQCS PREP oracle.
    ///
    /// Layout ``[subPrepReg | xReg | zReg]``.  Creates a one-hot superposition
    /// over the families (weighted by `absCoeffs`, `phases`) then spreads each
    /// family across the target register via its Dicke preparation.
    ///
    /// `phases` is passed separately from `params.phases` so the block encoding
    /// can supply the conjugated angles for the un-PREP.
    ///
    /// ## Controlled specialization
    /// Only the sub-PREP step is controlled; the Dicke routing is self-gated by
    /// the one-hot ancilla pattern and runs unconditionally.
    operation FoqcsPrepare(
        params : FoqcsParams,
        phases : Double[],
        ancilla : Qubit[]
    ) : Unit is Adj + Ctl {
        body ... {
            let numSubPrep = Length(params.absCoeffs);
            let L = params.numSites;
            let subPrepReg = ancilla[0..numSubPrep - 1];
            let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
            let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];

            UnbalancedDicke1Excitation(params.absCoeffs, phases, subPrepReg);
            for f in 0..numSubPrep - 1 {
                ApplyFamilyDicke(params.paulisPerFamily[f], params.offsets[f], [subPrepReg[f]], xReg, zReg);
            }
        }
        adjoint auto;
        controlled (ctls, ...) {
            let numSubPrep = Length(params.absCoeffs);
            let L = params.numSites;
            let subPrepReg = ancilla[0..numSubPrep - 1];
            let xReg = ancilla[numSubPrep..numSubPrep + L - 1];
            let zReg = ancilla[numSubPrep + L..numSubPrep + 2 * L - 1];

            Controlled UnbalancedDicke1Excitation(ctls, (params.absCoeffs, phases, subPrepReg));
            for f in 0..numSubPrep - 1 {
                ApplyFamilyDicke(params.paulisPerFamily[f], params.offsets[f], [subPrepReg[f]], xReg, zReg);
            }
        }
        controlled adjoint auto;
    }

    // ===================================================================
    // 5. Interop factories (PREPARE / SELECT callables)
    // ===================================================================

    /// Build the FOQCS PREPARE callable for a given phase vector.
    ///
    /// Partially applies :code:`FoqcsPrepare` over the ancilla register so it can
    /// be composed by the generic ``PrepSelPrep`` block-encoding framework.  The
    /// forward preparation passes ``params.phases``; the un-preparation passes the
    /// conjugated (negated) phases via :code:`NegatePhases`.
    function MakeFoqcsPrepareOp(
        params : FoqcsParams,
        phases : Double[]
    ) : (Qubit[] => Unit is Adj + Ctl) {
        FoqcsPrepare(params, phases, _)
    }

    /// Build the FOQCS SELECT callable.
    ///
    /// Partially applies :code:`SelectFoqcs` over the ancilla and system
    /// registers so it matches the ``(Qubit[], Qubit[]) => Unit`` signature
    /// expected by the generic ``PrepSelPrep`` framework.
    function MakeFoqcsSelectOp(
        numSubPrep : Int
    ) : ((Qubit[], Qubit[]) => Unit is Adj + Ctl) {
        SelectFoqcs(numSubPrep, _, _)
    }
}
