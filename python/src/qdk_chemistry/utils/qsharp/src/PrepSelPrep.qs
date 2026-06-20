// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Generic PREPARE-SELECT-PREPARE block encoding operations.
///
/// These operations compose arbitrary PREPARE and SELECT callables into
/// block encodings and quantum walk steps.  They are agnostic to the
/// concrete decomposition (LCU, double-factorized, etc.) — callers supply
/// the two callables and this module handles the stitching.
namespace QDKChemistry.Utils.PrepSelPrep {

    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Math.PI;
    import Std.Intrinsic.R;

    /// REFLECT oracle: reflection about the zero state on the ancilla register.
    ///
    /// Uses AND-ladder with measurement-based uncompute for n >= 2.
    /// Cost: n-2 Toffoli for n qubits (vs 2n-5 for standard multi-controlled-Z).
    ///
    /// $$
    ///     \mathrm{REFLECT} = 2|0\rangle\langle 0| - I
    /// $$
    operation Reflect(ancillaRegister : Qubit[]) : Unit is Adj + Ctl {
        body ... {
            let n = Length(ancillaRegister);
            if n == 0 {
                // No ancilla — reflection is a global phase (no-op).
            } elif n == 1 {
                Z(ancillaRegister[0]);
            } else {
                ReflectImpl([], ancillaRegister);
            }
        }
        adjoint self;
        controlled (ctls, ...) {
            let n = Length(ancillaRegister);
            if n == 0 {
                // No-op (global phase).
            } elif n == 1 {
                Controlled Z(ctls, ancillaRegister[0]);
            } else {
                ReflectImpl(ctls, ancillaRegister);
            }
        }
        controlled adjoint self;
    }

    /// AND-ladder implementation of 2|0⟩⟨0| - I (measurement-based uncompute).
    internal operation ReflectImpl(ctls : Qubit[], qs : Qubit[]) : Unit {
        let n = Length(qs);
        let nCtls = Length(ctls);
        let allQubits = ctls + qs;
        let nAll = nCtls + n;

        ApplyToEachCA(X, qs);

        if nAll == 2 {
            Controlled Z([allQubits[0]], allQubits[1]);
        } else {
            let nAnc = nAll - 2;
            use anc = Qubit[nAnc];

            AND(allQubits[0], allQubits[1], anc[0]);
            for i in 1..nAnc - 1 {
                AND(anc[i - 1], allQubits[i + 1], anc[i]);
            }

            Controlled Z([anc[nAnc - 1]], allQubits[nAll - 1]);

            for i in nAnc - 1..-1..1 {
                Adjoint AND(anc[i - 1], allQubits[i + 1], anc[i]);
            }
            Adjoint AND(allQubits[0], allQubits[1], anc[0]);
        }

        ApplyToEachCA(X, qs);

        if nCtls == 0 {
            R(PauliI, 2.0 * PI(), qs[0]);
        } elif nCtls == 1 {
            Z(ctls[0]);
        } else {
            Controlled Z(ctls[1...], ctls[0]);
        }
    }

    /// # Summary
    /// Block encoding: PREPARE† · SELECT · PREPARE.
    ///
    /// Takes `prepareOp` and `selectOp` as callables so they can be swapped
    /// for different implementations.
    ///
    /// When controlled (via `within/apply`), only SELECT is controlled while
    /// PREPARE and UNPREPARE run unconditionally.
    ///
    /// $$
    ///     B[H] = \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}
    /// $$
    operation PrepSelPrep(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        targetRegister : Qubit[],
        ancillaRegister : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            let numAncillaQubits = Length(ancillaRegister);
            if (numAncillaQubits == 0) {
                selectOp([], targetRegister);
            } else {
                within {
                    prepareOp(ancillaRegister);
                } apply {
                    selectOp(ancillaRegister, targetRegister);
                }
            }
        }
        adjoint auto;
        controlled (ctls, ...) {
            // Per Babbush et al. (arXiv:1805.03662): only SELECT is controlled;
            // PREPARE and PREPARE† run unconditionally.
            let numAncillaQubits = Length(ancillaRegister);
            if (numAncillaQubits == 0) {
                Controlled selectOp(ctls, ([], targetRegister));
            } else {
                prepareOp(ancillaRegister);
                Controlled selectOp(ctls, (ancillaRegister, targetRegister));
                Adjoint prepareOp(ancillaRegister);
            }
        }
        controlled adjoint auto;
    }

    /// # Summary
    /// Quantum walk step: W = REFLECT · B[H].
    ///
    /// When controlled, both SELECT (inside B[H]) and REFLECT are controlled,
    /// while PREPARE/PREPARE† run unconditionally (via within/apply semantics).
    /// This follows Babbush et al. (arXiv:1805.03662): c-W = c-R · (PREP† · c-SEL · PREP).
    ///
    /// $$
    ///     W = (2|0\rangle\langle 0| - I) \cdot \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}
    /// $$
    operation QuantumWalkStep(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        targetRegister : Qubit[],
        ancillaRegister : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            PrepSelPrep(prepareOp, selectOp, targetRegister, ancillaRegister);
            Reflect(ancillaRegister);
        }
        adjoint auto;
        controlled (ctls, ...) {
            Controlled PrepSelPrep(ctls, (prepareOp, selectOp, targetRegister, ancillaRegister));
            Controlled Reflect(ctls, (ancillaRegister));
        }
        controlled adjoint auto;
    }

    /// # Summary
    /// Creates a controlled block-encoding callable.
    ///
    /// The caller passes system + ancilla qubits together since the ancilla
    /// becomes entangled with the control qubit during the controlled operation.
    function MakeControlledPrepSelPrepOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        (control, allQubits) => {
            let systems = allQubits[0..numSystemQubits - 1];
            let ancilla = allQubits[numSystemQubits...];
            for _ in 0..power - 1 {
                Controlled PrepSelPrep([control], (prepareOp, selectOp, systems, ancilla));
            }
        }
    }

    /// # Summary
    /// Creates a controlled quantum-walk callable.
    ///
    /// System and ancilla qubits are passed together; the caller is responsible
    /// for allocation since the walk operator leaves ancilla entangled.
    function MakeControlledQuantumWalkOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        (control, allQubits) => {
            let systems = allQubits[0..numSystemQubits - 1];
            let ancilla = allQubits[numSystemQubits...];
            for _ in 0..power - 1 {
                Controlled QuantumWalkStep([control], (prepareOp, selectOp, systems, ancilla));
            }
        }
    }

    /// Circuit entry point for prep-sel-prep (allocates qubits).
    operation MakeControlledPrepSelPrepCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numAncillaQubits];
        let op = MakeControlledPrepSelPrepOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power);
        op(control, systems);
    }

    /// Circuit entry point for quantum walk (allocates qubits).
    operation MakeControlledQuantumWalkCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numAncillaQubits];
        let op = MakeControlledQuantumWalkOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power);
        op(control, systems);
    }
}
