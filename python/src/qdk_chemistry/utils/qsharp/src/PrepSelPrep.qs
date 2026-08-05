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
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.SingleVariant;

    /// No-op PREPARE callable for single-term Hamiltonians (0-ancilla case).
    operation NoOpPrepare(ancillaRegister : Qubit[]) : Unit is Adj + Ctl {}

    /// Reflection about the zero state on the ancilla register.
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
    /// PSP-based quantum walk: W = REFLECT · B[H].
    ///
    /// When controlled, both SELECT (inside B[H]) and REFLECT are controlled,
    /// while PREPARE/PREPARE† run unconditionally (via within/apply semantics).
    /// This follows Babbush et al. (arXiv:1805.03662): c-W = c-R · (PREP† · c-SEL · PREP).
    ///
    /// $$
    ///     W = (2|0\rangle\langle 0| - I) \cdot \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}
    /// $$
    operation PSPWalk(
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

    /// PREPARE†·SELECT·PREPARE on a flat `[systemReg | ancillaReg]` register, under `controls`.
    ///
    /// No reflection is applied here, so this is the block encoding B rather than the walk W.
    /// An empty `controls` therefore yields the plain uncontrolled B, which is what the
    /// unary-iteration signed-power schedule consumes: it applies B unconditionally and
    /// controls only the reflections.
    operation ControlledPSPOnRegister(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        controls : Qubit[],
        allQubits : Qubit[],
    ) : Unit is Adj {
        Controlled PrepSelPrep(
            controls,
            (prepareOp, selectOp, allQubits[0..numSystemQubits - 1], allQubits[numSystemQubits...])
        );
    }

    /// Trailing sub-register of `allQubits`, i.e. the block-encoding ancillas.
    ///
    /// These are the qubits whose all-zero state flags a successful block encoding, so they are
    /// also the register a qubitization walk reflects about.
    function TrailingAncillaRegister(numSystemQubits : Int, allQubits : Qubit[]) : Qubit[] {
        allQubits[numSystemQubits...]
    }

    /// Bind `TrailingAncillaRegister` into a selector over the flat register.
    function MakeTrailingAncillaSelector(numSystemQubits : Int) : (Qubit[] -> Qubit[]) {
        TrailingAncillaRegister(numSystemQubits, _)
    }

    /// # Summary
    /// Creates an uncontrolled block-encoding callable on the flat `[systemReg | ancillaReg]`
    /// register.
    ///
    /// This is `MakeControlledPrepSelPrepOp` with an empty control register, which is what the
    /// unary-iteration signed-power schedule applies between its reflections.
    function MakePrepSelPrepOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : (Qubit[] => Unit is Adj) {
        MakeControlledPrepSelPrepOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power)([], _)
    }

    /// Circuit entry point for the uncontrolled block encoding (allocates qubits).
    operation MakePrepSelPrepCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : Unit {
        use qs = Qubit[numSystemQubits + numAncillaQubits];
        MakePrepSelPrepOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power)(qs);
    }

    /// # Summary
    /// Creates a controlled block-encoding callable.
    ///
    /// The caller passes system + ancilla qubits together since the ancilla
    /// becomes entangled with the control qubits during the controlled operation.
    ///
    /// Passing an empty control register yields the plain uncontrolled block encoding, which is
    /// what the unary-iteration signed-power schedule applies between its reflections.
    function MakeControlledPrepSelPrepOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : ((Qubit[], Qubit[]) => Unit is Adj) {
        (controls, allQubits) => {
            for _ in 0..power - 1 {
                ControlledPSPOnRegister(prepareOp, selectOp, numSystemQubits, controls, allQubits);
            }
        }
    }

    /// # Summary
    /// Creates a controlled PSP-based quantum-walk callable.
    ///
    /// System and ancilla qubits are passed together; the caller is responsible
    /// for allocation since the walk operator leaves ancilla entangled.
    function MakeControlledPSPWalkOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : (Qubit[], Qubit[]) => Unit {
        (controls, allQubits) => {
            let systems = allQubits[0..numSystemQubits - 1];
            let ancilla = allQubits[numSystemQubits...];
            for _ in 0..power - 1 {
                if BeginEstimateCaching("ControlledPSPWalk", SingleVariant()) {
                    Controlled PSPWalk(controls, (prepareOp, selectOp, systems, ancilla));
                    EndEstimateCaching();
                }
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
        op([control], systems);
    }

    /// Circuit entry point for quantum walk (allocates qubits).
    operation MakeControlledPSPWalkCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numAncillaQubits];
        let op = MakeControlledPSPWalkOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power);
        op([control], systems);
    }

    /// Applies the PSP walk to a computational basis state, leaking the qubits.
    operation TestPSPWalkOnBasisState(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
        basisState : Int,
    ) : Unit {
        let qs = QIR.Runtime.AllocateQubitArray(numSystemQubits + numAncillaQubits);
        ApplyXorInPlace(basisState, qs);
        for _ in 1..power {
            PSPWalk(prepareOp, selectOp, qs[0..numSystemQubits - 1], qs[numSystemQubits...]);
        }
    }
}
