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
    /// Uncontrolled block encoding on the flat `[systemReg | ancillaReg]` register,
    /// applied `power` times.
    ///
    /// The returned callable is `Adj + Ctl`, so the Q# `Controlled` functor applied to it
    /// already yields the controlled block encoding with PREPARE left unconditional — that
    /// rule lives in `PrepSelPrep`'s own controlled specialization and never has to be
    /// restated. It is also what the unary-iteration signed-power schedule applies between
    /// its reflections, so both schedules consume the same callable.
    function MakePrepSelPrepOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        power : Int,
    ) : (Qubit[] => Unit is Adj + Ctl) {
        (allQubits) => {
            for _ in 1..power {
                PrepSelPrep(prepareOp, selectOp, allQubits[0..numSystemQubits - 1], allQubits[numSystemQubits...]);
            }
        }
    }

    /// # Summary
    /// Reflection about the all-zero state of the trailing block-encoding ancillas of a flat
    /// `[systemReg | ancillaReg]` register.
    ///
    /// Those are the qubits whose all-zero state flags a successful block encoding, so they
    /// are the register a qubitization walk reflects about. Handing the reflection out as a
    /// callable on the *whole* register is what lets a schedule apply it without knowing how
    /// the block encoding lays its ancillas out.
    function MakeAncillaReflectionOp(numSystemQubits : Int) : (Qubit[] => Unit is Adj + Ctl) {
        (allQubits) => Reflect(allQubits[numSystemQubits...])
    }

    /// # Summary
    /// Controlled block encoding, from an already-built uncontrolled one.
    ///
    /// The caller passes system + ancilla qubits together since the ancilla becomes
    /// entangled with the control qubits during the controlled operation.
    function MakeControlledBlockEncodingOp(
        blockEncoding : Qubit[] => Unit is Adj + Ctl
    ) : ((Qubit[], Qubit[]) => Unit is Adj) {
        (controls, allQubits) => Controlled blockEncoding(controls, allQubits)
    }

    /// # Summary
    /// Controlled qubitization walk `W = REFLECT · B`, applied `power` times.
    ///
    /// Both the block encoding and the reflection are controlled, while PREPARE/PREPARE†
    /// run unconditionally inside `B` — see Babbush et al. (arXiv:1805.03662),
    /// c-W = c-R · (PREP† · c-SEL · PREP).
    ///
    /// $$
    ///     W = (2|0\rangle\langle 0| - I) \cdot \mathrm{PREPARE}^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREPARE}
    /// $$
    function MakeControlledWalkOp(
        blockEncoding : Qubit[] => Unit is Adj + Ctl,
        applyReflection : Qubit[] => Unit is Adj + Ctl,
        power : Int,
    ) : (Qubit[], Qubit[]) => Unit {
        (controls, allQubits) => {
            for _ in 1..power {
                if BeginEstimateCaching("ControlledWalk", SingleVariant()) {
                    Controlled blockEncoding(controls, allQubits);
                    Controlled applyReflection(controls, allQubits);
                    EndEstimateCaching();
                }
            }
        }
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
        MakePrepSelPrepOp(prepareOp, selectOp, numSystemQubits, power)(qs);
    }

    /// Circuit entry point for the controlled block encoding (allocates qubits).
    operation MakeControlledBlockEncodingCircuit(
        blockEncoding : Qubit[] => Unit is Adj + Ctl,
        numQubits : Int,
    ) : Unit {
        use control = Qubit();
        use qs = Qubit[numQubits];
        MakeControlledBlockEncodingOp(blockEncoding)([control], qs);
    }

    /// Circuit entry point for the controlled quantum walk (allocates qubits).
    operation MakeControlledWalkCircuit(
        blockEncoding : Qubit[] => Unit is Adj + Ctl,
        applyReflection : Qubit[] => Unit is Adj + Ctl,
        numQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use qs = Qubit[numQubits];
        MakeControlledWalkOp(blockEncoding, applyReflection, power)([control], qs);
    }

    /// PREPARE fixture: a single-ancilla Ry rotation.
    internal operation TestRyPrepare(theta : Double, ancilla : Qubit[]) : Unit is Adj + Ctl {
        Ry(theta, ancilla[0]);
    }

    /// SELECT fixture: a sign flip on the system qubit.
    internal operation TestSignSelect(ancilla : Qubit[], system : Qubit[]) : Unit is Adj + Ctl {
        Controlled Z(ancilla, system[0]);
    }

    /// # Summary
    /// One-system-qubit, one-ancilla block encoding used to drive block-encoding-agnostic
    /// schedules from a test.
    ///
    /// `PREPARE = Ry(theta)` and `SELECT = c-Z` block-encode `diag(1, cos theta)`, which is
    /// Hermitian and therefore self-inverse, so pairing it with `MakeAncillaReflectionOp(1)`
    /// gives a genuine qubitization walk whose phase seen by `|1>` is exactly `theta`.
    function MakeTestBlockEncodingOp(theta : Double) : (Qubit[] => Unit is Adj + Ctl) {
        MakePrepSelPrepOp(TestRyPrepare(theta, _), TestSignSelect, 1, 1)
    }
}
