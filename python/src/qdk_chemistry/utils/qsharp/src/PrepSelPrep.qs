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

    import Std.Arrays.Subarray;
    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Intrinsic.AND;
    import Std.Intrinsic.R;
    import Std.Math.PI;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.SingleVariant;

    /// No-op PREPARE callable for single-term Hamiltonians (0-ancilla case).
    operation NoOpPrepare(ancillaRegister : Qubit[]) : Unit is Adj + Ctl {}

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

        if nAll <= 3 {
            Controlled Z(allQubits[0..nAll - 2], allQubits[nAll - 1]);
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
    /// # Description
    /// `prepareRegister` is everything PREPARE needs. SELECT controls on only its first
    /// `numSelectQubits` qubits, because a PREPARE oracle may need ancilla beyond the index
    /// it produces — alias sampling, for instance, leaves `2μ + 1` qubits of garbage
    /// entangled with a `numSelectQubits`-wide index. Handing that garbage to SELECT would
    /// make it control on qubits that carry no index information.
    /// Both the garbage ancillas and select ancillas are cleared by the `Adjoint prepareOp`,
    /// which uncomputes the whole register back to |0⟩.
    operation PrepSelPrep(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        targetRegister : Qubit[],
        prepareRegister : Qubit[],
        numSelectQubits : Int,
    ) : Unit is Adj + Ctl {
        body ... {
            if (Length(prepareRegister) == 0) {
                selectOp([], targetRegister);
            } else {
                within {
                    prepareOp(prepareRegister);
                } apply {
                    selectOp(prepareRegister[0..numSelectQubits - 1], targetRegister);
                }
            }
        }
        adjoint auto;
        controlled (ctls, ...) {
            // Per Babbush et al. (arXiv:1805.03662): only SELECT is controlled;
            // PREPARE and PREPARE† run unconditionally.
            if (Length(prepareRegister) == 0) {
                Controlled selectOp(ctls, ([], targetRegister));
            } else {
                prepareOp(prepareRegister);
                Controlled selectOp(ctls, (prepareRegister[0..numSelectQubits - 1], targetRegister));
                Adjoint prepareOp(prepareRegister);
            }
        }
        controlled adjoint auto;
    }

    /// Qubitization walk: PREPARE† · SELECT · PREPARE followed by REFLECT.
    operation PSPWalk(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        targetRegister : Qubit[],
        ancillaRegister : Qubit[],
    ) : Unit is Adj + Ctl {
        body ... {
            PrepSelPrep(prepareOp, selectOp, targetRegister, ancillaRegister, Length(ancillaRegister));
            Reflect(ancillaRegister);
        }
        adjoint auto;
        controlled (ctls, ...) {
            Controlled PrepSelPrep(
                ctls,
                (prepareOp, selectOp, targetRegister, ancillaRegister, Length(ancillaRegister))
            );
            Controlled Reflect(ctls, ancillaRegister);
        }
        controlled adjoint auto;
    }

    /// # Summary
    /// Reflection about the all-zero state of the block-encoding ancillas of a flat
    /// `[systemReg | blockAncilla]` register.
    function MakeAncillaReflectionOp(
        numSystemQubits : Int,
        numBlockAncillaQubits : Int,
    ) : (Qubit[] => Unit is Adj + Ctl) {
        (allQubits) => Reflect(allQubits[numSystemQubits..numSystemQubits + numBlockAncillaQubits - 1])
    }

    /// Block encoding on the flat `[systemReg | ancillaReg]` register.
    function MakePrepSelPrepOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
    ) : (Qubit[] => Unit is Adj + Ctl) {
        (allQubits) => PrepSelPrep(
            prepareOp,
            selectOp,
            allQubits[0..numSystemQubits - 1],
            allQubits[numSystemQubits...],
            numSelectQubits
        )
    }

    /// Qubitization walk `W = REFLECT · B` on the flat `[systemReg | ancillaReg]` register.
    function MakeWalkOp(
        blockEncoding : Qubit[] => Unit is Adj + Ctl,
        applyReflection : Qubit[] => Unit is Adj + Ctl,
    ) : (Qubit[] => Unit is Adj + Ctl) {
        (allQubits) => {
            blockEncoding(allQubits);
            applyReflection(allQubits);
        }
    }

    /// Circuit entry point: allocates the register and applies the block encoding `power` times.
    operation MakePrepSelPrepCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
        numBlockAncillaQubits : Int,
        power : Int,
        useWalk : Bool,
    ) : Unit {
        use register = Qubit[numSystemQubits + numBlockAncillaQubits];
        let systems = register[0..numSystemQubits - 1];
        let prepareRegister = register[numSystemQubits...];
        let blockAncilla = register[numSystemQubits..numSystemQubits + numBlockAncillaQubits - 1];
        for _ in 1..power {
            if BeginEstimateCaching(useWalk ? "PSPWalk" | "PrepSelPrep", SingleVariant()) {
                PrepSelPrep(prepareOp, selectOp, systems, prepareRegister, numSelectQubits);
                if useWalk {
                    Reflect(blockAncilla);
                }
                EndEstimateCaching();
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
    ) : (Qubit, Qubit[]) => Unit is Adj + Ctl {
        (control, allQubits) => {
            let systems = allQubits[0..numSystemQubits - 1];
            let ancilla = allQubits[numSystemQubits...];
            for _ in 0..power - 1 {
                if BeginEstimateCaching("ControlledPSPWalk", SingleVariant()) {
                    Controlled PSPWalk([control], (prepareOp, selectOp, systems, ancilla));
                    EndEstimateCaching();
                }
            }
        }
    }

    /// Circuit entry point for the singly-controlled block encoding
    operation MakeControlledPrepSelPrepCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
        numBlockAncillaQubits : Int,
        power : Int,
        useWalk : Bool,
    ) : Unit {
        use control = Qubit();
        use register = Qubit[numSystemQubits + numBlockAncillaQubits];
        let systems = register[0..numSystemQubits - 1];
        let prepareRegister = register[numSystemQubits...];
        let blockAncilla = register[numSystemQubits..numSystemQubits + numBlockAncillaQubits - 1];
        for _ in 1..power {
            if BeginEstimateCaching(useWalk ? "ControlledPSPWalk" | "ControlledPrepSelPrep", SingleVariant()) {
                Controlled PrepSelPrep(
                    [control],
                    (prepareOp, selectOp, systems, prepareRegister, numSelectQubits)
                );
                if useWalk {
                    Controlled Reflect([control], blockAncilla);
                }
                EndEstimateCaching();
            }
        }
    }

    /// # Summary
    /// One-system-qubit, one-ancilla block encoding used to drive block-encoding-agnostic
    /// schedules from a test.
    internal function MakeTestBlockEncodingOp(theta : Double) : (Qubit[] => Unit is Adj + Ctl) {
        MakePrepSelPrepOp(
            (ancilla) => Ry(theta, ancilla[0]),
            (ancilla, system) => Controlled Z(ancilla, system[0]),
            1,
            1
        )
    }

    /// Applies the PSP walk to a computational basis state, leaking the qubits.
    ///
    /// The qubits are leaked with `QIR.Runtime.AllocateQubitArray` so the full statevector
    /// survives until the caller dumps it, which lets a test reconstruct the walk operator
    /// column by column and check its spectrum without a Qiskit round trip. Register layout
    /// is `[systemReg | ancillaReg]`.
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
