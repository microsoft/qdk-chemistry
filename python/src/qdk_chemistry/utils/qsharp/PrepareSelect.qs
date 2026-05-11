// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Generic PREPARE-SELECT block encoding operations.
///
/// These operations compose arbitrary PREPARE and SELECT callables into
/// block encodings and quantum walk steps.  They are agnostic to the
/// concrete decomposition (LCU, double-factorized, etc.) — callers supply
/// the two callables and this module handles the stitching.
namespace QDKChemistry.Utils.PrepareSelect {

    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Math.PI;
    import Std.Intrinsic.R;

    /// REFLECT oracle: reflection about the zero state on the ancilla register.
    ///
    /// $$
    ///     \mathrm{REFLECT} = 2|0\rangle\langle 0| - I
    /// $$
    operation Reflect(ancillaRegister : Qubit[]) : Unit is Adj + Ctl {
        within {
            ApplyToEachCA(X, ancillaRegister);
        } apply {
            Controlled Z(ancillaRegister[1...], ancillaRegister[0]);
        }
        R(PauliI, 2.0 * PI(), ancillaRegister[0]);
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
    operation PrepareSelectPrepare(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        targetRegister : Qubit[],
        ancillaRegister : Qubit[],
    ) : Unit is Adj + Ctl {
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

    /// # Summary
    /// Quantum walk step: W = REFLECT · B[H].
    ///
    /// When controlled, only the block encoding is controlled; REFLECT runs
    /// unconditionally (it is a no-op on |0⟩ when control is |0⟩).
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
            PrepareSelectPrepare(prepareOp, selectOp, targetRegister, ancillaRegister);
            Reflect(ancillaRegister);
        }
        adjoint auto;
        controlled (ctls, ...) {
            Controlled PrepareSelectPrepare(ctls, (prepareOp, selectOp, targetRegister, ancillaRegister));
            Reflect(ancillaRegister);
        }
        controlled adjoint auto;
    }

    /// # Summary
    /// Creates a controlled block-encoding callable.
    ///
    /// The caller passes system + ancilla qubits together since the ancilla
    /// becomes entangled with the control qubit during the controlled operation.
    operation MakePrepareSelectPrepareOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        (control, allQubits) => {
            let systems = allQubits[0..numSystemQubits - 1];
            let ancilla = allQubits[numSystemQubits...];
            Controlled PrepareSelectPrepare([control], (prepareOp, selectOp, systems, ancilla));
        }
    }

    /// # Summary
    /// Creates a controlled quantum-walk callable.
    ///
    /// System and ancilla qubits are passed together; the caller is responsible
    /// for allocation since the walk operator leaves ancilla entangled.
    operation MakeQuantumWalkOp(
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

    /// Circuit entry point for prepare-select (allocates qubits).
    operation MakePrepareSelectCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numAncillaQubits];
        let op = MakePrepareSelectPrepareOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power);
        op(control, systems);
    }

    /// Circuit entry point for quantum walk (allocates qubits).
    operation MakeQuantumWalkCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numAncillaQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numAncillaQubits];
        let op = MakeQuantumWalkOp(prepareOp, selectOp, numSystemQubits, numAncillaQubits, power);
        op(control, systems);
    }
}
