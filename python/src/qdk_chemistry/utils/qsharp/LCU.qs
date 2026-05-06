// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.LCU {

    import Std.Arrays.Reversed;
    import Std.Canon.ApplyControlledOnInt;
    import Std.Canon.ApplyPauli;
    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Math.PI;
    import Std.Intrinsic.R;
    import Std.StatePreparation.PreparePureStateD;

    /// Parameters for the default Pauli PREPARE oracle.
    struct DefaultPrepareParams {
        amplitudes : Double[],
    }

    /// Parameters for the default Pauli SELECT oracle.
    struct DefaultSelectParams {
        pauliTerms : Pauli[][],
        signs : Int[],
    }

    /// Default PREPARE oracle: encodes normalized amplitudes into the ancilla register
    /// using PreparePureStateD.
    ///
    /// $$
    ///     \mathrm{PREPARE} |0\rangle_a = \sum_j \sqrt{\frac{|\alpha_j|}{\lambda}} |j\rangle_a
    /// $$
    operation DefaultPrepare(params : DefaultPrepareParams, ancillaRegister : Qubit[]) : Unit is Adj + Ctl {
        PreparePureStateD(params.amplitudes, Reversed(ancillaRegister));
    }

    /// Creates a PREPARE callable from parameters.
    function MakePrepareOp(params : DefaultPrepareParams) : Qubit[] => Unit is Adj + Ctl {
        DefaultPrepare(params, _)
    }

    /// Default SELECT oracle: applies Pauli string P_j controlled on ancilla state |j⟩,
    /// with a sign flip (global phase of π) for negative coefficients.
    ///
    /// $$
    ///     \mathrm{SELECT} = \sum_j |j\rangle\langle j| \otimes \mathrm{sign}(\alpha_j) \cdot P_j
    /// $$
    operation DefaultSelect(params : DefaultSelectParams, selectRegister : Qubit[], targets : Qubit[]) : Unit is Adj + Ctl {
        let numUnitary = Length(params.pauliTerms);
        for i in 0..numUnitary - 1 {
            let U = params.pauliTerms[i];
            ApplyControlledOnInt(i, ApplyPauli(U, _), selectRegister, targets);
            if params.signs[i] < 0 and Length(targets) > 0 {
                // Controlled global phase of -1 to encode the sign of the coefficient.
                ApplyControlledOnInt(i, R(PauliI, -2.0 * PI(), _), selectRegister, targets[0]);
            }
        }
    }

    /// Creates a SELECT callable from parameters.
    function MakeSelectOp(params : DefaultSelectParams) : (Qubit[], Qubit[]) => Unit is Adj + Ctl {
        DefaultSelect(params, _, _)
    }

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
    /// LCU block encoding: PREPARE† · SELECT · PREPARE.
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
    operation BlockEncoding(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        targetRegister : Qubit[],
        ancillaRegister : Qubit[],
    ) : Unit is Adj + Ctl {
        let numSelectQubits = Length(ancillaRegister);
        if (numSelectQubits == 0) {
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
            BlockEncoding(prepareOp, selectOp, targetRegister, ancillaRegister);
            Reflect(ancillaRegister);
        }
        adjoint auto;
        controlled (ctls, ...) {
            Controlled BlockEncoding(ctls, (prepareOp, selectOp, targetRegister, ancillaRegister));
            Reflect(ancillaRegister);
        }
        controlled adjoint auto;
    }

    /// # Summary
    /// Creates a controlled LCU block encoding callable.
    /// The caller passes system + ancilla qubits together since ancilla
    /// becomes entangled with the control qubit during the controlled operation.
    operation MakeLCUOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
        power : Int,
    ) : (Qubit, Qubit[]) => Unit {
        (control, allQubits) => {
            let systems = allQubits[0..numSystemQubits - 1];
            let ancilla = allQubits[numSystemQubits...];
            Controlled BlockEncoding([control], (prepareOp, selectOp, systems, ancilla));
        }
    }

    /// # Summary
    /// Creates a controlled LCU quantum walk callable.
    /// System and ancilla qubits are passed together; the caller is responsible
    /// for allocation since the walk operator leaves ancilla entangled.
    operation MakeLCUQuantumWalkOp(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
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

    /// Circuit entry point for LCU block encoding (allocates qubits).
    operation MakeLCUCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numSelectQubits];
        let op = MakeLCUOp(prepareOp, selectOp, numSystemQubits, numSelectQubits, power);
        op(control, systems);
    }

    /// Circuit entry point for LCU quantum walk (allocates qubits).
    operation MakeLCUQuantumWalkCircuit(
        prepareOp : Qubit[] => Unit is Adj + Ctl,
        selectOp : (Qubit[], Qubit[]) => Unit is Adj + Ctl,
        numSystemQubits : Int,
        numSelectQubits : Int,
        power : Int,
    ) : Unit {
        use control = Qubit();
        use systems = Qubit[numSystemQubits + numSelectQubits];
        let op = MakeLCUQuantumWalkOp(prepareOp, selectOp, numSystemQubits, numSelectQubits, power);
        op(control, systems);
    }
}
