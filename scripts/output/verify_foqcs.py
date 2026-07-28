"""Numeric verification of the FOQCS block encoding via the standalone FOQCS mapper."""

import numpy as np
from qdk_chemistry.algorithms.controlled_circuit_mapper import ControlledFoqcsMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.foqcs import (
    FoqcsBuilder,
)
from qdk_chemistry.data import QubitOperator
from qiskit.quantum_info import Operator


def extract_block(full_u, num_target, num_ancilla):
    n_total = num_target + num_ancilla + 1
    dim = 2**n_total
    assert full_u.shape == (dim, dim), (full_u.shape, dim)
    indices = []
    for i in range(dim):
        ctrl_bit = (i >> 0) & 1
        anc_bits = i >> (1 + num_target)
        if ctrl_bit == 1 and anc_bits == 0:
            indices.append(i)
    return full_u[np.ix_(indices, indices)]


def check(name, pauli_strings, coefficients):
    coefficients = np.array(coefficients, dtype=float)
    ham = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
    num_target = ham.num_qubits

    unitary_rep = FoqcsBuilder().run(ham)
    circuit = ControlledFoqcsMapper().run(unitary_rep)
    qc = circuit.get_qiskit_circuit()
    full_u = Operator(qc).data

    num_ancilla = qc.num_qubits - 1 - num_target
    block = extract_block(full_u, num_target, num_ancilla)

    lam = unitary_rep.get_container().lambda_
    expected = ham.to_matrix() / lam

    ok = np.allclose(block, expected, atol=1e-8)
    max_err = np.max(np.abs(block - expected))
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}: lambda={lam:.4f} max_err={max_err:.2e} "
        f"(qubits={qc.num_qubits}, target={num_target}, ancilla={num_ancilla})"
    )
    if not ok:
        print("  block[0,:4]   =", np.round(block[0, :4], 4))
        print("  expected[0,:4]=", np.round(expected[0, :4], 4))
    return ok


if __name__ == "__main__":
    results = []
    # Signed transverse-field Ising, L=3: ZZ chain + negative X field.
    results.append(
        check(
            "Ising L=3 (+ZZ, -X)",
            ["ZZI", "IZZ", "XII", "IXI", "IIX"],
            [0.5, 0.5, -0.3, -0.3, -0.3],
        )
    )
    # Ising L=3, all positive.
    results.append(
        check(
            "Ising L=3 (+ZZ, +X)",
            ["ZZI", "IZZ", "XII", "IXI", "IIX"],
            [0.7, 0.7, 0.4, 0.4, 0.4],
        )
    )
    # Heisenberg L=3: XX+YY+ZZ chain (tests Y families).
    results.append(
        check(
            "Heisenberg L=3 (XX+YY+ZZ)",
            ["XXI", "IXX", "YYI", "IYY", "ZZI", "IZZ"],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
    )
    # Anisotropic Heisenberg + fields, signed.
    results.append(
        check(
            "Heisenberg L=3 aniso + fields",
            ["XXI", "IXX", "YYI", "IYY", "ZZI", "IZZ", "ZII", "IZI", "IIZ"],
            [0.3, 0.3, -0.2, -0.2, 0.5, 0.5, -0.1, -0.1, -0.1],
        )
    )
    print()
    print("ALL PASS" if all(results) else "SOME FAILED")
