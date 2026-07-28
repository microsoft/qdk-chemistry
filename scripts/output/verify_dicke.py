"""Ad-hoc verification of DickeStatePreparation producing |D^n_1>."""

import numpy as np
from qdk import TargetProfile, qsharp
from qdk_chemistry.algorithms import registry
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS


def build_dicke1_wavefunction(n: int) -> Wavefunction:
    """Build a |D^n_1> wavefunction: uniform superposition of one-hot bitstrings."""
    dets = []
    for j in range(n):
        bits = ["0"] * n
        bits[j] = "1"
        dets.append(Configuration.from_bitstring("".join(bits)))
    coeffs = np.array([1.0 / np.sqrt(n)] * n)
    container = StateVectorContainer(coeffs, dets, ModelOrbitals(n))
    return Wavefunction(container)


def dump_dicke(n: int) -> np.ndarray:
    """Simulate the Dicke Q# op on n qubits and return the dense statevector."""
    current_profile = (
        qsharp.get_config().get_target_profile()
        if hasattr(qsharp, "get_config")
        else "Unrestricted"
    )
    try:
        qsharp.init(target_profile=TargetProfile.from_str(current_profile))
    except Exception:
        qsharp.init(target_profile=TargetProfile.Unrestricted)
    _ = QSHARP_UTILS.Dicke
    qsharp.eval(f"use qs = Qubit[{n}];")
    qsharp.eval(
        f"QDKChemistry.Utils.Dicke.PrepareDicke("
        f"new QDKChemistry.Utils.Dicke.DickeParams {{ numQubits = {n}, weight = 1 }}, qs);"
    )
    state = qsharp.dump_machine()
    return np.array(state.as_dense_state())


def main() -> None:
    n = 4
    wf = build_dicke1_wavefunction(n)
    prep = registry.create("state_prep", "dicke")
    circuit = prep.run(wf)
    print("Circuit built:", circuit is not None, "op:", circuit._qsharp_op is not None)

    sv = dump_dicke(n)
    weights = [bin(i).count("1") for i in range(len(sv))]
    support = {i: sv[i] for i in range(len(sv)) if abs(sv[i]) > 1e-9}
    print("Support indices (weight):", {i: weights[i] for i in support})
    mags = np.array([abs(v) for v in support.values()])
    print("magnitudes:", mags)
    all_weight1 = all(weights[i] == 1 for i in support)
    equal_mag = np.allclose(mags, mags[0]) if len(mags) else False
    full_support = len(support) == n
    print(
        f"all weight-1: {all_weight1}, equal magnitude: {equal_mag}, full support (n={n}): {full_support}"
    )
    assert all_weight1 and equal_mag and full_support, (
        "Dicke |D^n_1> verification FAILED"
    )
    print("PASS: DickeStatePreparation produces |D^4_1>")


if __name__ == "__main__":
    main()
