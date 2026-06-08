"""Verstraete-Cirac Fermion-to-Qubit Encoding for 2D lattice Hamiltonians.

References
----------
[VC2005]  F. Verstraete and J. I. Cirac, J. Stat. Mech. (2005) P09012.
[WHT2016] J. D. Whitfield, V. Havlicek, M. Troyer, Phys. Rev. A 94, 030301(R) (2016).
[HTW2017] V. Havlicek, M. Troyer, J. D. Whitfield, Phys. Rev. A 95, 032332 (2017).
"""

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import MajoranaMapping
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian, Symmetries

__all__ = ["VerstraeteCiracQubitMapper", "build_vc_majorana_mapping"]


def _pauli_str(n_qubits: int, ops: dict) -> str:
    """Build a Pauli string in QDK/Chemistry little-endian convention.

    String position 0 (leftmost) = qubit n_qubits-1 (most significant).
    String position -1 (rightmost) = qubit 0 (least significant).
    So qubit index qi maps to string position n_qubits - 1 - qi.
    This matches the convention used by QubitHamiltonian.to_matrix().
    """
    chars = ["I"] * n_qubits
    for qi, op in ops.items():
        chars[n_qubits - 1 - qi] = op
    return "".join(chars)


def build_vc_majorana_mapping(n_rows: int, n_cols: int) -> MajoranaMapping:
    """Return a MajoranaMapping for the Verstraete-Cirac encoding.

    For N = n_rows * n_cols physical modes, uses N auxiliary qubits.

    Qubit layout:
        Qubits 0..N-1     -- physical (row-major)
        Qubits N..2N-1    -- auxiliary (qubit N+n for site n)

    Majorana operators (JW on physical + stabilising Z on auxiliary):
        Gamma_{2n}   = Z_{p0}...Z_{p_{n-1}} X_{pn} . Z_{an}
        Gamma_{2n+1} = Z_{p0}...Z_{p_{n-1}} Y_{pn} . Z_{an}

    Satisfies {Gamma_a, Gamma_b} = 2 delta_{ab} exactly.
    Codespace: all Z_{an} = +1 (auxiliary qubits in |0>).
    In codespace: equals the Jordan-Wigner Hamiltonian on physical qubits.
    """
    if n_rows < 2 or n_cols < 2:
        raise ValueError(
            f"VC encoding requires at least a 2x2 lattice, got {n_rows}x{n_cols}."
        )
    N = n_rows * n_cols
    table = []
    for n in range(N):
        z_string = [(k, 3) for k in range(n)]          # Z on qubits 0..n-1
        table.append(z_string + [(n, 1), (N + n, 3)])  # gamma_{2n}  = ZZ..ZX.Za
        table.append(z_string + [(n, 2), (N + n, 3)])  # gamma_{2n+1}= ZZ..ZY.Za
    return MajoranaMapping.from_table(table, name="verstraete-cirac")


class VerstraeteCiracQubitMapper(QubitMapper):
    """Fermion-to-qubit mapper using the Verstraete-Cirac encoding.

    One auxiliary qubit per physical mode on a 2-D open square lattice.
    Uses Jordan-Wigner on physical qubits + auxiliary Z gate per site.
    Correct fermionic anticommutation is guaranteed in the full 2N-qubit space.
    Physical eigenvalues live in the codespace where all auxiliary qubits are |0>.
    """

    def __init__(self, lattice_shape, threshold=1e-12, integral_threshold=1e-12):
        super().__init__()
        if lattice_shape[0] < 2 or lattice_shape[1] < 2:
            raise ValueError(
                f"VC encoding requires at least a 2x2 lattice, "
                f"got {lattice_shape[0]}x{lattice_shape[1]}."
            )
        self._lattice_shape = lattice_shape
        self._mapping = build_vc_majorana_mapping(*lattice_shape)
        self._threshold = float(threshold)
        self._integral_threshold = float(integral_threshold)

    @property
    def lattice_shape(self):
        return self._lattice_shape

    @property
    def mapping(self):
        return self._mapping

    def name(self):
        return "verstraete-cirac"

    def _run_impl(self, hamiltonian: "Hamiltonian", mapping: "MajoranaMapping | None" = None) -> QubitHamiltonian:
        """Build the VC qubit Hamiltonian.

        For VC Majorana Gamma_{2n} = JW_{2n} . Z_{an}:

            h_{nm}(a†_n a_m + h.c.)
              = h/2 * (Xn Z_{n+1}..Z_{m-1} Xm + Yn Z_{n+1}..Z_{m-1} Ym) * Z_{an} Z_{am}

        In codespace (Z_{an}=+1) this equals the JW Hamiltonian exactly.
        """
        Logger.trace_entering()
        threshold = self._threshold
        integral_threshold = self._integral_threshold

        if mapping is not None and mapping is not self._mapping:
            raise ValueError(
                "The supplied mapping does not match the mapper's internal "
                "VC mapping. Pass mapper.mapping or omit the argument."
            )
        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        import numpy as _np
        if h1_beta is not None and not _np.allclose(
            h1_beta, h1_alpha, atol=self._integral_threshold
        ):
            raise ValueError(
                "VerstraeteCiracQubitMapper only supports restricted "
                "(spin-symmetric) one-body Hamiltonians. A non-trivial "
                "beta-spin channel was detected."
            )
        n_sites = h1_alpha.shape[0]
        n_rows, n_cols = self._lattice_shape

        if n_sites != n_rows * n_cols:
            raise ValueError(
                f"Hamiltonian has {n_sites} modes but the "
                f"{n_rows}x{n_cols} lattice has {n_rows * n_cols} sites."
            )

        N = n_sites
        n_qubits = 2 * N
        identity = "I" * n_qubits
        pauli_strs = []
        coeffs = []

        def _add(ops, coeff):
            if abs(coeff) < threshold:
                return
            pauli_strs.append(_pauli_str(n_qubits, ops))
            coeffs.append(coeff)

        # Diagonal: h_{nn} * n_n = h_{nn}/2 * (I - Z_{pn})
        const = 0.0
        for n in range(N):
            h_nn = float(h1_alpha[n, n].real)
            if abs(h_nn) < integral_threshold:
                continue
            const += h_nn / 2.0
            _add({n: "Z"}, complex(-h_nn / 2.0))
        if abs(const) >= threshold:
            pauli_strs.append(identity)
            coeffs.append(complex(const))

        # Off-diagonal hopping with JW Z-string + auxiliary Z's
        for n in range(N):
            for m in range(n + 1, N):
                h_nm = (complex(h1_alpha[n, m]) + complex(h1_alpha[m, n])) / 2.0
                if abs(h_nm) < integral_threshold:
                    continue
                scale = h_nm / 2.0
                # JW Z-string between n and m on physical qubits
                z_mid = {k: "Z" for k in range(n + 1, m)}
                # XX term: X_n Z..Z X_m * Z_{an} Z_{am}
                _add({n: "X", m: "X", N + n: "Z", N + m: "Z"} | z_mid, scale)
                # YY term: Y_n Z..Z Y_m * Z_{an} Z_{am}
                _add({n: "Y", m: "Y", N + n: "Z", N + m: "Z"} | z_mid, scale)

        if not pauli_strs:
            pauli_strs = [identity]
            coeffs = [0.0 + 0j]

        Logger.debug(f"VC mapper: {len(pauli_strs)} Pauli terms on {n_qubits} qubits")

        return QubitHamiltonian(
            pauli_strings=pauli_strs,
            coefficients=np.array(coeffs, dtype=complex),
            encoding="verstraete-cirac",
            fermion_mode_order=None,
        )
