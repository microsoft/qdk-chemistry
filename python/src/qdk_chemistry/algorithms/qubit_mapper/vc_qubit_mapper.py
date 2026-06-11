"""Verstraete-Cirac Fermion-to-Qubit Encoding for 2D lattice Hamiltonians.

References
----------
[VC2005]  F. Verstraete and J. I. Cirac, J. Stat. Mech. (2005) P09012.
          arXiv:cond-mat/0508353
[WHT2016] J. D. Whitfield, V. Havlicek, M. Troyer, Phys. Rev. A 94, 030301(R) (2016).
[HTW2017] V. Havlicek, M. Troyer, J. D. Whitfield, Phys. Rev. A 95, 032332 (2017).

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import MajoranaMapping
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian

__all__ = ["VerstraeteCiracQubitMapper", "build_vc_majorana_mapping"]


def _pauli_str(n_qubits: int, ops: dict) -> str:
    """Build a Pauli string in QDK/Chemistry little-endian convention.

    String position 0 (leftmost) = qubit n_qubits-1 (most significant).
    String position -1 (rightmost) = qubit 0 (least significant).
    Qubit index qi maps to string position n_qubits - 1 - qi.
    This matches the convention used by QubitHamiltonian.to_matrix().
    """
    chars = ["I"] * n_qubits
    for qi, op in ops.items():
        chars[n_qubits - 1 - qi] = op
    return "".join(chars)


def build_vc_majorana_mapping(n_sites: int) -> MajoranaMapping:
    """Return a MajoranaMapping for the Verstraete-Cirac encoding.

    Works for any 2D lattice geometry with n_sites fermionic modes.
    Introduces n_sites auxiliary qubits (one per physical mode), giving
    2 * n_sites qubits total.

    Qubit layout::

        Qubits 0..n_sites-1          physical modes (any site ordering)
        Qubits n_sites..2*n_sites-1  auxiliary (qubit n_sites+n for site n)

    Majorana operators (JW on physical + stabilising Z on auxiliary)::

        Gamma_{2n}   = Z_{p0}...Z_{p_{n-1}} X_{pn} . Z_{an}
        Gamma_{2n+1} = Z_{p0}...Z_{p_{n-1}} Y_{pn} . Z_{an}

    These satisfy the Clifford algebra {Gamma_a, Gamma_b} = 2 delta_{ab}
    exactly in the full 2*n_sites-qubit Hilbert space.

    Codespace: the +1 eigenspace of the auxiliary Z operators,
    i.e. all auxiliary qubits in state |0> (Z_{an} = +1 for all n).
    In this sector the mapped Hamiltonian equals the Jordan-Wigner form.
    The general VC stabiliser structure P_{j,j} = i * Gamma_{2j} * Gamma_{2j+1}
    (arXiv:cond-mat/0508353) is a planned follow-up.

    Args:
        n_sites: Total number of fermionic modes. Must be >= 1.

    Returns:
        A MajoranaMapping on 2 * n_sites qubits named "verstraete-cirac".

    Raises:
        ValueError: If n_sites < 1.

    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be >= 1, got {n_sites}.")
    n_modes = n_sites
    table = []
    for n in range(n_modes):
        z_string = [(k, 3) for k in range(n)]
        table.append([*z_string, (n, 1), (n_modes + n, 3)])
        table.append([*z_string, (n, 2), (n_modes + n, 3)])
    return MajoranaMapping.from_table(table, name="verstraete-cirac")


class VerstraeteCiracQubitMapper(QubitMapper):
    """Fermion-to-qubit mapper using the Verstraete-Cirac encoding.

    Introduces one auxiliary qubit per physical mode on a 2-D lattice.
    The Majorana operators use Jordan-Wigner on physical qubits combined
    with a stabilising Z gate on the corresponding auxiliary qubit.
    Correct fermionic anticommutation {Gamma_a, Gamma_b} = 2 delta_{ab}
    is guaranteed in the full 2N-qubit Hilbert space.

    Codespace: the +1 eigenspace of the auxiliary Z operators
    (Z_{an} = +1 for all n, auxiliary qubits in state |0>).
    Physical eigenvalues are recovered by restricting to this sector.

    Note: This is a Python-level prototype. A C++ backend implementation
    following the pattern of QdkQubitMapper is a planned follow-up.

    Args:
        lattice_shape: (n_rows, n_cols) of the open 2-D lattice.
            Both dimensions must be >= 2.
        threshold: Drop Pauli terms with |coeff| < threshold.
        integral_threshold: Skip one-body integrals below this value.

    """

    def __init__(
        self,
        lattice_shape: tuple[int, int],
        threshold: float = 1e-12,
        integral_threshold: float = 1e-12,
    ) -> None:
        """Initialise the mapper for the given lattice shape."""
        super().__init__()
        if lattice_shape[0] < 2 or lattice_shape[1] < 2:
            raise ValueError(f"VC encoding requires at least a 2x2 lattice, got {lattice_shape[0]}x{lattice_shape[1]}.")
        self._lattice_shape = lattice_shape
        self._mapping = build_vc_majorana_mapping(lattice_shape[0] * lattice_shape[1])
        self._threshold = float(threshold)
        self._integral_threshold = float(integral_threshold)

    @property
    def lattice_shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) of the lattice."""
        return self._lattice_shape

    @property
    def mapping(self) -> MajoranaMapping:
        """Return the underlying MajoranaMapping."""
        return self._mapping

    def name(self) -> str:
        """Return the encoding name."""
        return "verstraete-cirac"

    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping | None = None,
    ) -> QubitHamiltonian:
        """Build the VC qubit Hamiltonian.

        For VC Majorana Gamma_{2n} = JW_{2n} . Z_{an}:

            h_{nm}(a†_n a_m + h.c.)
              = h/2 * (Xn Z_{n+1}..Z_{m-1} Xm + Yn Z_{n+1}..Z_{m-1} Ym)
                    * Z_{an} Z_{am}

        In codespace (Z_{an}=+1) this equals the JW Hamiltonian exactly.

        Args:
            hamiltonian: Fermionic Hamiltonian. Alpha one-body integrals
                must be (N, N) with N = n_rows * n_cols. Only restricted
                real-valued one-body Hamiltonians are supported.
            mapping: Optional mapping for API compatibility. Must match
                self.mapping if supplied.

        Returns:
            QubitHamiltonian on 2N qubits with encoding "verstraete-cirac".

        Raises:
            ValueError: On mapping mismatch, beta-spin channel, complex
                hoppings, or mode count mismatch.

        """
        Logger.trace_entering()

        if mapping is not None and (
            mapping.name != self._mapping.name or mapping.num_qubits != self._mapping.num_qubits
        ):
            raise ValueError(
                f"Supplied mapping '{mapping.name}' "
                f"({mapping.num_qubits} qubits) does not match "
                f"the internal VC mapping '{self._mapping.name}' "
                f"({self._mapping.num_qubits} qubits)."
            )

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()

        if h1_beta is not None and not np.allclose(h1_beta, h1_alpha, atol=self._integral_threshold):
            raise ValueError(
                "VerstraeteCiracQubitMapper only supports restricted "
                "(spin-symmetric) one-body Hamiltonians. A non-trivial "
                "beta-spin channel was detected."
            )

        if not np.allclose(h1_alpha.imag, 0, atol=self._integral_threshold):
            raise ValueError(
                "VerstraeteCiracQubitMapper only supports real-valued "
                "hopping matrices. Complex off-diagonal elements such as "
                "Peierls phases are not currently supported."
            )
        if not np.allclose(h1_alpha.real, h1_alpha.real.T, atol=self._integral_threshold):
            raise ValueError(
                "h1_alpha must be symmetric (Hermitian for real matrices). "
                "The provided matrix is not close to its transpose."
            )

        n_sites = h1_alpha.shape[0]
        n_rows, n_cols = self._lattice_shape

        if n_sites != n_rows * n_cols:
            raise ValueError(
                f"Hamiltonian has {n_sites} modes but the {n_rows}x{n_cols} lattice has {n_rows * n_cols} sites."
            )

        n_modes = n_sites
        n_qubits = 2 * n_modes
        identity = "I" * n_qubits
        pauli_strs: list[str] = []
        coeffs: list[complex] = []

        threshold = self._threshold
        integral_threshold = self._integral_threshold

        def _add(ops: dict, coeff: complex) -> None:
            if abs(coeff) < threshold:
                return
            pauli_strs.append(_pauli_str(n_qubits, ops))
            coeffs.append(coeff)

        # Include constant/core energy contribution to the identity term
        try:
            core_energy = float(getattr(hamiltonian, "core_energy", 0.0) or 0.0)
        except (AttributeError, TypeError):
            core_energy = 0.0

        # Diagonal: h_{nn} * n_n = h_{nn}/2 * (I - Z_{pn})
        const = core_energy
        for n in range(n_modes):
            h_nn = float(h1_alpha[n, n].real)
            if abs(h_nn) < integral_threshold:
                continue
            const += h_nn / 2.0
            _add({n: "Z"}, complex(-h_nn / 2.0))

        if abs(const) >= threshold:
            pauli_strs.append(identity)
            coeffs.append(complex(const))

        # Off-diagonal hopping with JW Z-string + auxiliary Z's
        for n in range(n_modes):
            for m in range(n + 1, n_modes):
                h_nm = float(h1_alpha[n, m].real + h1_alpha[m, n].real) / 2.0
                if abs(h_nm) < integral_threshold:
                    continue
                scale = h_nm / 2.0
                z_mid = dict.fromkeys(range(n + 1, m), "Z")
                _add({n: "X", m: "X", n_modes + n: "Z", n_modes + m: "Z"} | z_mid, scale)
                _add({n: "Y", m: "Y", n_modes + n: "Z", n_modes + m: "Z"} | z_mid, scale)

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
