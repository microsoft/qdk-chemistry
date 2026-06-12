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

from qdk_chemistry._core.data import majorana_map_hamiltonian, sparse_pauli_word_to_label
from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import MajoranaMapping
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import Hamiltonian

__all__ = ["VerstraeteCiracQubitMapper", "build_vc_majorana_mapping"]


def build_vc_majorana_mapping(n_modes: int) -> MajoranaMapping:
    """Return a MajoranaMapping for the Verstraete-Cirac encoding.

    Works for any 2D lattice geometry with n_modes fermionic modes.
    Introduces n_modes auxiliary qubits (one per physical mode), giving
    2 * n_modes qubits total.

    Qubit layout::

        Qubits 0..n_modes-1          physical modes (any site ordering)
        Qubits n_modes..2*n_modes-1  auxiliary (qubit n_modes+n for site n)

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
        n_modes: Total number of fermionic modes. Must be >= 1.

    Returns:
        A MajoranaMapping on 2 * n_modes qubits named "verstraete-cirac".

    Raises:
        ValueError: If n_modes < 1.

    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
    table = []
    for n in range(n_modes):
        z_string = [(k, 3) for k in range(n)]
        table.append([*z_string, (n, 1), (n_modes + n, 3)])
        table.append([*z_string, (n, 2), (n_modes + n, 3)])
    return MajoranaMapping.from_table(table, name="verstraete-cirac")


class VerstraeteCiracQubitMapper(QubitMapper):
    """Fermion-to-qubit mapper using the Verstraete-Cirac encoding.

    For an n_rows x n_cols lattice with N = 2 * n_rows * n_cols modes
    (covering both spin-orbital blocks expected by the C++ engine), this
    mapper builds a MajoranaMapping via ``build_vc_majorana_mapping(N)``
    and delegates to the C++ Majorana-loop engine
    (``majorana_map_hamiltonian``) -- the same backend used by
    :class:`QdkQubitMapper` for Jordan-Wigner and Bravyi-Kitaev. The
    resulting QubitHamiltonian acts on 2N = 4 * n_rows * n_cols qubits.

    Codespace: the +1 eigenspace of the auxiliary Z operators
    (Z_{an} = +1 for all n, auxiliary qubits in state |0>).
    Restricting to this 2N/2-qubit sector recovers the Jordan-Wigner
    Hamiltonian for the (decoupled alpha + beta) spin-orbital blocks.

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
        n_sites = lattice_shape[0] * lattice_shape[1]
        self._mapping = build_vc_majorana_mapping(2 * n_sites)
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
        mapping: MajoranaMapping,
    ) -> QubitHamiltonian:
        """Map a fermionic Hamiltonian to a VC qubit Hamiltonian.

        Delegates to the generic C++ Majorana-loop engine
        (``majorana_map_hamiltonian``), passing the VC
        ``MajoranaMapping`` produced by ``build_vc_majorana_mapping``.
        This automatically handles one-body integrals, two-body
        integrals, and core energy via the standard fermion-to-Majorana
        substitution -- the same engine used by ``QdkQubitMapper`` for
        Jordan-Wigner and Bravyi-Kitaev.

        Args:
            hamiltonian: Fermionic Hamiltonian. One-body integrals must
                be (N, N) with N = n_rows * n_cols (single spin species).
            mapping: The VC MajoranaMapping (typically ``self.mapping``),
                with ``num_modes == 2 * N``. May include tapering, which
                is applied to the result via ``_taper_result``.

        Returns:
            QubitHamiltonian with encoding "verstraete-cirac" on
            2 * mapping.num_modes = 4 * N qubits (before tapering).

        Raises:
            ValueError: On mode count mismatch.

        """
        Logger.trace_entering()

        base_mapping = mapping.without_tapering() if mapping.tapering else mapping
        threshold = self._threshold
        integral_threshold = self._integral_threshold

        h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()
        n_spatial = h1_alpha.shape[0]
        n_rows, n_cols = self._lattice_shape

        if n_spatial != n_rows * n_cols:
            raise ValueError(
                f"Hamiltonian has {n_spatial} modes but the {n_rows}x{n_cols} lattice has {n_rows * n_cols} sites."
            )

        if base_mapping.num_modes != 2 * n_spatial:
            raise ValueError(
                f"MajoranaMapping has {base_mapping.num_modes} modes but "
                f"the Hamiltonian has {n_spatial} spatial orbitals "
                f"({2 * n_spatial} spin-orbitals required)."
            )

        spin_symmetric = hamiltonian.get_orbitals().is_restricted()

        h1_a_flat = np.ascontiguousarray(h1_alpha).ravel()
        h1_b_flat = h1_a_flat if spin_symmetric else np.ascontiguousarray(h1_beta).ravel()
        h2_aaaa_flat = np.ascontiguousarray(h2_aaaa).ravel()
        h2_aabb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_aabb).ravel()
        h2_bbbb_flat = h2_aaaa_flat if spin_symmetric else np.ascontiguousarray(h2_bbbb).ravel()

        core_energy = float(hamiltonian.get_core_energy())

        words, coefficients = majorana_map_hamiltonian(
            base_mapping,
            core_energy,
            h1_a_flat,
            h1_b_flat,
            h2_aaaa_flat,
            h2_aabb_flat,
            h2_bbbb_flat,
            n_spatial,
            spin_symmetric,
            threshold,
            integral_threshold,
        )

        n_qubits = base_mapping.num_qubits
        pauli_strings = [sparse_pauli_word_to_label(word, n_qubits) for word in words]

        if not pauli_strings:
            pauli_strings = ["I" * n_qubits]
            coefficients = [0.0 + 0j]

        Logger.debug(f"VC mapper: {len(pauli_strings)} Pauli terms on {n_qubits} qubits")

        qh = QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array(coefficients, dtype=complex),
            encoding="verstraete-cirac",
            fermion_mode_order=None,
        )
        return self._taper_result(qh, mapping)
