"""Conversion utilities for QDK Chemistry to OpenFermion interoperability.

This module provides functions to convert between QDK/Chemistry data structures
and OpenFermion operator representations:

- Hamiltonian to InteractionOperator / FermionOperator
- QubitOperator to/from QubitHamiltonian
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import openfermion as of

from qdk_chemistry import data
from qdk_chemistry.utils import Logger

__all__ = [
    "hamiltonian_to_fermion_operator",
    "hamiltonian_to_interaction_operator",
    "qubit_hamiltonian_to_qubit_operator",
    "qubit_operator_to_qubit_hamiltonian",
]


def _spatial_to_spinorb_one_body(
    h1_alpha: np.ndarray,
    h1_beta: np.ndarray,
) -> np.ndarray:
    """Build the full spin-orbital one-body integrals from spatial MO integrals.

    Uses interleaved spin-orbital ordering (OpenFermion convention):
    spin-orbital 2p = spatial orbital p alpha, 2p+1 = spatial orbital p beta.

    Args:
        h1_alpha: Alpha one-body integrals, shape ``(norb, norb)``.
        h1_beta: Beta one-body integrals, shape ``(norb, norb)``.

    Returns:
        numpy.ndarray: One-body spin-orbital integrals, shape ``(2*norb, 2*norb)``.

    """
    norb = h1_alpha.shape[0]
    n_spinorb = 2 * norb
    h1_so = np.zeros((n_spinorb, n_spinorb), dtype=float)

    for p in range(norb):
        for q in range(norb):
            # Alpha-alpha block (even indices)
            h1_so[2 * p, 2 * q] = h1_alpha[p, q]
            # Beta-beta block (odd indices)
            h1_so[2 * p + 1, 2 * q + 1] = h1_beta[p, q]

    return h1_so


def _spatial_to_spinorb_two_body(
    h2_aaaa: np.ndarray,
    h2_aabb: np.ndarray,
    h2_bbbb: np.ndarray,
) -> np.ndarray:
    """Build the full spin-orbital two-body integrals from spatial MO integrals.

    Converts from QDK chemist notation ``(pq|rs)`` (Mulliken) to the tensor
    layout expected by OpenFermion's ``InteractionOperator``, using interleaved
    spin-orbital ordering.

    OpenFermion defines ``h[p,q,r,s]`` via
    ``H₂ = ½ Σ h[p,q,r,s] a†_p a†_q a_r a_s``, which in physicist (Dirac)
    notation is ``⟨pq|sr⟩``, or equivalently ``(ps|rq)`` in chemist notation.

    Args:
        h2_aaaa: Alpha-alpha two-body integrals in chemist notation, shape ``(norb, norb, norb, norb)``.
        h2_aabb: Alpha-beta two-body integrals in chemist notation, shape ``(norb, norb, norb, norb)``.
        h2_bbbb: Beta-beta two-body integrals in chemist notation, shape ``(norb, norb, norb, norb)``.

    Returns:
        numpy.ndarray: Two-body spin-orbital integrals in physicist notation,
            shape ``(2*norb, 2*norb, 2*norb, 2*norb)``.

    """
    norb = h2_aaaa.shape[0]
    n_spinorb = 2 * norb

    # Convert chemist notation (pq|rs) to physicist notation <pr|qs>
    # OpenFermion InteractionOperator convention: h[p,q,r,s] for a†_p a†_q a_r a_s
    # which is <pq|sr> in physicist notation = (ps|rq) in chemist notation
    h2_so = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb), dtype=float)

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    # Chemist (pq|rs) → InteractionOperator h[p,q,r,s] = (ps|rq) in chemist
                    # Or equivalently, we can use spinorb_from_spatial which does the standard mapping

                    # alpha-alpha: spin-orbitals 2p, 2q, 2r, 2s
                    h2_so[2 * p, 2 * q, 2 * r, 2 * s] = h2_aaaa[p, s, r, q]

                    # beta-beta: spin-orbitals 2p+1, 2q+1, 2r+1, 2s+1
                    h2_so[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = h2_bbbb[p, s, r, q]

                    # alpha-beta: a†_(2p) a†_(2q+1) a_(2r+1) a_(2s)
                    h2_so[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = h2_aabb[p, s, r, q]

                    # beta-alpha: a†_(2p+1) a†_(2q) a_(2r) a_(2s+1)
                    h2_so[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = h2_aabb[q, r, s, p]

    return h2_so


def hamiltonian_to_interaction_operator(
    hamiltonian: data.Hamiltonian,
) -> "of.InteractionOperator":
    """Convert a QDK/Chemistry Hamiltonian to an OpenFermion InteractionOperator.

    Handles both restricted (RHF) and unrestricted (UHF) Hamiltonians.
    For restricted Hamiltonians, uses ``openfermion.chem.molecular_data.spinorb_from_spatial``
    for efficient conversion. For unrestricted Hamiltonians, explicitly constructs spin-orbital
    integrals from all spin channels.

    The resulting operator uses OpenFermion's interleaved spin-orbital ordering:
    spin-orbital ``2p`` = spatial orbital ``p`` alpha, ``2p+1`` = spatial orbital ``p`` beta.

    Args:
        hamiltonian: The QDK/Chemistry Hamiltonian to convert.

    Returns:
        openfermion.InteractionOperator: The electronic Hamiltonian as an InteractionOperator.

    Examples:
        >>> from qdk_chemistry.plugins.openfermion.conversion import hamiltonian_to_interaction_operator
        >>> iop = hamiltonian_to_interaction_operator(hamiltonian)
        >>> print(f"Number of spin-orbitals: {iop.n_qubits}")

    """
    Logger.trace_entering()

    h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()
    h2_aaaa_flat, h2_aabb_flat, h2_bbbb_flat = hamiltonian.get_two_body_integrals()
    core_energy = hamiltonian.get_core_energy()

    norb = h1_alpha.shape[0]
    h2_aaaa = h2_aaaa_flat.reshape((norb, norb, norb, norb))

    if hamiltonian.is_restricted():
        Logger.debug("Using restricted (RHF) integral conversion via spinorb_from_spatial.")
        # Convert chemist notation (pq|rs) to physicist notation <pr|qs>
        h2_phys = np.transpose(h2_aaaa, (0, 2, 3, 1))
        one_body_so, two_body_so = of.chem.molecular_data.spinorb_from_spatial(h1_alpha, h2_phys)
        return of.InteractionOperator(core_energy, one_body_so, 0.5 * two_body_so)
    Logger.debug("Using unrestricted (UHF) integral conversion with explicit spin channels.")
    h2_aabb = h2_aabb_flat.reshape((norb, norb, norb, norb))
    h2_bbbb = h2_bbbb_flat.reshape((norb, norb, norb, norb))

    one_body_so = _spatial_to_spinorb_one_body(h1_alpha, h1_beta)
    two_body_so = _spatial_to_spinorb_two_body(h2_aaaa, h2_aabb, h2_bbbb)

    return of.InteractionOperator(core_energy, one_body_so, 0.5 * two_body_so)


def hamiltonian_to_fermion_operator(
    hamiltonian: data.Hamiltonian,
) -> "of.FermionOperator":
    """Convert a QDK/Chemistry Hamiltonian to an OpenFermion FermionOperator.

    This is a convenience function that first creates an ``InteractionOperator``
    and then converts it to a ``FermionOperator`` using
    ``openfermion.transforms.get_fermion_operator``.

    Args:
        hamiltonian: The QDK/Chemistry Hamiltonian to convert.

    Returns:
        openfermion.FermionOperator: The electronic Hamiltonian as a FermionOperator.

    Examples:
        >>> from qdk_chemistry.plugins.openfermion.conversion import hamiltonian_to_fermion_operator
        >>> fop = hamiltonian_to_fermion_operator(hamiltonian)
        >>> print(f"Number of terms: {len(fop.terms)}")

    """
    Logger.trace_entering()
    iop = hamiltonian_to_interaction_operator(hamiltonian)
    return of.transforms.get_fermion_operator(iop)


def qubit_operator_to_qubit_hamiltonian(
    qubit_op: "of.QubitOperator",
    encoding: str | None = None,
) -> data.QubitHamiltonian:
    """Convert an OpenFermion QubitOperator to a QDK/Chemistry QubitHamiltonian.

    Translates OpenFermion's Pauli term format (e.g., ``((0, 'X'), (1, 'Z'))``)
    to the dense Pauli string format used by QDK/Chemistry (e.g., ``"XZI..."``).

    Args:
        qubit_op: The OpenFermion QubitOperator to convert.
        encoding: Optional encoding label (e.g., ``"jordan-wigner"``) to attach to the resulting QubitHamiltonian.

    Returns:
        QubitHamiltonian: A QDK/Chemistry QubitHamiltonian.

    Raises:
        ValueError: If the QubitOperator has no terms.

    Examples:
        >>> import openfermion as of
        >>> qop = of.QubitOperator("X0 Z1", 0.5) + of.QubitOperator("Y0 Y1", 0.3)
        >>> qh = qubit_operator_to_qubit_hamiltonian(qop, encoding="jordan-wigner")

    """
    Logger.trace_entering()

    # After compression, a zero operator may have no terms or only a zero identity
    epsilon = np.finfo(np.float64).eps
    non_zero_terms = {k: v for k, v in qubit_op.terms.items() if abs(v) > epsilon}
    if not non_zero_terms:
        msg = "QubitOperator is empty (no non-zero terms)."
        raise ValueError(msg)

    # Determine the number of qubits from the highest qubit index
    n_qubits = 0
    for term in qubit_op.terms:
        if term:  # Non-identity term
            max_idx = max(idx for idx, _ in term)
            n_qubits = max(n_qubits, max_idx + 1)

    pauli_strings = []
    coefficients = []

    for term, coeff in qubit_op.terms.items():
        # Build dense Pauli string in Qiskit/QDK little-endian convention
        # QDK uses little-endian: qubit 0 is the rightmost character
        pauli_list = ["I"] * n_qubits
        for qubit_idx, pauli_label in term:
            pauli_list[qubit_idx] = pauli_label
        # Reverse for little-endian (qubit 0 at rightmost position)
        pauli_str = "".join(reversed(pauli_list))
        pauli_strings.append(pauli_str)
        coefficients.append(coeff)

    return data.QubitHamiltonian(
        pauli_strings=pauli_strings,
        coefficients=np.array(coefficients, dtype=complex),
        encoding=encoding,
    )


def qubit_hamiltonian_to_qubit_operator(
    qubit_hamiltonian: data.QubitHamiltonian,
) -> "of.QubitOperator":
    """Convert a QDK/Chemistry QubitHamiltonian to an OpenFermion QubitOperator.

    Translates the dense Pauli string format (e.g., ``"XZII"``) to OpenFermion's
    sparse tuple format (e.g., ``((0, 'X'), (1, 'Z'))``).

    Args:
        qubit_hamiltonian: The QDK/Chemistry QubitHamiltonian to convert.

    Returns:
        openfermion.QubitOperator: The equivalent OpenFermion QubitOperator.

    Examples:
        >>> qop = qubit_hamiltonian_to_qubit_operator(qubit_hamiltonian)
        >>> print(qop)

    """
    Logger.trace_entering()

    qubit_op = of.QubitOperator()

    for pauli_str, coeff in zip(
        qubit_hamiltonian.pauli_strings,
        qubit_hamiltonian.coefficients,
        strict=True,
    ):
        # QDK uses little-endian: qubit 0 is the rightmost character
        # Reverse to get qubit 0 at index 0
        reversed_str = pauli_str[::-1]
        term_tuples = []
        for qubit_idx, pauli_char in enumerate(reversed_str):
            if pauli_char != "I":
                term_tuples.append((qubit_idx, pauli_char))
        qubit_op += of.QubitOperator(tuple(term_tuples), coeff)

    return qubit_op
