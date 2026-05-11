"""Internal qubit tapering utilities for symmetry-conserving encodings.

Provides functions that post-process a :class:`~qdk_chemistry.data.QubitHamiltonian`
by projecting out qubits whose Z₂ eigenvalues are fixed by symmetry, reducing the
qubit count without loss of information within the symmetry sector.

These are internal utilities used by the QDK mapper when a
:class:`~qdk_chemistry.data.MajoranaMapping` carries a
:class:`~qdk_chemistry.data.TaperingSpecification`.  Users should prefer the
one-step API::

    from qdk_chemistry.algorithms import create
    from qdk_chemistry.data import MajoranaMapping, Symmetries

    mapper = create("qubit_mapper")
    mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(
        num_modes=n_spin_orbitals, symmetries=Symmetries(n_alpha=2, n_beta=2)
    )
    qh = mapper.run(hamiltonian, mapping)  # already tapered
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitHamiltonian, Symmetries

__all__ = ["taper_qubits", "taper_to_scbk"]


def taper_qubits(
    qubit_hamiltonian: QubitHamiltonian,
    qubit_indices: Sequence[int],
    eigenvalues: Sequence[int],
) -> QubitHamiltonian:
    """Remove qubits with known Z eigenvalues from a qubit Hamiltonian.

    For each specified qubit, every Pauli term that has Z on that qubit gets
    its coefficient multiplied by the eigenvalue, and the Z is replaced with I.
    Terms with X or Y on a tapered qubit are dropped (they connect different
    symmetry sectors). After replacement, the tapered qubit positions are
    removed from all strings and the remaining qubits are renumbered.

    This implements the qubit tapering step of symmetry-conserving encodings
    such as SCBK (arXiv:1701.08213).

    Args:
        qubit_hamiltonian (QubitHamiltonian): The qubit Hamiltonian to taper.
        qubit_indices (Sequence[int]): Qubit indices to taper (0-indexed).
        eigenvalues (Sequence[int]): Corresponding Z eigenvalues (+1 or -1) for each qubit.

    Returns:
        QubitHamiltonian: A new QubitHamiltonian with the specified qubits removed.

    Raises:
        ValueError: If lengths don't match, indices are out of range, contain duplicates, or eigenvalues are not ±1.

    """
    from qdk_chemistry.data import QubitHamiltonian  # noqa: PLC0415

    qubit_indices = list(qubit_indices)
    eigenvalues = list(eigenvalues)

    if len(qubit_indices) != len(eigenvalues):
        raise ValueError(
            f"qubit_indices length ({len(qubit_indices)}) must match eigenvalues length ({len(eigenvalues)})"
        )
    if len(set(qubit_indices)) != len(qubit_indices):
        raise ValueError("qubit_indices must not contain duplicates")

    nq = qubit_hamiltonian.num_qubits
    for q in qubit_indices:
        if q < 0 or q >= nq:
            raise ValueError(f"Qubit index {q} out of range [0, {nq})")
    for ev in eigenvalues:
        if ev not in (1, -1):
            raise ValueError(f"Eigenvalue must be +1 or -1, got {ev}")

    # String positions to remove (little-endian: qubit q is at position nq-1-q)
    positions_to_remove = sorted([nq - 1 - q for q in qubit_indices])
    eigenvalue_map = dict(zip(qubit_indices, eigenvalues, strict=True))

    new_strings: list[str] = []
    new_coeffs: list[complex] = []

    for pauli_str, coeff in zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True):
        skip = False
        adjusted_coeff = complex(coeff)

        for q, ev in eigenvalue_map.items():
            pos = nq - 1 - q
            char = pauli_str[pos]
            if char == "Z":
                adjusted_coeff *= ev
            elif char in ("X", "Y"):
                skip = True
                break
            # 'I' → no change

        if skip:
            continue

        # Remove the tapered qubit positions from the string
        chars = [c for i, c in enumerate(pauli_str) if i not in positions_to_remove]
        new_str = "".join(chars)

        new_strings.append(new_str)
        new_coeffs.append(adjusted_coeff)

    if not new_strings:
        raise ValueError("All Pauli terms were eliminated by tapering")

    # Merge duplicate Pauli strings
    merged: dict[str, complex] = {}
    for s, c in zip(new_strings, new_coeffs, strict=True):
        merged[s] = merged.get(s, 0.0) + c

    # Filter near-zero terms
    final_strings = []
    final_coeffs = []
    for s, c in merged.items():
        if abs(c) > np.finfo(np.float64).eps:
            final_strings.append(s)
            final_coeffs.append(c)

    if not final_strings:
        raise ValueError("All Pauli terms cancelled after tapering")

    return QubitHamiltonian(
        pauli_strings=final_strings,
        coefficients=np.array(final_coeffs),
        encoding=qubit_hamiltonian.encoding,
        fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
    )


def taper_to_scbk(
    qubit_hamiltonian: QubitHamiltonian,
    symmetries: Symmetries,
) -> QubitHamiltonian:
    """Apply symmetry-conserving Bravyi-Kitaev tapering to a BK-encoded Hamiltonian.

    Tapers the two Z₂ symmetry qubits of the Bravyi-Kitaev encoding — total
    electron-number parity (qubit n-1) and alpha-spin parity (qubit n/2-1) —
    reducing the qubit count by 2.  The eigenvalues are determined by the
    electron counts following the convention of arXiv:1701.08213.

    The input must be a BK-encoded QubitHamiltonian with an even number of
    qubits (= 2 * n_spatial).

    Args:
        qubit_hamiltonian (QubitHamiltonian): A Bravyi-Kitaev encoded qubit Hamiltonian.
        symmetries (Symmetries): Symmetry information providing ``n_alpha`` and ``n_beta`` electron counts.

    Returns:
        QubitHamiltonian: Tapered Hamiltonian with 2 fewer qubits and encoding ``"symmetry-conserving-bravyi-kitaev"``.

    Raises:
        ValueError: If encoding is not ``"bravyi-kitaev"``, or qubit count is odd or < 4.

    """
    from qdk_chemistry.data import QubitHamiltonian  # noqa: PLC0415
    from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder  # noqa: PLC0415

    if qubit_hamiltonian.encoding != "bravyi-kitaev":
        raise ValueError(
            f"taper_to_scbk requires a Bravyi-Kitaev encoded QubitHamiltonian "
            f"(encoding='bravyi-kitaev'), got encoding={qubit_hamiltonian.encoding!r}. "
            f"Use MajoranaMapping.bravyi_kitaev() to produce a BK-encoded Hamiltonian first."
        )

    if (
        qubit_hamiltonian.fermion_mode_order is not None
        and qubit_hamiltonian.fermion_mode_order != FermionModeOrder.BLOCKED
    ):
        raise ValueError(
            f"taper_to_scbk requires blocked spin-orbital ordering "
            f"(fermion_mode_order='blocked'), got {qubit_hamiltonian.fermion_mode_order!r}. "
            f"The QDK and OpenFermion BK mappers produce blocked ordering by default."
        )

    n = qubit_hamiltonian.num_qubits
    if n < 4 or n % 2 != 0:
        raise ValueError(f"SCBK tapering requires an even qubit count >= 4, got {n}")

    # In blocked BK ordering [α₀,α₁,…,αₙ₋₁,β₀,β₁,…,βₙ₋₁]:
    #   qubit n/2-1 stores alpha electron number parity
    #   qubit n-1   stores total electron number parity
    ev_total = 1 if (symmetries.n_alpha + symmetries.n_beta) % 2 == 0 else -1
    ev_alpha = 1 if symmetries.n_alpha % 2 == 0 else -1

    q_total = n - 1
    q_alpha = n // 2 - 1

    result = taper_qubits(qubit_hamiltonian, [q_alpha, q_total], [ev_alpha, ev_total])
    return QubitHamiltonian(
        pauli_strings=result.pauli_strings,
        coefficients=result.coefficients,
        encoding="symmetry-conserving-bravyi-kitaev",
        fermion_mode_order=result.fermion_mode_order,
        term_partition=result.term_partition,
    )
