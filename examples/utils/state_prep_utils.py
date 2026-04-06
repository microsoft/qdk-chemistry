"""Utility functions for state preparation."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import QubitHamiltonian, Wavefunction
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_matrix import pauli_string_to_masks


def pauli_expectation(pauli_str: str, psi: np.ndarray) -> float:
    """Compute the expectation value ``<psi|P|psi>`` for a single Pauli string.

    Args:
        pauli_str: Pauli label of length *n* (characters in {I, X, Y, Z}), using Little-Endian convention.
        psi: Complex state vector of length ``2**n``.

    Returns:
        Expectation value.

    """
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    expected_dim = 1 << len(pauli_str)
    if psi.size != expected_dim:
        raise ValueError(
            f"State vector length {psi.size} does not match the expected "
            f"dimension 2**{len(pauli_str)} = {expected_dim} for Pauli string '{pauli_str}'."
        )
    x_mask, z_mask, phase = pauli_string_to_masks(pauli_str)
    dim = psi.shape[0]

    # Build the row-index array and the corresponding column-index array
    rows = np.arange(dim, dtype=np.int64)
    cols = rows ^ x_mask  # P maps |r> -> phase * |r ^ x_mask>

    # Parity of (col & z_mask) gives the sign: (-1)^popcount(col & z_mask)
    parity_bits = cols & z_mask
    signs = np.where(np.bitwise_count(parity_bits) & 1, -1.0, 1.0)

    val = np.sum(np.conj(psi[rows]) * phase * signs * psi[cols])
    return float(val.real)


def _filter_and_group_pauli_ops_from_statevector(
    hamiltonian: QubitHamiltonian,
    statevector: np.ndarray,
    abelian_grouping: bool = True,
    trimming: bool = True,
    trimming_tolerance: float = 1e-8,
) -> tuple[list[QubitHamiltonian], list[float]]:
    """Filter and group the Pauli operators respect to a given quantum state.

    This function evaluates each Pauli term in the Hamiltonian with respect to the
    provided statevector:

    * Terms with zero expectation value are discarded.
    * Terms with expectation ±1 are treated as classical and their contribution is
        added to the energy at the end.
    * Remaining terms with fractional expectation values are retained and grouped by
        shared expectation value to reduce measurement redundancy
        (e.g., due to symmetry).
    * The rest of Hamiltonian is grouped into qubit wise commuting terms.

    Args:
        hamiltonian (QubitHamiltonian): QubitHamiltonian to be filtered and grouped.
        statevector (numpy.ndarray): Statevector used to compute expectation values.
        abelian_grouping (bool): Whether to group into qubit-wise commuting subsets.
        trimming (bool): If True, discard or reduce terms with ±1 or 0 expectation value.
        trimming_tolerance (float): Numerical tolerance for determining zero or ±1 expectation (Default: 1e-8).

    Returns:
        A tuple of ``(list[QubitHamiltonian], list[float])``
            * A list of grouped QubitHamiltonian.
            * A list of classical coefficients for terms that were reduced to classical contributions.

    """
    Logger.trace_entering()
    psi = np.asarray(statevector, dtype=complex)
    norm = np.linalg.norm(psi)
    if norm < np.finfo(np.float64).eps:
        raise ValueError("Statevector has zero norm.")
    psi /= norm

    retained_paulis: list[str] = []
    retained_coeffs: list[complex] = []
    expectations: list[float] = []
    classical: list[float] = []

    for pauli_str, coeff in zip(
        hamiltonian.pauli_strings, hamiltonian.coefficients, strict=True
    ):
        expval = pauli_expectation(pauli_str, psi)

        if not trimming:
            retained_paulis.append(pauli_str)
            retained_coeffs.append(coeff)
            expectations.append(expval)
            continue

        if np.isclose(expval, 0.0, atol=trimming_tolerance):
            continue
        if np.isclose(expval, 1.0, atol=trimming_tolerance):
            classical.append(float(coeff.real))
        elif np.isclose(expval, -1.0, atol=trimming_tolerance):
            classical.append(float(-coeff.real))
        else:
            retained_paulis.append(pauli_str)
            retained_coeffs.append(coeff)
            expectations.append(expval)

    if not retained_paulis:
        return [], classical

    grouped: dict[int, list[tuple[str, complex, float]]] = {}
    key_counter = 0
    # Assign approximate groups based on tolerance
    for pauli, coeff, expval in zip(
        retained_paulis, retained_coeffs, expectations, strict=True
    ):
        matched_key = None
        for k, terms in grouped.items():
            if np.isclose(expval, terms[0][2], atol=trimming_tolerance):
                matched_key = k
                break
        if matched_key is None:
            grouped[key_counter] = [(pauli, coeff, expval)]
            key_counter += 1
        else:
            grouped[matched_key].append((pauli, coeff, expval))

    reduced_pauli: list[str] = []
    reduced_coeffs: list[complex] = []

    for _, terms in grouped.items():
        coeff_sum = sum(c for _, c, _ in terms)
        # Choose Pauli with maximum # of I (most diagonal)
        best_pauli = sorted(
            [p for p, _, _ in terms], key=lambda p: (-str(p).count("I"), str(p))
        )[0]
        reduced_pauli.append(best_pauli)
        reduced_coeffs.append(coeff_sum)

    reduced_hamiltonian = QubitHamiltonian(
        reduced_pauli,
        np.array(reduced_coeffs),
        encoding=hamiltonian.encoding,
        fermion_mode_order=hamiltonian.fermion_mode_order,
    )

    grouped_hamiltonians = (
        reduced_hamiltonian.group_commuting(qubit_wise=abelian_grouping)
        if abelian_grouping
        else [reduced_hamiltonian]
    )

    return grouped_hamiltonians, classical


def filter_and_group_pauli_ops_from_wavefunction(
    hamiltonian: QubitHamiltonian,
    wavefunction: Wavefunction,
    abelian_grouping: bool = True,
    trimming: bool = True,
    trimming_tolerance: float = 1e-8,
) -> tuple[list[QubitHamiltonian], list[float]]:
    """Filter and group the Pauli operators respect to a given quantum state.

    This function evaluates each Pauli term in the Hamiltonian with respect to the
    provided wavefunction:

    * Terms with zero expectation value are discarded.
    * Terms with expectation ±1 are treated as classical and their contribution is
        added to the energy at the end.
    * Remaining terms with fractional expectation values are retained and grouped by
        shared expectation value to reduce measurement redundancy
        (e.g., due to symmetry).
    * The rest of Hamiltonian is grouped into qubit wise commuting terms.

    Args:
        hamiltonian (QubitHamiltonian): QubitHamiltonian to be filtered and grouped.
        wavefunction (Wavefunction): Wavefunction used to compute expectation values.
        abelian_grouping (bool): Whether to group into qubit-wise commuting subsets.
        trimming (bool): If True, discard or reduce terms with ±1 or 0 expectation value.
        trimming_tolerance (float): Numerical tolerance for determining zero or ±1 expectation (Default: 1e-8).

    Returns:
        A tuple of ``(list[QubitHamiltonian], list[float])``
            * A list of grouped QubitHamiltonian.
            * A list of classical coefficients for terms that were reduced to classical contributions.

    """
    from qdk_chemistry.plugins.qiskit.conversion import (
        create_statevector_from_wavefunction,  # noqa: PLC0415
    )

    Logger.trace_entering()
    psi = create_statevector_from_wavefunction(wavefunction, normalize=True)
    return _filter_and_group_pauli_ops_from_statevector(
        hamiltonian, psi, abelian_grouping, trimming, trimming_tolerance
    )
