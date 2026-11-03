# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
"""Internal two-body permutation-symmetry helpers for Python consumers of the integrals.

These helpers are **not** part of the public API. They mirror the C++ library
(``cpp/src/qdk/chemistry/data/two_body_symmetry.hpp``) for plugins that hand the integrals to
other packages.
"""

import numpy as np

from qdk_chemistry.utils import Logger

# Eigen's default isApprox precision, as in the C++ library.
_PRECISION = 1e-12

_FOURFOLD_PARTNERS = "(pq|rs), (rs|pq), (qp|sr) and (sr|qp)"
_REVERSAL_PARTNERS = "(pq|rs) and (qp|sr)"

_EIGHTFOLD_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def _require_finite(eri: np.ndarray, consumer: str) -> None:
    for eri_slice in eri:
        if not np.all(np.isfinite(eri_slice)):
            raise ValueError(f"{consumer}: the two-body integrals contain a non-finite value.")


def _squared_norms(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    """Return ``||first - second||^2``, ``||first||^2`` and ``||second||^2``, one slice at a time."""
    difference = first_norm = second_norm = 0.0
    for first_slice, second_slice in zip(first, second, strict=True):
        difference += float(np.sum(np.square(first_slice - second_slice)))
        first_norm += float(np.sum(np.square(first_slice)))
        second_norm += float(np.sum(np.square(second_slice)))
    return difference, first_norm, second_norm


def _warn_if_not_hermitian(operator: np.ndarray, average: np.ndarray, consumer: str, partners: str) -> None:
    difference, operator_norm, average_norm = _squared_norms(operator, average)
    if difference <= _PRECISION**2 * min(operator_norm, average_norm):
        return
    Logger.warn(
        f"{consumer}: the two-body integrals differ from their Hermitian part, the average over {partners}, "
        f"by {np.sqrt(difference / operator_norm):.1e} relative to their norm; the Hermitian part is used. "
        "Symmetrize the integrals to avoid this warning."
    )


def fourfold_average(eri: np.ndarray, consumer: str) -> np.ndarray:
    """Average a restricted or same-spin ``(pq|rs)`` block over ``(pq|rs)``, ``(rs|pq)``, ``(qp|sr)``, ``(sr|qp)``.

    A warning is logged if the block is not Hermitian.

    Args:
        eri: Two-body integrals of shape ``(n, n, n, n)``.
        consumer: Name used in messages.

    Returns:
        The average, of the same shape.

    Raises:
        ValueError: If the tensor has a non-finite value.

    """
    _require_finite(eri, consumer)
    # The operator depends only on the bra-ket average.
    bra_ket = eri + eri.transpose(2, 3, 0, 1)
    average = 0.25 * (bra_ket + bra_ket.transpose(1, 0, 3, 2))
    _warn_if_not_hermitian(0.5 * bra_ket, average, consumer, _FOURFOLD_PARTNERS)
    return average


def reversal_average(eri: np.ndarray, consumer: str) -> np.ndarray:
    """Average an unrestricted alpha-beta ``(pq|rs)`` block over ``(pq|rs)`` and ``(qp|sr)``.

    A warning is logged if the block is not Hermitian.

    Args:
        eri: Two-body integrals of shape ``(n, n, n, n)``.
        consumer: Name used in messages.

    Returns:
        The average, of the same shape.

    Raises:
        ValueError: If the tensor has a non-finite value.

    """
    _require_finite(eri, consumer)
    average = 0.5 * (eri + eri.transpose(1, 0, 3, 2))
    _warn_if_not_hermitian(eri, average, consumer, _REVERSAL_PARTNERS)
    return average


def eightfold_average(eri: np.ndarray, consumer: str) -> np.ndarray:
    """Return the average of a ``(pq|rs)`` tensor over its 8-fold permutation classes.

    Args:
        eri: Two-body integrals of shape ``(n, n, n, n)``.
        consumer: Name used in error messages.

    Returns:
        The 8-fold average, of the same shape.

    Raises:
        ValueError: If the tensor has a non-finite value or is not 8-fold symmetric.

    """
    _require_finite(eri, consumer)
    average = np.empty_like(eri)
    views = [eri.transpose(permutation) for permutation in _EIGHTFOLD_PERMUTATIONS]
    for index, (a, b, c, d, e, f, g, h) in enumerate(zip(*views, strict=True)):
        # Summed pairwise, so that equal values average to themselves exactly.
        average[index] = (((a + b) + (c + d)) + ((e + f) + (g + h))) / 8
    # Eigen's isApprox: ||eri - average|| <= p min(||eri||, ||average||).
    difference, eri_norm, average_norm = _squared_norms(eri, average)
    if difference > _PRECISION**2 * min(eri_norm, average_norm):
        raise ValueError(
            f"{consumer} requires two-body integrals with 8-fold permutation symmetry, "
            "(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq); they differ from their 8-fold average by "
            f"{np.sqrt(difference / eri_norm):.1e} relative to their norm. If this is round-off, as in "
            "Hamiltonians saved by earlier versions, rebuild the Hamiltonian."
        )
    return average


def pair_average(eri: np.ndarray, consumer: str) -> np.ndarray:
    """Average an alpha-beta ``(pq|rs)`` block over ``(pq|rs)``, ``(qp|rs)``, ``(pq|sr)`` and ``(qp|sr)``.

    Args:
        eri: Two-body integrals of shape ``(n, n, n, n)``.
        consumer: Name used in error messages.

    Returns:
        The average, of the same shape.

    Raises:
        ValueError: If the tensor has a non-finite value or is not symmetric under these swaps.

    """
    _require_finite(eri, consumer)
    views = [eri, eri.transpose(1, 0, 2, 3), eri.transpose(0, 1, 3, 2), eri.transpose(1, 0, 3, 2)]
    average = np.empty_like(eri)
    for index, (a, b, c, d) in enumerate(zip(*views, strict=True)):
        average[index] = ((a + b) + (c + d)) / 4
    difference, eri_norm, average_norm = _squared_norms(eri, average)
    if difference > _PRECISION**2 * min(eri_norm, average_norm):
        raise ValueError(
            f"{consumer} requires two-body integrals with pair permutation symmetry, (pq|rs) = (qp|rs) = (pq|sr); "
            f"they differ from their pair average by {np.sqrt(difference / eri_norm):.1e} relative to their norm. "
            "If this is round-off, as in Hamiltonians saved by earlier versions, rebuild the Hamiltonian."
        )
    return average
