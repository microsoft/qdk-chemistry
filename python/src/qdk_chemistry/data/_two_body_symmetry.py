# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
"""Internal two-body permutation-symmetry helpers for Python consumers of the integrals.

These helpers are **not** part of the public API. They mirror the checks in the C++ library
(``cpp/src/qdk/chemistry/data/two_body_symmetry.hpp``) for plugins that hand the integrals to
other packages.
"""

import numpy as np

# Symmetry violations up to this fraction of the largest integral are treated as round-off;
# keep it equal to ``two_body_symmetry_tolerance`` in the C++ library.
TWO_BODY_SYMMETRY_TOLERANCE = 1e-5

_EIGHTFOLD_PERMUTATIONS = (
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def eightfold_average(eri: np.ndarray, consumer: str) -> np.ndarray:
    """Return the average of a ``(pq|rs)`` tensor over its 8-fold permutation classes.

    Consumers that keep one value per 8-fold class discard the difference between the tensor and
    this average, so a tensor that differs from it by more than ``TWO_BODY_SYMMETRY_TOLERANCE``
    times its largest element is rejected, as in FCIDUMP export. Temporaries are one slice at a
    time, so the peak memory is the input and the returned average.

    Args:
        eri: Two-body integrals of shape ``(n, n, n, n)``.
        consumer: Name used in error messages.

    Returns:
        The 8-fold average, of the same shape.

    Raises:
        ValueError: If the tensor has a non-finite value or differs from its 8-fold average by more
            than the tolerance.

    """
    scale = 0.0
    for eri_slice in eri:
        if not np.all(np.isfinite(eri_slice)):
            raise ValueError(f"{consumer}: the two-body integrals contain a non-finite value.")
        scale = max(scale, float(np.max(np.abs(eri_slice), initial=0.0)))
    average = eri.copy()
    for permutation in _EIGHTFOLD_PERMUTATIONS:
        average += eri.transpose(permutation)
    average /= len(_EIGHTFOLD_PERMUTATIONS) + 1
    deviation = max(
        (
            float(np.max(np.abs(eri_slice - average_slice), initial=0.0))
            for eri_slice, average_slice in zip(eri, average, strict=True)
        ),
        default=0.0,
    )
    if deviation > TWO_BODY_SYMMETRY_TOLERANCE * scale:
        raise ValueError(
            f"{consumer} requires two-body integrals with 8-fold permutation symmetry, "
            "(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq); reduced-symmetry integrals, such as downfolded "
            "ones, are not supported."
        )
    return average
