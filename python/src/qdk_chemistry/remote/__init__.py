"""Caching for QDK/Chemistry algorithm results.

This module provides result caching via the ``cache`` kwarg
on ``algorithm.run()``.

Usage:
    >>> from qdk_chemistry.algorithms import create
    >>>
    >>> scf = create("scf_solver")
    >>>
    >>> # Local with caching
    >>> energy, wfn = scf.run(structure, 0, 1, "cc-pvdz", cache="./cache")

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.remote.cache import resolve_cache

__all__ = [
    "resolve_cache",
]
