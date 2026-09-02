"""Hashing utilities for algorithm run content hashing."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import hashlib

from qdk_chemistry.data import Settings
from qdk_chemistry.data._hashing import _hash_arg, _hash_str, _hash_uint

__all__ = ["run_content_hash"]


def run_content_hash(
    algorithm_type: str,
    algorithm_name: str,
    settings: Settings,
    *args,
    **kwargs,
) -> str:
    """Compute a deterministic content hash for an algorithm run.

    This function has the same logical signature as ``Algorithm.run()`` but
    returns a hex hash string instead of executing the algorithm. Identical
    inputs produce identical hashes.

    Args:
        algorithm_type: The algorithm type name (e.g., "scf_solver").
        algorithm_name: The algorithm name (e.g., "pyscf").
        settings: The Settings object for the algorithm.
        args: Positional arguments that would be passed to run().
        kwargs: Keyword arguments that would be passed to run().

    Returns:
        str: A 16-character hex string content hash.

    Examples:
        >>> from qdk_chemistry.algorithms.hashing import run_content_hash
        >>> hash_val = run_content_hash(
        ...     "scf_solver", "pyscf", scf.settings(),
        ...     structure, charge=0, spin_multiplicity=1,
        ... )

    """
    h = hashlib.sha256()
    _hash_str(h, algorithm_type)
    _hash_str(h, algorithm_name)

    # Hash settings in sorted key order
    setting_keys = sorted(settings.keys())
    _hash_uint(h, len(setting_keys))
    for key in setting_keys:
        _hash_str(h, key)
        val = settings.get(key)
        _hash_arg(h, val)

    # Hash positional args
    _hash_uint(h, len(args))
    for arg in args:
        _hash_arg(h, arg)

    # Hash keyword args in sorted order
    sorted_keys = sorted(kwargs.keys())
    _hash_uint(h, len(sorted_keys))
    for key in sorted_keys:
        _hash_str(h, key)
        _hash_arg(h, kwargs[key])

    return h.hexdigest()[:16]
