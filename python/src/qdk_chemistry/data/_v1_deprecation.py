"""Deprecation warning wrappers for v1 Hamiltonian and Wavefunction accessors.

This module patches deprecated C++ methods on the _core data classes to emit
Python DeprecationWarning at call time.  It is imported by data/__init__.py
so the warnings fire regardless of how users import the classes.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings
from typing import Any

from qdk_chemistry._core.data import (
    CanonicalFourCenterHamiltonianContainer,
    CholeskyHamiltonianContainer,
    Hamiltonian,
    HamiltonianContainer,
    SparseHamiltonianContainer,
    WavefunctionContainer,
)


def _wrap_deprecated(cls: type, method_name: str, replacement: str) -> None:
    """Replace *method_name* on *cls* with a wrapper that emits DeprecationWarning.

    Also patches the corresponding property if a legacy property binding still
    exists, so both call styles warn during the deprecation period.
    """
    original = getattr(cls, method_name, None)
    if original is None:
        return

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{cls.__name__}.{method_name} is deprecated. Use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        bound = original.__get__(self, type(self)) if hasattr(original, "__get__") else original
        return bound(*args, **kwargs)

    wrapper.__name__ = method_name
    wrapper.__doc__ = getattr(original, "__doc__", None)
    setattr(cls, method_name, wrapper)

    # Patch a matching legacy property if one is still present
    # (e.g. "get_one_body_integrals" → "one_body_integrals").
    prop_name = method_name[4:] if method_name.startswith("get_") else None

    if prop_name:
        # Narrow prop_name to a non-Optional local so the default-argument
        # binding below has a concrete `str` type for mypy.
        prop_name_str: str = prop_name
        # Properties live on the base class and propagate via Python MRO
        # (unlike pybind11 methods). Check the class itself AND its bases.
        for klass in cls.__mro__:
            existing = klass.__dict__.get(prop_name_str)
            if existing is not None and isinstance(existing, property):

                def prop_wrapper(
                    self: Any,
                    _orig: Any = original,
                    _pn: str = prop_name_str,
                    _cls: type = cls,
                    _repl: str = replacement,
                ) -> Any:
                    warnings.warn(
                        f"{_cls.__name__}.{_pn} is deprecated. Use {_repl} instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    bound = _orig.__get__(self, type(self)) if hasattr(_orig, "__get__") else _orig
                    return bound()

                setattr(klass, prop_name_str, property(prop_wrapper, doc=existing.__doc__))
                break


def _install_deprecation_warnings() -> None:
    """Patch all deprecated v1 accessors to emit DeprecationWarning."""
    # -- HamiltonianContainer base methods --
    _wrap_deprecated(HamiltonianContainer, "get_one_body_integrals", "one_body_integrals()")
    _wrap_deprecated(HamiltonianContainer, "get_inactive_fock_matrix", "inactive_fock()")

    # -- CanonicalFourCenterHamiltonianContainer --
    _wrap_deprecated(CanonicalFourCenterHamiltonianContainer, "get_two_body_integrals", "two_body_integrals()")

    # -- CholeskyHamiltonianContainer --
    _wrap_deprecated(CholeskyHamiltonianContainer, "get_three_center_integrals", "three_center()")
    _wrap_deprecated(CholeskyHamiltonianContainer, "get_two_body_integrals", "three_center()")

    # -- SparseHamiltonianContainer --
    _wrap_deprecated(SparseHamiltonianContainer, "sparse_two_body_integrals", "two_body_integrals_sparse()")
    _wrap_deprecated(SparseHamiltonianContainer, "get_two_body_integrals", "two_body_integrals_sparse()")

    # -- Hamiltonian wrapper --
    _wrap_deprecated(Hamiltonian, "get_one_body_integrals", "get_container().one_body_integrals()")
    _wrap_deprecated(
        Hamiltonian,
        "get_two_body_integrals",
        "get_container().two_body_integrals()",
    )
    _wrap_deprecated(Hamiltonian, "get_inactive_fock_matrix", "get_container().inactive_fock()")

    # -- WavefunctionContainer (abstract, but used via Wavefunction.get_container()) --
    _wrap_deprecated(WavefunctionContainer, "get_active_one_rdm_spin_dependent", "active_one_rdm()")
    _wrap_deprecated(WavefunctionContainer, "get_active_two_rdm_spin_dependent", "active_two_rdm()")
    _wrap_deprecated(
        WavefunctionContainer,
        "get_active_one_rdm_spin_traced",
        "active_one_rdm() and trace over the spin variant",
    )
    _wrap_deprecated(
        WavefunctionContainer,
        "get_active_two_rdm_spin_traced",
        "active_two_rdm() and trace over the spin variant",
    )
