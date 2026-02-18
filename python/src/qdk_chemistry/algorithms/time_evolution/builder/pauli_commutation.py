"""Backward-compatible re-export — canonical location is :mod:`qdk_chemistry.utils.pauli_commutation`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.utils.pauli_commutation import (  # noqa: F401
    commutator_bound_first_order,
    do_pauli_strings_commute,
    do_pauli_strings_qw_commute,
    do_pauli_terms_commute,
    do_pauli_terms_qw_commute,
    get_commutation_checker,
)

__all__: list[str] = [
    "commutator_bound_first_order",
    "do_pauli_strings_commute",
    "do_pauli_strings_qw_commute",
    "do_pauli_terms_commute",
    "do_pauli_terms_qw_commute",
    "get_commutation_checker",
]
