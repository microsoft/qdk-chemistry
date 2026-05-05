"""Term-grouper algorithms for :class:`~qdk_chemistry.data.QubitHamiltonian`.

A *term grouper* takes a :class:`~qdk_chemistry.data.QubitHamiltonian` and
returns a new one with a populated
:attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition` that downstream
algorithms can exploit.

Example:
    >>> from qdk_chemistry.algorithms import registry
    >>> grouper = registry.create("term_grouper", "qubit_wise_commuting")
    >>> grouped = grouper.run(my_hamiltonian)
    >>> grouped.term_partition  # FlatPartition with strategy="qubit_wise_commuting"

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper, TermGrouperFactory
from qdk_chemistry.algorithms.term_grouper.commuting import (
    FullCommutingTermGrouper,
    QubitWiseCommutingTermGrouper,
)
from qdk_chemistry.algorithms.term_grouper.identity import IdentityTermGrouper

__all__ = [
    "FullCommutingTermGrouper",
    "IdentityTermGrouper",
    "QubitWiseCommutingTermGrouper",
    "TermGrouper",
    "TermGrouperFactory",
]
