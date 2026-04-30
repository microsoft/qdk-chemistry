"""Term-grouper algorithms for :class:`~qdk_chemistry.data.QubitHamiltonian`.

A *term grouper* takes a :class:`~qdk_chemistry.data.QubitHamiltonian` and
returns a new one with a populated
:attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition` that downstream
algorithms can exploit.

Currently registered strategies:

* ``"identity"`` — every term in its own single-element group.  Useful to
  override existing groupings or to disable optimisation.
* ``"commuting"`` — fully commuting groups (``[A,B]=0``).  Used by Trotter
  decompositions to reduce the number of exponentials per step.
* ``"qubit_wise_commuting"`` — qubit-wise commuting groups.  Used by
  measurement workflows where a single basis change suffices for all members
  of a group.

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
    CommutingTermGrouper,
    QubitWiseCommutingTermGrouper,
)
from qdk_chemistry.algorithms.term_grouper.identity import IdentityTermGrouper

__all__ = [
    "CommutingTermGrouper",
    "IdentityTermGrouper",
    "QubitWiseCommutingTermGrouper",
    "TermGrouper",
    "TermGrouperFactory",
]
