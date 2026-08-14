"""QDK native boson-to-qubit mapper built on the C++ mapping engine.

This module provides the :class:`QdkBosonQubitMapper` class for transforming
bosonic Hamiltonians into qubit operators. The encoding and the occupation
cutoff are specified by a :class:`~qdk_chemistry.data.BosonMapping` passed to
``run()``, making the mapper encoding-agnostic.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry._core.data import (
    boson_map_hamiltonian,
    sparse_pauli_word_to_label,
)
from qdk_chemistry.algorithms.boson_qubit_mapper.boson_qubit_mapper import (
    BosonQubitMapper,
    BosonQubitMapperSettings,
)
from qdk_chemistry.data.qubit_operator import QubitOperator
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import BosonMapping, Hamiltonian

__all__ = ["QdkBosonQubitMapper", "QdkBosonQubitMapperSettings"]


class QdkBosonQubitMapperSettings(BosonQubitMapperSettings):
    """Settings configuration for a QdkBosonQubitMapper.

    Settings:
        threshold (double, default=1e-12): Threshold for pruning small Pauli coefficients.
        integral_threshold (double, default=1e-12): Threshold for filtering small integrals.

    """

    def __init__(self) -> None:
        """Initialize QdkBosonQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "threshold",
            "double",
            1e-12,
            "Threshold for pruning small Pauli coefficients",
        )
        self._set_default(
            "integral_threshold",
            "double",
            1e-12,
            "Threshold for filtering small integrals (improves performance)",
        )


class QdkBosonQubitMapper(BosonQubitMapper):
    r"""QDK native boson-to-qubit mapper using the C++ mapping engine.

    The Hamiltonian is read in the same chemist notation used for every other
    Hamiltonian in the library,

    .. math::

        H = \sum_{pq} h_{pq}\, b_p^\dagger b_q
          + \tfrac{1}{2} \sum_{pqrs} (pq|rs)\, b_p^\dagger b_r^\dagger b_s b_q ,

    so no bosonic-specific container is required.  With :math:`h_{ii} = -\mu`,
    :math:`h_{ij} = -t` on bonds and :math:`(ii|ii) = U` this reproduces the
    Bose-Hubbard model exactly, the two-body contraction collapsing to
    :math:`\tfrac{U}{2}\sum_i n_i(n_i-1)`.

    Each mode is truncated to a local dimension ``d`` and encoded on
    ``nq = log2(d)`` qubits.  Because ``d`` is a power of two the encoding is
    surjective: the encoded subspace is the entire qubit Hilbert space, so
    there is no leakage and no penalty term is needed.

    The Hamiltonian's constant energy shift is **not** included in the result,
    matching :class:`~qdk_chemistry.algorithms.QdkQubitMapper`.

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> from qdk_chemistry.data import BosonicModes, BosonMapping, LatticeGraph
        >>> from qdk_chemistry.utils.model_hamiltonians import create_bose_hubbard_hamiltonian
        >>> lattice = LatticeGraph.chain(2)
        >>> hamiltonian = create_bose_hubbard_hamiltonian(lattice, t=1.0, U=4.0, mu=0.0, mode_dimension=2)
        >>> mapper = create("boson_qubit_mapper")
        >>> mapping = BosonMapping.standard_binary(num_modes=2, mode_dimension=2)
        >>> qh = mapper.run(hamiltonian, mapping)

    """

    def __init__(
        self,
        threshold: float = 1e-12,
        integral_threshold: float = 1e-12,
    ) -> None:
        """Initialize the QdkBosonQubitMapper with default settings.

        Args:
            threshold: Threshold for pruning small Pauli coefficients. Default: 1e-12.
            integral_threshold: Threshold for filtering small integrals. Default: 1e-12.

        """
        super().__init__()
        self._settings = QdkBosonQubitMapperSettings()
        self._settings.set("threshold", threshold)
        self._settings.set("integral_threshold", integral_threshold)

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The algorithm name.

        """
        return "qdk"

    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: BosonMapping,
    ) -> QubitOperator:
        """Transform a bosonic Hamiltonian to a qubit operator.

        Args:
            hamiltonian: The bosonic Hamiltonian with one-body and two-body integrals.
            mapping: The boson-to-qubit encoding; supplies both the encoding family and the occupation cutoff.

        Returns:
            QubitOperator: The qubit operator with Pauli strings and coefficients.

        Raises:
            ValueError: If the mapping's mode count or cutoff disagrees with the Hamiltonian.

        """
        Logger.trace_entering()

        threshold = float(self.settings().get("threshold"))
        integral_threshold = float(self.settings().get("integral_threshold"))

        n_modes = hamiltonian.get_one_body_integrals()[0].shape[0]
        if mapping.num_modes() != n_modes:
            raise ValueError(
                f"BosonMapping has {mapping.num_modes()} modes but the Hamiltonian has "
                f"{n_modes} modes. Use BosonMapping.standard_binary(num_modes={n_modes}, ...) "
                f"or BosonMapping.standard_binary(hamiltonian.get_orbitals())."
            )

        # The C++ engine validates the cutoff against a BosonicModes basis and
        # reads sparse integrals natively when the container provides them.
        words, coefficients = boson_map_hamiltonian(
            mapping,
            hamiltonian,
            threshold,
            integral_threshold,
        )

        n_qubits = mapping.num_qubits()
        pauli_strings = [sparse_pauli_word_to_label(word, n_qubits) for word in words]

        Logger.debug(f"Generated {len(pauli_strings)} Pauli terms for {n_qubits} qubits")

        return QubitOperator(
            pauli_strings=pauli_strings,
            coefficients=np.array(coefficients, dtype=complex),
            encoding=mapping.name,
        )
