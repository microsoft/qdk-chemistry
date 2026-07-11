"""Qiskit-based qubit mappers to map electronic structure Hamiltonians to qubit operators.

This module provides a QiskitQubitMapper class to convert Hamiltonians to QubitOperators
using different mapping strategies ("jordan-wigner", "bravyi-kitaev", and "parity").
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)

from qdk_chemistry.algorithms.qubit_mapper import QubitMapper, QubitMapperSettings
from qdk_chemistry.data import Hamiltonian, QubitOperator
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import MajoranaMapping

__all__ = ["QiskitQubitMapper", "QiskitQubitMapperSettings"]

_SUPPORTED_ENCODINGS: dict[str, type] = {
    "jordan-wigner": JordanWignerMapper,
    "bravyi-kitaev": BravyiKitaevMapper,
    "parity": ParityMapper,
}


class QiskitQubitMapperSettings(QubitMapperSettings):
    """Settings configuration for a QiskitQubitMapper."""

    def __init__(self):
        """Initialize QiskitQubitMapperSettings."""
        Logger.trace_entering()
        super().__init__()


class QiskitQubitMapper(QubitMapper):
    """Map an electronic structure Hamiltonian to a QubitOperator using Qiskit.

    This is a **third-party** backend: it reads
    ``mapping.base_encoding`` to select the corresponding Qiskit Nature
    mapper class and **ignores** ``mapping.table``.  The qubit operator
    is built from scratch using Qiskit Nature's own fermion-to-qubit
    pipeline.

    .. warning::

       Because this backend chooses its transform by encoding *name*
       rather than from the Pauli table, it relies on the mapping's
       ``base_encoding`` string being consistent with its table.
       This is guaranteed for factory-produced mappings
       (``MajoranaMapping.jordan_wigner()``, ``.bravyi_kitaev()``, etc.)
       and is verified by cross-backend eigenvalue tests in the test
       suite.  Manually built mappings with mismatched names will
       produce silently incorrect results.

    Tapering-based encodings (e.g. parity two-qubit reduction) are
    supported — each backend handles tapering in its own ``_run_impl()``
    via the ``QubitMapper._taper_result()``
    helper.

    Both restricted (RHF) and unrestricted (UHF) Hamiltonians are supported.
    For unrestricted systems, separate alpha and beta one-body and two-body
    integrals are forwarded to Qiskit Nature's ``ElectronicEnergy``.

    Supported base encodings:
        - ``"jordan-wigner"`` → :class:`qiskit_nature.second_q.mappers.JordanWignerMapper`
        - ``"bravyi-kitaev"`` → :class:`qiskit_nature.second_q.mappers.BravyiKitaevMapper`
        - ``"parity"`` → :class:`qiskit_nature.second_q.mappers.ParityMapper`

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> from qdk_chemistry.data import MajoranaMapping
        >>> mapper = create("qubit_mapper", "qiskit")
        >>> mapping = MajoranaMapping.jordan_wigner(num_modes=n_spin_orbitals)
        >>> qh = mapper.run(hamiltonian, mapping)

    """

    QubitMappers: ClassVar = _SUPPORTED_ENCODINGS

    def __init__(self):
        """Initialize QiskitQubitMapper."""
        Logger.trace_entering()
        super().__init__()
        self._settings = QiskitQubitMapperSettings()

    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
    ) -> QubitOperator:
        """Build a qubit operator via Qiskit Nature.

        Reads ``mapping.base_encoding`` to select a Qiskit Nature mapper
        class.  ``mapping.table`` is **not used** — the qubit operator
        is built entirely by Qiskit's own pipeline.

        If *mapping* carries tapering metadata, the base encoding is
        extracted first, mapped, and tapering is applied to the result
        via :meth:`~QubitMapper._taper_result`.

        Args:
            hamiltonian: The fermionic Hamiltonian (restricted or unrestricted).
            mapping: Encoding selector — only ``base_encoding`` is read.

        Returns:
            QubitOperator: An instance of the QubitOperator.

        Raises:
            NotImplementedError: If ``mapping.base_encoding`` is not a supported Qiskit encoding.

        """
        Logger.trace_entering()

        # --- Select transform by encoding name (see QubitMapper class docstring) ---
        encoding_name = mapping.base_encoding

        if encoding_name not in _SUPPORTED_ENCODINGS:
            raise NotImplementedError(
                f"Qiskit plugin does not support base encoding {encoding_name!r}. "
                f"Supported encodings: {sorted(_SUPPORTED_ENCODINGS.keys())}."
            )

        h1_a, h1_b = hamiltonian.get_one_body_integrals()
        h2_aa, h2_ab, h2_bb = hamiltonian.get_two_body_integrals()
        num_orbs = len(hamiltonian.get_orbitals().get_active_space_indices()[0])
        is_restricted = hamiltonian.get_orbitals().is_restricted()

        if is_restricted:
            electronic_hamiltonian = ElectronicEnergy.from_raw_integrals(
                h1_a=h1_a, h2_aa=h2_aa.reshape(num_orbs, num_orbs, num_orbs, num_orbs)
            )
        else:
            # h2_ab is eri_aabb in chemist notation: (aa|bb).
            # Qiskit's h2_ba parameter expects (bb|aa) = eri_aabb transposed.
            # By Coulomb symmetry (pq|rs)=(rs|pq), this is eri_aabb[r,s,p,q].
            h2_ab_4d = h2_ab.reshape(num_orbs, num_orbs, num_orbs, num_orbs)
            electronic_hamiltonian = ElectronicEnergy.from_raw_integrals(
                h1_a=h1_a,
                h2_aa=h2_aa.reshape(num_orbs, num_orbs, num_orbs, num_orbs),
                h1_b=h1_b,
                h2_bb=h2_bb.reshape(num_orbs, num_orbs, num_orbs, num_orbs),
                h2_ba=h2_ab_4d.transpose(2, 3, 0, 1),
            )

        fermionic_op = electronic_hamiltonian.second_q_op()
        qubit_mapper = _SUPPORTED_ENCODINGS[encoding_name]()
        qubit_op = qubit_mapper.map(fermionic_op)
        qh = QubitOperator(
            pauli_strings=qubit_op.paulis.to_labels(),
            coefficients=qubit_op.coeffs,
            encoding=encoding_name,
            fermion_mode_order=FermionModeOrder.BLOCKED,
        )
        return self._taper_result(qh, mapping)

    def name(self) -> str:
        """Return the algorithm name ``qiskit``."""
        Logger.trace_entering()
        return "qiskit"
