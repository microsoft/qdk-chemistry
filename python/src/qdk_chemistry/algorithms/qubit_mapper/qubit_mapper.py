"""QDK/Chemistry qubit mapper abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Settings
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:  # Only needed for type annotations; avoid importing into module namespace
    from collections.abc import Sequence

    from qdk_chemistry.data import Hamiltonian, MajoranaMapping, QubitHamiltonian

__all__: list[str] = []


def _taper_qubits(
    qubit_hamiltonian: QubitHamiltonian,
    qubit_indices: Sequence[int],
    eigenvalues: Sequence[int],
) -> QubitHamiltonian:
    import numpy as np  # noqa: PLC0415

    from qdk_chemistry.data import QubitHamiltonian  # noqa: PLC0415

    qubit_indices = list(qubit_indices)
    eigenvalues = list(eigenvalues)

    if len(qubit_indices) != len(eigenvalues):
        raise ValueError(
            f"qubit_indices length ({len(qubit_indices)}) must match eigenvalues length ({len(eigenvalues)})"
        )
    if len(set(qubit_indices)) != len(qubit_indices):
        raise ValueError("qubit_indices must not contain duplicates")

    nq = qubit_hamiltonian.num_qubits
    for q in qubit_indices:
        if q < 0 or q >= nq:
            raise ValueError(f"Qubit index {q} out of range [0, {nq})")
    for ev in eigenvalues:
        if ev not in (1, -1):
            raise ValueError(f"Eigenvalue must be +1 or -1, got {ev}")

    positions_to_remove = sorted([nq - 1 - q for q in qubit_indices])
    eigenvalue_map = dict(zip(qubit_indices, eigenvalues, strict=True))

    new_strings: list[str] = []
    new_coeffs: list[complex] = []

    for pauli_str, coeff in zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True):
        skip = False
        adjusted_coeff = complex(coeff)

        for q, ev in eigenvalue_map.items():
            pos = nq - 1 - q
            char = pauli_str[pos]
            if char == "Z":
                adjusted_coeff *= ev
            elif char in ("X", "Y"):
                skip = True
                break

        if skip:
            continue

        chars = [c for i, c in enumerate(pauli_str) if i not in positions_to_remove]
        new_strings.append("".join(chars))
        new_coeffs.append(adjusted_coeff)

    new_nq = nq - len(qubit_indices)

    if not new_strings:
        return QubitHamiltonian(
            pauli_strings=["I" * new_nq],
            coefficients=np.array([0.0]),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
        )

    merged: dict[str, complex] = {}
    for s, c in zip(new_strings, new_coeffs, strict=True):
        merged[s] = merged.get(s, 0.0) + c

    final_strings = []
    final_coeffs = []
    for s, c in merged.items():
        if abs(c) > 1e-12:
            final_strings.append(s)
            final_coeffs.append(c)

    if not final_strings:
        final_strings = ["I" * new_nq]
        final_coeffs = [0.0]

    return QubitHamiltonian(
        pauli_strings=final_strings,
        coefficients=np.array(final_coeffs),
        encoding=qubit_hamiltonian.encoding,
        fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
    )


class QubitMapperSettings(Settings):
    """Base settings for all QubitMapper implementations.

    Settings are variant-specific (thresholds, etc.). The encoding is
    determined by the :class:`~qdk_chemistry.data.MajoranaMapping` passed
    to :meth:`~qdk_chemistry.algorithms.QubitMapper.run`.

    """

    def __init__(self) -> None:
        """Initialize QubitMapperSettings."""
        super().__init__()


class QubitMapper(Algorithm):
    """Abstract base class for mapping a Hamiltonian to a QubitHamiltonian.

    .. rubric:: How backends use the MajoranaMapping

    ``QubitMapper`` backends fall into two groups, and they use the
    :class:`~qdk_chemistry.data.MajoranaMapping` argument differently:

    *  **Table-driven backends** (e.g. :class:`QdkQubitMapper`) read
       ``mapping.table`` — the Pauli strings that define the encoding —
       and feed them directly to the mapping engine.  Any valid table
       works, including custom encodings with no standard name.

    *  **Third-party backends** (e.g. ``OpenFermionQubitMapper``,
       ``QiskitQubitMapper``) **ignore** ``mapping.table``.  Instead
       they read ``mapping.base_encoding`` (a string like
       ``"jordan-wigner"``) and pass it to their own library to select
       the matching transform.  The qubit operator is then built entirely
       by the third-party library's own pipeline.

    .. rubric:: Third-party backends

    Because third-party backends choose their transform by encoding
    *name*, the Pauli table in the ``MajoranaMapping`` is not used.
    Consistency between the table and the name is not checked at runtime.
    If a ``MajoranaMapping`` is manually built with a table that does not
    match its ``base_encoding`` name, a third-party backend will silently
    use the wrong transform.

    Factory-produced mappings (``MajoranaMapping.jordan_wigner()``,
    ``.bravyi_kitaev()``, etc.) always keep the table and name in sync.
    Cross-backend eigenvalue tests in the test suite verify this for every
    supported factory × backend combination.  Custom or manually built
    mappings with non-standard names cannot be used with third-party
    backends.

    .. rubric:: Tapering

    Each backend handles tapering in its own ``_run_impl()``.  The
    static helper ``_taper_result`` provides shared taper-then-relabel
    logic so backends don't have to reimplement it.  All shipped
    backends (QDK, OpenFermion, Qiskit) use this helper.

    """

    def __init__(self):
        """Initialize the QubitMapper."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``qubit_mapper`` as the algorithm type name."""
        return "qubit_mapper"

    def run(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
    ) -> QubitHamiltonian:
        """Map a fermionic Hamiltonian to a qubit Hamiltonian.

        Delegates entirely to ``_run_impl``.  Each backend is
        responsible for handling tapering (if ``mapping.tapering`` is set).

        Args:
            hamiltonian: The fermionic Hamiltonian.
            mapping: The Majorana-to-Pauli encoding (may include tapering).

        Returns:
            QubitHamiltonian with encoding and tapering metadata set.

        """
        self._settings.lock()
        return self._run_impl(hamiltonian, mapping)

    @staticmethod
    def _taper_result(qh: QubitHamiltonian, mapping: MajoranaMapping) -> QubitHamiltonian:
        """Apply post-mapping tapering if the mapping specifies it.

        Convenience helper for backends.  If ``mapping.tapering`` is
        ``None``, returns *qh* unchanged.  Otherwise, applies
        the result with the mapping's final encoding name.

        Args:
            qh: The untapered qubit Hamiltonian from the base mapping.
            mapping: The original mapping (with tapering metadata).

        Returns:
            QubitHamiltonian: Tapered and relabelled, or *qh* unchanged.

        """
        tapering = mapping.tapering
        if tapering is None:
            return qh

        from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian  # noqa: PLC0415

        tapered = _taper_qubits(qh, tapering.qubit_indices, tapering.eigenvalues)
        result = QubitHamiltonian(
            pauli_strings=tapered.pauli_strings,
            coefficients=tapered.coefficients,
            encoding=mapping.name,
            fermion_mode_order=tapered.fermion_mode_order,
            tapering=tapering,
        )
        Logger.debug(f"Tapered {tapering.num_tapered} qubits → {result.num_qubits} qubits")
        return result

    @abstractmethod
    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
    ) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the given mapping.

        Implementations receive the **full** mapping, which may include
        tapering.  Each backend handles tapering — typically by stripping
        it via ``mapping.without_tapering()``, performing the base mapping,
        and calling ``_taper_result`` to apply tapering to the output.

        .. important::

           **Table-driven** backends (e.g. :class:`QdkQubitMapper`) read
           ``mapping.table`` and pass the Pauli strings to the native engine.

           **Third-party** backends (e.g. ``OpenFermionQubitMapper``)
           read ``mapping.base_encoding`` to choose a transform function
           from their own library.  They do **not** use ``mapping.table``.
           See the class docstring for details and caveats.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            mapping: The Majorana-to-Pauli encoding (may include tapering).

        Returns:
           QubitHamiltonian: An instance of the QubitHamiltonian.

        """


class QubitMapperFactory(AlgorithmFactory):
    """Factory class for creating QubitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return ``qubit_mapper`` as the algorithm type name."""
        return "qubit_mapper"

    def default_algorithm_name(self) -> str:
        """Return ``qdk`` as the default algorithm name."""
        return "qdk"
