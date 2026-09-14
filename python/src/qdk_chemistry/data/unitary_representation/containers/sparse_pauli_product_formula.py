"""Sparse Pauli words and construction of the existing Pauli product formula representation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np

from qdk_chemistry._core.data import sparse_pauli_word_to_label

from .pauli_product_formula import ExponentiatedPauliTerm, PauliProductFormulaContainer

__all__ = ["SparsePauliProductFormulaContainer", "SparsePauliTerms"]


@dataclass(frozen=True, eq=False, init=False)
class SparsePauliTerms(Sequence[str]):
    """Immutable non-identity words with lazy, qubit-zero-rightmost labels.

    ``words`` retains ordered terms, duplicates, and empty identity words. Each
    word is a sorted tuple of ``(qubit, X/Y/Z)`` pairs; only label access expands
    to register width. Storage is proportional to the non-identity factors.
    """

    num_qubits: int
    words: tuple[tuple[tuple[int, str], ...], ...]

    def __init__(self, num_qubits: int, terms: Iterable[Mapping[int, str] | Iterable[tuple[int, str]]]) -> None:
        """Copy and validate sparse words without combining terms or constructing dense labels."""
        if (
            isinstance(num_qubits, bool | np.bool_)
            or not isinstance(num_qubits, int | np.integer)
            or not 1 <= num_qubits <= 2**32
        ):
            raise ValueError("num_qubits must be an integer in 1..2**32.")
        words = []
        for term in terms:
            word = []
            for qubit, axis in term.items() if isinstance(term, Mapping) else term:
                if (
                    isinstance(qubit, bool | np.bool_)
                    or not isinstance(qubit, int | np.integer)
                    or not 0 <= qubit < num_qubits
                    or axis not in ("X", "Y", "Z")
                ):
                    raise ValueError("Sparse Pauli factors require in-range integer indices and X/Y/Z axes.")
                word.append((int(qubit), axis))
            word.sort()
            if len({q for q, _ in word}) != len(word):
                raise ValueError("Sparse Pauli factors must have unique qubit indices.")
            words.append(tuple(word))
        object.__setattr__(self, "num_qubits", int(num_qubits))
        object.__setattr__(self, "words", tuple(words))

    def factors(self, index: int) -> tuple[tuple[int, str], ...]:
        """Access one word without materializing a label."""
        return self.words[index]

    def __len__(self) -> int:
        """Count terms, including identities and duplicates."""
        return len(self.words)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        """Materialize only the requested labels."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return sparse_pauli_word_to_label([(q, "IXYZ".index(axis)) for q, axis in self.words[index]], self.num_qubits)

    def __eq__(self, other: object) -> bool:
        """Compare sparse words directly, or labels with a legacy sequence."""
        if isinstance(other, SparsePauliTerms):
            return self.num_qubits == other.num_qubits and self.words == other.words
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(a == b for a, b in zip(self, other, strict=True))
        )

    __hash__ = None  # type: ignore[assignment]


class SparsePauliProductFormulaContainer(PauliProductFormulaContainer):
    """Construct sparse words using the existing formula operations and wire format.

    There is no second formula representation: hashing, fusion, reordering and
    persistence are inherited. Parent loaders restore the canonical parent type.
    """

    @classmethod
    def from_sparse_terms(
        cls,
        terms: SparsePauliTerms,
        angles: Iterable[float],
        *,
        step_reps: int,
        scale: float = 1.0,
    ) -> SparsePauliProductFormulaContainer:
        """Pair immutable sparse words with angles, retaining identity and term order."""
        return cls(
            [ExponentiatedPauliTerm(dict(word), float(angle)) for word, angle in zip(terms.words, angles, strict=True)],
            step_reps,
            terms.num_qubits,
            scale,
        )
