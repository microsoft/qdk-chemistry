"""Second-order stabilizer Renyi entropy from sparse, real wavefunctions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from itertools import combinations
from math import fsum, isfinite, log2, sqrt
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Settings, StabilizerRenyiEntropyResult, Wavefunction

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

__all__ = [
    "ReplicaStabilizerRenyiEntropy",
    "StabilizerRenyiEntropy",
    "StabilizerRenyiEntropyFactory",
    "StabilizerRenyiEntropySettings",
]

_NORMALIZATION_TOLERANCE = 1e-10
_MOMENT_TOLERANCE = 1e-12
_SCALAR_CONFIGURATION_LIMIT = 16


class StabilizerRenyiEntropySettings(Settings):
    """Settings for real, second-order stabilizer Renyi entropy."""

    def __init__(self) -> None:
        """Initialize optional amplitude truncation, disabled by default."""
        super().__init__()
        self._set_default("truncate", "bool", False, "Truncate and renormalize the coefficient expansion.")
        self._set_default(
            "coefficient_cutoff",
            "double",
            1e-12,
            "Retain only abs(coefficient) > cutoff when truncation is enabled.",
        )


class StabilizerRenyiEntropy(Algorithm):
    """Base interface for stabilizer Renyi entropy algorithms."""

    def type_name(self) -> str:
        """Return the algorithm type identifier."""
        return "stabilizer_renyi_entropy"

    @abstractmethod
    def _run_impl(self, wavefunction: Wavefunction) -> StabilizerRenyiEntropyResult:
        """Compute stabilizer Renyi entropy of an active-space wavefunction."""


def _sparse_coefficients(wavefunction: Wavefunction) -> tuple[int, list[int], "NDArray[np.float64]"]:
    """Extract real, normalized occupation coefficients without a dense vector."""
    if not isinstance(wavefunction, Wavefunction):
        raise TypeError("wavefunction must be a Wavefunction.")
    if wavefunction.get_container_type() != "state_vector":
        raise TypeError("Stabilizer Renyi entropy requires a coefficient expansion, not an amplitude container.")

    determinants = wavefunction.get_active_determinants()
    coefficients = np.asarray(wavefunction.get_coefficients())
    if not determinants or coefficients.size == 0:
        raise ValueError("The wavefunction coefficient expansion must not be empty.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Wavefunction coefficients must be finite.")
    if np.iscomplexobj(coefficients) and np.any(coefficients.imag != 0):
        raise ValueError("The replica backend currently supports only real coefficients.")

    coefficients = np.asarray(coefficients.real, dtype=np.float64)
    norm_squared = fsum(float(c) * float(c) for c in coefficients)
    if not isfinite(norm_squared) or abs(norm_squared - 1.0) > _NORMALIZATION_TOLERANCE:
        raise ValueError(
            "The input wavefunction must be normalized: "
            f"sum(abs(c)**2) = {norm_squared!r}, tolerance = {_NORMALIZATION_TOLERANCE}."
        )

    bits_per_mode = determinants[0].bits_per_mode()
    if bits_per_mode not in (1, 2):
        raise ValueError("Only binary spinless or spin-half occupation configurations are supported.")
    num_qubits = wavefunction.get_configuration_set().num_modes() * bits_per_mode
    labels = [sum(int(bit) << q for q, bit in enumerate(det.to_bits(num_qubits))) for det in determinants]
    if len(set(labels)) != len(labels):
        raise ValueError("The coefficient expansion must not contain duplicate configurations.")
    return num_qubits, labels, coefficients


def _scalar_replica_moment(labels: list[int], coefficients: "NDArray[np.float64]") -> float:
    """Evaluate tiny expansions without NumPy batching overhead."""
    amplitudes = dict(sorted(zip(labels, coefficients.tolist(), strict=True)))
    squared = {x: c * c for x, c in amplitudes.items()}
    fourth = [c * c for c in squared.values()]
    configurations = list(amplitudes)

    def distinct_terms() -> "Iterator[float]":
        for x, y, z, w in combinations(configurations, 4):
            partner = (
                amplitudes.get(x ^ y ^ z, 0.0)
                * amplitudes.get(x ^ y ^ w, 0.0)
                * amplitudes.get(x ^ z ^ w, 0.0)
                * amplitudes.get(y ^ z ^ w, 0.0)
            )
            yield amplitudes[x] * amplitudes[y] * amplitudes[z] * amplitudes[w] * partner

    def repeated_terms() -> "Iterator[float]":
        for i, x in enumerate(configurations):
            for y in configurations[i + 1 :]:
                for z in configurations:
                    if z not in (x, y):
                        yield squared[x] * squared[y] * squared[z] * squared.get(x ^ y ^ z, 0.0)

    # Fully ordered distinct indices have 24 permutations. The remaining factors
    # count one-pair, two-pair/triple, and all-equal replica index patterns.
    return fsum(
        (
            24.0 * fsum(distinct_terms()),
            12.0 * fsum(repeated_terms()),
            14.0 * fsum(a * b for a, b in combinations(fourth, 2)),
            fsum(c * c for c in fourth),
        )
    )


def _replica_moment(labels: list[int], coefficients: "NDArray[np.float64]") -> float:
    """Evaluate permutation-grouped replicas with quadratic index storage."""
    count = len(labels)
    if count <= _SCALAR_CONFIGURATION_LIMIT:
        return _scalar_replica_moment(labels, coefficients)

    amplitudes = sorted(zip(labels, coefficients.tolist(), strict=True))
    configurations = [x for x, _ in amplitudes]
    values = np.array([c for _, c in amplitudes] + [0.0])
    squared = values * values
    fourth = squared * squared

    # Equal pair XORs identify triple-XOR partners without a register-sized table.
    pair_ids = np.zeros((count, count), dtype=np.intp)
    xor_groups = {0: 0}
    for i, x in enumerate(configurations):
        for j in range(i + 1, count):
            group = xor_groups.setdefault(x ^ configurations[j], len(xor_groups))
            pair_ids[i, j] = pair_ids[j, i] = group
    lookup = np.full(len(xor_groups), count, dtype=np.intp)
    del xor_groups
    indices = np.arange(count)
    partners = np.full((count + 1, count + 1), count, dtype=np.intp)
    distinct_sums = []
    repeated_sums = []
    for y in range(1, count):
        # partners[i,j] indexes c[y ^ i ^ j] in configuration-label notation.
        # The sentinel count indexes zero, including further sentinel lookups.
        lookup.fill(count)
        lookup[pair_ids[y]] = indices
        partners[:count, :count] = lookup[pair_ids]

        repeated = squared[partners[:y, :count]]
        repeated[np.arange(y), np.arange(y)] = 0.0
        repeated[:, y] = 0.0
        repeated_sums.append(float(squared[y] * np.sum(repeated * squared[:y, None] * squared[None, :count])))

        z, w = np.triu_indices(count - y - 1, 1)
        z += y + 1
        w += y + 1
        q = partners[z, w]
        present = q != count
        z, w, q = z[present], w[present], q[present]
        if q.size == 0:
            continue
        common = values[z] * values[w] * values[q]
        for x in range(y):
            # x<y<z<w. For q=y^z^w, x^z^w=x^y^q.
            row = partners[x]
            distinct_sums.append(
                float(values[x] * values[y] * np.sum(common * values[row[z]] * values[row[w]] * values[row[q]]))
            )

    return fsum(
        (
            24.0 * fsum(distinct_sums),
            12.0 * fsum(repeated_sums),
            14.0 * fsum(a * b for a, b in combinations(fourth[:count].tolist(), 2)),
            fsum(c * c for c in fourth[:count].tolist()),
        )
    )


class ReplicaStabilizerRenyiEntropy(StabilizerRenyiEntropy):
    """Exact real 2-SRE in bits, using sparse Jordan-Wigner occupations.

    No state-preparation circuit, Pauli enumeration, tapering, or dense
    ``2**n`` allocation is used. Worst-case runtime is quartic in the retained
    number of configurations; auxiliary storage is quadratic in that number.
    """

    def __init__(self) -> None:
        """Initialize the replica backend with truncation disabled."""
        super().__init__()
        self._settings = StabilizerRenyiEntropySettings()

    def name(self) -> str:
        """Return the backend identifier."""
        return "replica"

    def _run_impl(self, wavefunction: Wavefunction) -> StabilizerRenyiEntropyResult:
        """Validate, optionally truncate, and evaluate the second-order entropy.

        The original squared norm must differ from one by at most ``1e-10``.
        Only this bounded numerical drift is corrected. Truncation, when
        requested, uses the supplied amplitudes and always renormalizes.
        """
        num_qubits, labels, coefficients = _sparse_coefficients(wavefunction)
        support = coefficients != 0
        num_configurations = int(np.count_nonzero(support))
        truncation_enabled = self._settings.get("truncate")
        cutoff = None
        retained = support
        if truncation_enabled:
            cutoff = self._settings.get("coefficient_cutoff")
            if not isfinite(cutoff) or cutoff < 0:
                raise ValueError("coefficient_cutoff must be finite and non-negative.")
            retained = np.abs(coefficients) > cutoff
        if not np.any(retained):
            raise ValueError("Truncation removed every nonzero coefficient.")

        # Sum the discarded mass directly to preserve weights much smaller than
        # machine epsilon, instead of subtracting the retained norm from one.
        norm_squared = fsum(float(c) * float(c) for c in coefficients)
        discarded_weight = fsum(float(c) * float(c) for c in coefficients[~retained]) / norm_squared
        retained_coefficients = coefficients[retained]
        retained_norm = sqrt(fsum(float(c) * float(c) for c in retained_coefficients))
        retained_coefficients = retained_coefficients / retained_norm
        retained_labels = [x for x, keep in zip(labels, retained, strict=True) if keep]
        moment = _replica_moment(retained_labels, retained_coefficients)
        if not isfinite(moment) or moment <= 0 or moment > 1.0 + _MOMENT_TOLERANCE:
            raise RuntimeError(f"The replica calculation produced an invalid Pauli fourth moment: {moment!r}.")
        entropy = -log2(min(moment, 1.0))
        if entropy > num_qubits + _MOMENT_TOLERANCE:
            raise RuntimeError(f"The replica calculation produced entropy {entropy!r} above n = {num_qubits}.")
        return StabilizerRenyiEntropyResult(
            entropy=min(entropy, float(num_qubits)),
            num_qubits=num_qubits,
            num_configurations=num_configurations,
            num_retained_configurations=len(retained_labels),
            truncation_enabled=truncation_enabled,
            coefficient_cutoff=cutoff,
            discarded_weight=discarded_weight,
        )


class StabilizerRenyiEntropyFactory(AlgorithmFactory):
    """Factory for stabilizer Renyi entropy backends."""

    def algorithm_type_name(self) -> str:
        """Return the algorithm type identifier."""
        return "stabilizer_renyi_entropy"

    def default_algorithm_name(self) -> str:
        """Return the sparse replica backend as the default."""
        return "replica"
