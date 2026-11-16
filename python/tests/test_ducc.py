"""Compare public DUCC output with an independent, finite-Fock-space T1/T2 oracle.

All determinants use bit p for mode p, ordered [alpha, beta]. The BCH is never
truncated between commutators; only its final reference-normal-ordered scalar,
one-body and two-body terms survive. No quantum mapper or contraction engine is used.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from itertools import combinations, product
from math import factorial
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AmplitudeContainer,
    AmplitudeType,
    CanonicalFourCenterHamiltonianContainer,
    Configuration,
    Hamiltonian,
    ModelOrbitals,
    Orbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data import (
    symmetry as sym,
)

from .helpers import create_test_basis_set

_Array: TypeAlias = NDArray[np.float64]


def _apply_word(det: int, creators: tuple[int, ...], holes: tuple[int, ...]) -> tuple[int, int]:
    """Apply a†p a†q ... a_j a_i to a bit determinant, including fermionic signs."""
    phase = 1
    for orbital in holes:
        if not det & (1 << orbital):
            return det, 0
        phase *= (-1) ** (det & ((1 << orbital) - 1)).bit_count()
        det ^= 1 << orbital
    for orbital in reversed(creators):
        if det & (1 << orbital):
            return det, 0
        phase *= (-1) ** (det & ((1 << orbital) - 1)).bit_count()
        det ^= 1 << orbital
    return det, phase


def _add_word(matrix: _Array, value: float, creators: tuple[int, ...], holes: tuple[int, ...]) -> None:
    """Accumulate a fermion word in the complete determinant basis."""
    if value == 0.0:
        return
    for column in range(matrix.shape[0]):
        row, phase = _apply_word(column, creators, holes)
        if phase:
            matrix[row, column] += value * phase


def _spin_integrals(hamiltonian: Hamiltonian) -> tuple[_Array, _Array]:
    """Expand chemist blocks to [alpha, beta], without assuming eightfold symmetry."""
    ha, hb = hamiltonian.get_one_body_integrals()
    n = ha.shape[0]
    gaa, gab, gbb = (block.reshape((n,) * 4) for block in hamiltonian.get_two_body_integrals())
    one = np.zeros((2 * n, 2 * n))
    two = np.zeros((2 * n,) * 4)
    one[:n, :n], one[n:, n:] = ha, hb
    two[:n, :n, :n, :n], two[n:, n:, n:, n:] = gaa, gbb
    two[:n, :n, n:, n:] = gab
    two[n:, n:, :n, :n] = gab.transpose(2, 3, 0, 1)
    return one, two


def _matrix(one: _Array, two: _Array, scalar: float) -> _Array:
    """Build E + h[p,q] a†p a_q + (pq|rs) a†p a†r a_s a_q / 2 directly."""
    matrix = scalar * np.eye(1 << one.shape[0])
    for p, q in np.ndindex(one.shape):
        _add_word(matrix, float(one[p, q]), (p,), (q,))
    for p, q, r, s in np.ndindex(two.shape):
        _add_word(matrix, float(0.5 * two[p, q, r, s]), (p, r), (q, s))
    return matrix


def _external_cluster(t1: tuple[_Array, _Array], t2: tuple[_Array, _Array, _Array], kept: tuple[int, ...]) -> _Array:
    """Build T_ext; ordered same-spin pairs absorb 1/(2!)², mixed-spin pairs have unit weight."""
    creators: tuple[int, ...]
    holes: tuple[int, ...]
    n = sum(t1[0].shape)
    cluster = np.zeros((1 << (2 * n),) * 2)
    tab, taa, tbb = t2
    for spin, (singles, doubles) in enumerate(zip(t1, (taa, tbb), strict=True)):
        no, nv = singles.shape
        for i, a in np.ndindex(singles.shape):
            creators, holes = (spin * n + no + a,), (spin * n + i,)
            if not all(p in kept for p in creators + holes):
                _add_word(cluster, float(singles[i, a]), creators, holes)
        for (i, j), (a, b) in product(combinations(range(no), 2), combinations(range(nv), 2)):
            creators, holes = (spin * n + no + a, spin * n + no + b), (spin * n + i, spin * n + j)
            if not all(p in kept for p in creators + holes):
                _add_word(cluster, float(doubles[i, j, a, b]), creators, holes)
    for i, j, a, b in np.ndindex(tab.shape):
        creators, holes = (t1[0].shape[0] + a, n + t1[1].shape[0] + b), (i, n + j)
        if not all(p in kept for p in creators + holes):
            _add_word(cluster, float(tab[i, j, a, b]), creators, holes)
    return cluster


def _normal_order_two_body(matrix: _Array, reference: int) -> _Array:
    """Extract quasiparticle degrees 0, 2, 4 relative to the exact reference bit mask."""
    creators: tuple[int, ...]
    holes: tuple[int, ...]
    size = matrix.shape[0]
    n = size.bit_length() - 1
    occupied = {p for p in range(n) if reference & (1 << p)}
    indices = np.empty(size, dtype=int)
    phases = np.ones(size)
    for quasiparticles in range(size):
        det = reference
        for p in reversed(range(n)):
            if quasiparticles & (1 << p):
                creators, holes = ((), (p,)) if p in occupied else ((p,), ())
                det, phase = _apply_word(det, creators, holes)
                assert phase != 0
                phases[quasiparticles] *= phase
        indices[quasiparticles] = det
    transformed = phases[:, None] * matrix[np.ix_(indices, indices)] * phases[None, :]
    retained = transformed[0, 0] * np.eye(size)
    for rank in (1, 2):
        increment = np.zeros_like(matrix)
        pairs = list(combinations(range(n), rank))
        for physical_creators, physical_holes in product(pairs, repeat=2):
            creators = tuple(
                sorted(
                    [p for p in physical_creators if p not in occupied] + [p for p in physical_holes if p in occupied]
                )
            )
            holes = tuple(
                sorted(
                    [p for p in physical_creators if p in occupied] + [p for p in physical_holes if p not in occupied]
                )
            )
            row, column = (sum(1 << p for p in slots) for slots in (creators, holes))
            # At this degree only this monomial connects |holes> to |creators>.
            value = transformed[row, column] - retained[row, column]
            _add_word(increment, float(value), creators, holes)
        retained += increment
    result = np.empty_like(matrix)
    result[np.ix_(indices, indices)] = phases[:, None] * retained * phases[None, :]
    return result


@pytest.mark.usefixtures("warn_logger")
@pytest.mark.parametrize("level", range(3))
@pytest.mark.parametrize(
    ("restricted", "nalpha", "nbeta"),
    [
        pytest.param(True, 2, 2, id="restricted"),
        pytest.param(False, 2, 1, id="unrestricted-alpha-majority"),
        pytest.param(False, 1, 2, id="unrestricted-beta-majority"),
    ],
)
def test_ducc_full_active_operator(
    restricted: bool, nalpha: int, nbeta: int, level: int, capfd: pytest.CaptureFixture[str]
) -> None:
    """Compare every active-Fock matrix element, including the scalar and all number sectors."""
    n = 4
    rng = np.random.default_rng(20261030)
    syms = sym.SymmetryProduct([sym.axes.spin(1, restricted)])
    labels = [sym.SymmetryLabel([sym.axes.alpha()]), sym.SymmetryLabel([sym.axes.beta()])]
    extents = dict.fromkeys(labels, n)
    window = sym.SymmetryBlockedIndexSet(syms, extents, {label: list(range(n)) for label in labels})
    orbitals = ModelOrbitals(window)
    kept_spatial = ([nalpha - 1, 2], [nbeta - 1, 2])
    p_indices = sym.SymmetryBlockedIndexSet(syms, extents, dict(zip(labels, kept_spatial, strict=True)))
    kept = tuple(spin * n + p for spin in range(2) for p in kept_spatial[spin])
    occupied = (*range(nalpha), *(n + p for p in range(nbeta)))
    reference_mask = sum(1 << p for p in occupied)
    # "2200", "2u00", "2d00" correspond to masks 0x33, 0x13, 0x31 respectively.
    occupation = "".join(
        "2" if p < min(nalpha, nbeta) else "u" if p < nalpha else "d" if p < nbeta else "0" for p in range(n)
    )
    reference = Wavefunction(StateVectorContainer(Configuration.from_spin_half_string(occupation), orbitals))

    raw = rng.normal(scale=0.07, size=(n, n))
    ha = np.diag([-1.2, -0.5, 0.3, 1.1]) + (raw + raw.T) / 2
    raw = rng.normal(scale=0.12, size=(3, n, n))
    la = (raw + raw.swapaxes(1, 2)) / 2
    # An antisymmetric Cholesky factor makes the integrals fourfold but not eightfold symmetric.
    la[2] = (raw[2] - raw[2].T) / 2
    rotation = np.eye(n) if restricted else np.linalg.qr(rng.normal(size=(n, n)))[0]
    hb = rotation.T @ ha @ rotation
    lb = np.einsum("pi,xpq,qj->xij", rotation, la, rotation)
    gaa, gab, gbb = (np.einsum("xpq,xrs->pqrs", left, right).ravel() for left, right in ((la, la), (la, lb), (lb, lb)))
    empty = np.zeros((0, 0))
    if restricted:
        container = CanonicalFourCenterHamiltonianContainer(ha, gaa, orbitals, 0.37, empty)
    else:
        container = CanonicalFourCenterHamiltonianContainer(ha, hb, gaa, gab, gbb, orbitals, 0.37, empty, empty)
    hamiltonian = Hamiltonian(container)

    t1a = rng.normal(scale=0.04, size=(nalpha, n - nalpha))
    t1b = t1a.copy() if restricted else rng.normal(scale=0.04, size=(nbeta, n - nbeta))
    tab = rng.normal(scale=0.04, size=(nalpha, nbeta, n - nalpha, n - nbeta))
    if restricted:
        tab = (tab + tab.transpose(1, 0, 3, 2)) / 2
        taa = tab - tab.swapaxes(2, 3)
        tbb = taa.copy()
        amplitudes = AmplitudeContainer(orbitals, reference, AmplitudeType.CoupledCluster, t1a.ravel(), tab.ravel())
    else:
        same_spin = []
        for no in (nalpha, nbeta):
            raw = rng.normal(scale=0.04, size=(no, no, n - no, n - no))
            antisymmetric = raw - raw.swapaxes(0, 1)
            same_spin.append((antisymmetric - antisymmetric.swapaxes(2, 3)) / 4)
        taa, tbb = same_spin
        amplitudes = AmplitudeContainer(
            orbitals,
            reference,
            AmplitudeType.CoupledCluster,
            t1a.ravel(),
            t1b.ravel(),
            tab.ravel(),
            taa.ravel(),
            tbb.ravel(),
        )
    # Keep internal amplitudes in the native input: masking belongs only in T_ext.
    assert t1a[nalpha - 1, 2 - nalpha] != 0.0
    assert tab[nalpha - 1, nbeta - 1, 2 - nalpha, 2 - nbeta] != 0.0
    cluster = _external_cluster((t1a, t1b), (tab, taa, tbb), kept)
    sigma = cluster - cluster.T
    one, two = _spin_integrals(hamiltonian)
    bare = _matrix(one, two, 0.37)
    fock: _Array = one + sum((two[:, :, i, i] - two[:, i, i, :] for i in occupied), np.zeros_like(one))
    normal_fock = _matrix(fock, np.zeros_like(two), -sum(float(fock[i, i]) for i in occupied))
    dressed = bare.copy()
    if level:
        # ad_sigma(A) = [A, sigma]; H through level, then F at level+1, with k! weights.
        nested_h, nested_f = bare, normal_fock
        for order in range(1, level + 2):
            nested_f = nested_f @ sigma - sigma @ nested_f
            if order <= level:
                nested_h = nested_h @ sigma - sigma @ nested_h
                dressed += nested_h / factorial(order)
            else:
                dressed += nested_f / factorial(order)
    no2 = _normal_order_two_body(dressed, reference_mask)

    # Freeze Q to its reference occupancy. The embedding is a†P...|Q>, not unsigned bit insertion.
    core_mask = sum(1 << p for p in occupied if p not in kept)
    embedding = [
        _apply_word(core_mask, tuple(p for j, p in enumerate(kept) if det & (1 << j)), ())
        for det in range(1 << len(kept))
    ]
    indices = [det for det, _ in embedding]
    phases = np.array([phase for _, phase in embedding])
    expected = phases[:, None] * no2[np.ix_(indices, indices)] * phases[None, :]
    constructor = create("effective_hamiltonian_constructor", "qdk_ducc", level=level)
    result = constructor.run(Wavefunction(amplitudes), hamiltonian, p_indices)
    observed = _matrix(*_spin_integrals(result), result.get_core_energy())
    np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(observed, observed.T, atol=2e-10, rtol=2e-10)
    assert result.is_restricted() == restricted
    output_orbitals = result.get_orbitals()
    assert output_orbitals.is_restricted() == restricted
    if restricted:
        # MACIS accepts only restricted Hamiltonians; its active sector holds one alpha and one beta electron.
        capfd.readouterr()
        energy, _ = create("multi_configuration_calculator", "macis_cas").run(result, 1, 1)
        assert "Hermitian part" not in capfd.readouterr().out
        sector = [det for det in range(len(observed)) if (det & 3).bit_count() == 1 and (det >> 2).bit_count() == 1]
        assert energy == pytest.approx(np.linalg.eigvalsh(observed[np.ix_(sector, sector)])[0], abs=1e-10)
    for spin, label in enumerate(labels):
        assert output_orbitals.active_indices().indices(label) == tuple(kept_spatial[spin])
        frozen = tuple(p for p in range((nalpha, nbeta)[spin]) if p not in kept_spatial[spin])
        assert output_orbitals.inactive_indices().indices(label) == frozen


@pytest.mark.parametrize("level", [-1, 3, 4])
def test_ducc_rejects_unsupported_level(level: int) -> None:
    """The public factory accepts only DUCC levels zero through two."""
    with pytest.raises(ValueError, match="level"):
        create("effective_hamiltonian_constructor", "qdk_ducc", level=level)


def test_restricted_amplitude_conventions() -> None:
    """Preserve spatial and expanded restricted amplitudes, including serialization."""
    orbitals = Orbitals(np.eye(2), None, None, create_test_basis_set(2, "ducc-round-trip"))
    determinant = Wavefunction(StateVectorContainer(Configuration.from_spin_half_string("20"), orbitals))
    hamiltonian = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(np.diag([-1.0, 0.4]), np.full(16, 0.1), orbitals, 0.2, np.zeros((0, 0)))
    )
    target = sym.spin_index_set(2, [0], [0], equivalent=True)
    singles, doubles, zeros = np.array([0.03]), np.array([0.1]), np.zeros(1)

    def run(wavefunction: Wavefunction) -> _Array:
        constructor = create("effective_hamiltonian_constructor", "qdk_ducc", level=1)
        result = constructor.run(wavefunction, hamiltonian, target)
        return _matrix(*_spin_integrals(result), result.get_core_energy())

    spatial = Wavefunction(AmplitudeContainer(orbitals, determinant, AmplitudeType.CoupledCluster, singles, doubles))
    expected = run(spatial)
    expanded = AmplitudeContainer(
        orbitals, determinant, AmplitudeType.CoupledCluster, singles, singles, doubles, zeros, zeros
    )
    np.testing.assert_array_equal(run(Wavefunction(expanded)), expected)
    restored = Wavefunction.from_json(spatial.to_json())
    constructor = create("effective_hamiltonian_constructor", "qdk_ducc", level=1)
    result = constructor.run(restored, hamiltonian, target)
    np.testing.assert_array_equal(_matrix(*_spin_integrals(result), result.get_core_energy()), expected)
    for beta, same in [(np.array([0.04]), zeros), (singles, np.array([0.05]))]:
        incompatible = AmplitudeContainer(
            orbitals, determinant, AmplitudeType.CoupledCluster, singles, beta, doubles, same, zeros
        )
        with pytest.raises(ValueError, match="Restricted DUCC"):
            run(Wavefunction(incompatible))


def test_nonaufbau_reference_uses_occupied_orbital_positions() -> None:
    """The active expectation follows the actual determinant, not Aufbau counts."""
    syms = sym.SymmetryProduct([sym.axes.spin(1, False)])
    orbitals = ModelOrbitals(2, syms)
    reference = Wavefunction(StateVectorContainer(Configuration.from_spin_half_string("02"), orbitals))
    one = np.diag([1.0, 0.0])
    zeros = np.zeros(16)
    hamiltonian = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            one, one, zeros, zeros, zeros, orbitals, 0.0, np.zeros((0, 0)), np.zeros((0, 0))
        )
    )
    amplitudes = AmplitudeContainer(
        orbitals,
        reference,
        AmplitudeType.CoupledCluster,
        np.array([0.2]),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
    )
    target = sym.spin_index_set(2, [1], [1], equivalent=False)
    constructor = create("effective_hamiltonian_constructor", "qdk_ducc", level=2)
    result = constructor.run(Wavefunction(amplitudes), hamiltonian, target)
    matrix = _matrix(*_spin_integrals(result), result.get_core_energy())
    assert matrix[3, 3] == pytest.approx(0.2**2, abs=1e-12)
