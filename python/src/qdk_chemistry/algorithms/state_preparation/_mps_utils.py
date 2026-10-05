"""Shared validation and encoding helpers for MPS state preparation algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.data import Configuration, MPSContainer, MPSSite
from qdk_chemistry.data.symmetry import (
    SymmetryBlockedTensorRank3,
    SymmetryBlockedTensorRank3Complex,
    SymmetryLabel,
    SymmetryProduct,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdk_chemistry.data import Wavefunction

__all__: list[str] = []

# Supported physical bases by the number of physical states per site. A site with two states
# is one spinless mode on one qubit; a site with four states is one spatial orbital on two
# qubits, its alpha and beta modes.
_PHYSICAL_BASIS_LABELS = {2: ("0", "1"), 4: ("0", "u", "d", "2")}
_PHYSICAL_BASES = {
    2: [Configuration.from_bitstring(state) for state in _PHYSICAL_BASIS_LABELS[2]],
    4: [Configuration.from_spin_half_string(state) for state in _PHYSICAL_BASIS_LABELS[4]],
}
_DIMENSION_ERROR = "MPS state preparation requires two or four physical states per site."


def validate_mps_physical_basis(sites: Sequence[MPSSite]) -> int:
    """Require a physical dimension and basis order supported by the Q# operations.

    Args:
        sites: MPS sites to check.

    Returns:
        The number of physical states per site.

    Raises:
        ValueError: If the sites do not share one physical dimension, the dimension is not two or
            four, or a site does not order its basis as ``('0', '1')`` or ``('0', 'u', 'd', '2')``.

    """
    dimensions = {site.physical_dimension for site in sites}
    if len(dimensions) > 1:
        raise ValueError("MPS state preparation requires the same physical dimension on every site.")
    (dimension,) = dimensions
    if dimension not in _PHYSICAL_BASES:
        raise ValueError(_DIMENSION_ERROR)
    if any(site.physical_basis != _PHYSICAL_BASES[dimension] for site in sites):
        labels = ", ".join(f"'{label}'" for label in _PHYSICAL_BASIS_LABELS[dimension])
        raise ValueError(f"MPS state preparation requires physical basis ordering ({labels}).")
    return dimension


def validate_mps_wavefunction(wavefunction: Wavefunction, algorithm: str, description: str) -> MPSContainer:
    """Return the MPS container of a wavefunction after checking it can be prepared.

    Args:
        wavefunction: Wavefunction to prepare.
        algorithm: Class name used in the type error.
        description: Algorithm description used in value errors.

    Returns:
        The validated MPS container.

    Raises:
        TypeError: If the wavefunction is not backed by an :class:`~qdk_chemistry.data.MPSContainer`.
        ValueError: If the MPS is complex, does not use one supported physical basis on every
            site, is not right-canonical with orthogonality center zero, or does not have exactly
            one site per molecular orbital.

    """
    container = wavefunction.get_container()
    if not isinstance(container, MPSContainer):
        raise TypeError(f"{algorithm} requires an MPSContainer, got {type(container)}.")
    if container.is_complex:
        raise ValueError(f"{description} currently supports only real-valued MPS tensors.")
    validate_mps_physical_basis(container.sites)
    if container.orthogonality_center != 0:
        raise ValueError(f"{description} requires a right-canonical MPS with center zero.")
    if container.num_sites != container.orbitals.get_num_molecular_orbitals():
        raise ValueError(f"{description} requires exactly one MPS site per molecular orbital.")
    return container


def as_mps_sites(tensors: Sequence[np.ndarray | MPSSite], description: str) -> list[MPSSite]:
    """Wrap dense inputs as MPS sites and check the chain structure without densifying it.

    Args:
        tensors: MPS sites or dense ``(chi_left, d, chi_right)`` arrays in chain order.
        description: Algorithm description used in error messages.

    Returns:
        The MPS sites.

    Raises:
        ValueError: If a dense input is not rank three, does not have two or four physical
            states, or is not finite, or if the chain is empty, complex, not open-boundary, or
            uses an unsupported physical basis.

    """
    sites = []
    for tensor in tensors:
        if isinstance(tensor, MPSSite):
            sites.append(tensor)
            continue
        # Dense arrays become single-block trivial-symmetry sites with the default basis.
        values = np.asarray(tensor)
        if values.ndim != 3:
            raise ValueError("Dense MPS site tensors must have shape (chi_left, d, chi_right).")
        left, physical, right = values.shape
        if physical not in _PHYSICAL_BASES:
            raise ValueError(_DIMENSION_ERROR)
        if not np.isfinite(values).all():
            raise ValueError("MPS sites must contain finite amplitudes with nonzero norm.")
        product, label = SymmetryProduct([]), SymmetryLabel([])
        tensor_type = SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else SymmetryBlockedTensorRank3
        blocked = tensor_type(
            [product] * 3,
            [{label: left}, {label: physical}, {label: right}],
            [((label,) * 3, values.reshape(left * physical, right))],
        )
        sites.append(MPSSite(blocked, [label], [label], [label]))
    if not sites:
        raise ValueError(f"{description} requires at least one site.")
    if any(site.is_complex for site in sites):
        raise ValueError(f"{description} currently supports only real-valued MPS tensors.")
    if sites[0].left_bond_dimension != 1 or sites[-1].right_bond_dimension != 1:
        raise ValueError(f"{description} requires open boundary bonds of dimension one.")
    validate_mps_physical_basis(sites)
    return sites


def first_site_amplitudes(sites: Sequence[MPSSite]) -> np.ndarray:
    """Return the amplitudes of the first site after checking every site has a nonzero norm.

    Sites are densified one at a time, so absent symmetry blocks are materialized as zeros
    without holding the whole chain in dense form.

    Args:
        sites: Open-boundary MPS sites in chain order.

    Returns:
        The ``(d, chi_1)`` amplitudes of the first site.

    Raises:
        ValueError: If any site has non-finite amplitudes or zero norm.

    """
    for site in sites:
        amplitudes = site.to_dense()
        if not np.isfinite(amplitudes).all() or np.linalg.norm(amplitudes) <= 1e-15:
            raise ValueError("MPS sites must contain finite amplitudes with nonzero norm.")
    # With a left bond of dimension one the packed (left * d, right) matrix is (d, chi_1).
    return sites[0].to_dense()


def qubits_per_site(sites: Sequence[MPSSite]) -> int:
    """Return the physical qubits per site of a chain with a validated physical basis.

    Args:
        sites: MPS sites that share a physical dimension of two or four.

    Returns:
        One for the ``('0', '1')`` basis and two for the ``('0', 'u', 'd', '2')`` basis.

    """
    return (sites[0].physical_dimension - 1).bit_length()


def ancilla_bits_for(sites: Sequence[MPSSite]) -> int:
    """Return the width of a bond register that holds every bond of the chain.

    Args:
        sites: MPS sites in chain order.

    Returns:
        ``ceil(log2(max bond dimension))``, and at least one.

    """
    max_bond = max(max(site.left_bond_dimension, site.right_bond_dimension) for site in sites)
    return max(1, (max_bond - 1).bit_length())


def initial_state_vector(first_site_amplitudes: np.ndarray, ancilla_dim: int) -> list[float]:
    """Return the normalized amplitudes that prepare the first site and its right bond.

    Args:
        first_site_amplitudes: ``(d, chi_1)`` amplitudes of the first site.
        ancilla_dim: Dimension of the bond register.

    Returns:
        Amplitudes indexed by ``physical * ancilla_dim + bond``.

    Raises:
        ValueError: If the amplitudes are non-finite or have zero norm.

    """
    padded = np.zeros((first_site_amplitudes.shape[0], ancilla_dim))
    padded[:, : first_site_amplitudes.shape[1]] = first_site_amplitudes
    vector = padded.flatten()
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")
    return (vector / norm).tolist()


def validate_site_to_orbital_order(site_to_orbital_order: Sequence[int] | None, num_sites: int) -> list[int]:
    """Return a site-to-orbital map, defaulting to the identity.

    Args:
        site_to_orbital_order: Orbital that holds each chain site, or ``None`` for the identity.
        num_sites: Number of MPS sites.

    Returns:
        The site-to-orbital map.

    Raises:
        ValueError: If the map does not contain one unique nonnegative index per site.

    """
    order = list(range(num_sites)) if site_to_orbital_order is None else list(site_to_orbital_order)
    if len(order) != num_sites or len(set(order)) != num_sites or any(index < 0 for index in order):
        raise ValueError("site_to_orbital_order must contain one unique nonnegative index per MPS site.")
    return order
