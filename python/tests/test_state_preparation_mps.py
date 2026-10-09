"""Tests for matrix product state preparation with dense and block-sparse unitary synthesis.

Tests the native site synthesis (each site unitary reproduces its site isometry), input
validation, the Q# circuits (state preparation fidelity via statevector simulation), and their
resource estimates.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import functools
import itertools
from collections.abc import Sequence

import numpy as np
import pytest
from qdk.qsharp import QSharpError

from qdk_chemistry._core.utils.unitary_synthesis import (
    DenseSiteSynthesis,
    SparseSiteSynthesis,
    block_sparse_unitary_synthesis,
    dense_unitary_synthesis,
    matrix_product_state_synthesis,
)
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation.matrix_product_state import (
    MatrixProductStatePreparation,
    MatrixProductStatePreparationData,
)
from qdk_chemistry.data import Configuration, MPSContainer, MPSSite, Orbitals, Wavefunction
from qdk_chemistry.data import symmetry as sym
from qdk_chemistry.utils.qsharp import create_qsharp_context, get_qsharp_context, use_qsharp_context

from .test_helpers import create_test_basis_set, create_test_orbitals, create_test_wavefunction

# Q# operation and site struct of each synthesis method.
_QSHARP_OPERATIONS = {
    "dense": ("QDKChemistry.Utils.MPSSequential.MPSSequential", "QDKChemistry.Utils.MPSSequential.DenseSiteSynthesis"),
    "block_sparse": ("QDKChemistry.Utils.MPSSparse.MPSSparse", "QDKChemistry.Utils.MPSSparse.SparseSiteSynthesis"),
}
_METHODS = pytest.mark.parametrize("method", list(_QSHARP_OPERATIONS))

_PHYSICAL_PARTICLE_NUMBERS = {
    2: (0, 1),  # Particle numbers of the ('0', '1') physical basis states.
    4: (0, 1, 1, 2),  # Particle numbers of the ('0', 'u', 'd', '2') physical basis states.
}

_SINGLE_SITE = np.array([[[1.0], [0.0], [0.0], [0.0]]])


def make_site(values: np.ndarray, physical_basis: Sequence[Configuration] | None = None) -> MPSSite:
    """Wrap a dense ``(left, physical, right)`` array as one trivial-symmetry MPS block."""
    left, physical, right = values.shape
    product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
    tensor_type = sym.SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else sym.SymmetryBlockedTensorRank3
    tensor = tensor_type(
        [product] * 3,
        [{label: left}, {label: physical}, {label: right}],
        [((label,) * 3, values.reshape(left * physical, right))],
    )
    return MPSSite(tensor, [label], [label], [label], list(physical_basis or []))


def particle_number_blocked_sites(tensors: Sequence[np.ndarray], tolerance: float = 1e-12) -> list[MPSSite]:
    """Store dense sites as particle-number-blocked MPS sites.

    Bond particle numbers are propagated from left to right using every entry above
    ``tolerance``. Bond indices must already be grouped by particle number. Each block
    copies its dense slice exactly, and exactly-zero blocks are omitted. Sites use the
    ``('0', '1')`` or ``('0', 'u', 'd', '2')`` physical basis by their physical dimension.
    """
    physical_particle_numbers = _PHYSICAL_PARTICLE_NUMBERS[tensors[0].shape[1]]
    max_particles = max(physical_particle_numbers) * len(tensors)
    product = sym.SymmetryProduct([sym.axes.particle_number(max_particles)])

    def label(count: int) -> sym.SymmetryLabel:
        return sym.SymmetryLabel([sym.axes.particle_number_value(count)])

    def sectors(numbers: Sequence[int]) -> list[tuple[int, int, int]]:
        """Return ``(particle_number, start, stop)`` for contiguous bond sectors."""
        result: list[tuple[int, int, int]] = []
        for index, count in enumerate(numbers):
            if result and result[-1][0] == count:
                result[-1] = (count, result[-1][1], index + 1)
            else:
                if any(existing == count for existing, _, _ in result):
                    raise ValueError("Bond indices must be grouped by particle number.")
                result.append((count, index, index + 1))
        return result

    physical_sectors = sectors(physical_particle_numbers)
    left_numbers = [0]
    sites = []
    for tensor in tensors:
        right_numbers: list[int | None] = [None] * tensor.shape[2]
        for left, physical, right in zip(*np.nonzero(np.abs(tensor) > tolerance), strict=True):
            count = left_numbers[left] + physical_particle_numbers[physical]
            if right_numbers[right] not in (None, count):
                raise ValueError("Dense tensor does not conserve particle number.")
            right_numbers[right] = count
        resolved_right_numbers = [count for count in right_numbers if count is not None]
        if len(resolved_right_numbers) != len(right_numbers):
            raise ValueError("Every right-bond index needs a nonzero entry.")
        left_sectors, right_sectors = sectors(left_numbers), sectors(resolved_right_numbers)
        blocks = []
        for left_count, left_start, left_stop in left_sectors:
            for physical_count, physical_start, physical_stop in physical_sectors:
                for right_count, right_start, right_stop in right_sectors:
                    block = tensor[left_start:left_stop, physical_start:physical_stop, right_start:right_stop]
                    if np.any(block != 0.0):
                        key = (label(left_count), label(physical_count), label(right_count))
                        blocks.append((key, block.reshape(-1, right_stop - right_start)))
        extents = [
            {label(count): stop - start for count, start, stop in slot_sectors}
            for slot_sectors in (left_sectors, physical_sectors, right_sectors)
        ]
        sites.append(
            MPSSite(
                sym.SymmetryBlockedTensorRank3([product] * 3, extents, blocks),
                [label(count) for count, _, _ in left_sectors],
                [label(count) for count, _, _ in physical_sectors],
                [label(count) for count, _, _ in right_sectors],
            )
        )
        left_numbers = resolved_right_numbers
    return sites


def make_mps(
    tensors: Sequence[np.ndarray | MPSSite],
    orthogonality_center: int | None = 0,
    site_to_orbital_order: Sequence[int] | None = None,
) -> MPSContainer:
    """Construct a native MPS container from dense test tensors or prebuilt sites."""
    sites = [tensor if isinstance(tensor, MPSSite) else make_site(np.asarray(tensor)) for tensor in tensors]
    return MPSContainer(
        sites,
        create_test_orbitals(max(1, len(sites))),
        orthogonality_center=orthogonality_center,
        site_to_orbital_order=list(site_to_orbital_order or []),
    )


def preparation_data(mps: MPSContainer, unitary_synthesis: str = "dense") -> MatrixProductStatePreparationData:
    """Generate preparation data from a container using the selected algorithm setting."""
    preparer = MatrixProductStatePreparation()
    preparer.settings().update("unitary_synthesis", unitary_synthesis)
    return preparer.generate_matrix_product_state_preparation_data(mps)


def right_normalized_tensors(tensors: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return right-canonical dense tensors preserving the normalized state."""
    normalized = [np.array(tensor, dtype=float, copy=True) for tensor in tensors]
    for site in range(len(normalized) - 1, 0, -1):
        chi_left, physical, chi_right = normalized[site].shape
        matrix = normalized[site].reshape(chi_left, physical * chi_right)
        q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
        normalized[site] = q_matrix.T.reshape(chi_left, physical, chi_right)
        previous_left, previous_physical, _ = normalized[site - 1].shape
        previous = normalized[site - 1].reshape(previous_left * previous_physical, chi_left)
        normalized[site - 1] = (previous @ r_matrix.T).reshape(previous_left, previous_physical, chi_left)
    normalized[0] /= np.linalg.norm(normalized[0])
    return normalized


def right_normalized_mps(tensors: Sequence[np.ndarray]) -> MPSContainer:
    """Construct a right-canonical MPS preserving the normalized state."""
    return make_mps(right_normalized_tensors(tensors))


def random_mps(
    num_sites: int,
    bond_dim: int,
    site_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> MPSContainer:
    """Construct a right-normalized random native MPS for algorithm tests."""
    rng = np.random.default_rng() if rng is None else rng
    bond_dims = [1]
    for site in range(1, num_sites):
        max_left = bond_dims[-1] * site_dim
        max_right = site_dim ** min(site, num_sites - site)
        bond_dims.append(min(bond_dim, max_left, max_right))
    bond_dims.append(1)

    tensors = [rng.standard_normal((bond_dims[site], site_dim, bond_dims[site + 1])) for site in range(num_sites)]
    return right_normalized_mps(tensors)


def random_particle_number_tensors(
    num_sites: int,
    num_particles: int,
    max_bond: int,
    site_dim: int = 4,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Return right-canonical dense tensors of a random state with a fixed particle number.

    Each bond index carries the particle number to its left, and bond indices are grouped by
    that number, so the tensors are block-sparse and :func:`particle_number_blocked_sites`
    can store them as particle-number-blocked sites. Each bond sector is as large as the
    configurations on both sides of the bond allow, capped at ``max_bond``.
    """
    rng = np.random.default_rng() if rng is None else rng
    physical_numbers = _PHYSICAL_PARTICLE_NUMBERS[site_dim]

    # configurations[sites][particles]: physical-index strings of `sites` sites with `particles` particles.
    configurations = [np.zeros(num_particles + 1, dtype=int) for _ in range(num_sites + 1)]
    configurations[0][0] = 1
    for sites in range(1, num_sites + 1):
        for count in physical_numbers:
            configurations[sites][count:] += configurations[sites - 1][: num_particles + 1 - count]

    def bond_sectors(bond: int) -> dict[int, int]:
        """Return the sector dimensions of the bond after ``bond`` sites, keyed by particle number."""
        return {
            count: min(
                max_bond, int(configurations[bond][count]), int(configurations[num_sites - bond][num_particles - count])
            )
            for count in range(num_particles + 1)
            if configurations[bond][count] and configurations[num_sites - bond][num_particles - count]
        }

    def offsets(sectors: dict[int, int]) -> dict[int, int]:
        return dict(zip(sectors, np.cumsum([0, *sectors.values()])[:-1].tolist(), strict=True))

    tensors = []
    for site in range(num_sites):
        left_sectors, right_sectors = bond_sectors(site), bond_sectors(site + 1)
        left_offsets, right_offsets = offsets(left_sectors), offsets(right_sectors)
        tensor = np.zeros((sum(left_sectors.values()), site_dim, sum(right_sectors.values())))
        for left_count, left_dim in left_sectors.items():
            columns = [
                (physical, right_offsets[left_count + count] + index)
                for physical, count in enumerate(physical_numbers)
                if left_count + count in right_sectors
                for index in range(right_sectors[left_count + count])
            ]
            # Orthonormal rows within the sector keep the site right-orthonormal.
            rows, _ = np.linalg.qr(rng.standard_normal((len(columns), left_dim)))
            for row in range(left_dim):
                for column, (physical, right) in enumerate(columns):
                    tensor[left_offsets[left_count] + row, physical, right] = rows[column, row]
        tensors.append(tensor)
    return tensors


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random real orthogonal matrix."""
    matrix, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return matrix


def reconstruct_givens(givens) -> np.ndarray:
    """Reconstruct ``D · L_l · ... · L_1`` from a ``GivensDecomposition``."""
    result = np.eye(len(givens.phases))
    for angles, shifted in zip(givens.layer_angles, givens.layer_shifted, strict=True):
        upper = (1 if shifted else 0) + 2 * np.arange(len(angles))
        cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
        first, second = result[upper], result[upper + 1]
        result[upper] = cosine * first - sine * second
        result[upper + 1] = sine * first + cosine * second
    return np.where(np.asarray(givens.phases, dtype=bool), -1.0, 1.0)[:, None] * result


def site_isometry(tensor: np.ndarray, ancilla_dim: int) -> np.ndarray:
    """Return the isometry of a ``(left, physical, right)`` site on the joint register.

    Row ``physical * ancilla_dim + right`` and column ``left`` hold ``tensor[left, physical, right]``,
    with the right bond padded to ``ancilla_dim``.
    """
    left, physical, right = tensor.shape
    padded = np.zeros((physical, ancilla_dim, left))
    padded[:, :right, :] = tensor.transpose(1, 2, 0)
    return padded.reshape(physical * ancilla_dim, left)


def dense_site(site: MPSSite) -> np.ndarray:
    """Return a site's ``(left, physical, right)`` array from its packed dense matrix."""
    return site.to_dense().reshape(site.shape)


def contract_mps(wavefunction: MPSContainer) -> np.ndarray:
    """Contract a native MPS into a normalized dense state vector."""
    state = dense_site(wavefunction.sites[0])
    for site in wavefunction.sites[1:]:
        tensor = dense_site(site)
        left, num_states, previous_bond = state.shape
        incoming_bond, physical, outgoing_bond = tensor.shape
        state = (state.reshape(left * num_states, previous_bond) @ tensor.reshape(incoming_bond, -1)).reshape(
            left, num_states * physical, outgoing_bond
        )
    vector = state.sum(axis=0).flatten()
    norm = np.linalg.norm(vector)
    return vector if norm <= 1e-15 else vector / norm


def blocked_jordan_wigner_state(
    chain_state: np.ndarray,
    site_to_orbital_order: Sequence[int] | None = None,
    qubits_per_site: int = 2,
) -> np.ndarray:
    """Map a contracted MPS state to the blocked Jordan-Wigner qubit basis.

    ``chain_state`` is indexed like :func:`contract_mps`: chain site 0 is the most-significant
    group of ``qubits_per_site`` bits holding the physical index. With two qubits per site the
    physical index is in the ``('0', 'u', 'd', '2')`` basis and its basis states create the alpha
    and then the beta mode of each site in chain order; with one qubit per site it is in the
    ``('0', '1')`` basis of one spinless mode. The result is indexed by the qubit occupations of
    :meth:`Configuration.to_bits`, qubit 0 most significant, and carries the sign of reordering
    the creation operators into qubit order.
    """
    num_sites = (len(chain_state).bit_length() - 1) // qubits_per_site
    order = list(range(num_sites)) if site_to_orbital_order is None else list(site_to_orbital_order)
    num_qubits = qubits_per_site * num_sites
    labels, parse = (
        ("01", Configuration.from_bitstring) if qubits_per_site == 1 else ("0ud2", Configuration.from_spin_half_string)
    )
    blocked = np.zeros(1 << num_qubits, dtype=chain_state.dtype)
    for index in np.flatnonzero(chain_state):
        orbital_states = ["0"] * num_sites
        created: list[int] = []
        for site, orbital in enumerate(order):
            physical = (int(index) >> (qubits_per_site * (num_sites - 1 - site))) & ((1 << qubits_per_site) - 1)
            orbital_states[orbital] = labels[physical]
            # Physical bit k is the mode on qubit k * num_sites + orbital (alpha, then beta).
            created += [channel * num_sites + orbital for channel in range(qubits_per_site) if physical >> channel & 1]
        bits = parse("".join(orbital_states)).to_bits(num_qubits)
        assert sorted(created) == [qubit for qubit, bit in enumerate(bits) if bit]
        inversions = sum(first > second for i, first in enumerate(created) for second in created[i + 1 :])
        blocked[int("".join(map(str, bits)), 2)] = (-1) ** inversions * chain_state[index]
    return blocked


def simulate_mps_preparation(
    operation: str,
    site_struct: str,
    params: dict,
    num_sites: int,
    num_ancilla_qubits: int,
) -> tuple[float, np.ndarray]:
    """Run an MPS preparation operation and post-select the bond register on ``|0>``.

    Args:
        operation: Fully qualified Q# operation with signature
            ``(initialStateVec, numSites, siteToOrbitalOrder, siteDecompositions, state, ancilla, phaseGradient)``.
        site_struct: Fully qualified Q# struct of one site decomposition.
        params: Parameters produced by ``to_qsharp_params``.
        num_sites: Number of MPS sites.
        num_ancilla_qubits: Width of the bond register.

    Returns:
        ``P(ancilla = 0)`` and the normalized post-selected state in the blocked Jordan-Wigner
        qubit basis, qubit 0 most significant (see :func:`blocked_jordan_wigner_state`).

    """
    parameter_names = ("initialStateVec", "numSites", "siteToOrbitalOrder", "siteDecompositions")
    arguments = ", ".join(_to_qsharp_literal(params[name], site_struct) for name in parameter_names)
    num_state_qubits = params["numQubitsPerSite"] * num_sites
    context = get_qsharp_context()
    context.eval(f"use state = Qubit[{num_state_qubits}];")
    context.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
    context.eval(
        f"{{ use phaseGradient = Qubit[{params['rotationBitPrecision']}]; "
        "within { QDKChemistry.Utils.PhaseGradient.PreparePhaseGradientState(phaseGradient); } "
        f"apply {{ {operation}({arguments}, state, ancilla, phaseGradient); }} }}"
    )
    dump = context.dump_machine()
    context.eval("ResetAll(state + ancilla);")

    # DumpMachine orders qubits state[0]..state[N-1], ancilla[0].. from the most-significant
    # bit, so the bond register occupies the lowest bits. Basis states with any other qubit
    # set (scratch qubits that were not uncomputed) are excluded.
    num_relevant_qubits = num_state_qubits + num_ancilla_qubits
    ancilla_mask = (1 << num_ancilla_qubits) - 1
    amplitudes = np.zeros(1 << num_state_qubits, dtype=complex)
    for idx in dump:
        if idx >> num_relevant_qubits == 0 and idx & ancilla_mask == 0:
            amplitudes[idx >> num_ancilla_qubits] = dump[idx]
    ancilla_zero_probability = float(np.sum(np.abs(amplitudes) ** 2))
    return ancilla_zero_probability, amplitudes / np.sqrt(ancilla_zero_probability)


_GIVENS_STRUCT = "QDKChemistry.Utils.UnitarySynthesis.GivensDecomposition"
_GIVENS_FIELDS = {"layerAngles", "layerShifted", "phases"}


def _to_qsharp_literal(value, site_struct: str) -> str:
    """Serialize nested numeric and Boolean data as a Q# literal.

    Dictionaries with the fields of the Q# ``GivensDecomposition`` struct become that struct, and
    every other dictionary becomes ``site_struct``.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.15f}"
    if isinstance(value, dict):
        fields = ", ".join(f"{name} = {_to_qsharp_literal(item, site_struct)}" for name, item in value.items())
        struct = _GIVENS_STRUCT if value.keys() == _GIVENS_FIELDS else site_struct
        return f"new {struct} {{ {fields} }}"
    if isinstance(value, list):
        return f"[{', '.join(_to_qsharp_literal(item, site_struct) for item in value)}]"
    return str(value)


def multiplexed_ry(angles, physical: int, chi: int, target_bit: int, control_bit: int | None = None) -> np.ndarray:
    """Ry rotations of physical qubit ``target_bit`` addressed by the bond state.

    Basis state ``p * chi + a`` holds physical state ``p``, whose bit ``k`` is physical qubit ``k``,
    and bond state ``a``. An optional physical qubit ``control_bit`` must be set.
    """
    result = np.eye(physical * chi)
    for p in range(physical):
        if p >> target_bit & 1 or (control_bit is not None and not p >> control_bit & 1):
            continue
        flipped = p | 1 << target_bit
        for a, angle in enumerate(angles):
            cosine, sine = np.cos(angle / 2), np.sin(angle / 2)
            zero, one = p * chi + a, flipped * chi + a
            result[[zero, zero, one, one], [zero, one, zero, one]] = [cosine, -sine, sine, cosine]
    return result


def controlled_bond_unitary(unitary: np.ndarray, control_bit: int) -> np.ndarray:
    """Apply ``unitary`` to the bond register when physical qubit ``control_bit`` is set."""
    chi = len(unitary)
    result = np.eye(4 * chi)
    for p in range(4):
        if p >> control_bit & 1:
            result[p * chi : (p + 1) * chi, p * chi : (p + 1) * chi] = unitary
    return result


def site_unitary(synthesis: DenseSiteSynthesis | SparseSiteSynthesis, chi: int) -> np.ndarray:
    """Return the unitary that a synthesized site applies to the joint physical and bond register.

    Dense sites follow the ``MPSSequential`` circuit (Fig. 5 of Rupprecht & Wölk). Block-sparse
    sites apply the block-diagonal unitary between their row and column permutations.
    """
    block = reconstruct_givens(synthesis.block_givens)
    if isinstance(synthesis, SparseSiteSynthesis):
        return block[np.ix_(np.argsort(synthesis.row_permutation), synthesis.column_permutation)]
    rotation_angles, mixing_givens = synthesis.rotation_angles, synthesis.mixing_givens
    if len(block) == 2 * chi:
        return block @ multiplexed_ry(rotation_angles[0], 2, chi, 0)
    # CNOT with physical qubit 1 as control and physical qubit 0 as target.
    cnot = np.zeros((4 * chi, 4 * chi))
    for p in range(4):
        mapped = p ^ 1 if p & 2 else p
        cnot[mapped * chi : (mapped + 1) * chi, p * chi : (p + 1) * chi] = np.eye(chi)
    return (
        block
        @ multiplexed_ry(rotation_angles[2], 4, chi, 0, 1)
        @ controlled_bond_unitary(reconstruct_givens(mixing_givens[1]), 1)
        @ cnot
        @ multiplexed_ry(rotation_angles[1], 4, chi, 1, 0)
        @ controlled_bond_unitary(reconstruct_givens(mixing_givens[0]), 0)
        @ cnot
        @ multiplexed_ry(rotation_angles[0], 4, chi, 0)
    )


# Fixed four-site MPS from the Qualtran MPSPreparation tests (Apache-2.0).
REFERENCE_MPS_TENSORS = (
    np.array(
        [
            [
                [0.01650572, 0.0, 0.0, 0.0],
                [0.0, -0.52929781, 0.0, 0.0],
                [0.0, 0.0, -0.84462254, 0.0],
                [0.0, 0.0, 0.0, -0.07863941],
            ]
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.05969264, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9973967, 0.04045497, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.08381532, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.98376348, 0.15869598, 0.0],
            ],
            [
                [-0.0421477, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.46961402, 0.0265522, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.41109095, 0.03268939, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.77904869],
            ],
        ]
    ),
    np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.19640516, 0.0, 0.0, 0.0], [0.0, -0.98052283, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.98052283, 0.0, 0.0, 0.0], [0.0, 0.19640516, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [-0.02411236, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.99970925, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [-0.99970925, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.02411236, 0.0]],
            [
                [-0.17695837, 0.0, 0.0, 0.0],
                [0.0, -0.58052668, 0.0, 0.0],
                [0.0, 0.0, -0.53176612, 0.0],
                [0.0, 0.0, 0.0, -0.59067698],
            ],
        ]
    ),
    np.array(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[1.0], [0.0], [0.0], [0.0]],
        ]
    ),
)

REFERENCE_MPS_EXPECTED_STATE = np.array(
    [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.01650572, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.03159519, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.12468186, 0.        , 0.        ,
     0.51343194, 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.07079231, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.15403441, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.82743524, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00331447, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00930066, 0.        , 0.        ,
     0.03580077, 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00334943, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.03225657, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.01084116, 0.        , 0.        ,
     0.03556534, 0.        , 0.        , 0.03257808, 0.        , 0.        ,
     0.03618719, 0.        , 0.        , 0.        ])  # fmt: skip

# Non-zero spin MPS from the Qualtran MPSPreparation tests (Apache-2.0): a 4-site system
# whose first site carries a left bond of dimension 3 (singlet embedding).
_NON_ZERO_SPIN_RAW_TENSORS = (
    np.array(
        [
            [
                [-0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.00316609, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054, 0.0, 0.0],
            ],
            [
                [0.0, 0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.00223876, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.00223876, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054, 0.0],
            ],
            [
                [0.0, 0.0, -0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.00316609, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054],
            ],
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.55872176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.82920795, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01562571, 0.0],
            ],
            [
                [0.0, -0.55872176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.82920795, -0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01562571],
            ],
            [
                [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710678],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.99960484, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.02810986],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.70710678, 0.0, 0.0],
                [0.0, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    ),
    np.array(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[1.0], [0.0], [0.0], [0.0]],
        ]
    ),
)

NON_ZERO_SPIN_EXPECTED_STATE = np.array(
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00110206,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00110206,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.00176896,  0.        ,  0.        ,  0.        ,
       -0.00125085,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00262431,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.00185567,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.00125085,  0.        ,  0.        ,  0.        ,  0.00176896,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.00185567,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00262431,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.57734054,  0.        ,  0.        ,
        0.        , -0.40824141,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.40824141,  0.        ,
        0.        ,  0.        ,  0.57734054,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ])  # fmt: skip

# Contracting the singlet-embedding left bond with the all-ones boundary vector (as
# ``contract_mps`` does) gives an equivalent open-boundary MPS.
NON_ZERO_SPIN_TENSORS = (_NON_ZERO_SPIN_RAW_TENSORS[0].sum(axis=0, keepdims=True), *_NON_ZERO_SPIN_RAW_TENSORS[1:])

# Qualtran resource estimates (QROM mode), from QubitCount and QECGatesCost (and_bloq + cswap):
# the dense-mode qubit count and the sparse-mode Toffoli count. The non-zero spin estimates use
# the left bond of dimension 3.
_QUALTRAN_COSTS = [
    pytest.param(REFERENCE_MPS_TENSORS, 26, 321, id="standard"),
    pytest.param(NON_ZERO_SPIN_TENSORS, 29, 258, id="non_zero_spin"),
]

_SQRT_HALF = float(np.sqrt(0.5))
_SINGLET_TENSORS = (
    np.array([[[0.0, 0.0], [_SQRT_HALF, 0.0], [0.0, -_SQRT_HALF], [0.0, 0.0]]]),
    np.array([[[0.0], [0.0], [1.0], [0.0]], [[0.0], [1.0], [0.0], [0.0]]]),
)
_DOUBLE_OCCUPANCY_TENSORS = (
    np.array([[[0.0, 0.0], [0.0, _SQRT_HALF], [0.0, 0.0], [_SQRT_HALF, 0.0]]]),
    np.array([[[0.0], [1.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [1.0]]]),
)
_SPINLESS_PAIR_TENSORS = (
    np.array([[[_SQRT_HALF, 0.0], [0.0, _SQRT_HALF]]]),
    np.array([[[1.0], [0.0]], [[0.0], [1.0]]]),
)
_SPINLESS_HOP_TENSORS = (
    np.array([[[0.0, -_SQRT_HALF], [_SQRT_HALF, 0.0]]]),
    np.array([[[1.0], [0.0]], [[0.0], [1.0]]]),
)

# Two-site MPSs with hand-derived blocked Jordan-Wigner amplitudes, keyed by the qubit
# occupations with qubit 0 most significant: (alpha_0, alpha_1, beta_0, beta_1) for spatial
# orbitals and (mode_0, mode_1) for spinless modes.
_JORDAN_WIGNER_CASES = {
    # (|ud> - |du>)/sqrt(2) in chain order is the singlet: reordering b0 a1 into a1 b0 flips |du>.
    "singlet": (_SINGLET_TENSORS, [0, 1], {0b1001: _SQRT_HALF, 0b0110: _SQRT_HALF}),
    # Chain site 0 on orbital 1: |ud> creates a1 b0 (ordered); |du> creates b1 a0 (one swap).
    "singlet-swapped": (_SINGLET_TENSORS, [1, 0], {0b0110: _SQRT_HALF, 0b1001: _SQRT_HALF}),
    # (|2u> + |u2>)/sqrt(2): |2u> creates a0 b0 a1 (one swap); |u2> creates a0 a1 b1 (ordered).
    "double": (_DOUBLE_OCCUPANCY_TENSORS, [0, 1], {0b1110: -_SQRT_HALF, 0b1101: _SQRT_HALF}),
    # Chain site 0 on orbital 1: |2u> creates a1 b1 a0 (two swaps); |u2> creates a1 a0 b0 (one swap).
    "double-swapped": (_DOUBLE_OCCUPANCY_TENSORS, [1, 0], {0b1101: _SQRT_HALF, 0b1110: -_SQRT_HALF}),
    # (|00> + |11>)/sqrt(2) in chain order: |11> creates c0 c1, already in qubit order.
    "spinless-pair": (_SPINLESS_PAIR_TENSORS, [0, 1], {0b00: _SQRT_HALF, 0b11: _SQRT_HALF}),
    # Chain site 0 on orbital 1: |11> creates c1 c0 (one swap).
    "spinless-pair-swapped": (_SPINLESS_PAIR_TENSORS, [1, 0], {0b00: _SQRT_HALF, 0b11: -_SQRT_HALF}),
    # (|10> - |01>)/sqrt(2): one particle per term, so only the placement changes.
    "spinless-hop": (_SPINLESS_HOP_TENSORS, [0, 1], {0b10: _SQRT_HALF, 0b01: -_SQRT_HALF}),
    "spinless-hop-swapped": (_SPINLESS_HOP_TENSORS, [1, 0], {0b01: _SQRT_HALF, 0b10: -_SQRT_HALF}),
}


def _qubits_per_site(mps: MPSContainer) -> int:
    return 1 if mps.sites[0].physical_dimension == 2 else 2


def _chain_target(mps: MPSContainer, chain_state: np.ndarray | None = None) -> np.ndarray:
    """Map a chain-order state, by default the contracted MPS, to the blocked Jordan-Wigner basis."""
    state = contract_mps(mps) if chain_state is None else chain_state / np.linalg.norm(chain_state)
    return blocked_jordan_wigner_state(state, mps.site_to_orbital_order, _qubits_per_site(mps))


def _random_case(num_sites, bond_dim, site_dim, seed, site_to_orbital_order=None):
    mps = random_mps(num_sites, bond_dim, site_dim, np.random.default_rng(seed))
    if site_to_orbital_order is not None:
        mps = make_mps(mps.sites, site_to_orbital_order=site_to_orbital_order)
    return mps, _chain_target(mps)


def _particle_number_case(site_dim):
    tensors = random_particle_number_tensors(4, 2, max_bond=3, site_dim=site_dim, rng=np.random.default_rng(4))
    mps = make_mps(particle_number_blocked_sites(tensors))
    return mps, _chain_target(mps)


def _qualtran_case(tensors, expected_state, site_to_orbital_order=None):
    mps = make_mps(right_normalized_tensors(tensors), site_to_orbital_order=site_to_orbital_order)
    return mps, _chain_target(mps, expected_state)


def _jordan_wigner_case(name):
    tensors, site_to_orbital_order, amplitudes = _JORDAN_WIGNER_CASES[name]
    mps = make_mps(tensors, site_to_orbital_order=site_to_orbital_order)
    target = np.zeros(1 << (_qubits_per_site(mps) * mps.num_sites))
    for index, amplitude in amplitudes.items():
        target[index] = amplitude
    return mps, target


# Builders of (mps, target state in the blocked Jordan-Wigner qubit basis).
_PREPARATION_CASES = [
    *(
        pytest.param(functools.partial(_random_case, num_sites, bond_dim, 4, 42), id=f"random-{num_sites}x{bond_dim}")
        for num_sites, bond_dim in [(2, 4), (3, 4), (4, 2)]
    ),
    *(
        pytest.param(functools.partial(_random_case, num_sites, bond_dim, 2, 11), id=f"spinless-{num_sites}x{bond_dim}")
        for num_sites, bond_dim in [(2, 2), (4, 4), (5, 3)]
    ),
    pytest.param(functools.partial(_random_case, 4, 4, 2, 19, [3, 1, 0, 2]), id="spinless-permuted-order"),
    *(
        pytest.param(functools.partial(_particle_number_case, site_dim), id=f"particle-number-blocked-{site_dim}")
        for site_dim in [2, 4]
    ),
    pytest.param(functools.partial(_qualtran_case, REFERENCE_MPS_TENSORS, REFERENCE_MPS_EXPECTED_STATE), id="qualtran"),
    pytest.param(
        functools.partial(_qualtran_case, NON_ZERO_SPIN_TENSORS, NON_ZERO_SPIN_EXPECTED_STATE),
        id="qualtran-non-zero-spin",
    ),
    pytest.param(
        functools.partial(_qualtran_case, REFERENCE_MPS_TENSORS, REFERENCE_MPS_EXPECTED_STATE, [2, 0, 3, 1]),
        id="qualtran-permuted-order",
    ),
    *(
        pytest.param(functools.partial(_jordan_wigner_case, name), id=f"jordan-wigner-{name}")
        for name in _JORDAN_WIGNER_CASES
    ),
]


def _active_space_mps() -> MPSContainer:
    """Two right-canonical sites on three molecular orbitals."""
    tensors = right_normalized_tensors(REFERENCE_MPS_TENSORS[2:])
    tensors[0] = tensors[0][:1] / np.linalg.norm(tensors[0][:1])
    orbitals = Orbitals(
        np.eye(3),
        None,
        None,
        create_test_basis_set(3),
        sym.spin_index_set(3, [0, 1], [0, 1]),
        sym.spin_index_set(3, [], []),
    )
    return MPSContainer([make_site(tensor) for tensor in tensors], orbitals, orthogonality_center=0)


def _with_basis(tensors: Sequence[np.ndarray], index: int, states: Sequence[str]) -> MPSContainer:
    """Give site ``index`` the physical basis ``states`` in place of its canonical one."""
    parse = Configuration.from_bitstring if len(states) == 2 else Configuration.from_spin_half_string
    sites = [make_site(np.asarray(tensor)) for tensor in tensors]
    sites[index] = make_site(np.asarray(tensors[index]), [parse(state) for state in states])
    return make_mps(sites)


# Builders of MPS containers that cannot be prepared, with the expected error.
_UNPREPARABLE_CASES = [
    pytest.param(
        lambda: make_mps(right_normalized_tensors(REFERENCE_MPS_TENSORS), orthogonality_center=None),
        "right-canonical MPS with center zero",
        id="unknown-center",
    ),
    pytest.param(
        lambda: make_mps(right_normalized_tensors(REFERENCE_MPS_TENSORS), orthogonality_center=1),
        "right-canonical MPS with center zero",
        id="nonzero-center",
    ),
    pytest.param(_active_space_mps, "exactly one MPS site per molecular orbital", id="missing-orbitals"),
    pytest.param(
        lambda: make_mps([tensor.astype(complex) for tensor in right_normalized_tensors(REFERENCE_MPS_TENSORS)]),
        "only real-valued",
        id="complex",
    ),
    pytest.param(
        lambda: _with_basis(right_normalized_tensors(REFERENCE_MPS_TENSORS), 2, ("0", "d", "u", "2")),
        r"physical basis ordering \('0', 'u', 'd', '2'\)",
        id="permuted-basis",
    ),
    pytest.param(
        lambda: _with_basis(right_normalized_tensors([np.ones((1, 2, 2)), np.eye(2).reshape(2, 2, 1)]), 1, ("1", "0")),
        r"physical basis ordering \('0', '1'\)",
        id="permuted-spinless-basis",
    ),
    pytest.param(
        lambda: make_mps(
            [
                make_site(
                    np.array([[[1.0], [0.0], [0.0]]]),
                    [Configuration.from_spin_half_string(state) for state in ("0", "u", "d")],
                )
            ]
        ),
        "two or four physical states per site",
        id="three-physical-states",
    ),
    pytest.param(
        lambda: make_mps([np.array([[[1.0, 0.0], [0.0, 0.0]]]), np.ones((2, 4, 1)) / np.sqrt(8)]),
        "same physical dimension on every site",
        id="mixed-physical-dimensions",
    ),
    pytest.param(lambda: make_mps([np.zeros((1, 4, 1))]), "finite amplitudes with nonzero norm", id="zero-state"),
    pytest.param(lambda: make_mps([np.ones((1, 4, 2)), np.ones((2, 4, 1))]), "isometric", id="non-isometric-site"),
]


class TestSiteSynthesis:
    """Test the native site synthesis through its Python bindings."""

    @_METHODS
    @pytest.mark.parametrize("physical", [2, 4])
    @pytest.mark.parametrize("structure", ["dense", "particle_number"])
    def test_site_unitaries_reconstruct_site_isometries(self, method, physical, structure):
        """Each site unitary maps every left-bond state to its column of the site isometry.

        A dense site absorbs the right factor of the following site and hands its own right
        factor to the preceding site, so its target is rotated on both bonds.
        """
        rng = np.random.default_rng(physical)
        if structure == "dense":
            bonds = [1, 3, 2, 4, 1] if physical == 4 else [1, 3, 2, 4, 2, 1]
            tensors = [
                random_orthogonal(physical * right, rng)[:, :left].reshape(physical, right, left).transpose(2, 0, 1)
                for left, right in itertools.pairwise(bonds)
            ]
        else:
            tensors = random_particle_number_tensors(4, 2, max_bond=3, site_dim=physical, rng=rng)
        chi = 1 << int(np.ceil(np.log2(max(max(tensor.shape[0], tensor.shape[2]) for tensor in tensors))))

        syntheses = matrix_product_state_synthesis(make_mps(tensors), chi, method)

        assert len(syntheses) == len(tensors) - 1
        for index, (tensor, synthesis) in enumerate(zip(tensors[1:], syntheses, strict=True)):
            expected = site_isometry(tensor, chi)
            if method == "dense":
                following = syntheses[index + 1].right_factor if index + 1 < len(syntheses) else np.eye(tensor.shape[2])
                expected = site_isometry(tensor @ following.T, chi) @ synthesis.right_factor.T
            unitary = site_unitary(synthesis, chi)
            np.testing.assert_allclose(unitary.T @ unitary, np.eye(physical * chi), atol=1e-11)
            np.testing.assert_allclose(unitary[:, : tensor.shape[0]], expected, atol=1e-10)

    @pytest.mark.parametrize(
        ("num_sites", "chi", "seed", "site_index"),
        [pytest.param(5, 16, 2, 3, id="chi-16"), pytest.param(12, 256, 1, 8, id="chi-256")],
    )
    def test_dense_synthesis_of_degenerate_csd_spectra(self, num_sites, chi, seed, site_index):
        """Sites whose CSD blocks have many zero singular values still synthesize exactly.

        Regression test: Eigen 3.4's divide-and-conquer SVD returned non-finite singular vectors
        for the chi = 16 site (12 zero singular values) and crashed on the chi = 256 site (192).
        """
        site = random_mps(num_sites, chi, rng=np.random.default_rng(seed)).sites[site_index]
        synthesis = dense_unitary_synthesis(site, chi)

        tensor = dense_site(site)
        unitary = site_unitary(synthesis, chi)
        np.testing.assert_allclose(unitary.T @ unitary, np.eye(4 * chi), atol=1e-10)
        expected = site_isometry(tensor, chi) @ synthesis.right_factor.T
        np.testing.assert_allclose(unitary[:, : tensor.shape[0]], expected, atol=1e-10)

    @pytest.mark.parametrize("synthesize", [dense_unitary_synthesis, block_sparse_unitary_synthesis])
    def test_rejects_invalid_sites(self, synthesize):
        """Sites that do not fit the bond register, are not isometric, or are complex are rejected."""
        tensor = random_orthogonal(8, np.random.default_rng(5))[:, :2].reshape(4, 2, 2).transpose(2, 0, 1)
        with pytest.raises(ValueError, match="bond"):
            synthesize(make_site(tensor), 1)
        with pytest.raises(ValueError, match="isometric"):
            synthesize(make_site(2.0 * tensor), 2)
        with pytest.raises(ValueError, match="real"):
            synthesize(make_site(tensor.astype(complex)), 2)

    def test_dense_synthesis_rejects_invalid_inputs(self):
        """Dense synthesis needs two or four physical states and an orthogonal successor factor."""
        three_states = [Configuration.from_spin_half_string(state) for state in ("0", "u", "d")]
        with pytest.raises(ValueError, match="two or four physical states"):
            dense_unitary_synthesis(make_site(np.ones((1, 3, 1)) / np.sqrt(3), three_states), 2)
        site = make_site(random_orthogonal(8, np.random.default_rng(5))[:, :2].reshape(4, 2, 2).transpose(2, 0, 1))
        with pytest.raises(ValueError, match="successor factor"):
            dense_unitary_synthesis(site, 2, np.eye(3))
        with pytest.raises(ValueError, match="orthogonal successor"):
            dense_unitary_synthesis(site, 2, np.zeros((2, 2)))

    @_METHODS
    def test_container_synthesis_skips_initial_site(self, method):
        """The initial site need not be an isometry and a one-site MPS has no site syntheses."""
        assert matrix_product_state_synthesis(make_mps([2.0 * _SINGLE_SITE]), 2, method) == []
        mps = make_mps([np.ones((1, 4, 2)), np.eye(2, 4).reshape(2, 4, 1)])
        assert len(matrix_product_state_synthesis(mps, 2, method)) == 1
        with pytest.raises(ValueError, match="bond"):
            matrix_product_state_synthesis(mps, 1, method)
        with pytest.raises(ValueError, match="isometric"):
            matrix_product_state_synthesis(make_mps([np.ones((1, 4, 2)), np.ones((2, 4, 1))]), 2, method)
        with pytest.raises(ValueError, match="real"):
            matrix_product_state_synthesis(make_mps([_SINGLE_SITE.astype(complex)]), 2, method)

    @pytest.mark.parametrize("method", ["unknown", "general"])
    def test_container_synthesis_rejects_unknown_method(self, method):
        """Only the dense and block-sparse methods are available."""
        mps = make_mps([np.ones((1, 4, 2)), np.eye(2, 4).reshape(2, 4, 1)])
        with pytest.raises(ValueError, match="must be"):
            matrix_product_state_synthesis(mps, 2, method)


class TestStatePreparationAlgorithm:
    """Test the registered algorithm's settings, validation, and circuit registers."""

    def test_settings(self):
        """Dense synthesis is the default and unsupported settings are rejected."""
        settings = create("state_prep", "matrix_product_state").settings()
        assert settings.get("unitary_synthesis") == "dense"
        with pytest.raises(ValueError, match="out of allowed options"):
            settings.update("unitary_synthesis", "general")
        for rotation_bit_precision in (0, 31):
            with pytest.raises(ValueError, match="out of allowed range"):
                settings.update("rotation_bit_precision", rotation_bit_precision)

    def test_requires_mps_container(self):
        """Non-MPS wavefunctions and raw sites or arrays are rejected."""
        with pytest.raises(TypeError, match="requires an MPSContainer"):
            create("state_prep", "matrix_product_state").run(create_test_wavefunction())
        mps = right_normalized_mps(REFERENCE_MPS_TENSORS)
        preparer = MatrixProductStatePreparation()
        for invalid in (right_normalized_tensors(REFERENCE_MPS_TENSORS), mps.sites, Wavefunction(mps)):
            with pytest.raises(TypeError, match="requires an MPSContainer"):
                preparer.generate_matrix_product_state_preparation_data(invalid)

    @_METHODS
    @pytest.mark.parametrize(("make_unpreparable", "message"), _UNPREPARABLE_CASES)
    def test_rejects_unpreparable_mps(self, method, make_unpreparable, message):
        """MPSs outside the supported real, right-canonical, one-site-per-orbital form are rejected."""
        prep = create("state_prep", "matrix_product_state", unitary_synthesis=method)
        with pytest.raises(ValueError, match=message):
            prep.run(Wavefunction(make_unpreparable()))

    @_METHODS
    @pytest.mark.parametrize("allocate_phase_gradient", [True, False])
    def test_circuit_registers(self, method, allocate_phase_gradient):
        """The circuit acts on two qubits per site plus any phase gradient the caller must own."""
        prep = create(
            "state_prep",
            "matrix_product_state",
            unitary_synthesis=method,
            rotation_bit_precision=6,
            allocate_phase_gradient=allocate_phase_gradient,
        )
        circuit = prep.run(Wavefunction(right_normalized_mps(REFERENCE_MPS_TENSORS)))
        num_gradient_ancillas = 0 if allocate_phase_gradient else 6
        assert circuit.encoding == "jordan-wigner"
        assert circuit.num_qubits == 8 + num_gradient_ancillas
        assert circuit.metadata.num_phase_gradient_ancillas == num_gradient_ancillas


class TestQSharpPreparation:
    """Test that the Q# circuits prepare the target state."""

    @_METHODS
    @pytest.mark.parametrize("build_case", _PREPARATION_CASES)
    def test_prepares_mps_state(self, method, build_case):
        """The post-selected state matches the MPS in the blocked Jordan-Wigner basis.

        Six rotation bits keep statevector simulation small while retaining enough accuracy to
        detect synthesis regressions.
        """
        mps, target = build_case()
        data = preparation_data(mps, method)
        operation, site_struct = _QSHARP_OPERATIONS[method]
        params = data.to_qsharp_params(rotation_bit_precision=6)

        ancilla_zero_prob, prepared = simulate_mps_preparation(
            operation, site_struct, params, mps.num_sites, data.ancilla_bits
        )

        assert ancilla_zero_prob > 0.99, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"
        fidelity = np.abs(np.vdot(target, prepared)) ** 2
        assert fidelity > 0.97, f"Fidelity {fidelity:.4f} too low"

    @_METHODS
    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("layerShifted", "one shift flag per layer"),
            ("phases", "one entry per basis state of the target register"),
        ],
    )
    def test_rejects_givens_data_with_mismatched_lengths(self, method, field, message):
        """The Givens operations reject shift flags or phases that do not match the layers and register."""
        data = preparation_data(random_mps(num_sites=2, bond_dim=4, rng=np.random.default_rng(42)), method)
        params = data.to_qsharp_params(rotation_bit_precision=6)
        block_givens = params["siteDecompositions"][0]["blockGivens"]
        block_givens[field] = block_givens[field][:-1]
        operation, site_struct = _QSHARP_OPERATIONS[method]
        # A Q# runtime failure leaves the interpreter unusable, so run it on a throwaway context.
        with use_qsharp_context(create_qsharp_context()), pytest.raises(QSharpError, match=message):
            simulate_mps_preparation(operation, site_struct, params, 2, data.ancilla_bits)

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_permutation_via_qroam_with_measurement_uncompute(self, num_bits):
        """The lookup, SWAP, and measurement-based unlookup apply the permutation exactly.

        Each repetition samples new X-basis outcomes in the unlookup, so the phase fixup is
        exercised on different measurement records.
        """

        def table(values: list[int]) -> str:
            rows = (", ".join("true" if value >> bit & 1 else "false" for bit in range(num_bits)) for value in values)
            return "[" + ", ".join(f"[{row}]" for row in rows) + "]"

        context = get_qsharp_context()
        rng = np.random.default_rng(num_bits)
        for _ in range(6):
            permutation = rng.permutation(1 << num_bits).tolist()
            angles = rng.uniform(0.2, np.pi - 0.2, num_bits)
            amplitudes = np.ones(1)
            for angle in angles:
                # Little-endian: qubit k is bit k of the register value.
                amplitudes = np.kron([np.cos(angle / 2), np.sin(angle / 2)], amplitudes)
            expected = np.zeros_like(amplitudes)
            expected[permutation] = amplitudes

            # Target qubit k is prepared by Ry(angles[k]) before the permutation.
            context.eval(f"use target = Qubit[{num_bits}];")
            for qubit, angle in enumerate(angles):
                context.eval(f"Ry({float(angle):.15f}, target[{qubit}]);")
            inverse = np.argsort(permutation).tolist()
            context.eval(
                f"QDKChemistry.Utils.MPSSparse.PermutationViaQROAM({table(permutation)}, {table(inverse)}, target);"
            )
            dump = context.dump_machine()
            context.eval("ResetAll(target);")

            # DumpMachine shows target[0] as the most-significant bit; scratch qubits must be released clean.
            state = np.zeros(1 << num_bits, dtype=complex)
            for index in dump:
                assert index >> num_bits == 0 or abs(dump[index]) < 1e-10, "scratch qubits were not uncomputed"
                if index >> num_bits == 0:
                    state[int(format(index, f"0{num_bits}b")[::-1], 2)] = dump[index]
            assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)
            assert np.abs(np.vdot(expected, state)) ** 2 > 1 - 1e-10


class TestResourceEstimation:
    """Test resource estimates of the prepared circuits."""

    @_METHODS
    @pytest.mark.parametrize(("tensors", "qualtran_qubits", "qualtran_toffolis"), _QUALTRAN_COSTS)
    def test_resources_are_consistent_with_qualtran(self, method, tensors, qualtran_qubits, qualtran_toffolis):
        """Qubit counts are comparable to Qualtran's dense mode and Toffolis to its sparse mode."""
        prep = create("state_prep", "matrix_product_state", unitary_synthesis=method)
        counts = prep.run(Wavefunction(right_normalized_mps(tensors))).estimate().logical_counts

        assert qualtran_qubits <= counts["numQubits"] <= 2 * qualtran_qubits
        # The CCZ count includes every QROAM and Select decomposition, so it exceeds
        # Qualtran's sparse Toffoli count.
        assert 0 < counts["cczCount"] <= 10 * qualtran_toffolis

    @_METHODS
    def test_spinless_mps_needs_fewer_resources(self, method):
        """A spinless MPS needs fewer qubits and Toffolis than a spatial-orbital MPS of equal bond dimension."""
        prep = create("state_prep", "matrix_product_state", unitary_synthesis=method, rotation_bit_precision=6)
        spinless, spatial = (
            prep.run(
                Wavefunction(random_mps(num_sites=4, bond_dim=4, site_dim=site_dim, rng=np.random.default_rng(42)))
            )
            .estimate()
            .logical_counts
            for site_dim in (2, 4)
        )
        assert spinless["numQubits"] < spatial["numQubits"]
        assert 0 < spinless["cczCount"] < spatial["cczCount"]
