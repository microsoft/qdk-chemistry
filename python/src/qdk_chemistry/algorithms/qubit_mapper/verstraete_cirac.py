"""Verstraete-Cirac fermion-to-qubit encoding for 2D lattices.

This module provides a factory function that constructs a
:class:`~qdk_chemistry.data.MajoranaMapping` using the Verstraete-Cirac (VC)
auxiliary fermion method :cite:`Verstraete2005,Whitfield2016,Havlicek2017`.

The VC encoding places one qubit on each lattice vertex and one auxiliary qubit
on each edge (including "phantom" edges at open boundaries to ensure uniform
vertex degree).  This produces bounded-weight Pauli strings for nearest-neighbor
interactions, independent of system size — a key advantage over Jordan-Wigner
and Bravyi-Kitaev for 2D lattice models.

For *num_species* spin species (e.g. 2 for spin-1/2 Fermi-Hubbard), each
species gets an independent copy of the lattice encoding on its own set of
qubits.

References
----------
.. [VC2005] Verstraete & Cirac, J. Stat. Mech. (2005) P09012.
.. [Whitfield2016] Whitfield, Havlicek & Troyer, Phys. Rev. A 94, 030301 (2016).
.. [Havlicek2017] Havlicek, Troyer & Whitfield, Phys. Rev. A 95, 032332 (2017).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

__all__ = [
    "verstraete_cirac",
    "codespace_effective_hamiltonian",
]

# Pauli type codes used by SparsePauliWord
_X = 1
_Y = 2
_Z = 3


# ---------------------------------------------------------------------------
# Lattice + qubit layout helpers
# ---------------------------------------------------------------------------


class _VCLatticeLayout:
    """Computes qubit assignments for the padded VC encoding on an R×C lattice.

    Every vertex is padded to degree 4 using phantom edges at open boundaries,
    ensuring that all Majorana operators have the same Pauli weight (5) and
    all nearest-neighbour bilinears have the same weight (8).

    Qubit numbering within a single-species block (starting at *offset*):
        0 .. N_sites-1                          : vertex (site) qubits
        N_sites .. N_sites+N_real_edges-1       : real edge qubits
        N_sites+N_real_edges .. N_total_qubits-1: phantom edge qubits

    Real edges are ordered: horizontal edges (row-major, left-to-right within
    each row, rows bottom-to-top), then vertical edges (column-major,
    bottom-to-top within each column, columns left-to-right).
    """

    def __init__(self, rows: int, cols: int, offset: int = 0) -> None:
        if rows < 2 or cols < 2:
            raise ValueError(f"Lattice must be at least 2×2, got {rows}×{cols}")
        self.rows = rows
        self.cols = cols
        self.offset = offset
        self.n_sites = rows * cols

        # Real edges
        self.n_h_edges = rows * (cols - 1)  # horizontal
        self.n_v_edges = (rows - 1) * cols  # vertical
        self.n_real_edges = self.n_h_edges + self.n_v_edges

        # Phantom edges: each boundary site gets one per missing direction
        self.n_phantom_left = rows       # x=0 sites missing left
        self.n_phantom_right = rows      # x=cols-1 missing right
        self.n_phantom_bottom = cols     # y=0 missing bottom
        self.n_phantom_top = cols        # y=rows-1 missing top
        self.n_phantom = (
            self.n_phantom_left + self.n_phantom_right
            + self.n_phantom_bottom + self.n_phantom_top
        )

        self.n_total_qubits = self.n_sites + self.n_real_edges + self.n_phantom

    # -- qubit index helpers --------------------------------------------------

    def site_qubit(self, x: int, y: int) -> int:
        """Qubit index for the vertex at grid position (x, y)."""
        return self.offset + y * self.cols + x

    def _h_edge_qubit(self, x: int, y: int) -> int:
        """Qubit for the real horizontal edge between (x,y) and (x+1,y)."""
        return self.offset + self.n_sites + y * (self.cols - 1) + x

    def _v_edge_qubit(self, x: int, y: int) -> int:
        """Qubit for the real vertical edge between (x,y) and (x,y+1)."""
        return self.offset + self.n_sites + self.n_h_edges + x * (self.rows - 1) + y

    def _phantom_base(self) -> int:
        return self.offset + self.n_sites + self.n_real_edges

    def _phantom_left_qubit(self, y: int) -> int:
        return self._phantom_base() + y

    def _phantom_right_qubit(self, y: int) -> int:
        return self._phantom_base() + self.n_phantom_left + y

    def _phantom_bottom_qubit(self, x: int) -> int:
        return self._phantom_base() + self.n_phantom_left + self.n_phantom_right + x

    def _phantom_top_qubit(self, x: int) -> int:
        return (
            self._phantom_base()
            + self.n_phantom_left
            + self.n_phantom_right
            + self.n_phantom_bottom
            + x
        )

    def incident_edge_qubits(self, x: int, y: int) -> list[int]:
        """Return the 4 auxiliary qubit indices for the 4 cardinal edges of (x, y).

        Order: left, right, bottom, top.  Phantom qubits are used at boundaries.
        """
        left = self._h_edge_qubit(x - 1, y) if x > 0 else self._phantom_left_qubit(y)
        right = self._h_edge_qubit(x, y) if x < self.cols - 1 else self._phantom_right_qubit(y)
        bottom = self._v_edge_qubit(x, y - 1) if y > 0 else self._phantom_bottom_qubit(x)
        top = self._v_edge_qubit(x, y) if y < self.rows - 1 else self._phantom_top_qubit(x)
        return [left, right, bottom, top]

    # -- stabiliser helpers ---------------------------------------------------

    def real_edge_stabilizers(self) -> list[tuple[int, int, int]]:
        """Return (site_u, site_v, edge_qubit) for each real edge."""
        stabs = []
        # Horizontal edges
        for y in range(self.rows):
            for x in range(self.cols - 1):
                u = self.site_qubit(x, y)
                v = self.site_qubit(x + 1, y)
                e = self._h_edge_qubit(x, y)
                stabs.append((u, v, e))
        # Vertical edges
        for x in range(self.cols):
            for y in range(self.rows - 1):
                u = self.site_qubit(x, y)
                v = self.site_qubit(x, y + 1)
                e = self._v_edge_qubit(x, y)
                stabs.append((u, v, e))
        return stabs

    def phantom_qubit_indices(self) -> list[int]:
        """Return all phantom qubit indices."""
        base = self._phantom_base()
        return list(range(base, base + self.n_phantom))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def verstraete_cirac(
    rows: int,
    cols: int,
    num_species: int = 2,
) -> "MajoranaMapping":
    """Construct a Verstraete-Cirac MajoranaMapping for a 2D square lattice.

    The encoding pads all vertices to degree 4 using phantom boundary edges
    so that every nearest-neighbour hopping term has constant Pauli weight
    regardless of lattice size.

    For spin-1/2 models (``num_species=2``), each spin species gets an
    independent VC copy on its own qubit register.  The modes follow blocked
    spin-orbital ordering (all alpha first, then all beta), matching the
    convention used by :class:`~qdk_chemistry.algorithms.qubit_mapper.QdkQubitMapper`.

    Args:
        rows: Number of lattice rows (≥ 2).
        cols: Number of lattice columns (≥ 2).
        num_species: Number of independent spin species (default 2 for
            spin-1/2 Fermi-Hubbard; use 1 for spinless models).

    Returns:
        MajoranaMapping: A table-based mapping with
            ``num_modes = num_species * rows * cols`` and
            ``num_qubits = num_species * (rows*cols + 2*rows*cols + rows + cols)``.

    Raises:
        ValueError: If ``rows < 2``, ``cols < 2``, or ``num_species < 1``.

    """
    from qdk_chemistry.data import MajoranaMapping  # noqa: PLC0415

    if num_species < 1:
        raise ValueError(f"num_species must be >= 1, got {num_species}")

    n_sites = rows * cols
    n_modes = num_species * n_sites
    table: list[list[tuple[int, int]]] = []

    qubits_per_species = _VCLatticeLayout(rows, cols).n_total_qubits
    for species in range(num_species):
        layout = _VCLatticeLayout(rows, cols, offset=species * qubits_per_species)

        for site_idx in range(n_sites):
            x = site_idx % cols
            y = site_idx // cols
            sq = layout.site_qubit(x, y)
            edge_qs = layout.incident_edge_qubits(x, y)

            # γ_{2*mode} = X_site · Z_e1 · Z_e2 · Z_e3 · Z_e4
            word_x: list[tuple[int, int]] = [(sq, _X)]
            for eq in sorted(edge_qs):
                word_x.append((eq, _Z))
            table.append(sorted(word_x, key=lambda t: t[0]))

            # γ_{2*mode+1} = Y_site · Z_e1 · Z_e2 · Z_e3 · Z_e4
            word_y: list[tuple[int, int]] = [(sq, _Y)]
            for eq in sorted(edge_qs):
                word_y.append((eq, _Z))
            table.append(sorted(word_y, key=lambda t: t[0]))

    return MajoranaMapping.from_table(table, name="verstraete-cirac")


# ---------------------------------------------------------------------------
# Codespace projection utilities (for eigenvalue comparison tests)
# ---------------------------------------------------------------------------


def codespace_effective_hamiltonian(
    qubit_hamiltonian: "QubitHamiltonian",
    rows: int,
    cols: int,
    num_species: int = 2,
) -> np.ndarray:
    """Project a VC-encoded QubitHamiltonian onto its codespace.

    Returns the effective Hamiltonian as a dense matrix of dimension
    ``2**(num_species * rows * cols)``, suitable for exact diagonalisation.

    The codespace is defined by the stabiliser constraints:

    *  Real edge (u, v): ``Z_u · Z_v · X_e = +1``
    *  Phantom edge: ``Z_phantom = +1``

    Given a computational-basis state of the *site* qubits, each auxiliary
    qubit is uniquely determined by the stabiliser it belongs to, so the
    codespace dimension equals ``2**(num_sites_total)`` where
    ``num_sites_total = num_species * rows * cols``.

    The matrix is built *without* constructing the full ``2**num_qubits``
    Hilbert-space matrix: for each Pauli string the matrix element between
    every pair of codespace basis states is computed analytically.

    Args:
        qubit_hamiltonian: A VC-encoded qubit Hamiltonian.
        rows: Lattice rows.
        cols: Lattice columns.
        num_species: Number of spin species.

    Returns:
        numpy.ndarray: Dense Hermitian matrix of shape
            ``(2**n_site_qubits, 2**n_site_qubits)``.

    """
    n_sites = rows * cols
    n_site_qubits = num_species * n_sites
    dim = 2**n_site_qubits

    # Guard against infeasible dense allocations.  A dim×dim complex128
    # matrix requires dim² × 16 bytes; cap at ~4 GiB (n_site_qubits ≤ 14).
    _MAX_SITE_QUBITS = 14
    if n_site_qubits > _MAX_SITE_QUBITS:
        raise ValueError(
            f"codespace_effective_hamiltonian is a dense exact-diagonalisation "
            f"utility intended for small systems only.  Received "
            f"n_site_qubits={n_site_qubits} (dim={dim}), which would require "
            f"~{dim * dim * 16 / 1e9:.1f} GB.  Maximum supported: "
            f"n_site_qubits={_MAX_SITE_QUBITS}."
        )

    # Precompute layout info for each species
    qubits_per_species = _VCLatticeLayout(rows, cols).n_total_qubits
    layouts = []
    for species in range(num_species):
        lay = _VCLatticeLayout(rows, cols, offset=species * qubits_per_species)
        real_stabs = list(lay.real_edge_stabilizers())
        phantom_qs = lay.phantom_qubit_indices()
        site_qs = [lay.site_qubit(x, y) for y in range(rows) for x in range(cols)]
        layouts.append((lay, real_stabs, phantom_qs, site_qs))

    # Build the set of all site qubit indices and auxiliary qubit indices
    all_site_qs: list[int] = []
    all_real_stabs: list[tuple[int, int, int]] = []  # (site_u, site_v, edge_qubit)
    all_phantom_qs: list[int] = []
    for _lay, real_stabs, phantom_qs, site_qs in layouts:
        all_site_qs.extend(site_qs)
        all_real_stabs.extend(real_stabs)
        all_phantom_qs.extend(phantom_qs)

    site_qubit_set = set(all_site_qs)

    # Map global qubit index → site-qubit bit position (for computational basis)
    site_q_to_bit = {q: i for i, q in enumerate(sorted(all_site_qs))}

    H_eff = np.zeros((dim, dim), dtype=complex)  # noqa: N806

    pauli_char_to_code = {"I": 0, "X": _X, "Y": _Y, "Z": _Z}

    for term_idx in range(len(qubit_hamiltonian.pauli_strings)):
        pauli_str = qubit_hamiltonian.pauli_strings[term_idx]
        coeff = qubit_hamiltonian.coefficients[term_idx]
        nq = len(pauli_str)

        # Parse the Pauli string into per-qubit operators
        # Label convention: label[nq - 1 - qubit] = operator on qubit
        qubit_ops: dict[int, int] = {}
        for pos, ch in enumerate(pauli_str):
            qubit = nq - 1 - pos
            code = pauli_char_to_code[ch]
            if code != 0:
                qubit_ops[qubit] = code

        # Separate into site-qubit ops and auxiliary-qubit ops
        site_ops: dict[int, int] = {}
        aux_ops: dict[int, int] = {}
        for q, op in qubit_ops.items():
            if q in site_qubit_set:
                site_ops[q] = op
            else:
                aux_ops[q] = op

        # For each pair of codespace basis states |s⟩, |s'⟩, compute:
        # ⟨ψ_{s'}|P|ψ_s⟩ = ⟨s'|P_site|s⟩ · ∏_aux ⟨aux(s')|P_aux|aux(s)⟩
        #
        # We iterate over |s⟩ and compute where P_site sends it, plus
        # the auxiliary contribution.
        for s in range(dim):
            # Compute P_site|s⟩ = phase_site · |s'⟩
            s_prime = s
            phase_site = 1.0 + 0.0j
            for q, op in site_ops.items():
                bit = site_q_to_bit[q]
                b = (s >> bit) & 1
                if op == _Z:
                    phase_site *= (-1) ** b
                elif op == _X:
                    s_prime ^= (1 << bit)
                elif op == _Y:
                    s_prime ^= (1 << bit)
                    phase_site *= 1j * ((-1) ** b)  # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩

            if abs(phase_site) < 1e-15:
                continue

            # Compute auxiliary contribution
            aux_phase = 1.0 + 0.0j
            valid = True

            # Phantom qubits: in |0⟩, so ⟨0|P|0⟩ = 1 for I,Z; 0 for X,Y
            for pq in all_phantom_qs:
                op = aux_ops.get(pq, 0)
                if op == _X or op == _Y:
                    valid = False
                    break
                # I and Z both give 1 on |0⟩

            if not valid:
                continue

            # Real edge qubits: in X eigenstate with eigenvalue λ = (-1)^(s_u⊕s_v)
            for site_u, site_v, edge_q in all_real_stabs:
                op = aux_ops.get(edge_q, 0)
                if op == 0:
                    # Identity: need λ_s == λ_{s'}
                    bit_u = site_q_to_bit[site_u]
                    bit_v = site_q_to_bit[site_v]
                    lam_s = (-1) ** (((s >> bit_u) & 1) ^ ((s >> bit_v) & 1))
                    lam_sp = (-1) ** (((s_prime >> bit_u) & 1) ^ ((s_prime >> bit_v) & 1))
                    if lam_s != lam_sp:
                        valid = False
                        break
                elif op == _X:
                    # X eigenstate: ⟨λ'|X|λ⟩ = λ · δ(λ, λ')
                    bit_u = site_q_to_bit[site_u]
                    bit_v = site_q_to_bit[site_v]
                    lam_s = (-1) ** (((s >> bit_u) & 1) ^ ((s >> bit_v) & 1))
                    lam_sp = (-1) ** (((s_prime >> bit_u) & 1) ^ ((s_prime >> bit_v) & 1))
                    if lam_s != lam_sp:
                        valid = False
                        break
                    aux_phase *= lam_s
                elif op == _Z:
                    # Z flips X eigenvalue: ⟨λ'|Z|λ⟩ = δ(λ', -λ)
                    bit_u = site_q_to_bit[site_u]
                    bit_v = site_q_to_bit[site_v]
                    lam_s = (-1) ** (((s >> bit_u) & 1) ^ ((s >> bit_v) & 1))
                    lam_sp = (-1) ** (((s_prime >> bit_u) & 1) ^ ((s_prime >> bit_v) & 1))
                    if lam_sp != -lam_s:
                        valid = False
                        break
                elif op == _Y:
                    # Y: ⟨λ'|Y|λ⟩ = -i·λ · δ(λ', -λ)
                    bit_u = site_q_to_bit[site_u]
                    bit_v = site_q_to_bit[site_v]
                    lam_s = (-1) ** (((s >> bit_u) & 1) ^ ((s >> bit_v) & 1))
                    lam_sp = (-1) ** (((s_prime >> bit_u) & 1) ^ ((s_prime >> bit_v) & 1))
                    if lam_sp != -lam_s:
                        valid = False
                        break
                    aux_phase *= -1j * lam_s

            if not valid:
                continue

            H_eff[s_prime, s] += coeff * phase_site * aux_phase

    return H_eff
