"""Build ``SymmetryBlockedTensor`` JSON from dense spin-channel arrays.

Legacy serializations stored integrals and orbital coefficients as plain dense
arrays; the current schema stores them as ``SymmetryBlockedTensor`` documents.
These helpers rebuild the tensor JSON using only the live
``qdk_chemistry.data.symmetry`` bindings and serializing the result, so the
migration never hand-writes the symmetry-blocked JSON layout.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from qdk_chemistry.data import symmetry as _sym


def _spin(restricted: bool) -> _sym.SymmetryProduct:
    """Spin SymmetryProduct, restricted-equivalent iff ``restricted``."""
    return _sym.SymmetryProduct([_sym.axes.spin(1, restricted)])


_ALPHA = _sym.SymmetryLabel([_sym.axes.alpha()])
_BETA = _sym.SymmetryLabel([_sym.axes.beta()])
_AUX = _sym.SymmetryLabel([])


def _serialize(tensor) -> dict:
    """Serialize an SBT object to its JSON dict via a temporary file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.json"
        tensor.to_json_file(str(path))
        return json.loads(path.read_text(encoding="utf-8"))


def _as_matrix(array) -> np.ndarray:
    """Return ``array`` as a contiguous float64 array of its own shape."""
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64).reshape(np.asarray(array).shape))


def _as_2d(array) -> np.ndarray:
    """Return ``array`` as a contiguous 2-D float64 array."""
    return np.ascontiguousarray(np.atleast_2d(np.asarray(array, dtype=np.float64)))


def _as_1d(array) -> np.ndarray:
    """Return ``array`` as a contiguous 1-D float64 array."""
    return np.ascontiguousarray(np.asarray(array, dtype=np.float64).ravel())


def rank2_dict(alpha, beta: np.ndarray | None = None) -> dict:
    """Rank-2 spin-diagonal SBT JSON (e.g. one-body integrals, MO coefficients).

    ``alpha`` may be rectangular (AO x MO); the per-slot extents are taken from
    its shape. ``beta is None`` builds a spin-restricted tensor.
    """
    a = _as_2d(alpha)
    restricted = beta is None
    rows, cols = a.shape
    syms = [_spin(restricted), _spin(restricted)]
    extents = [{_ALPHA: rows, _BETA: rows}, {_ALPHA: cols, _BETA: cols}]
    blocks = [((_ALPHA, _ALPHA), a)]
    if not restricted:
        blocks.append(((_BETA, _BETA), _as_2d(beta)))
    return _serialize(_sym.SymmetryBlockedTensorRank2(syms, extents, blocks))


def rank1_dict(alpha, beta: np.ndarray | None = None) -> dict:
    """Rank-1 spin-diagonal SBT JSON (e.g. orbital energies)."""
    a = _as_1d(alpha)
    restricted = beta is None
    n = a.shape[0]
    syms = [_spin(restricted)]
    extents = [{_ALPHA: n, _BETA: n}]
    blocks = [((_ALPHA,), a)]
    if not restricted:
        blocks.append(((_BETA,), _as_1d(beta)))
    return _serialize(_sym.SymmetryBlockedTensorRank1(syms, extents, blocks))


def rank4_dict(
    aaaa,
    aabb: np.ndarray | None = None,
    bbbb: np.ndarray | None = None,
) -> dict:
    """Rank-4 spin-diagonal SBT JSON for two-body integrals.

    ``aabb`` and ``bbbb is None`` build a spin-restricted tensor: a single
    underlying block aliased to all four ``(aaaa, aabb, bbbb, bbaa)`` spin keys,
    matching the canonical restricted layout produced by the C++ core.
    """
    a = _as_1d(aaaa)
    n = round(a.shape[0] ** 0.25)
    if n**4 != a.shape[0]:
        raise ValueError(f"two-body block size {a.shape[0]} is not a perfect fourth power")
    restricted = bbbb is None
    extents = [{_ALPHA: n, _BETA: n}] * 4

    if not restricted:
        syms = [_spin(False)] * 4
        blocks = [
            ((_ALPHA, _ALPHA, _ALPHA, _ALPHA), a),
            ((_ALPHA, _ALPHA, _BETA, _BETA), _as_1d(aabb)),
            ((_BETA, _BETA, _BETA, _BETA), _as_1d(bbbb)),
        ]
        return _serialize(_sym.SymmetryBlockedTensorRank4(syms, extents, blocks))

    # Restricted: the binding's same-spin orbit aliasing only fills (bbbb) from
    # (aaaa). Build that single-block tensor, then alias the one block to all
    # four restricted spin keys (the C++ core stores one block for aaaa==aabb==
    # bbbb==bbaa).
    syms = [_spin(True)] * 4
    single = _serialize(_sym.SymmetryBlockedTensorRank4(syms, extents, [((_ALPHA, _ALPHA, _ALPHA, _ALPHA), a)]))
    keys = single["blocks"][0]["keys"]
    la, lb = keys[0][0], keys[1][0]
    single["blocks"][0]["keys"] = [
        [la, la, la, la],
        [lb, lb, lb, lb],
        [la, la, lb, lb],
        [lb, lb, la, la],
    ]
    return single


def rank3_three_center_dict(aa, bb: np.ndarray | None = None) -> dict:
    """Rank-3 SBT JSON for Cholesky three-center MO integrals.

    Each spin block is the dense ``[norb**2, naux]`` matrix (row index ``p*norb+q``)
    over two spin-orbital axes and a trivial-symmetry auxiliary axis. ``bb is None``
    builds the restricted form (the beta block aliases alpha).
    """
    a = _as_2d(aa)
    norb2, naux = a.shape
    norb = round(norb2**0.5)
    if norb * norb != norb2:
        raise ValueError(f"three-center block rows {norb2} is not a perfect square")
    restricted = bb is None
    aux = _sym.SymmetryProduct([])
    syms = [_spin(restricted), _spin(restricted), aux]
    extents = [{_ALPHA: norb, _BETA: norb}, {_ALPHA: norb, _BETA: norb}, {_AUX: naux}]
    blocks = [((_ALPHA, _ALPHA, _AUX), a)]
    if not restricted:
        blocks.append(((_BETA, _BETA, _AUX), _as_2d(bb)))
    return _serialize(_sym.SymmetryBlockedTensorRank3(syms, extents, blocks))


def sparse_rank4_dict(entries, norb: int) -> dict:
    """Restricted rank-4 ``SymmetryBlockedSparseMap`` JSON from ``[p, q, r, s, v]`` entries."""
    extent = {_ALPHA: norb, _BETA: norb}
    block = {(int(p), int(q), int(r), int(s)): float(v) for p, q, r, s, v in entries}
    syms = _spin(True)
    sparse_map = _sym.SymmetryBlockedSparseMapRank4(
        [syms, syms, syms, syms],
        [extent, extent, extent, extent],
        [((_ALPHA, _ALPHA, _ALPHA, _ALPHA), block)],
    )
    return _serialize(sparse_map)


def rank4_rdm_dict(aaaa, aabb, bbbb: np.ndarray | None = None) -> dict:
    """Rank-4 SBT JSON for a spin-dependent active 2-RDM.

    Unlike two-body integrals (where the restricted channels are equal), the
    same-spin (``aaaa``) and opposite-spin (``aabb``) 2-RDM channels differ.
    ``bbbb is None`` builds the restricted form: two distinct blocks aliased by
    the restricted spin orbit (``bbbb`` from ``aaaa``, ``bbaa`` from ``aabb``).
    """
    a = _as_1d(aaaa)
    n = round(a.shape[0] ** 0.25)
    if n**4 != a.shape[0]:
        raise ValueError(f"2-RDM block size {a.shape[0]} is not a perfect fourth power")
    extents = [{_ALPHA: n, _BETA: n}] * 4
    restricted = bbbb is None
    syms = [_spin(restricted)] * 4
    blocks = [
        ((_ALPHA, _ALPHA, _ALPHA, _ALPHA), a),
        ((_ALPHA, _ALPHA, _BETA, _BETA), _as_1d(aabb)),
    ]
    if not restricted:
        blocks.append(((_BETA, _BETA, _BETA, _BETA), _as_1d(bbbb)))
    return _serialize(_sym.SymmetryBlockedTensorRank4(syms, extents, blocks))
