# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Spin-integrated DUCC with ambit tensor backend (active-only output).

Evaluates wicked-generated BCH equations via ambit's blocked tensor library.
The design mirrors the einsum solver's ``generate_equation`` + ``exec`` pattern:

1. wicked's ``eq.compile('ambit')`` produces ambit C-style code
2. A mechanical bracket substitution makes it valid Python
3. ``exec()`` evaluates it against ambit ``BlockedTensor`` objects

Input tensors are decomposed into elementary sub-spaces (core/active/external
per spin) and merged into **4 tensors** (H rank-2, H rank-4, T rank-2, T rank-4).
A lightweight ``TensorDispatch`` wrapper resolves the rank from the index count,
so wicked's single ``"H"`` label maps transparently to both.

Output tensors are built with **only the active block** — ambit's expert mode
treats missing blocks as structural zeros and skips them entirely.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm
from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_common import (
    ScalarProxy,
    assemble_active_hamiltonian,
    build_ccsd_amplitudes,
    build_ducc_bch,
    build_integrals,
    require_wicked,
    zero_all_active_and_transpose,
)

logger = logging.getLogger(__name__)

_ambit = None


def _require_ambit():
    global _ambit
    if _ambit is None:
        try:
            import ambit
            _ambit = ambit
        except ImportError:
            raise ImportError("ambit is required for WickedDuccSIAmbitSolver.")
    return _ambit


# ── TensorDispatch: single name, rank-aware ──────────────────────────────────


class TensorDispatch:
    """Wraps a rank-2 and rank-4 BlockedTensor under one subscript interface.

    Resolves rank from the number of comma-separated indices::

        H = TensorDispatch(H_rank2, H_rank4)
        H["o0,v0"]          # → H_rank2["o0,v0"]
        H["o0,o1,v0,v1"]    # → H_rank4["o0,o1,v0,v1"]

    This lets ``exec(eq.compile('ambit'))`` use a single ``H`` variable
    for both 1-body and 2-body references without parsing.
    """

    __slots__ = ("_r2", "_r4")

    def __init__(self, r2, r4):
        self._r2 = r2
        self._r4 = r4

    def __getitem__(self, idx):
        return (self._r2 if idx.count(",") < 2 else self._r4)[idx]


# ── Ambit space & tensor setup ───────────────────────────────────────────────

_MAX_IDX = 6
_ELEM_MAP = {"o": ["c", "a"], "v": ["b", "e"], "O": ["C", "A"], "V": ["B", "E"]}
_ACTIVE_ELEM = {"o": "a", "v": "b", "O": "A", "V": "B"}


def _idx(prefix):
    return ",".join(f"{prefix}{i}" for i in range(_MAX_IDX))


def _expand(comp_key):
    """Expand composite key to elementary block keys."""
    return ["".join(c) for c in itertools.product(*[_ELEM_MAP[k] for k in comp_key])]


def _active_key(spaces):
    """Map composite output spaces to the active-only elementary block key."""
    return "".join(_ACTIVE_ELEM[c] for c in spaces)


def _fill(tensor, comp_key, arr, sl_a, sl_b, exclude=frozenset()):
    """Fill a BlockedTensor from a numpy array at elementary block level."""
    def _sl(s):
        return sl_a.get(s, sl_b.get(s))
    for combo in itertools.product(*[_ELEM_MAP[c] for c in comp_key]):
        blk = "".join(combo)
        if blk in exclude:
            continue
        slices = tuple(_sl(c) for c in combo)
        np.asarray(tensor.block(blk))[:] = arr[slices]


def setup_spaces(ambit, nocc_a, nocc_b, nvir_a, nvir_b, noa, nva, nob, nvb):
    """Register 8 elementary + 4 composite MO spaces with ambit."""
    nmo = nocc_a + nvir_a
    ncore_a, ncore_b = nocc_a - noa, nocc_b - nob

    ambit.BlockedTensor.reset_mo_space()
    for lbl, orbs, spin in [
        ("c", list(range(ncore_a)), ambit.SpinType.AlphaSpin),
        ("a", list(range(ncore_a, nocc_a)), ambit.SpinType.AlphaSpin),
        ("b", list(range(nocc_a, nocc_a + nva)), ambit.SpinType.AlphaSpin),
        ("e", list(range(nocc_a + nva, nmo)), ambit.SpinType.AlphaSpin),
        ("C", list(range(nmo, nmo + ncore_b)), ambit.SpinType.BetaSpin),
        ("A", list(range(nmo + ncore_b, nmo + nocc_b)), ambit.SpinType.BetaSpin),
        ("B", list(range(nmo + nocc_b, nmo + nocc_b + nvb)), ambit.SpinType.BetaSpin),
        ("E", list(range(nmo + nocc_b + nvb, 2 * nmo)), ambit.SpinType.BetaSpin),
    ]:
        ambit.BlockedTensor.add_mo_space(lbl, _idx(lbl), orbs, spin)
    ambit.BlockedTensor.add_composite_mo_space("o", _idx("o"), ["c", "a"])
    ambit.BlockedTensor.add_composite_mo_space("v", _idx("v"), ["b", "e"])
    ambit.BlockedTensor.add_composite_mo_space("O", _idx("O"), ["C", "A"])
    ambit.BlockedTensor.add_composite_mo_space("V", _idx("V"), ["B", "E"])

    sl_a = {"c": slice(0, ncore_a), "a": slice(ncore_a, nocc_a),
            "b": slice(0, nva), "e": slice(nva, nvir_a)}
    sl_b = {"C": slice(0, ncore_b), "A": slice(ncore_b, nocc_b),
            "B": slice(0, nvb), "E": slice(nvb, nvir_b)}
    return sl_a, sl_b


def build_tensors(ambit, H_dict, T_dict, sl_a, sl_b):
    """Build 4 merged BlockedTensors (all spins per rank).

    Returns ``(H_dispatch, T_dispatch)`` — TensorDispatch wrappers that
    resolve rank from the index count.
    """
    BT, CT = ambit.BlockedTensor, ambit.TensorType.CoreTensor
    active_set = {"a", "A", "b", "B"}

    # ── H: rank-2 (Fock, all spins) ──
    h1_keys = [c1 + c2 for c1 in "ov" for c2 in "ov"] + \
              [c1 + c2 for c1 in "OV" for c2 in "OV"]
    H2 = BT.build(CT, "H", [b for ck in h1_keys for b in _expand(ck)])
    for ck in h1_keys:
        _fill(H2, ck, H_dict[ck], sl_a, sl_b)

    # ── H: rank-4 (2-body, αα + ββ + αβ) ──
    h2_aa = [c1 + c2 + c3 + c4 for c1 in "ov" for c2 in "ov" for c3 in "ov" for c4 in "ov"]
    h2_bb = [k.upper() for k in h2_aa]
    h2_ab = [c1 + c2.upper() + c3 + c4.upper()
             for c1 in "ov" for c2 in "ov" for c3 in "ov" for c4 in "ov"]
    h4_keys = h2_aa + h2_bb + h2_ab
    H4 = BT.build(CT, "H", [b for ck in h4_keys for b in _expand(ck)])
    for ck in h4_keys:
        _fill(H4, ck, H_dict[ck], sl_a, sl_b)

    # ── T: rank-2 (T1, all spins, excl. active zeros) ──
    t1_keys = ["ov", "vo", "OV", "VO"]
    t1_zeros = frozenset(_active_key(ck) for ck in t1_keys)
    T2 = BT.build(CT, "T", [b for ck in t1_keys for b in _expand(ck) if b not in t1_zeros])
    for ck in t1_keys:
        _fill(T2, ck, T_dict[ck], sl_a, sl_b, exclude=t1_zeros)

    # ── T: rank-4 (T2, all spins, excl. active zeros) ──
    t2_keys = ["oovv", "vvoo", "OOVV", "VVOO", "oOvV", "vVoO", "VvOo"]
    t2_zeros = frozenset(_active_key(ck) for ck in t2_keys)
    T4 = BT.build(CT, "T", [b for ck in t2_keys for b in _expand(ck) if b not in t2_zeros])
    for ck in t2_keys:
        _fill(T4, ck, T_dict[ck], sl_a, sl_b, exclude=t2_zeros)

    return TensorDispatch(H2, H4), TensorDispatch(T2, T4)


# ── Compilation & Evaluation ─────────────────────────────────────────────────


def compile_ambit_blocks(mbeq, osi):
    """Compile wicked equations into exec-ready ambit code strings (per block).

    This is the ambit analogue of the einsum solver's ``compile_bch_equations``.
    The result is molecule-independent and cacheable per BCH level.

    Returns a list of ``(output_spaces, code_str)`` tuples for rank-2/4 blocks,
    plus the scalar block key (if present).
    """
    blocks = []
    scalar_key = None

    for key, eqs in mbeq.items():
        if not eqs:
            continue

        lhs_t = eqs[0].lhs().tensors()[0]
        lhs_idx = lhs_t.upper() + lhs_t.lower()
        ndim = len(lhs_idx)
        output_spaces = "".join(osi.label(i.space()) for i in lhs_idx)

        if ndim == 0:
            scalar_key = key
            continue

        # Compile all equations for this block into one exec-ready string
        lines = []
        for eq in eqs:
            lines.append(eq.compile("ambit").replace("[", '["').replace("]", '"]'))
        blocks.append((output_spaces, "\n".join(lines)))

    return blocks, scalar_key


def evaluate_bch_ambit(ambit, compiled_blocks, scalar_key, mbeq, osi,
                       H, T, E0, H_dict, T_dict, nocc_a, nvir_a, nocc_b, nvir_b):
    """Evaluate pre-compiled ambit blocks with active-only output.

    Args:
        compiled_blocks: Output of :func:`compile_ambit_blocks`.
        scalar_key: Block key for the E0 scalar term (or None).
    """
    BT, CT = ambit.BlockedTensor, ambit.TensorType.CoreTensor
    exec_ns = {"H": H, "T": T, "E0": ScalarProxy(E0)}

    fbar, vbar = {}, {}
    E0_bch = 0.0

    # Scalar block: einsum fallback
    if scalar_key is not None:
        from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_si import (
            generate_equation,
        )
        dim_map = {"o": nocc_a, "v": nvir_a, "O": nocc_b, "V": nvir_b}
        func_str, _ = generate_equation(mbeq, osi, scalar_key)
        ns = {}
        exec(func_str, {"np": np, "_dims": dim_map}, ns)  # noqa: S102
        E0_bch = ns["_eval"](ScalarProxy(E0), H_dict, T_dict)

    # Tensor blocks: exec pre-compiled ambit code
    for output_spaces, code in compiled_blocks:
        ndim = len(output_spaces)
        active = _active_key(output_spaces)
        R = BT.build(CT, "R", [active])
        exec_ns["R"] = R

        exec(code, exec_ns)  # noqa: S102

        result = np.asarray(R.block(active)).copy()
        if ndim == 2:
            fbar[output_spaces] = result
        elif ndim == 4:
            vbar[output_spaces] = result

    return fbar, vbar, E0_bch


# ── Solver class ─────────────────────────────────────────────────────────────


class WickedDuccSIAmbitSolver(Algorithm):
    """Spin-integrated DUCC Hamiltonian downfolder (ambit active-only).

    Uses ambit's blocked tensor library with expert mode to compute only
    the active sub-block of Hbar.  Evaluation uses ``exec(eq.compile('ambit'))``
    directly — no regex parsing, no tensor dispatch logic.

    Usage::

        solver = create("hamiltonian_downfolder", "wicked_ducc_si_ambit",
                         nactive_oa=2, nactive_va=3, ducc_level=2)
        downfolded = solver.run(hamiltonian, n_alpha, n_beta)
    """

    def __init__(self) -> None:
        super().__init__()
        s = self.settings()
        s._set_default("nactive_oa", "int", 0, "Number of active occupied alpha orbitals")
        s._set_default("nactive_ob", "int", 0, "Number of active occupied beta orbitals")
        s._set_default("nactive_va", "int", 0, "Number of active virtual alpha orbitals")
        s._set_default("nactive_vb", "int", 0, "Number of active virtual beta orbitals")
        s._set_default("ducc_level", "int", 2, "BCH truncation level (0, 1, or 2)")

    @staticmethod
    def type_name() -> str:
        return "hamiltonian_downfolder"

    @staticmethod
    def name() -> str:
        return "wicked_ducc_si_ambit"

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        w = require_wicked()
        ambit = _require_ambit()
        s = self.settings()
        noa_act = s["nactive_oa"]
        nob_act = s["nactive_ob"]
        nva_act = s["nactive_va"]
        nvb_act = s["nactive_vb"]
        ducc_level = s["ducc_level"]
        nocc_a, nocc_b = n_alpha, n_beta

        # Steps 1-3: integrals, CCSD, zero all-active (shared with other solvers)
        H_dict, E0, nmo = build_integrals(hamiltonian, nocc_a, nocc_b)
        nvir_a, nvir_b = nmo - nocc_a, nmo - nocc_b
        T_dict = build_ccsd_amplitudes(hamiltonian, nmo, nocc_a, nocc_b)
        zero_all_active_and_transpose(T_dict, nocc_a, nocc_b, noa_act, nob_act, nva_act, nvb_act)

        # Step 4: symbolic BCH (cacheable per level)
        mbeq, osi = build_ducc_bch(w, ducc_level)

        # Step 5a: compile ambit code (cacheable — independent of molecule)
        compiled_blocks, scalar_key = compile_ambit_blocks(mbeq, osi)

        # Step 5b: ambit evaluation
        ambit.initialize()
        try:
            ambit.BlockedTensor.set_expert_mode(True)
            sl_a, sl_b = setup_spaces(ambit, nocc_a, nocc_b, nvir_a, nvir_b,
                                      noa_act, nva_act, nob_act, nvb_act)
            H, T = build_tensors(ambit, H_dict, T_dict, sl_a, sl_b)

            fbar, vbar, E0_bch = evaluate_bch_ambit(
                ambit, compiled_blocks, scalar_key, mbeq, osi,
                H, T, E0, H_dict, T_dict,
                nocc_a, nvir_a, nocc_b, nvir_b)

            ambit.BlockedTensor.set_expert_mode(False)
        finally:
            ambit.finalize()

        # Step 6: assemble Hamiltonian
        return assemble_active_hamiltonian(fbar, vbar, E0_bch, noa_act, nva_act)
