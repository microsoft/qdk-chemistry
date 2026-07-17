# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Spin-integrated DUCC Hamiltonian downfolding using wicked (full-space).

Computes the full Hbar in all orbital blocks, then extracts the active
sub-block.  This is the reference implementation; for the optimised version
that computes only active-space output see :mod:`wicked_ducc_si_presliced`.
"""

from __future__ import annotations

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


# ── Step 5a: Full-space einsum evaluation ────────────────────────────────────


def generate_equation(mbeq, osi, block_key):
    """Generate an ``exec``-ready function string for one Hbar block.

    Mirrors wicked's ``generate_equation`` helper: produces a Python function
    that evaluates all einsum terms for *block_key* and returns the result
    array (or scalar).  The function takes ``(E0, H, T)`` where ``E0`` is a
    :class:`ScalarProxy` and ``H``/``T`` are space-keyed dicts.

    Uses ``eq.compile('einsum')`` directly — no modification of the compiled
    einsum strings.

    The generated function uses ``nocc``/``nvir`` as free variables resolved
    through the closure namespace at ``exec`` time.  Call with the appropriate
    dimension dict.

    Args:
        mbeq: Many-body equation dict from ``Expression.to_manybody_equation``.
        osi: ``OrbitalSpaceInfo`` from ``w.osi()``.
        block_key: Block key string, e.g. ``"oo|vv"`` or ``"|"``.

    Returns:
        ``(func_str, output_spaces)`` — the function source and the list of
        space labels for the output axes.

    """
    eqs = mbeq[block_key]
    lhs_tensor = eqs[0].lhs().tensors()[0]
    lhs_indices = lhs_tensor.upper() + lhs_tensor.lower()
    ndim = len(lhs_indices)
    output_spaces = [osi.label(idx.space()) for idx in lhs_indices]

    fc = eqs[0].compile("einsum")
    rv = fc.split("+=")[0].strip()

    lines = ["def _eval(E0, H, T):"]
    if ndim == 0:
        lines.append(f"    {rv} = 0.0")
    else:
        dims = ",".join(f"_dims['{s}']" for s in output_spaces)
        lines.append(f"    {rv} = np.zeros(({dims}))")

    for eq in eqs:
        lines.append(f"    {eq.compile('einsum')}")

    lines.append(f"    return {rv}")
    return "\n".join(lines), output_spaces


def evaluate_bch_full(mbeq, osi, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b):
    """Evaluate all BCH blocks at full orbital size.

    For each non-empty block in *mbeq*, calls :func:`generate_equation` to
    produce a function string, ``exec``s it, and evaluates with the given
    tensors.

    Args:
        mbeq: Many-body equation dict.
        osi: ``OrbitalSpaceInfo``.
        H, T: Spin-blocked integral/amplitude dicts.
        E0: Scalar reference energy.
        nocc_a, nvir_a, nocc_b, nvir_b: Orbital counts.

    Returns:
        ``(fbar, vbar, E0_bch)`` — full-sized 1-body/2-body dicts and scalar.

    """
    dim_map = {"o": nocc_a, "v": nvir_a, "O": nocc_b, "V": nvir_b}
    E0_proxy = ScalarProxy(E0)

    fbar, vbar, E0_bch = {}, {}, 0.0
    for key, eqs in mbeq.items():
        if not eqs:
            continue

        func_str, output_spaces = generate_equation(mbeq, osi, key)
        ndim = len(output_spaces)
        ic = "".join(output_spaces)

        ns = {}
        exec(func_str, {"np": np, "_dims": dim_map}, ns)
        result = ns["_eval"](E0_proxy, H, T)

        if ndim == 0:
            E0_bch = result
        elif ndim == 2:
            fbar[ic] = np.array(result)
        elif ndim == 4:
            vbar[ic] = np.array(result)

    return fbar, vbar, E0_bch


# ── Solver class ─────────────────────────────────────────────────────────────


class WickedDuccSISolver(Algorithm):
    """Spin-integrated DUCC Hamiltonian downfolder (full-space evaluation).

    Computes Hbar at full orbital size, then extracts the active sub-block.

    Usage::

        solver = create("hamiltonian_downfolder", "wicked_ducc_si",
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
        return "wicked_ducc_si"

    def _run_impl(self, hamiltonian, n_alpha, n_beta):
        w = require_wicked()
        s = self.settings()
        noa_act = s["nactive_oa"]
        nob_act = s["nactive_ob"]
        nva_act = s["nactive_va"]
        nvb_act = s["nactive_vb"]
        ducc_level = s["ducc_level"]

        nocc_a, nocc_b = n_alpha, n_beta

        # Step 1: integrals
        H, E0, nmo = build_integrals(hamiltonian, nocc_a, nocc_b)
        nvir_a, nvir_b = nmo - nocc_a, nmo - nocc_b
        logger.info(
            "WickedDuccSISolver: nmo=%d, nocc=(%d,%d), active=(%d,%d,%d,%d), level=%d",
            nmo,
            nocc_a,
            nocc_b,
            noa_act,
            nob_act,
            nva_act,
            nvb_act,
            ducc_level,
        )

        # Step 2: CCSD amplitudes
        T = build_ccsd_amplitudes(hamiltonian, nmo, nocc_a, nocc_b)

        # Step 3: zero all-active
        act_oa, act_va, act_ob, act_vb = zero_all_active_and_transpose(
            T, nocc_a, nocc_b, noa_act, nob_act, nva_act, nvb_act
        )

        # Step 4: symbolic BCH
        mbeq, osi = build_ducc_bch(w, ducc_level)

        # Step 5: full-space evaluation
        fbar_full, vbar_full, E0_bch = evaluate_bch_full(mbeq, osi, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b)

        # Extract active sub-blocks from full arrays
        def _act(c):
            return list(range(act_oa.start, act_oa.stop)) if c in ("o", "O") else list(range(act_va.start, act_va.stop))

        fbar = {}
        for key, arr in fbar_full.items():
            fbar[key] = arr[np.ix_(_act(key[0]), _act(key[1]))]

        vbar = {}
        for key, arr in vbar_full.items():
            vbar[key] = arr[np.ix_(*[_act(c) for c in key])]

        # Step 6: assemble Hamiltonian
        return assemble_active_hamiltonian(fbar, vbar, E0_bch, noa_act, nva_act)

    # Keep as static method for tests that call it directly.
    @staticmethod
    def _wicked_bch_si(w, bch_order, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b):
        """Run full-space BCH. Returns (fbar_dict, vbar_dict, E0_scalar)."""
        mbeq, osi = build_ducc_bch(w, bch_order)
        return evaluate_bch_full(mbeq, osi, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b)
