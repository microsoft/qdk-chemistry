# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests validating free-index pre-slicing against full 2-space SI computation.

The pre-slicing optimization computes only the active sub-block of Hbar output
by slicing input tensors along free-index dimensions (those appearing in the
einsum output subscript) to the active range, while keeping contracted
dimensions at their full size.

This test validates that:
    pre_slice_bch(H, T, active_ranges, bch_level) == full_bch(H, T)[active_slice]

for all output blocks (scalar, 1-body, 2-body) at BCH levels 0, 1, and 2.
"""

import itertools
import os

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

wickd = pytest.importorskip("wickd")
pytest.importorskip("pyscf")

import qdk_chemistry.algorithms.hamiltonian_downfolder  # noqa: E402
from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

qdk_chemistry.algorithms.hamiltonian_downfolder.load()


# ─── Pre-slicing implementation ───────────────────────────────────────────────


def _preslice_bch_si(w, bch_order, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b, act_ranges):
    """Wicked BCH with free-index pre-slicing.

    Only computes the active sub-block of each output tensor by slicing
    input tensors along dimensions corresponding to free indices.

    Args:
        act_ranges: dict mapping space char -> (slice_in_space, length)
            e.g. {'o': (slice(1,3), 2), 'v': (slice(0,2), 2),
                   'O': (slice(1,3), 2), 'V': (slice(0,2), 2)}

    """
    w.reset_space()
    w.add_space("o", "fermion", "occupied", list("ijklmn")[: max(nocc_a, 1)])
    w.add_space("v", "fermion", "unoccupied", list("abcdef")[: max(nvir_a, 1)])
    w.add_space("O", "fermion", "occupied", list("IJKLMN")[: max(nocc_b, 1)])
    w.add_space("V", "fermion", "unoccupied", list("ABCDEF")[: max(nvir_b, 1)])

    Top = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "v+ V+ O o"], unique=True)
    Hops = []
    for i in itertools.product(["v+", "o+"], ["v", "o"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["V+", "O+"], ["V", "O"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["v+", "o+"], ["v+", "o+"], ["v", "o"], ["v", "o"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["V+", "O+"], ["V+", "O+"], ["V", "O"], ["V", "O"]):
        Hops.append(" ".join(i))
    for i in itertools.product(["v+", "o+"], ["V+", "O+"], ["v", "o"], ["V", "O"]):
        Hops.append(" ".join(i))
    Hop = w.op("H", Hops, unique=True)
    Fops = []
    for i in itertools.product(["v+", "o+"], ["v", "o"]):
        Fops.append(" ".join(i))
    for i in itertools.product(["V+", "O+"], ["V", "O"]):
        Fops.append(" ".join(i))
    Fop = w.op("H", Fops, unique=True)

    sigma = w.op("T", ["v+ o", "V+ O", "v+ v+ o o", "V+ V+ O O", "v+ V+ O o"], unique=True)
    sigma.add2(Top.adjoint(), w.rational(-1))

    E0op = w.op("E0", [""])
    if bch_order == 0:
        Hbar = E0op + Hop
    elif bch_order == 1:
        Hbar = E0op + Hop + w.commutator(Hop, sigma)
        Hbar.add2(w.commutator(Fop, sigma, sigma), w.rational(1, 2))
    elif bch_order == 2:
        Hbar = E0op + Hop + w.commutator(Hop, sigma)
        Hbar.add2(w.commutator(Hop, sigma, sigma), w.rational(1, 2))
        Hbar.add2(w.commutator(Fop, sigma, sigma, sigma), w.rational(1, 6))
    else:
        raise ValueError(f"Unsupported BCH order {bch_order}")

    expr = w.WickTheorem().contract(w.rational(1), Hbar, 0, 4)
    mbeq = expr.to_manybody_equation("R")

    # Index → space char mapping (wicked convention)
    # NOTE: This mapping is NOT reliable for multi-index blocks because wicked
    # reuses index names across spaces. Instead, we derive the mapping per-block
    # from the output subscript + space key.

    class _S:
        def __init__(self, v):
            self.val = v

        def __getitem__(self, k):
            return self.val

    def _eval_presliced(eqs, output_space_chars, act_ranges, H, T, E0):
        """Evaluate einsum equations with free-index pre-slicing."""
        if not output_space_chars:
            # Scalar
            result = 0.0
            for eq in eqs:
                code = eq.compile("einsum")
                # code is like: R += 1.0 * np.einsum(...)
                rhs = code.split("+=")[1].strip()
                ns = {"np": np, "H": H, "T": T, "E0": _S(E0)}
                exec(f"_r = {rhs}", ns)
                result += ns["_r"]
            return result

        # Determine output shape from active ranges
        shape = [act_ranges[c][1] for c in output_space_chars]
        result = np.zeros(shape)

        for eq in eqs:
            code = eq.compile("einsum")
            # Format: Rxxx += coeff * np.einsum("subs", T1, T2, ..., optimize="optimal")
            rhs = code.split("+=")[1].strip()
            parts = rhs.split("*", 1)
            coeff = float(parts[0].strip())
            einsum_call = parts[1].strip()

            # Extract subscript string
            sub_start = einsum_call.index('"') + 1
            sub_end = einsum_call.index('"', sub_start)
            subscripts = einsum_call[sub_start:sub_end]
            inputs_str, output_str = subscripts.split("->")
            input_parts = inputs_str.split(",")

            # Free chars are those in the output subscript
            free_chars = set(output_str)

            # Build per-equation mapping: output index char → space char
            idx_to_space = {}
            for idx_ch, space_ch in zip(output_str, output_space_chars):
                idx_to_space[idx_ch] = space_ch

            # Extract tensor references
            after_subs = einsum_call[sub_end + 2 :]  # skip closing quote and comma
            tensor_str = after_subs.rsplit(",optimize=", 1)[0]
            tensor_refs = [t.strip() for t in tensor_str.split(",")]

            # Resolve tensors
            tensors = []
            for ref in tensor_refs:
                if ref.startswith("H["):
                    key = ref[3:-2]
                    tensors.append(H[key])
                elif ref.startswith("T["):
                    key = ref[3:-2]
                    tensors.append(T[key])
                elif ref.startswith("E0["):
                    tensors.append(E0)
                else:
                    raise ValueError(f"Unknown tensor ref: {ref}")

            # Slice tensors along free-index dimensions
            sliced = []
            for part, tensor in zip(input_parts, tensors):
                if np.isscalar(tensor):
                    sliced.append(tensor)
                    continue
                t = tensor
                idx = [slice(None)] * len(part)
                needs_slice = False
                for axis, ch in enumerate(part):
                    if ch in free_chars:
                        space = idx_to_space[ch]
                        idx[axis] = act_ranges[space][0]
                        needs_slice = True
                if needs_slice:
                    t = t[tuple(idx)]
                sliced.append(t)

            result += coeff * np.einsum(subscripts, *sliced, optimize="optimal")

        return result

    fbar, vbar, E0_bch = {}, {}, 0.0
    for key, eqs in mbeq.items():
        if not eqs:
            continue
        fc = eqs[0].compile("einsum")
        rv = fc.split("+=")[0].strip()
        ic = rv[1:]  # output index chars
        ndim = len(ic)

        result = _eval_presliced(eqs, ic, act_ranges, H, T, E0)

        if ndim == 0:
            E0_bch = result
        elif ndim == 2:
            fbar[ic] = np.array(result)
        elif ndim == 4:
            vbar[ic] = np.array(result)

    return fbar, vbar, E0_bch


# ─── Test helpers ─────────────────────────────────────────────────────────────


def _setup_system(xyz, basis, nocc):
    """Set up molecular system and return (H_dict, T_dict, metadata)."""
    from pyscf import cc

    from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf

    structure = Structure.from_xyz(xyz)
    _, wfn = create("scf_solver").run(structure, charge=0, spin_multiplicity=1, basis_or_guess=basis)
    full_ham = create("hamiltonian_constructor").run(wfn.get_orbitals())

    orbitals = full_ham.get_orbitals()
    nmo = orbitals.get_num_molecular_orbitals()
    nocc_a = nocc_b = nocc
    nvir_a = nvir_b = nmo - nocc

    h1_list, _ = full_ham.get_one_body_integrals()
    h1 = np.array(h1_list).reshape(nmo, nmo)
    eri_list, _, _ = full_ham.get_two_body_integrals()
    eri = np.array(eri_list).reshape(nmo, nmo, nmo, nmo)
    core_energy = full_ham.get_core_energy()

    V = eri.swapaxes(1, 2)
    V_asym = V - V.swapaxes(2, 3)

    E0 = core_energy
    for m in range(nocc_a):
        E0 += h1[m, m]
    for m in range(nocc_b):
        E0 += h1[m, m]
    for m in range(nocc_a):
        for n in range(nocc_a):
            E0 += 0.5 * V_asym[m, n, m, n]
    for m in range(nocc_b):
        for n in range(nocc_b):
            E0 += 0.5 * V_asym[m, n, m, n]
    for m in range(nocc_a):
        for n in range(nocc_b):
            E0 += V[m, n, m, n]

    F = h1.copy()
    for m in range(nocc_a):
        F += V_asym[:, m, :, m]
    for m in range(nocc_b):
        F += V[:, m, :, m]

    oa, va = slice(0, nocc_a), slice(nocc_a, nmo)
    ob, vb = slice(0, nocc_b), slice(nocc_b, nmo)
    sl_map = {"o": oa, "v": va, "O": ob, "V": vb}

    H = {}
    for c1 in ["o", "v"]:
        for c2 in ["o", "v"]:
            H[c1 + c2] = F[sl_map[c1], sl_map[c2]]
            H[c1.upper() + c2.upper()] = F[sl_map[c1.upper()], sl_map[c2.upper()]]
    for c1 in ["o", "v"]:
        for c2 in ["o", "v"]:
            for c3 in ["o", "v"]:
                for c4 in ["o", "v"]:
                    H[c1 + c2 + c3 + c4] = V_asym[sl_map[c1], sl_map[c2], sl_map[c3], sl_map[c4]]
                    H[c1.upper() + c2.upper() + c3.upper() + c4.upper()] = V_asym[
                        sl_map[c1.upper()], sl_map[c2.upper()], sl_map[c3.upper()], sl_map[c4.upper()]
                    ]
                    H[c1 + c2.upper() + c3 + c4.upper()] = V[
                        sl_map[c1], sl_map[c2.upper()], sl_map[c3], sl_map[c4.upper()]
                    ]

    # CCSD amplitudes
    alpha_occ = np.zeros(nmo)
    alpha_occ[:nocc_a] = 1.0
    beta_occ = np.zeros(nmo)
    beta_occ[:nocc_b] = 1.0
    mycc = cc.CCSD(hamiltonian_to_scf(full_ham, alpha_occ, beta_occ)).run()

    t2_asym = mycc.t2 - mycc.t2.swapaxes(2, 3)
    T = {
        "ov": mycc.t1.copy(),
        "OV": mycc.t1.copy(),
        "oovv": t2_asym.copy(),
        "OOVV": t2_asym.copy(),
        "oOvV": mycc.t2.copy(),
    }

    return H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b


def _zero_all_active(T, noa_act, nob_act, nva_act, nvb_act, nocc_a, nocc_b):
    """Zero all-active T blocks and build transposes."""
    ncore_a = nocc_a - noa_act
    ncore_b = nocc_b - nob_act
    act_oa = slice(ncore_a, nocc_a)
    act_va = slice(0, nva_act)
    act_ob = slice(ncore_b, nocc_b)
    act_vb = slice(0, nvb_act)

    T["ov"][act_oa, act_va] = 0.0
    T["OV"][act_ob, act_vb] = 0.0
    T["oovv"][act_oa, act_oa, act_va, act_va] = 0.0
    T["OOVV"][act_ob, act_ob, act_vb, act_vb] = 0.0
    T["oOvV"][act_oa, act_ob, act_va, act_vb] = 0.0

    T["vo"] = T["ov"].T.copy()
    T["VO"] = T["OV"].T.copy()
    T["vvoo"] = T["oovv"].transpose(2, 3, 0, 1).copy()
    T["VVOO"] = T["OOVV"].transpose(2, 3, 0, 1).copy()
    T["vVoO"] = T["oOvV"].transpose(2, 3, 0, 1).copy()
    T["VvOo"] = T["oOvV"].transpose(3, 2, 1, 0).copy()

    return T


def _full_bch_si(w, bch_order, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b):
    """Reference: full-size BCH (same as existing WickedDuccSISolver._wicked_bch_si)."""
    from qdk_chemistry.algorithms.hamiltonian_downfolder.wicked_ducc_si import WickedDuccSISolver

    return WickedDuccSISolver._wicked_bch_si(w, bch_order, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b)


def _compare_full_vs_preslice(xyz, basis, nocc, noa, nva, ducc_level, atol=1e-10):
    """Compare full BCH + post-slice vs pre-sliced BCH at the tensor level."""
    w = wickd

    H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b = _setup_system(xyz, basis, nocc)
    T = _zero_all_active(T, noa, noa, nva, nva, nocc_a, nocc_b)

    ncore_a = nocc_a - noa
    ncore_b = nocc_b - noa
    act_oa = slice(ncore_a, nocc_a)
    act_va = slice(0, nva)
    act_ob = slice(ncore_b, nocc_b)
    act_vb = slice(0, nva)

    # ── Method 1: Full computation (reference) ──
    fbar_full, vbar_full, E0_full = _full_bch_si(w, ducc_level, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b)

    # ── Method 2: Pre-sliced computation ──
    act_ranges = {
        "o": (act_oa, noa),
        "v": (act_va, nva),
        "O": (act_ob, noa),
        "V": (act_vb, nva),
    }
    fbar_pre, vbar_pre, E0_pre = _preslice_bch_si(w, ducc_level, H, T, E0, nocc_a, nvir_a, nocc_b, nvir_b, act_ranges)

    # ── Compare scalar ──
    assert abs(E0_full - E0_pre) < atol, (
        f"E0 mismatch: full={E0_full:.12f} pre={E0_pre:.12f} diff={abs(E0_full - E0_pre):.2e}"
    )

    # ── Compare 1-body blocks ──
    # Map from space chars (used as dict keys) to active slices
    def _act_slice(c):
        return {"o": act_oa, "v": act_va, "O": act_ob, "V": act_vb}[c]

    for key in fbar_full:
        full_active = fbar_full[key][_act_slice(key[0]), _act_slice(key[1])]
        pre = fbar_pre[key]
        assert full_active.shape == pre.shape, f"fbar[{key}] shape mismatch: {full_active.shape} vs {pre.shape}"
        err = np.max(np.abs(full_active - pre))
        assert err < atol, f"fbar[{key}] mismatch: max_err={err:.2e}"

    # ── Compare 2-body blocks ──
    for key in vbar_full:
        sl = tuple(_act_slice(c) for c in key)
        full_active = vbar_full[key][sl]
        pre = vbar_pre[key]
        assert full_active.shape == pre.shape, f"vbar[{key}] shape mismatch: {full_active.shape} vs {pre.shape}"
        err = np.max(np.abs(full_active - pre))
        assert err < atol, f"vbar[{key}] mismatch: max_err={err:.2e}"

    # Ensure all keys match
    assert set(fbar_full.keys()) == set(fbar_pre.keys()), (
        f"fbar key mismatch: full={set(fbar_full.keys())} pre={set(fbar_pre.keys())}"
    )
    assert set(vbar_full.keys()) == set(vbar_pre.keys()), (
        f"vbar key mismatch: full={set(vbar_full.keys())} pre={set(vbar_pre.keys())}"
    )


# ─── Test class ───────────────────────────────────────────────────────────────


class TestWickedDuccPreslice:
    """Validate pre-slicing matches full computation at all BCH levels."""

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2_sto3g_cas11(self, ducc_level):
        """H2/STO-3G CAS(1,1) — minimal system, all orbitals active."""
        _compare_full_vs_preslice("2\nH2\nH 0 0 0\nH 0 0 0.740848\n", "sto-3g", 1, 1, 1, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas11(self, ducc_level):
        """LiH/STO-3G CAS(1,1) — frozen core + frozen virtuals."""
        _compare_full_vs_preslice("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 1, 1, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas12(self, ducc_level):
        """LiH/STO-3G CAS(1,2) — asymmetric active space."""
        _compare_full_vs_preslice("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 1, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_lih_sto3g_cas22(self, ducc_level):
        """LiH/STO-3G CAS(2,2) — all occupied active."""
        _compare_full_vs_preslice("2\nLiH\nLi 0 0 0\nH 0 0 1.595\n", "sto-3g", 2, 2, 2, ducc_level)

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_cas11(self, ducc_level):
        """H2O/STO-3G CAS(1,1) — large frozen space on both sides."""
        _compare_full_vs_preslice(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "sto-3g",
            5,
            1,
            1,
            ducc_level,
        )

    @pytest.mark.parametrize("ducc_level", [0, 1, 2])
    def test_h2o_sto3g_cas22(self, ducc_level):
        """H2O/STO-3G CAS(2,2) — medium active space."""
        _compare_full_vs_preslice(
            "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n",
            "sto-3g",
            5,
            2,
            2,
            ducc_level,
        )
