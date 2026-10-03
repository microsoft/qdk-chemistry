"""Logical resource estimates for dense sequential MPS state preparation on an L x L lattice.

Sweeps the bond dimension chi over logspace(2, 4, 9) for spin (d = 2) and fermionic (d = 4)
sites on square lattices with n = L^2 sites (1D snake ordering, open boundaries), and reports
logical qubits and non-Clifford counts of the sequential (dense, not block-sparse) algorithm.

The circuit is traced shape-only with placeholder rotation data (``MPSSequentialEstimate.qs``):
non-Clifford counts depend only on the register sizes and the number of Givens layers per
orthogonal factor, calibrated on decompositions of random right-canonical MPS tensors:

* every site is padded to ``dim = 2^a`` bond states, ``a = ceil(log2(max bond))``;
* d = 4: W0, W1 and the block-diagonal U each need ``dim`` Clements layers;
* d = 2: U needs ``clements(max(chi_left, chi_right))`` layers (1 for a 2 x 2 block).

Bonds are ``min(chi, d^i, d^(n-i))``. Non-Clifford and measurement counts are additive over
sites, so the program is traced as an initial segment plus one trace per distinct site
(identical to tracing the whole program; check with ``--check``).

Usage::

    python mps_sequential_estimates.py              # run sweep, write CSV + plot
    python mps_sequential_estimates.py --plot-only  # re-plot the saved CSV
    python mps_sequential_estimates.py --check      # composed vs. monolithic trace
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import qdk
from qdk._native import TargetProfile

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "mps_sequential_estimates.csv"
RESULTS_PNG = HERE / "mps_sequential_estimates.png"

D_VALUES = (2, 4)
L_VALUES = (4, 8, 12)
CHI_VALUES = tuple(round(x) for x in np.logspace(2, 4, 9))
ROTATION_BITS = 10

ADDITIVE = ("tCount", "rotationCount", "cczCount", "ccixCount", "measurementCount")
COLUMNS = (
    "d",
    "L",
    "n",
    "chi",
    "max_bond",
    "ancilla_qubits",
    "numQubits",
    "cczCount",
    "tCount",
    "rotationCount",
    "measurementCount",
    "bulk_site_cczCount",
    "rotation_bits",
)


def library_qsharp_dir() -> Path:
    """Locate the QDK/Chemistry Q# utility sources (in-repo checkout first, then installed package)."""
    in_repo = (
        HERE.parents[1]
        / "python"
        / "src"
        / "qdk_chemistry"
        / "utils"
        / "qsharp"
        / "src"
    )
    if in_repo.is_dir():
        return in_repo
    spec = importlib.util.find_spec("qdk_chemistry")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("qdk_chemistry Q# sources not found")
    return (
        Path(next(iter(spec.submodule_search_locations))) / "utils" / "qsharp" / "src"
    )


def make_context(workdir: Path) -> qdk.Context:
    """Compile the estimator together with the library Q# utilities it uses."""
    src = workdir / "src"
    shutil.copytree(library_qsharp_dir(), src)
    shutil.copy(HERE / "MPSSequentialEstimate.qs", src / "MPSSequentialEstimate.qs")
    (workdir / "qsharp.json").write_text("{}", encoding="utf-8")
    return qdk.Context(
        project_root=str(workdir), target_profile=TargetProfile.Adaptive_RIF
    )


def clements_layers(k: int) -> int:
    """Number of Clements Givens layers for a dense k x k real orthogonal matrix."""
    if k <= 1:
        return 0
    return 1 if k == 2 else k


def schedule(d: int, n: int, chi: int) -> dict:
    """Bond dimensions, ancilla width and per-site Givens layer counts."""
    bonds = [1] + [min(chi, d**i, d ** (n - i)) for i in range(1, n)] + [1]
    a = max(
        max(1, math.ceil(math.log2(max(bonds[i], bonds[i + 1], 2)))) for i in range(n)
    )
    dim = 1 << a
    layers_w = [clements_layers(dim) if d == 4 else 0 for _ in range(1, n)]
    if d == 4:
        layers_u = [clements_layers(dim) for _ in range(1, n)]
    else:
        layers_u = [
            clements_layers(min(dim, max(bonds[i], bonds[i + 1]))) for i in range(1, n)
        ]
    return {
        "bonds": bonds,
        "a": a,
        "dim": dim,
        "layers_w": layers_w,
        "layers_u": layers_u,
    }


def initial_state(d: int, sched: dict, seed: int = 0) -> list[float]:
    """Random normalized first-site amplitudes, laid out as (d, dim) with zero padding."""
    rng = np.random.default_rng(seed)
    vec = np.zeros((d, sched["dim"]))
    vec[:, : sched["bonds"][1]] = rng.standard_normal((d, sched["bonds"][1]))
    return (vec / np.linalg.norm(vec)).reshape(-1).tolist()


def estimate(
    ctx: qdk.Context, d: int, n: int, chi: int, rotation_bits: int = ROTATION_BITS
) -> dict:
    """Logical counts of the full program, composed from the initial segment and distinct sites."""
    ops = ctx.code.MPSSequentialEstimate
    s = schedule(d, n, chi)
    a = s["a"]
    base = dict(
        ctx.logical_counts(
            ops.MPSSequentialStatePrep,
            d,
            n,
            rotation_bits,
            a,
            initial_state(d, s),
            [],
            [],
            [],
        )
    )
    total = {k: base.get(k, 0) for k in ADDITIVE}
    peak = base["numQubits"]
    variants: dict[tuple[int, int], int] = {}
    for key in zip(s["layers_w"], s["layers_u"], strict=True):
        variants[key] = variants.get(key, 0) + 1
    bulk_ccz = 0
    for (lw, lu), mult in sorted(variants.items(), key=lambda kv: kv[1]):
        seg = dict(
            ctx.logical_counts(ops.SiteSegment, d, n, rotation_bits, a, a, lw, lu)
        )
        for k in ADDITIVE:
            total[k] += mult * seg.get(k, 0)
        peak = max(peak, seg["numQubits"])
        bulk_ccz = seg["cczCount"]  # most frequent site variant (sorted last)
    total.update(
        {
            "d": d,
            "L": math.isqrt(n),
            "n": n,
            "chi": chi,
            "max_bond": max(s["bonds"]),
            "ancilla_qubits": a,
            "numQubits": peak,
            "bulk_site_cczCount": bulk_ccz,
            "rotation_bits": rotation_bits,
        }
    )
    return total


def estimate_monolithic(
    ctx: qdk.Context, d: int, n: int, chi: int, rotation_bits: int = ROTATION_BITS
) -> dict:
    """Logical counts from a single trace of the whole program (for small cross-checks)."""
    s = schedule(d, n, chi)
    args = (
        d,
        n,
        rotation_bits,
        s["a"],
        initial_state(d, s),
        [s["a"]] * (n - 1),
        s["layers_w"],
        s["layers_u"],
    )
    return dict(
        ctx.logical_counts(ctx.code.MPSSequentialEstimate.MPSSequentialStatePrep, *args)
    )


def run_sweep(ctx: qdk.Context) -> list[dict]:
    """Estimate every (d, L, chi) combination and write the CSV."""
    rows = []
    t0 = time.time()
    for d in D_VALUES:
        for size in L_VALUES:
            for chi in CHI_VALUES:
                r = estimate(ctx, d, size * size, chi)
                rows.append({k: r[k] for k in COLUMNS})
                print(
                    f"d={d} L={size:2d} chi={chi:5d} a={r['ancilla_qubits']:2d} "
                    f"qubits={r['numQubits']:5d} ccz={r['cczCount']:.3e}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_rows() -> list[dict]:
    """Read the saved CSV."""
    with RESULTS_CSV.open(encoding="utf-8") as f:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]


def plot(rows: list[dict]) -> None:
    """CCZ count and logical qubits versus bond dimension, one curve per lattice size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for j, d in enumerate(D_VALUES):
        for size in L_VALUES:
            sel = sorted(
                (r for r in rows if r["d"] == d and r["L"] == size),
                key=lambda r: r["chi"],
            )
            chi = [r["chi"] for r in sel]
            axes[0, j].loglog(
                chi,
                [r["cczCount"] for r in sel],
                "o-",
                label=f"L={size} (n={size * size})",
            )
            axes[1, j].semilogx(
                chi,
                [r["numQubits"] for r in sel],
                "o-",
                label=f"L={size} (n={size * size})",
            )
        axes[0, j].set_title(f"d = {d} ({'spin' if d == 2 else 'fermion'})")
        axes[0, j].set_ylabel("CCZ (Toffoli) count")
        axes[1, j].set_ylabel("logical qubits")
        axes[1, j].set_xlabel(r"bond dimension $\chi$")
        for ax in axes[:, j]:
            ax.grid(True, which="both", alpha=0.3)
        axes[0, j].legend(fontsize=8)
    fig.suptitle(
        f"Dense sequential MPS state preparation, n = L$^2$ sites, {ROTATION_BITS}-bit rotations"
    )
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=130)
    print(f"wrote {RESULTS_PNG.name}")


def check(ctx: qdk.Context) -> None:
    """Compare the composed estimate with a single monolithic trace on small cases."""
    for d, n, chi in [
        (2, 16, 100),
        (2, 36, 316),
        (2, 64, 1000),
        (4, 16, 178),
        (4, 36, 1000),
    ]:
        composed = estimate(ctx, d, n, chi)
        mono = estimate_monolithic(ctx, d, n, chi)
        diff = {
            k: (composed[k], mono[k])
            for k in ("numQubits", *ADDITIVE)
            if composed[k] != mono.get(k, 0)
        }
        print(f"d={d} n={n} chi={chi}: {'match' if not diff else diff}")


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot-only",
        action="store_true",
        help="re-plot the saved CSV without re-running",
    )
    group.add_argument(
        "--check", action="store_true", help="validate composed vs. monolithic traces"
    )
    args = parser.parse_args()
    if args.plot_only:
        plot(load_rows())
        return
    with tempfile.TemporaryDirectory() as tmp:
        ctx = make_context(Path(tmp))
        if args.check:
            check(ctx)
            return
        rows = run_sweep(ctx)
    print(f"wrote {RESULTS_CSV.name}")
    plot(rows)


if __name__ == "__main__":
    main()
