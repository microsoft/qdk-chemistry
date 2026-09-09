"""Run the notebook-convention Kitaev lattice/field sweep in separate processes.

The full grid has 456 cases: n = 2, 3, 4, 5, 10, 20, ..., 200 and H_b = 0..18 T.
All other Hamiltonian parameters use kitaev_resource_estimate.py's notebook
defaults. Evolution is fixed at total_time=1000, dt=0.04, Trotter order=4, with
a final Y-basis rotation. QRE settings remain fixed in the benchmark script.

Tables and logs are saved beside this script in results/kitaev, independently
of the working directory, e.g. kitaev_200x200_Hb_18T.csv and its .log companion.
Completed CSVs are skipped unless --overwrite is supplied. An infeasible
estimate is recorded as a header-only CSV with its explanation in the log.
Failed jobs retain their logs but never publish a partial CSV, and the driver
exits nonzero after reporting failures.

The first pending case runs alone to initialize QRE's shared factory cache
before concurrent readers start. Remaining cases run with --workers processes
(default 2). Allow roughly 9 GiB per worker for the largest lattices. BLAS,
OpenMP, and Rayon threads are limited to one per child to avoid oversubscription.
Use --sizes and --fields for a subset, or --dry-run to inspect the plan.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

LATTICE_SIZES = (2, 3, 4, 5, *range(10, 201, 10))
FIELDS_TESLA = tuple(range(19))
_BENCHMARKS = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse execution/output options without exposing model or QRE settings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--workers", type=int, default=2, help="Concurrent processes; allow about 9 GiB RAM each.")
    parser.add_argument(
        "--sizes", type=int, nargs="+", choices=LATTICE_SIZES, default=list(LATTICE_SIZES), help="Lattice sizes to run."
    )
    parser.add_argument(
        "--fields", type=int, nargs="+", choices=FIELDS_TESLA, default=list(FIELDS_TESLA), help="H_b values in tesla."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_BENCHMARKS / "results" / "kitaev", help="Directory for CSV tables and logs."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rerun completed cases and replace their CSVs on success."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the planned sweep without running cases or writing files."
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive.")
    args.sizes = list(dict.fromkeys(args.sizes))
    args.fields = list(dict.fromkeys(args.fields))
    args.output_dir = args.output_dir.resolve()
    return args


def _run_case(n: int, field_b: int, output_csv: Path, env: dict[str, str]) -> float:
    """Run one fresh interpreter and publish its table only after success."""
    start = perf_counter()
    with NamedTemporaryFile(dir=output_csv.parent, prefix=f".{output_csv.stem}.", suffix=".tmp", delete=False) as temp:
        partial_csv = Path(temp.name)
    command = [
        sys.executable,
        str(_BENCHMARKS / "kitaev_resource_estimate.py"),
        "--nx",
        str(n),
        "--ny",
        str(n),
        "--magnetic-field-abc",
        "0",
        str(field_b),
        "0",
        "--total-time",
        "1000",
        "--dt",
        "0.04",
        "--trotter-order",
        "4",
        "--spin-direction",
        "0",
        "1",
        "0",
        "--output-csv",
        str(partial_csv),
    ]
    try:
        with output_csv.with_suffix(".log").open("w", encoding="utf-8") as log:
            print(f"Command: {shlex.join(command)}", file=log, flush=True)
            subprocess.run(command, cwd=_BENCHMARKS, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if partial_csv.stat().st_size == 0:
            raise RuntimeError("The benchmark exited successfully without writing a CSV table.")
        partial_csv.replace(output_csv)
    finally:
        partial_csv.unlink(missing_ok=True)
    return perf_counter() - start


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested Cartesian sweep, preserving completed results on restart."""
    args = parse_args(argv)
    cases = [
        (n, field_b, args.output_dir / f"kitaev_{n}x{n}_Hb_{field_b:02d}T.csv")
        for n, field_b in product(args.sizes, args.fields)
    ]
    pending = [case for case in cases if args.overwrite or not case[2].is_file() or case[2].stat().st_size == 0]
    print(f"Cases: {len(cases)}; pending: {len(pending)}; skipped: {len(cases) - len(pending)}", flush=True)
    print(f"Lattice sizes: {args.sizes}; H_b (T): {args.fields}")
    print(f"Workers: {args.workers}; output: {args.output_dir}", flush=True)
    if args.dry_run or not pending:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        env[variable] = "1"

    # The factory cache writes are not synchronized by QRE. Finish one writer
    # before starting concurrent cases that share the same fixed architecture.
    first_n, first_field, first_csv = pending[0]
    print(f"Initializing the factory cache with {first_csv.stem}", flush=True)
    try:
        elapsed = _run_case(first_n, first_field, first_csv, env)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAILED {first_csv.stem}: {error}; log: {first_csv.with_suffix('.log')}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    print(f"[1/{len(pending)}] saved {first_csv.name} ({elapsed:.1f} s)", flush=True)

    failures = 0
    # Threads only supervise subprocesses: Q#/QRE state and memory never cross
    # case boundaries, and memory is released when each interpreter exits.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_case, n, field_b, output_csv, env): output_csv
            for n, field_b, output_csv in pending[1:]
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=2):
                output_csv = futures[future]
                try:
                    elapsed = future.result()
                except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
                    failures += 1
                    print(
                        f"[{completed}/{len(pending)}] FAILED {output_csv.stem}: {error}; "
                        f"log: {output_csv.with_suffix('.log')}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"[{completed}/{len(pending)}] saved {output_csv.name} ({elapsed:.1f} s)", flush=True)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    print(
        f"Finished: {len(pending) - failures} saved; {failures} failed; {len(cases) - len(pending)} skipped.",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
