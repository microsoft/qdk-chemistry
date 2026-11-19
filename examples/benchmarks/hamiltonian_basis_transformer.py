"""Compare active-orbital Cholesky transformation with a fresh Hamiltonian rebuild."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time


def _cases() -> dict[str, tuple[str, str, int, int]]:
    """Return XYZ geometry, basis, frozen-orbital count, and active-orbital count."""
    benzene = "12\nbenzene\n" + "".join(
        f"{element} {radius * math.cos(i * math.pi / 3):.12f} {radius * math.sin(i * math.pi / 3):.12f} 0\n"
        for element, radius in (("C", 1.397), ("H", 2.477))
        for i in range(6)
    )
    return {
        "lih": ("2\nLiH\nLi 0 0 0\nH 0 0 1.60\n", "cc-pvdz", 1, 6),
        "water": (
            "3\nwater\nO 0 0 0\nH 0 0.757 0.587\nH 0 -0.757 0.587\n",
            "cc-pvdz",
            1,
            8,
        ),
        "benzene": (benzene, "6-31g", 15, 12),
    }


def _parse_args() -> argparse.Namespace:
    """Parse and validate controls before loading numerical libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *_cases()], default="all")
    parser.add_argument(
        "--threads", type=int, default=1, help="Requested OpenMP and BLAS thread count."
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed calls to each method before measurement.",
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Measured calls to each method."
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def _timing_summary(samples: list[float]) -> dict[str, float | list[float]]:
    """Summarize elapsed milliseconds without asserting a performance threshold."""
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _run_case(
    name: str, case: tuple[str, str, int, int], args: argparse.Namespace
) -> dict[str, object]:
    """Measure one fixed molecular case and verify the returned payloads agree."""
    import numpy as np  # noqa: PLC0415 - thread controls must precede numerical imports

    from qdk_chemistry.algorithms import create  # noqa: PLC0415
    from qdk_chemistry.data import Orbitals, Structure  # noqa: PLC0415
    from qdk_chemistry.data.symmetry import SymmetryLabel, axes, spin_index_set  # noqa: PLC0415

    xyz, basis_name, inactive_count, active_count = case
    structure = Structure.from_xyz(xyz)
    scf = create("scf_solver", "qdk")
    start = time.perf_counter_ns()
    _, wavefunction = scf.run(
        structure, charge=0, spin_multiplicity=1, basis_or_guess=basis_name
    )
    scf_ms = (time.perf_counter_ns() - start) / 1e6
    orbitals = wavefunction.get_orbitals()
    alpha = SymmetryLabel([axes.alpha()])
    coefficients = orbitals.coefficients().block([alpha, alpha])
    nao, nmo = coefficients.shape
    active_indices = list(range(inactive_count, inactive_count + active_count))
    inactive_indices = list(range(inactive_count))
    active = spin_index_set(nmo, active_indices, active_indices)
    inactive = spin_index_set(nmo, inactive_indices, inactive_indices)
    source_orbitals = Orbitals(
        coefficients,
        None,
        orbitals.get_overlap_matrix(),
        orbitals.get_basis_set(),
        active,
        inactive,
    )
    seed = 588
    rotation, _ = np.linalg.qr(
        np.random.default_rng(seed).normal(size=(active_count, active_count))
    )
    target_coefficients = coefficients.copy()
    target_coefficients[:, active_indices] = coefficients[:, active_indices] @ rotation
    target_orbitals = Orbitals(
        target_coefficients,
        None,
        orbitals.get_overlap_matrix(),
        orbitals.get_basis_set(),
        active,
        inactive,
    )
    constructor = create(
        "hamiltonian_constructor",
        "qdk_cholesky",
        cholesky_tolerance=1e-8,
        eri_threshold=1e-12,
    )
    transformer = create("hamiltonian_basis_transformer")
    start = time.perf_counter_ns()
    source = constructor.run(source_orbitals)
    source_build_ms = (time.perf_counter_ns() - start) / 1e6

    operations = [
        ("transform", lambda: transformer.run(source, target_orbitals)),
        ("rebuild", lambda: constructor.run(target_orbitals)),
    ]
    for _ in range(args.warmups):
        for _, operation in operations:
            operation()
    samples: dict[str, list[float]] = {"transform": [], "rebuild": []}
    results = {}
    for iteration in range(args.repeats):
        order = operations if iteration % 2 == 0 else operations[::-1]
        for operation_name, operation in order:
            start = time.perf_counter_ns()
            result = operation()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            samples[operation_name].append(elapsed_ms)
            results[operation_name] = result

    transformed, rebuilt = results["transform"], results["rebuild"]
    comparison_tolerance = 1e-9
    errors = {}
    for payload, actual, expected in (
        (
            "one_body",
            transformed.get_one_body_integrals()[0],
            rebuilt.get_one_body_integrals()[0],
        ),
        (
            "three_center",
            transformed.get_container().get_three_center_integrals()[0],
            rebuilt.get_container().get_three_center_integrals()[0],
        ),
        (
            "inactive_fock",
            transformed.get_inactive_fock_matrix()[0],
            rebuilt.get_inactive_fock_matrix()[0],
        ),
        ("core_energy", transformed.get_core_energy(), rebuilt.get_core_energy()),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            atol=comparison_tolerance,
            rtol=0,
            equal_nan=False,
            err_msg=f"{name}: {payload}",
        )
        errors[payload] = float(
            np.max(np.abs(np.asarray(actual) - np.asarray(expected)))
        )
    return {
        "case": name,
        "basis": basis_name,
        "num_atomic_orbitals": nao,
        "num_molecular_orbitals": nmo,
        "num_active_orbitals": active_count,
        "num_inactive_orbitals": inactive_count,
        "active_indices": active_indices,
        "inactive_indices": inactive_indices,
        "cholesky_rank": source.get_container()
        .get_three_center_integrals()[0]
        .shape[1],
        "cholesky_tolerance": constructor.settings().get("cholesky_tolerance"),
        "eri_threshold": constructor.settings().get("eri_threshold"),
        "validation_tolerance": transformer.settings().get("validation_tolerance"),
        "comparison_tolerance": comparison_tolerance,
        "rotation_seed": seed,
        "scf_ms": scf_ms,
        "source_build_ms": source_build_ms,
        "transform": _timing_summary(samples["transform"]),
        "rebuild": _timing_summary(samples["rebuild"]),
        "speedup": statistics.median(samples["rebuild"])
        / statistics.median(samples["transform"]),
        "max_absolute_errors": errors,
    }


def main() -> None:
    """Run the benchmark with explicit thread controls and machine-readable results."""
    args = _parse_args()
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for variable in thread_variables:
        os.environ[variable] = str(args.threads)
    configuration = {
        "package_version": importlib.metadata.version("qdk-chemistry"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requested_threads": args.threads,
        "thread_environment": {
            variable: os.environ[variable] for variable in thread_variables
        },
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    print("BENCHMARK_CONFIG", json.dumps(configuration, allow_nan=False), flush=True)
    for name, case in _cases().items():
        if args.case in ("all", name):
            print(
                "BENCHMARK_RESULT",
                json.dumps(_run_case(name, case, args), allow_nan=False),
                flush=True,
            )


if __name__ == "__main__":
    main()
