"""Sample a one-step-scaled resource estimate for the 2D Fermi-Hubbard model.

For each requested ``L``, the script builds the periodic ``L x L`` Hubbard Hamiltonian
under Jordan-Wigner, sizes standard phase estimation from a target ground-state energy
accuracy, and traces one controlled Trotter step through the resource estimator. The
step estimate is scaled over the rescaled-time QPE ladder without materializing the
full circuit.

Examples:
    Sample 10 x 10, 20 x 20, and 50 x 50 lattices into separate CSVs::

        python sample_hubbard_resources.py --size 10 20 50 -o cost.csv

    This writes ``cost_L10.csv``, ``cost_L20.csv``, and ``cost_L50.csv``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import math
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import resource
except ImportError:
    resource = None

from qdk.qre import PSSPC, LatticeSurgery, estimate
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, LatticeGraph, MajoranaMapping
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.qsharp import (
    create_qsharp_context,
    use_qsharp_context,
)

#: U = 8t for the strong-coupling regime
HOPPING_T = 1.0
U_OVER_T = 8.0

#: Electrons per SITE
FILLING = 0.875

#: Per-site ground-state energy accuracy.
TARGET_PRECISION_PER_SITE = 0.0051

#: Number of phase-register precision bits.
QPE_PRECISION_BITS = 10

#: Share of the energy budget allocated to phase estimation. Minimizing the total
#: Trotter step count sum_k r_k, which scales as 1 / (delta * sqrt(1 - delta)), gives
#: delta = 2/3; this reproduces the analytic optimum of Campbell (arXiv:2012.09238,
#: App. F), whose Eqs. (F5)-(F7) split the combined budget as Delta_PE = 2/3 and
#: Delta_TS = 1/3.
QPE_BUDGET_FRACTION = 2.0 / 3.0

#: Sine-window phase error as a multiple of pi / (N tau). The minimum Holevo variance
#: for N queries is tan^2(pi/(N + 2)) (Babbush et al., PRX 8, 041015, Eq. (17); Berry
#: et al., PRA 80, 052114, Eq. (2.3)), i.e. a one-sigma constant of 1.0. The benchmark
#: requires 90% confidence rather than one sigma, so we use the 90% confidence
#: half-width of the sine-window error density, computed as in Lee et al.
#: (PRX Quantum 2, 030305, App. D, Eqs. (D26)-(D29)).
QPE_CONFIDENCE_CONSTANT = 1.553

#: Classical precision per site assumed available as prior knowledge, in units of t.
#: Evolution times beyond 2 pi / (CLASSICAL_PRECISION_PER_SITE * L^2) would alias into
#: bits the classical estimate cannot already fix (arXiv:2609.05316, App. C).
CLASSICAL_PRECISION_PER_SITE = 0.05

#: The plaquette trotter is second order only.
TROTTER_ORDER = 2

#: Majorana architecture physical error rate.
MAJORANA_ERROR_RATE = 1e-6

#: Error budget for the resource estimator.
MAX_ESTIMATE_ERROR = 0.01


def target_precision(size: int) -> float:
    """Return the ground-state energy accuracy required of an L x L lattice."""
    return TARGET_PRECISION_PER_SITE * size * size


def num_electrons(size: int) -> int:
    """Return the electron count nearest the requested per-site filling."""
    return round(FILLING * size * size)


def qpe_parameters(
    one_norm: float,
    energy_budget: float,
    size: int,
) -> tuple[float, int, dict[str, float | int | str]]:
    """Split the energy budget and size the QPE evolution time from the QPE share.

    The total budget is divided as ``eps = eps_QPE + eps_T``. The phase error of a
    sine-windowed register with ``N = 2^bits - 1`` queries is ``C pi / (N tau)``, so
    meeting ``eps_QPE`` fixes the base evolution time. The remainder is handed to the
    plaquette builder, which sizes its own step count against it.

    Args:
        one_norm: Hamiltonian coefficient one-norm, used only to report the aliasing
            factor relative to the conservative no-aliasing time ``pi / lambda``.
        energy_budget: Total ground-state energy accuracy required.
        size: Lattice side length, used for the classical-precision time cap.

    Returns:
        Base evolution time, precision bits, and builder settings carrying the Trotter
        share of the budget as ``target_accuracy``.

    Raises:
        ValueError: If the required evolution time exceeds the classical-precision cap,
            where aliasing could no longer be resolved by prior classical knowledge.

    """
    qpe_budget = QPE_BUDGET_FRACTION * energy_budget
    trotter_budget = energy_budget - qpe_budget
    num_queries = 2**QPE_PRECISION_BITS - 1
    base_time = QPE_CONFIDENCE_CONSTANT * math.pi / (num_queries * qpe_budget)

    # Aliasing is expected and permitted here: the conservative bound pi / lambda is far
    # more restrictive than necessary when a classical estimate already fixes the leading
    # bits. The binding constraint is the classical precision instead.
    max_time = 2.0 * math.pi / (CLASSICAL_PRECISION_PER_SITE * HOPPING_T * size * size)
    if base_time > max_time:
        raise ValueError(
            f"L={size}: base evolution time {base_time:.4g} exceeds the classical-precision "
            f"cap {max_time:.4g}. Raise QPE_PRECISION_BITS above {QPE_PRECISION_BITS} so the "
            "same accuracy is reached with more, shorter queries."
        )
    Logger.debug(
        f"L={size}: eps_QPE={qpe_budget:.4g}, eps_T={trotter_budget:.4g}, "
        f"tau={base_time:.4g}, aliasing factor {base_time * one_norm / math.pi:.3g}x "
        f"the conservative pi/lambda bound, {base_time / max_time:.3g}x the classical cap."
    )
    return base_time, QPE_PRECISION_BITS, {"target_accuracy": trotter_budget}


def resolve_num_divisions(
    operator,
    evolution_time: float,
    trotter_settings: dict[str, float | int | str],
    size: int,
) -> int:
    """Return the Trotter step count the plaquette builder derives from its settings.

    Args:
        operator: The qubit Hamiltonian.
        evolution_time: Base Hamiltonian evolution time.
        trotter_settings: Builder settings containing target_accuracy or num_divisions.
        size: Lattice side length.

    Returns:
        The step count the builder will use, which is the larger of the explicit
        num_divisions and the value its error bound derives from target_accuracy.

    """
    builder = create(
        "hamiltonian_unitary_builder",
        "plaquette",
        order=TROTTER_ORDER,
        time=evolution_time,
        lattice_width=size,
        lattice_height=size,
        **trotter_settings,
    )
    return builder._resolve_num_divisions(operator, evolution_time)


def one_trotter_step_circuit(
    context,
    operator,
    step_time: float,
    size: int,
    repetitions: int = 1,
) -> Circuit:
    """Return a controlled second-order plaquette Trotter step, optionally repeated.

    With ``repetitions`` at 1 the circuit is a single step and the caller scales the
    estimate by hand. Above 1 the repetition is carried in the container instead, so
    the estimator sees the whole ladder as one workload and sizes its factories
    against it rather than against a single step. The body is emitted once either
    way: ``step_reps`` drives ``RepeatEstimates`` rather than unrolling.

    Args:
        context: Q# context to build in.
        operator: The qubit Hamiltonian.
        step_time: Evolution time represented by one step.
        size: Lattice side length.
        repetitions: Number of identical steps the container should repeat.

    Returns:
        The controlled circuit.

    Raises:
        RuntimeError: If the unitary builder does not emit the requested repetitions.

    """
    with use_qsharp_context(context):
        unitary = create(
            "hamiltonian_unitary_builder",
            "plaquette",
            order=TROTTER_ORDER,
            time=step_time,
            lattice_width=size,
            lattice_height=size,
            num_divisions=1,
            target_accuracy=0.0,
            power=repetitions,
            power_strategy="repeat",
        ).run(operator)
        step_repetitions = unitary.get_container().step_reps
        if step_repetitions != repetitions:
            raise RuntimeError(
                f"expected {repetitions} Trotter step(s), got {step_repetitions}"
            )
        return create("controlled_circuit_mapper", "pauli_sequence").run(unitary)


def record_no_result(log_path: Path, size: int, reason: str) -> None:
    """Append a timestamped line for a lattice size that yielded no frontier point.

    Args:
        log_path: File to append to.
        size: Lattice side length.
        reason: Why the size produced nothing.

    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{stamp} L={size} produced no result: {reason}\n")


def estimate_physical(
    circuit: Circuit,
    name: str,
    max_error: float,
    cache_dir: Path | None = None,
):
    """Return the qubit/runtime Pareto frontier on the Majorana architecture.

    The default ISA grid is tried first. When it admits no configuration the search is
    retried over a wider rotation-synthesis, slow-down, and code-distance grid.

    Args:
        circuit: The circuit to trace.
        name: Label for the estimate.
        max_error: Error budget for this circuit.
        cache_dir: Directory holding cached Q# application traces.

    Returns:
        The estimator's result table, which may still be empty.

    """
    application = circuit.get_qre_application(cache_dir=cache_dir)
    architecture = Majorana(error_rate=MAJORANA_ERROR_RATE)
    table = estimate(
        application,
        architecture,
        ThreeAux.q() * RoundBasedFactory.q(code_query=ThreeAux.q()),
        max_error=max_error,
        name=name,
    )
    if not table.as_frame().empty:
        return table

    # Higher slow_down trades runtime for fewer factories, and hence fewer qubits.
    trace_query = (
        application.q()
        * PSSPC.q(num_ts_per_rotation=[16, 17, 18, 19])
        * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 35)])
    )
    isa_query = ThreeAux.q(distance=[11, 13, 15, 17, 19]) * RoundBasedFactory.q(
        code_query=ThreeAux.q(distance=[5, 7, 11, 13, 15, 17, 19])
    )
    return estimate(
        application,
        architecture,
        isa_query,
        trace_query,
        max_error=max_error,
        name=name,
    )


def peak_memory_gb() -> float:
    """Return peak resident set size in GB, or 0.0 where the platform cannot report it."""
    if resource is None:  # pragma: no cover - platform dependent
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 1e6 if sys.platform != "darwin" else peak / 1e9


def current_memory_gb() -> float:
    """Return current resident set size in GB, or the process peak as a fallback."""
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
    except OSError:  # pragma: no cover - Linux-specific interface
        pass
    return peak_memory_gb()


def run_sampling(
    context,
    size: int,
    cache_dir: Path | None = None,
    repeat_in_circuit: bool = False,
) -> pd.DataFrame:
    """Measure one lattice size.

    Args:
        context: Q# context to build in.
        size: Lattice side length.
        cache_dir: Directory holding cached Q# application traces.
        repeat_in_circuit: Carry the ladder's step count in the container so the
            estimator costs the whole workload, instead of tracing a single step
            and multiplying its frontier afterwards.

    Returns:
        The estimator's frontier with per-step and accumulated ladder costs, or an
        empty frame when no configuration was admitted.

    """
    started = time.monotonic()
    num_sites = size * size
    lattice = LatticeGraph.square(size, size, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(
        lattice, epsilon=0.0, t=HOPPING_T, U=U_OVER_T * HOPPING_T
    )
    mapper_started = time.monotonic()
    operator = create("qubit_mapper").run(
        hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites)
    )
    mapper_elapsed = time.monotonic() - mapper_started
    rss_after_mapper = current_memory_gb()
    one_norm = operator.schatten_norm
    energy_budget = target_precision(size)
    base_time, resolution_bits, trotter_settings = qpe_parameters(
        one_norm, energy_budget, size
    )
    max_power = 2 ** (resolution_bits - 1)
    steps_per_bit = [
        resolve_num_divisions(
            operator, base_time * 2**bit, trotter_settings, size
        )
        for bit in range(resolution_bits)
    ]
    total_steps = sum(steps_per_bit)
    num_divisions_for_largest_step = steps_per_bit[-1]
    step_time = base_time * max_power / num_divisions_for_largest_step

    circuit_repetitions = total_steps if repeat_in_circuit else 1
    step_budget = MAX_ESTIMATE_ERROR / (1 if repeat_in_circuit else total_steps)

    circuit_started = time.monotonic()
    circuit = one_trotter_step_circuit(
        context, operator, step_time, size, repetitions=circuit_repetitions
    )
    circuit_elapsed = time.monotonic() - circuit_started
    rss_after_circuit = current_memory_gb()

    qre_started = time.monotonic()
    table = estimate_physical(
        circuit,
        f"{size}x{size}-step",
        max_error=step_budget,
        cache_dir=cache_dir,
    )
    qre_elapsed = time.monotonic() - qre_started
    rss_after_qre = current_memory_gb()
    if table.as_frame().empty:
        return pd.DataFrame()
    table.add_qubit_partition_column()
    table.add_factory_summary_column()

    # Constant per run, repeated on every row so each estimate is self-describing.
    parameters = {
        "L": size,
        "sites": size * size,
        "system_qubits": operator.num_qubits,
        "terms": len(operator.pauli_strings),
        "electrons": num_electrons(size),
        "lambda": one_norm,
        "sigma": energy_budget,
        "target_precision": energy_budget,
        "qpe_budget": QPE_BUDGET_FRACTION * energy_budget,
        "qpe_budget_fraction": QPE_BUDGET_FRACTION,
        "qpe_confidence_constant": QPE_CONFIDENCE_CONSTANT,
        # Names the error model used to size base_time, not a traced subcircuit: the
        # phase register and inverse QFT are outside the one-step trace.
        "qpe_error_model": "sine-window-90pct",
        "base_time": base_time,
        "t_max": base_time * 2**resolution_bits,
        "qpe_type": "standard-one-step-scaled",
        "max_power": max_power,
        "num_unitary_queries": 2**resolution_bits - 1,
        "power_strategy": "rescale",
        "effective_evolution_time": base_time * max_power,
        "trotter_budget": trotter_settings.get("target_accuracy", 0.0),
        "num_divisions": trotter_settings.get("num_divisions", 0),
        "resolved_num_divisions": num_divisions_for_largest_step,
        "num_divisions_for_largest_step": num_divisions_for_largest_step,
        "steps_per_bit": str(steps_per_bit),
        "total_trotter_steps": total_steps,
        "trotter_step_time": step_time,
        "step_max_error": step_budget,
        "circuit_repetitions": circuit_repetitions,
        "repeat_in_circuit": repeat_in_circuit,
        "num_bits": resolution_bits,
        "qubit_mapper_elapsed_s": round(mapper_elapsed, 3),
        "trotter_step_elapsed_s": round(circuit_elapsed, 3),
        "qpe_circuit_elapsed_s": round(circuit_elapsed, 3),
        "qre_elapsed_s": round(qre_elapsed, 3),
        "elapsed_s": round(time.monotonic() - started, 1),
        "rss_after_qubit_mapper_gb": round(rss_after_mapper, 3),
        "rss_after_qpe_circuit_gb": round(rss_after_circuit, 3),
        "rss_after_qre_gb": round(rss_after_qre, 3),
        "peak_rss_gb": round(peak_memory_gb(), 2),
    }
    for name, value in parameters.items():
        table.add_column(name, lambda _entry, value=value: value)

    frame = table.as_frame()
    step_seconds = pd.to_timedelta(frame["runtime"]).dt.total_seconds()
    ladder_factor = 1 if repeat_in_circuit else total_steps
    frame["step_runtime_s"] = step_seconds / ladder_factor
    frame["ladder_runtime_s"] = step_seconds * ladder_factor
    frame["ladder_runtime_days"] = frame["ladder_runtime_s"] / 86400
    frame["step_error"] = frame["error"] / ladder_factor
    frame["error"] = frame["error"] * ladder_factor
    frame["runtime"] = pd.to_timedelta(frame["ladder_runtime_s"], unit="s")
    return frame


def output_path(output: Path, size: int, multiple_sizes: bool) -> Path:
    """Return the output path for a lattice size."""
    if "{L}" in output.name:
        return output.with_name(output.name.replace("{L}", str(size)))
    if multiple_sizes:
        suffix = output.suffix or ".csv"
        return output.with_name(f"{output.stem}_L{size}{suffix}")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Run the resource estimate.

    Args:
        argv: Command-line arguments, or None to read from sys.argv.

    Returns:
        Process exit status.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        required=True,
        help="one or more even lattice side lengths of at least 2",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hubbard_resources.csv"),
        help="CSV path; multiple sizes add _L<size> before the suffix",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="directory for cached QRE application traces (default: .qre_cache beside the output)",
    )
    parser.add_argument(
        "--repeat-in-circuit",
        action="store_true",
        help="carry the ladder's step count as container repetitions so the estimator "
        "costs the whole workload, instead of scaling a one-step frontier afterwards",
    )
    args = parser.parse_args(argv)

    Logger.set_global_level(Logger.LogLevel.off)

    if any(size < 2 or size % 2 for size in args.size):
        parser.error("each --size must be an even integer of at least 2")

    # QDK interpreters are thread-affine, so this context belongs to the calling thread.
    context = create_qsharp_context()

    cache_dir = args.cache_dir or args.output.parent / ".qre_cache"
    log_path = args.output.parent / "no_result.log"

    multiple_sizes = len(args.size) > 1
    for size in args.size:
        destination = output_path(args.output, size, multiple_sizes)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Sampling L={size}; writing {destination}", flush=True)
        try:
            frame = run_sampling(
                context,
                size,
                cache_dir=cache_dir,
                repeat_in_circuit=args.repeat_in_circuit,
            )
        except Exception as error:  # noqa: BLE001 - keep the sweep alive; the log explains the gap
            record_no_result(log_path, size, f"{type(error).__name__}: {error}")
            print(f"L={size} produced no result; see {log_path}", flush=True)
            continue
        if frame.empty:
            record_no_result(log_path, size, "estimator admitted no configuration")
            print(f"L={size} produced no result; see {log_path}", flush=True)
            continue
        frame.to_csv(destination, index=False)
        print(f"Finished L={size}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
