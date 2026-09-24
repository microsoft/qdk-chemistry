"""Sample logical resources for the 2D Fermi-Hubbard model.

By default the full standard QPE circuit is built on an identity reference state and
traced as a single workload. ``--one-step-scaled`` instead traces one Trotter step and
multiplies its logical counts across the whole QPE ladder.

Examples:
    Trace the full QPE circuit for several lattices into one table::

        python sample_hubbard_resources.py --size 2 4 6 8 10 20 \
            -o hubbard_logical_resources.csv

    Scale a single traced Trotter step across the ladder instead::

        python sample_hubbard_resources.py --one-step-scaled --size 110 120 \
            -o hubbard_logical_resources.csv
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import math
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, LatticeGraph, MajoranaMapping
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.qsharp import (
    create_qsharp_context,
    get_qsharp_context,
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

# Plaquette Trotter order.
TROTTER_ORDER = 2

#: Largest Hamming-weight phasing batch; zero keeps batching unbounded.
HWP_MAX_BATCH = 0


def target_precision(size: int) -> float:
    """Return the ground-state energy accuracy required of an L x L lattice."""
    return TARGET_PRECISION_PER_SITE * size * size


def num_electrons(size: int) -> int:
    """Return the electron count nearest the requested per-site filling."""
    return round(FILLING * size * size)


def error_partition(
    energy_budget: float,
) -> tuple[float, int, dict[str, float | int | str]]:
    """Split the energy budget and size the QPE evolution time from the QPE share.

    The total budget is divided as ``eps = eps_QPE + eps_T``. A sine-windowed register
    of ``N = 2^bits - 1`` queries has phase spread ``tan(pi / (N + 2))``, so requiring
    ``eps_QPE tau`` to equal that spread fixes the base evolution time. The remainder is
    handed to the plaquette builder, which sizes its own step count against it.

    Eqn. 8 in https://arxiv.org/pdf/2609.05316.

    Args:
        energy_budget: Total ground-state energy accuracy required.

    Returns:
        Base evolution time, precision bits, and builder settings carrying the Trotter
        share of the budget as ``target_accuracy``.

    """
    qpe_budget = QPE_BUDGET_FRACTION * energy_budget
    trotter_budget = energy_budget - qpe_budget
    num_queries = 2**QPE_PRECISION_BITS - 1
    base_time = math.tan(math.pi / (num_queries + 2)) / qpe_budget

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
        max_batch=HWP_MAX_BATCH,
        **trotter_settings,
    )
    return builder._resolve_num_divisions(operator, evolution_time)


def full_qpe_circuit(
    context,
    operator,
    base_time: float,
    resolution_bits: int,
    trotter_settings: dict[str, float | int | str],
    size: int,
):
    """Return the whole standard QPE circuit over an identity reference state.

    The phase register, the controlled-U ladder, and the inverse QFT are all materialized,
    so the counts describe the algorithm rather than an extrapolated step.

    Args:
        context: Q# context to build in.
        operator: The qubit Hamiltonian.
        base_time: Evolution time controlled by the least significant phase bit.
        resolution_bits: Number of phase-register qubits.
        trotter_settings: Builder settings carrying the Trotter share of the budget.
        size: Lattice side length.

    Returns:
        The standard QPE circuit.

    """
    unitary_builder = AlgorithmRef(
        "hamiltonian_unitary_builder",
        "plaquette",
        order=TROTTER_ORDER,
        time=base_time,
        lattice_width=size,
        lattice_height=size,
        max_batch=HWP_MAX_BATCH,
        # Bit k evolves for base_time * 2^k rather than repeating the block 2^k times.
        power_strategy="rescale",
        **trotter_settings,
    )
    circuit_builder = create(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=resolution_bits,
        unitary_builder=unitary_builder,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper", "pauli_sequence"
        ),
    )
    # Matches the Holevo spread error_partition used to size base_time.
    circuit_builder.settings().set("phase_window", "sine")
    with use_qsharp_context(context):
        state_prep = identity_state_prep(num_qubits=operator.num_qubits)
        return circuit_builder.run(state_prep, operator)[0]


def traced_step_counts(context, operator, step_time: float, size: int, num_divisions: int):
    """Return logical counts for a controlled evolution of ``num_divisions`` Trotter steps.

    Args:
        context: Q# context to build in.
        operator: The qubit Hamiltonian.
        step_time: Evolution time of a single step.
        size: Lattice side length.
        num_divisions: Number of Trotter steps to materialize.

    Returns:
        The traced logical counts.

    """
    with use_qsharp_context(context):
        unitary = create(
            "hamiltonian_unitary_builder",
            "plaquette",
            order=TROTTER_ORDER,
            time=step_time * num_divisions,
            lattice_width=size,
            lattice_height=size,
            max_batch=HWP_MAX_BATCH,
            num_divisions=num_divisions,
            target_accuracy=0.0,
        ).run(operator)
        circuit = create("controlled_circuit_mapper", "pauli_sequence").run(unitary)
        application = circuit.get_qre_application()
        return dict(
            get_qsharp_context().logical_counts(application.entry_expr, *application.args)
        )


def scaled_step_counts(
    step_counts: dict[str, int],
    total_steps: int,
    resolution_bits: int,
) -> dict[str, int]:
    """Return whole-ladder logical counts by scaling one traced Trotter step.

    Every logical operation count is multiplied by the total number of steps in the
    ladder. This intentionally ignores boundary merging between adjacent second-order
    steps, so it overestimates the ladder rather than inferring a lower cost from a
    multi-step trace.

    The traced block carries one control qubit, whereas the full algorithm carries a
    ``resolution_bits``-wide phase register, so the remaining phase qubits are added back.
    The inverse QFT and window preparation are still omitted; both are negligible against
    the query cost.

    Args:
        step_counts: Logical counts for one traced controlled Trotter step.
        total_steps: Number of Trotter steps in the ladder.
        resolution_bits: Width of the phase register.

    Returns:
        Whole-ladder logical counts keyed by the estimator's count names.

    """
    return {
        "numQubits": int(step_counts["numQubits"]) + (resolution_bits - 1),
        **{
            key: int(step_counts.get(key, 0)) * total_steps
            for key in (
                "rotationCount",
                "rotationDepth",
                "tCount",
                "cczCount",
                "ccixCount",
                "measurementCount",
            )
        },
    }


def run_sampling(
    context,
    size: int,
    one_step_scaled: bool = False,
) -> pd.DataFrame:
    """Measure the logical resources of one lattice size.

    Args:
        context: Q# context to build in.
        size: Lattice side length.
        one_step_scaled: Multiply a single traced Trotter step across the ladder instead
            of building and tracing the whole QPE circuit.

    Returns:
        One row of logical resources for this lattice.

    """
    started = time.monotonic()
    num_sites = size * size
    lattice = LatticeGraph.square(size, size, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(
        lattice, epsilon=0.0, t=HOPPING_T, U=U_OVER_T * HOPPING_T
    )
    operator = create("qubit_mapper").run(
        hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites)
    )
    one_norm = operator.schatten_norm
    energy_budget = target_precision(size)
    base_time, resolution_bits, trotter_settings = error_partition(energy_budget)

    # Bit k evolves for base_time * 2^k, so each bit resolves its own step count. The
    # schedule is reported in both modes, and its sum is the ladder multiplier below.
    steps_per_bit = [
        resolve_num_divisions(operator, base_time * 2**bit, trotter_settings, size)
        for bit in range(resolution_bits)
    ]
    total_steps = sum(steps_per_bit)
    step_time = base_time * 2 ** (resolution_bits - 1) / steps_per_bit[-1]

    circuit_started = time.monotonic()
    # Traced in both modes: it reports the one-step cost and drives the ladder scaling.
    step_counts = traced_step_counts(context, operator, step_time, size, 1)
    if one_step_scaled:
        logical_counts = scaled_step_counts(step_counts, total_steps, resolution_bits)
    else:
        circuit = full_qpe_circuit(
            context, operator, base_time, resolution_bits, trotter_settings, size
        )
        logical_counts = dict(circuit.estimate().logical_counts)
    circuit_elapsed = time.monotonic() - circuit_started

    ccz_count = int(logical_counts.get("cczCount", 0))
    ccix_count = int(logical_counts.get("ccixCount", 0))
    step_ccz_count = int(step_counts.get("cczCount", 0))
    step_ccix_count = int(step_counts.get("ccixCount", 0))
    return pd.DataFrame(
        [
            {
                "L": size,
                "sites": num_sites,
                "system_qubits": operator.num_qubits,
                "terms": len(operator.pauli_strings),
                "electrons": num_electrons(size),
                "lambda": one_norm,
                "target_precision": energy_budget,
                "qpe_budget": QPE_BUDGET_FRACTION * energy_budget,
                "qpe_budget_fraction": QPE_BUDGET_FRACTION,
                "trotter_budget": trotter_settings.get("target_accuracy", 0.0),
                "qpe_bits": resolution_bits,
                "num_unitary_queries": 2**resolution_bits - 1,
                "base_time": base_time,
                "t_max": base_time * 2**resolution_bits,
                "power_strategy": "rescale",
                "qpe_error_model": "sine-window-1sigma",
                "qpe_type": (
                    "standard-one-step-scaled"
                    if one_step_scaled
                    else "standard-full-circuit"
                ),
                "hwp_enabled": HWP_MAX_BATCH != 1,
                "hwp_max_batch": HWP_MAX_BATCH,
                "trotter_steps_per_qpe_bit": str(steps_per_bit),
                "one_trotter_step_time": step_time,
                "one_trotter_step_ccz_count": step_ccz_count,
                "one_trotter_step_ccix_count": step_ccix_count,
                "one_trotter_step_toffolis": step_ccz_count + step_ccix_count,
                "logical_qubits": int(logical_counts["numQubits"]),
                "rotations": int(logical_counts.get("rotationCount", 0)),
                "rotation_depth": int(logical_counts.get("rotationDepth", 0)),
                "t_gates": int(logical_counts.get("tCount", 0)),
                "ccz_count": ccz_count,
                "ccix_count": ccix_count,
                "toffolis": ccz_count + ccix_count,
                "measurements": int(logical_counts.get("measurementCount", 0)),
                "logical_estimate_elapsed_s": round(circuit_elapsed, 3),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Sample logical resources for each requested lattice size.

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
        default=Path("hubbard_logical_resources.csv"),
        help="CSV path for the combined table of logical counts",
    )
    parser.add_argument(
        "--one-step-scaled",
        action="store_true",
        help="trace one Trotter step and multiply its logical counts across the ladder "
        "instead of building and tracing the full QPE circuit; this conservatively "
        "ignores boundary merging between adjacent steps",
    )
    args = parser.parse_args(argv)

    Logger.set_global_level(Logger.LogLevel.off)

    if any(size < 2 or size % 2 for size in args.size):
        parser.error("each --size must be an even integer of at least 2")

    # QDK interpreters are thread-affine, so this context belongs to the calling thread.
    context = create_qsharp_context()

    log_path = args.output.parent / "no_result.log"
    mode = "one-step-scaled" if args.one_step_scaled else "full-circuit"
    print(f"Estimation mode: {mode}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for size in args.size:
        print(f"Sampling L={size}; mode={mode}; writing {args.output}", flush=True)
        try:
            frame = run_sampling(context, size, one_step_scaled=args.one_step_scaled)
        except Exception as error:  # noqa: BLE001 - keep the sweep alive; the log explains the gap
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"{stamp} L={size} produced no result: "
                    f"{mode}: {type(error).__name__}: {error}\n"
                )
            print(f"L={size} produced no result; see {log_path}", flush=True)
            continue
        frames.append(frame)
        # Rewritten after every size so a long sweep is resumable from partial output.
        pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)
        print(f"Finished L={size}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
