"""Sample logical resources for the 2D Fermi-Hubbard model.

Examples:
    Trace the full QPE circuit for several lattices into one table::

        python sample_hubbard_resources.py --size 2 4 6 8 10 20 \
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
from qdk import qsharp
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, LatticeGraph, QubitOperator
from qdk_chemistry.data.qubit_operator.containers.lattice import LatticeContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import (
    create_qsharp_context,
    use_qsharp_context,
)

#: U = 8t for the strong-coupling regime
HOPPING_T = 1.0
U_OVER_T = 8.0

#: Electrons per SITE
FILLING = 0.875

#: Per-site ground-state energy accuracy, in units of the hopping amplitude t.
#: Campbell (arXiv:2012.09238v4) uses an additive error eps = 0.0051 * L^2 in the FIG. 2
#: caption -- i.e. 0.0051 per site at t = 1 -- quoted there as roughly half a percent of the
#: total system energy. This is the *total* budget: it covers phase estimation, Trotter, and
#: gate synthesis together. Campbell states it for u/t = 4, whereas this sweep runs at
#: u/t = 8, where the ground-state energy per site differs; the coefficient is reused as is.
TARGET_PRECISION_PER_SITE = 0.0051

#: Number of phase-register precision bits.
QPE_PRECISION_BITS = 10

#: Share of the energy budget allocated to phase estimation. Minimizing the total Trotter
#: step count sum_k r_k, which scales as 1 / (f * sqrt(1 - f)) in the phase-estimation share
#: f, gives f = 2/3. Campbell (arXiv:2012.09238v4, App. F) reaches the same optimum by
#: minimizing over the step size instead; the sentence immediately following Eq. (F6)
#: records it as Delta_TS = (1/3) delta and Delta_PE = (2/3) delta. Campbell's delta there is
#: the combined phase-estimation-plus-Trotter budget in energy units, not a fraction, and the
#: 1:2 ratio is forced by minimizing a*t^2 + b/t rather than chosen freely.
QPE_BUDGET_FRACTION = 2.0 / 3.0

# Plaquette Trotter order.
TROTTER_ORDER = 2


def run_sampling(context, size: int) -> pd.DataFrame:
    """Measure the logical resources of one lattice size.

    Args:
        context: Q# context to build in.
        size: Lattice side length.

    Returns:
        One row of logical resources for this lattice.

    """
    started = time.monotonic()
    num_sites = size * size
    lattice = LatticeGraph.square(size, size, periodic_x=True, periodic_y=True)
    operator = QubitOperator(container=LatticeContainer(lattice))
    # The plaquette builder reads this lattice Hamiltonian's integrals directly, so the
    # Jordan-Wigner mapping is never materialized. That mapping was the one step whose
    # cost grew with the lattice rather than with the circuit being traced.
    num_qubits = 2 * num_sites

    # The total budget splits as eps = eps_QPE + eps_T. A sine-windowed register of
    # N = 2^bits - 1 queries has phase spread tan(pi / (N + 2)), so requiring eps_QPE * tau
    # to equal that spread fixes the base evolution time. The remainder is handed to the
    # plaquette builder, which sizes its own step count against it.
    # Eqn. 8 in https://arxiv.org/pdf/2609.05316.
    #
    # Note that the step count the builder derives is knowingly optimistic: it reuses
    # Campbell's IPG commutator bound for a PIG-ordered circuit. See the warning on
    # HubbardPlaquetteTrotter._step_count for the size and direction of the bias.
    energy_budget = TARGET_PRECISION_PER_SITE * num_sites
    resolution_bits = QPE_PRECISION_BITS
    qpe_budget = QPE_BUDGET_FRACTION * energy_budget
    trotter_budget = energy_budget - qpe_budget
    base_time = math.tan(math.pi / (2**resolution_bits - 1 + 2)) / qpe_budget
    trotter_settings: dict[str, float | int | str] = {"target_accuracy": trotter_budget}

    circuit_started = time.monotonic()
    unitary_builder = AlgorithmRef(
        "hamiltonian_unitary_builder",
        "plaquette",
        order=TROTTER_ORDER,
        time=base_time,
        t=HOPPING_T,
        U=U_OVER_T * HOPPING_T,
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
            "controlled_circuit_mapper", "hubbard_plaquette"
        ),
    )
    # Matches the Holevo spread used to size base_time above.
    circuit_builder.settings().set("phase_state", "sine")
    with use_qsharp_context(context):
        state_prep = identity_state_prep(num_qubits=num_qubits)
        circuit = circuit_builder.run(state_prep, operator)[0]
    qsharp_factory = circuit._qsharp_factory
    if qsharp_factory is None:
        raise RuntimeError("The QPE circuit does not have Q# factory data.")
    qsharp_context = getattr(qsharp_factory.program, "_qdk_context", qsharp)
    logical_counts = dict(
        qsharp_context.logical_counts(
            qsharp_factory.program,
            *qsharp_factory.parameter.values(),
        )
    )
    circuit_elapsed = time.monotonic() - circuit_started

    ccz_count = int(logical_counts.get("cczCount", 0))
    ccix_count = int(logical_counts.get("ccixCount", 0))
    return pd.DataFrame(
        [
            {
                "L": size,
                "sites": num_sites,
                "system_qubits": num_qubits,
                "electrons": round(FILLING * num_sites),
                "target_precision": energy_budget,
                "qpe_budget": qpe_budget,
                "trotter_budget": trotter_budget,
                "qpe_bits": resolution_bits,
                "num_unitary_queries": 2**resolution_bits - 1,
                "base_time": base_time,
                "t_max": base_time * 2**resolution_bits,
                "power_strategy": "rescale",
                "qpe_error_model": "sine-window-1sigma",
                "qpe_type": "standard-full-circuit",
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
    args = parser.parse_args(argv)

    Logger.set_global_level(Logger.LogLevel.off)

    if any(size < 2 or size % 2 for size in args.size):
        parser.error("each --size must be an even integer of at least 2")

    # QDK interpreters are thread-affine, so this context belongs to the calling thread.
    context = create_qsharp_context()

    log_path = args.output.parent / "no_result.log"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for size in args.size:
        print(f"Sampling L={size}; writing {args.output}", flush=True)
        try:
            frame = run_sampling(context, size)
        except Exception as error:  # noqa: BLE001 - keep the sweep alive; the log explains the gap
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"{stamp} L={size} produced no result: "
                    f"{type(error).__name__}: {error}\n"
                )
            print(f"L={size} produced no result; see {log_path}", flush=True)
            continue
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)
        print(f"Finished L={size}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
