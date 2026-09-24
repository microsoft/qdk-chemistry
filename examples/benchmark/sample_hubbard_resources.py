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

import numpy as np
import pandas as pd
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import AlgorithmRef, LatticeGraph
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


def jordan_wigner_profile(hamiltonian, size: int) -> tuple[int, float]:
    r"""Return the Pauli term count and one-norm of the mapped uniform Hubbard model.

    Both are read from the Hamiltonian's own integrals, so the benchmark reports them
    without paying for a fermion-to-qubit mapping that the plaquette builder no longer
    needs. With spin-blocked Jordan-Wigner modes and :math:`n_p = (I - Z_p)/2`:

    * each off-diagonal integral :math:`h_{ij}` becomes an ``XZ...ZX`` and a
      ``YZ...ZY`` string per spin, of magnitude :math:`|h_{ij}|/2`, giving four terms
      of weight :math:`2|h_{ij}|`;
    * site :math:`i` with on-site energy :math:`e_i` and interaction :math:`U_i`
      contributes :math:`Z_\uparrow` and :math:`Z_\downarrow` of magnitude
      :math:`|e_i/2 + U_i/4|`, one :math:`Z_\uparrow Z_\downarrow` of magnitude
      :math:`U_i/4`, and an identity share :math:`e_i + U_i/4`.

    Reading the integrals rather than re-deriving them from ``HOPPING_T`` keeps this
    correct on the 2x2 torus, where the two wrap-around edges of each axis coincide and
    so carry twice the weight. Checked against ``qubit_mapper`` for ``L = 2`` to ``16``.

    Args:
        hamiltonian: The lattice Hamiltonian being sampled.
        size: Lattice side length.

    Returns:
        The number of Pauli terms and the coefficient one-norm.

    """
    num_sites = size * size
    one_body, _ = hamiltonian.get_one_body_integrals()

    hopping = np.triu(np.abs(np.asarray(one_body)), k=1)
    num_bonds = int(np.count_nonzero(hopping))
    hopping_weight = 2.0 * float(hopping.sum())

    energies = np.asarray(one_body).diagonal()
    interactions = np.array(
        [hamiltonian.get_two_body_element(i, i, i, i) for i in range(num_sites)]
    )
    single_z = np.abs(0.5 * energies + 0.25 * interactions)
    pair_z = np.abs(0.25 * interactions)
    identity = float(np.sum(energies + 0.25 * interactions))

    num_terms = 4 * num_bonds + 3 * num_sites + 1
    one_norm = (
        hopping_weight + 2.0 * float(single_z.sum()) + float(pair_z.sum()) + abs(identity)
    )
    return num_terms, one_norm


def traced_step_counts(context, hamiltonian, step_time: float, size: int, num_divisions: int):
    """Return logical counts for a controlled evolution of ``num_divisions`` Trotter steps.

    Args:
        context: Q# context to build in.
        hamiltonian: The lattice Hamiltonian being evolved.
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
        ).run(hamiltonian)
        circuit = create("controlled_circuit_mapper", "pauli_sequence").run(unitary)
        application = circuit.get_qre_application()
        return dict(
            get_qsharp_context().logical_counts(application.entry_expr, *application.args)
        )


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
    # The plaquette builder reads this lattice Hamiltonian's integrals directly, so the
    # Jordan-Wigner mapping is never materialized. That mapping was the one step whose
    # cost grew with the lattice rather than with the circuit being traced.
    num_qubits = 2 * num_sites
    num_terms, one_norm = jordan_wigner_profile(hamiltonian, size)

    # The total budget splits as eps = eps_QPE + eps_T. A sine-windowed register of
    # N = 2^bits - 1 queries has phase spread tan(pi / (N + 2)), so requiring eps_QPE * tau
    # to equal that spread fixes the base evolution time. The remainder is handed to the
    # plaquette builder, which sizes its own step count against it.
    # Eqn. 8 in https://arxiv.org/pdf/2609.05316.
    energy_budget = TARGET_PRECISION_PER_SITE * num_sites
    resolution_bits = QPE_PRECISION_BITS
    qpe_budget = QPE_BUDGET_FRACTION * energy_budget
    trotter_budget = energy_budget - qpe_budget
    base_time = math.tan(math.pi / (2**resolution_bits - 1 + 2)) / qpe_budget
    trotter_settings: dict[str, float | int | str] = {"target_accuracy": trotter_budget}

    # Bit k evolves for base_time * 2^k, so each bit resolves its own step count from the
    # builder's error bound. The schedule is reported in both modes, and its sum is the
    # ladder multiplier in one-step mode.
    steps_per_bit = []
    for bit in range(resolution_bits):
        evolution_time = base_time * 2**bit
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
        steps_per_bit.append(builder._resolve_num_divisions(hamiltonian, evolution_time))
    total_steps = sum(steps_per_bit)
    step_time = base_time * 2 ** (resolution_bits - 1) / steps_per_bit[-1]

    circuit_started = time.monotonic()
    step_counts = traced_step_counts(context, hamiltonian, step_time, size, 1)
    if one_step_scaled:
        # Every count is multiplied by the step total. This ignores boundary merging
        # between adjacent second-order steps, so it overestimates the ladder rather than
        # inferring a lower cost from a multi-step trace. The traced block carries one
        # control qubit whereas the full algorithm carries a resolution_bits-wide phase
        # register, so the remaining phase qubits are added back. The inverse QFT and
        # window preparation are omitted; both are negligible against the query cost.
        logical_counts: dict[str, int] = {
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
    else:
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
        # Matches the Holevo spread used to size base_time above.
        circuit_builder.settings().set("phase_window", "sine")
        with use_qsharp_context(context):
            state_prep = identity_state_prep(num_qubits=num_qubits)
            circuit = circuit_builder.run(state_prep, hamiltonian)[0]
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
                "system_qubits": num_qubits,
                "terms": num_terms,
                "electrons": round(FILLING * num_sites),
                "lambda": one_norm,
                "target_precision": energy_budget,
                "qpe_budget": qpe_budget,
                "qpe_budget_fraction": QPE_BUDGET_FRACTION,
                "trotter_budget": trotter_budget,
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
