"""Sample a quantum resource estimate for the 2D Fermi-Hubbard model.

For the requested ``L``, the script builds the periodic ``L x L`` Hubbard Hamiltonian
under Jordan-Wigner, sizes an iterative phase estimation from a target ground-state
energy accuracy, and traces the resulting circuit through the resource estimator for
a qubit/runtime Pareto frontier.

Large lattices are bound by memory. Measured Hamiltonian-build peak RSS is about 2 GB
at ``L=60`` and 24 GB at ``L=120``.

Examples:
    Sample a 50 x 50 lattice::

        python sample_hubbard_resources.py --size 50 -o cost.csv
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
from pathlib import Path

try:  # Unix only; the sweep's memory reporting is best-effort on other platforms.
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

from qdk.qre import PSSPC, EstimationTable, LatticeSurgery, estimate
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, Circuit, LatticeGraph, MajoranaMapping
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.qsharp import (
    QSHARP_UTILS,
    create_qsharp_context,
    use_qsharp_context,
)

#: Hopping amplitude. Energies are quoted in units of this.
HOPPING_T = 1.0

#: Interaction strength as a multiple of the hopping. U = 8t is the strong-coupling
#: regime of the LANL Fermi-Hubbard chapter (arXiv:2406.06625 Ch. 4) and of Campbell's
#: Table II (arXiv:2012.09238v4).
U_OVER_T = 8.0

#: Electrons per SITE, not per orbital. n = 1 is half filling
#: (Kivlichan arXiv:1902.10673 Sec. 3.2).
FILLING = 0.875

#: Per-site ground-state energy accuracy, in units of T. Chosen for this benchmark;
#: the LANL appendix instead specifies accuracy on the order parameter.
TARGET_PRECISION_PER_SITE = 0.0051

#: Suzuki-Trotter product-formula order. The plaquette builder implements order 2 only.
TROTTER_ORDER = 2

#: One minus the confidence that a readout meets the target precision.
QPE_FAILURE_PROBABILITY = 0.1

#: Iteration 0 carries the largest power, 2**(m-1), so it is the expensive one.
IQPE_ITERATION = 0

#: Majorana architecture physical error rate, for the physical estimate.
MAJORANA_ERROR_RATE = 1e-5

#: Largest relative error the resource estimator may report.
MAX_ESTIMATE_ERROR = 0.01


def target_precision(size: int) -> float:
    """Return the absolute ground-state energy accuracy required of an L x L lattice."""
    return TARGET_PRECISION_PER_SITE * size * size


def num_electrons(size: int) -> int:
    """Return the electron count nearest the requested per-site filling."""
    return round(FILLING * size * size)


def qubit_operator(size: int):
    """Return the Jordan-Wigner qubit Hamiltonian of the periodic size x size lattice."""
    num_sites = size * size
    lattice = LatticeGraph.square(size, size, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(
        lattice, epsilon=0.0, t=HOPPING_T, U=U_OVER_T * HOPPING_T
    )
    return create("qubit_mapper").run(
        hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites)
    )


def reference_state_prep(context, num_sites: int, electrons: int) -> Circuit:
    """Return an occupation-number determinant, one X gate per occupied spin-orbital.

    Electrons are split as evenly as possible between the spin-up block (qubits
    ``0..N-1``) and the spin-down block (``N..2N-1``).

    Args:
        context: Q# context to build in.
        num_sites: Number of lattice sites.
        electrons: Total electron count.

    Returns:
        The state-preparation circuit.

    """
    num_up = (electrons + 1) // 2
    num_down = electrons // 2
    occupations = (
        [1] * num_up
        + [0] * (num_sites - num_up)
        + [1] * num_down
        + [0] * (num_sites - num_down)
    )
    with use_qsharp_context(context):
        state_preparation = QSHARP_UTILS.StatePreparation
        params = state_preparation.SingleReferenceParams(
            bitStrings=occupations, numQubits=2 * num_sites
        )
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=state_preparation.MakeSingleReferenceStateCircuit,
                parameter=vars(params),
            ),
            qsharp_op=state_preparation.MakePrepareSingleReferenceStateOp(params),
            encoding="jordan-wigner",
        )


def qpe_circuit(
    context,
    operator,
    evolution_time: float,
    trotter_budget: float,
    initial_state: Circuit,
    size: int,
    num_bits: int,
) -> Circuit:
    """Return a single IQPE iteration with the requested evolution and resolution.

    One round is built rather than the whole ladder, so exactly one controlled unitary
    is compiled instead of ``num_bits`` of them.

    Args:
        context: Q# context to build in.
        operator: The qubit Hamiltonian.
        evolution_time: Base Hamiltonian evolution time.
        trotter_budget: Accuracy allocated to Trotter error.
        initial_state: The reference state circuit.
        size: Lattice side length.
        num_bits: Number of QPE resolution bits, including guard bits.

    Returns:
        The circuit for one IQPE iteration.

    """
    with use_qsharp_context(context):
        builder = create(
            "qpe_circuit_builder",
            "qdk_iterative",
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder",
                "plaquette",
                order=TROTTER_ORDER,
                time=evolution_time,
                target_accuracy=trotter_budget,
                lattice_width=size,
                lattice_height=size,
            ),
            controlled_circuit_mapper=AlgorithmRef(
                "controlled_circuit_mapper", "pauli_sequence"
            ),
            num_bits=num_bits,
            num_iteration=IQPE_ITERATION,
        )
        return builder.run(initial_state, operator)[0]


def estimate_physical(circuit: Circuit, name: str):
    """Return the qubit/runtime Pareto frontier on the Majorana architecture.

    Args:
        circuit: The circuit to trace.
        name: Label for the estimate.

    Returns:
        The estimator's result table.

    """
    application = circuit.get_qre_application()
    trace_query = (
        application.q()
        * PSSPC.q(num_ts_per_rotation=list(range(20, 45, 2)))
        * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 20)])
    )
    isa_query = ThreeAux.q() * RoundBasedFactory.q(code_query=ThreeAux.q())
    return estimate(
        application,
        Majorana(error_rate=MAJORANA_ERROR_RATE),
        isa_query,
        trace_query,
        max_error=MAX_ESTIMATE_ERROR,
        name=name,
    )


def peak_memory_gb() -> float:
    """Return peak resident set size in GB, or 0.0 where the platform cannot report it."""
    if resource is None:  # pragma: no cover - platform dependent
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 1e6 if sys.platform != "darwin" else peak / 1e9


def run_sampling(context, size: int) -> EstimationTable:
    """Measure one lattice size.

    Args:
        context: Q# context to build in.
        size: Lattice side length.

    Returns:
        The estimator's table, with this run's parameters attached as columns.

    """
    started = time.monotonic()
    operator = qubit_operator(size)
    one_norm = operator.schatten_norm
    energy_budget = target_precision(size)
    qpe_budget = energy_budget / 2
    trotter_budget = energy_budget
    base_time = math.pi / one_norm / 2
    resolution_bits = math.ceil(
        math.log2(2 * math.pi / qpe_budget / base_time)
    )

    initial_state = reference_state_prep(context, size * size, num_electrons(size))
    circuit = qpe_circuit(
        context,
        operator,
        base_time,
        trotter_budget,
        initial_state,
        size,
        resolution_bits,
    )
    table = estimate_physical(circuit, f"{size}x{size}")
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
        "target_precision": energy_budget,
        "trotter_budget": trotter_budget,
        "num_bits": resolution_bits,
        "elapsed_s": round(time.monotonic() - started, 1),
        "peak_rss_gb": round(peak_memory_gb(), 2),
    }
    for name, value in parameters.items():
        table.add_column(name, lambda _entry, value=value: value)
    return table


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
        required=True,
        help="even lattice side length of at least 4",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hubbard_resources.csv"),
        help="CSV to write (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    Logger.set_global_level(Logger.LogLevel.off)

    if args.size < 4 or args.size % 2:
        parser.error("--size must be an even integer of at least 4")

    print(
        f"Hubbard U/T = {U_OVER_T:g}, filling {FILLING:g} e/site, target {TARGET_PRECISION_PER_SITE:g} T/site"
    )
    print(f"size: {args.size}")

    # QDK interpreters are thread-affine, so this context belongs to the calling thread.
    context = create_qsharp_context()

    frame = run_sampling(context, args.size).as_frame()
    frame.to_csv(args.output, index=False)

    first = frame.iloc[0]
    print(
        f"L={args.size:>3}: {first['system_qubits']:>6} qubits, {first['terms']:>7} terms, "
        f"m={first['num_bits']}, "
        f"{len(frame)} physical estimates "
        f"[{first['elapsed_s']}s, peak {first['peak_rss_gb']} GB]"
    )

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
