"""Sample quantum resource estimates for the 2D Fermi-Hubbard model across lattice sizes.

Scripted form of ``examples/benchmark/fermi_hubbard.ipynb``, for sweeping many lattice
sizes unattended. The notebook is the place to read the derivations; this is the place
to run them.

For each ``L`` the script builds the periodic ``L x L`` Hubbard Hamiltonian under
Jordan-Wigner, sizes an iterative phase estimation from a target ground-state energy
accuracy, and reports what that costs. ``--physical`` additionally traces the circuit
through the resource estimator for a qubit/runtime Pareto frontier, which is minutes
per size rather than seconds.

The sweep is bound by memory, not by the mapper's qubit ceiling. Measured
Hamiltonian-build peak RSS grows roughly like ``L^3.7``: about 2 GB at ``L=60`` and
24 GB at ``L=120``, which extrapolates to a few hundred GB by ``L=200``. Sizes are
therefore run one at a time, each row is written as soon as it is known, and
``--resume`` skips sizes already present in the output file, so a run that is killed
partway through keeps everything it had finished.

Examples:
    Algorithmic counts for a few small lattices::

        python sample_hubbard_resources.py --sizes 4,6,8,10 -o costs.csv

    A full sweep with physical estimation, resumable::

        python sample_hubbard_resources.py --sizes 20:200:10 --physical --resume -o sweep.csv

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import csv
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:  # Unix only; the sweep's memory reporting is best-effort on other platforms.
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.plaquette_trotter import (
    PlaquetteTrotter,
)
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

#: Widest phase register the planner will consider.
MAX_RESOLUTION_BITS = 64

#: Iteration 0 carries the largest power, 2**(m-1), so it is the expensive one.
IQPE_ITERATION = 0

#: Majorana architecture physical error rate, for the physical estimate.
MAJORANA_ERROR_RATE = 1e-5

#: Largest relative error the resource estimator may report.
MAX_ESTIMATE_ERROR = 0.01

#: The sparse on-demand Majorana mapper's ceiling, which bounds L at 2*L^2 qubits.
MAPPER_QUBIT_LIMIT = 131_072

COULOMB_U = U_OVER_T * HOPPING_T
GUARD_BITS = math.ceil(math.log2(2 + 1 / (2 * QPE_FAILURE_PROBABILITY)))


@dataclass(frozen=True)
class QpeParameters:
    """Algorithm parameters derived from a target precision."""

    one_norm: float  # lambda
    evolution_time: float  # t_0
    num_bits: int  # m, including guard bits
    num_divisions: int  # r
    error_constant: float  # Campbell's W_PLAQ (arXiv:2012.09238v4, Eq. (20) and App. D)
    trotter_budget: float  # the epsilon actually allocated to Trotter error


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
        lattice, epsilon=0.0, t=HOPPING_T, U=COULOMB_U
    )
    return create("qubit_mapper").run(
        hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites)
    )


def plan_qpe(one_norm: float, precision: float, error_constant: float) -> QpeParameters:
    """Choose the (m, r) minimizing the ladder cost 2**m * r for a target precision.

    Implements Campbell's Appendix F optimization (arXiv:2012.09238v4) with ``m``
    constrained to integers: for every candidate resolution the leftover budget goes
    entirely to the Trotter term, which maximizes the step time and so minimizes ``r``.

    Args:
        one_norm: The Hamiltonian's Schatten-1 norm, lambda.
        precision: Absolute ground-state energy accuracy required.
        error_constant: Campbell's W_PLAQ for this lattice.

    Returns:
        The chosen parameters.

    Raises:
        ValueError: If no resolution meets the precision.

    """
    # H*t_0 spectrum fits in [-pi, pi]. The endpoints +/-lambda alias, but the ground
    # state sits strictly inside.
    evolution_time = math.pi / one_norm

    best = None
    for resolution_bits in range(1, MAX_RESOLUTION_BITS):
        readout_budget = 2 * one_norm / 2**resolution_bits  # = 2*pi/(t_0 * 2**m)
        if readout_budget >= precision:
            continue  # nothing left for Trotter
        trotter_budget = precision - readout_budget
        # Campbell's Eq. (F2) (arXiv:2012.09238v4, App. F) / Kivlichan Eq. (8): a
        # second-order step of duration s has ENERGY error W*s^2, not the unitary-norm
        # bound W*s^3. It does not accumulate: the r repetitions realize e^{-i H_eff s r}
        # for one fixed H_eff whose distance from H is W*s^2, and QPE reads an eigenvalue
        # of that H_eff. So the largest step meeting the budget is s = sqrt(budget / W).
        step_time = math.sqrt(trotter_budget / error_constant)
        divisions = max(1, math.ceil(evolution_time / step_time))
        cost = (2**resolution_bits) * divisions
        if best is None or cost < best[0]:
            best = (cost, resolution_bits, divisions, trotter_budget)

    if best is None:
        raise ValueError(
            f"no resolution meets precision {precision:g} for lambda {one_norm:g}"
        )

    _, resolution_bits, divisions, trotter_budget = best
    return QpeParameters(
        one_norm=one_norm,
        evolution_time=evolution_time,
        num_bits=resolution_bits + GUARD_BITS,
        num_divisions=divisions,
        error_constant=error_constant,
        trotter_budget=trotter_budget,
    )


def qpe_parameters(operator, precision: float, size: int) -> QpeParameters:
    """Derive the phase register width and Trotter step count for a target precision.

    The error constant is Campbell's W_PLAQ (arXiv:2012.09238v4, Eq. (20) and App. D),
    derived for exactly the two-section plaquette splitting used here. A generic
    term-by-term commutator bound would not know that each section is internally exact,
    and would be far more conservative.

    Args:
        operator: The qubit Hamiltonian.
        precision: Absolute ground-state energy accuracy required.
        size: Lattice side length.

    Returns:
        The chosen parameters.

    """
    constant = PlaquetteTrotter._plaquette_error_constant(
        size, size, HOPPING_T, COULOMB_U
    )
    return plan_qpe(operator.schatten_norm, precision, constant)


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
    context, operator, parameters: QpeParameters, initial_state: Circuit, size: int
) -> Circuit:
    """Return a single IQPE iteration, Trotterized according to *parameters*.

    One round is built rather than the whole ladder, so exactly one controlled unitary
    is compiled instead of ``num_bits`` of them.

    Args:
        context: Q# context to build in.
        operator: The qubit Hamiltonian.
        parameters: The planned QPE parameters.
        initial_state: The reference state circuit.
        size: Lattice side length.

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
                time=parameters.evolution_time,
                num_divisions=parameters.num_divisions,
                lattice_width=size,
                lattice_height=size,
            ),
            controlled_circuit_mapper=AlgorithmRef(
                "controlled_circuit_mapper", "pauli_sequence"
            ),
            num_bits=parameters.num_bits,
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
    from qdk.qre import PSSPC, LatticeSurgery, estimate  # noqa: PLC0415
    from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux  # noqa: PLC0415

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


def sample_size(context, size: int, *, physical: bool) -> dict:
    """Measure one lattice size.

    Args:
        context: Q# context to build in.
        size: Lattice side length.
        physical: Whether to run the physical resource estimate.

    Returns:
        One row of results.

    """
    started = time.monotonic()
    operator = qubit_operator(size)
    parameters = qpe_parameters(operator, target_precision(size), size)

    row = {
        "L": size,
        "sites": size * size,
        "qubits": operator.num_qubits,
        "terms": len(operator.pauli_strings),
        "electrons": num_electrons(size),
        "lambda": parameters.one_norm,
        "W_PLAQ_per_site": parameters.error_constant / (size * size),
        "target_precision": target_precision(size),
        "num_bits": parameters.num_bits,
        "num_divisions": parameters.num_divisions,
        "physical_qubits": "",
        "runtime_hours": "",
    }

    if physical:
        initial_state = reference_state_prep(context, size * size, num_electrons(size))
        circuit = qpe_circuit(context, operator, parameters, initial_state, size)
        table = estimate_physical(circuit, f"{size}x{size}")
        fastest = min(table, key=lambda entry: entry.runtime)
        row["physical_qubits"] = fastest.qubits
        row["runtime_hours"] = fastest.runtime / 3.6e12

    row["elapsed_s"] = round(time.monotonic() - started, 1)
    row["peak_rss_gb"] = round(peak_memory_gb(), 2)
    return row


def parse_sizes(text: str) -> list[int]:
    """Parse a size specification into lattice sizes.

    Accepts comma-separated values and ``start:stop:step`` ranges, in any mix, for
    example ``4,6,8`` or ``20:200:10`` or ``4,6,20:60:10``.

    Args:
        text: The specification.

    Returns:
        Sorted unique sizes.

    Raises:
        ValueError: If a size is not a positive even integer, or exceeds the mapper's
            qubit ceiling.

    """
    sizes: set[int] = set()
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" in piece:
            bounds = piece.split(":")
            if len(bounds) not in (2, 3):
                raise ValueError(
                    f"malformed range {piece!r}, expected start:stop[:step]"
                )
            start, stop = int(bounds[0]), int(bounds[1])
            step = int(bounds[2]) if len(bounds) == 3 else 1
            sizes.update(range(start, stop + 1, step))
        else:
            sizes.add(int(piece))

    for size in sizes:
        # The plaquette tiling needs both sides even and at least four, and folds a 2x2
        # onto itself; the builder rejects anything else, so catch it before the build.
        if size < 4 or size % 2:
            raise ValueError(
                f"lattice size {size} must be an even integer of at least 4"
            )
        if 2 * size * size > MAPPER_QUBIT_LIMIT:
            raise ValueError(
                f"lattice size {size} needs {2 * size * size} qubits, past the mapper's "
                f"{MAPPER_QUBIT_LIMIT} limit"
            )
    return sorted(sizes)


def completed_sizes(path: Path) -> set[int]:
    """Return the sizes already present in an output file, empty if it does not exist."""
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["L"]) for row in csv.DictReader(handle) if row.get("L")}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep.

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
        "--sizes",
        default="4,6,8,10",
        help="lattice sizes: comma-separated values and/or start:stop[:step] ranges (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("hubbard_resources.csv"),
        help="CSV to write, one row per size, flushed as each size finishes (default: %(default)s)",
    )
    parser.add_argument(
        "--physical",
        action="store_true",
        help="also trace the circuit for a physical qubit/runtime estimate; minutes per size",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip sizes already present in the output file",
    )
    args = parser.parse_args(argv)

    Logger.set_global_level(Logger.LogLevel.off)

    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as error:
        parser.error(str(error))

    done = completed_sizes(args.output) if args.resume else set()
    pending = [size for size in sizes if size not in done]
    if done:
        print(
            f"resuming: {len(done)} size(s) already in {args.output}, {len(pending)} to go"
        )
    if not pending:
        print("nothing to do")
        return 0

    print(
        f"Hubbard U/T = {U_OVER_T:g}, filling {FILLING:g} e/site, target {TARGET_PRECISION_PER_SITE:g} T/site"
    )
    print(f"sizes: {pending}")
    print(f"physical estimation: {'on' if args.physical else 'off'}")

    # QDK interpreters are thread-affine, so this context belongs to the calling thread.
    context = create_qsharp_context()

    # Opened once and flushed per row: the sweep is memory-bound and may be killed
    # partway through, and a partial file is worth far more than none.
    is_new = not args.output.exists() or not done
    with args.output.open("a" if done else "w", newline="", encoding="utf-8") as handle:
        writer = None
        for size in pending:
            try:
                row = sample_size(context, size, physical=args.physical)
            except MemoryError:
                print(
                    f"L={size}: out of memory, stopping. Completed sizes are in {args.output}."
                )
                return 1
            except Exception as error:  # noqa: BLE001 - one bad size must not lose the rest
                print(f"L={size}: FAILED ({type(error).__name__}: {error})")
                continue

            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if is_new:
                    writer.writeheader()
            writer.writerow(row)
            handle.flush()

            physical = (
                f", {row['physical_qubits']} physical qubits, {row['runtime_hours']:.3g} h"
                if args.physical and row["physical_qubits"] != ""
                else ""
            )
            print(
                f"L={size:>3}: {row['qubits']:>6} qubits, {row['terms']:>7} terms, "
                f"m={row['num_bits']}, r={row['num_divisions']}{physical} "
                f"[{row['elapsed_s']}s, peak {row['peak_rss_gb']} GB]"
            )

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
