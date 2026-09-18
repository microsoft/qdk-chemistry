"""Estimate resources for standard QPE on an open square-lattice Heisenberg model.

Defaults match heisenberg_nxn_resource_estimate.ipynb: a grouped 200 x 200
J1-J2 lattice with J1 = 1 and J2 = 0.5, ten phase bits, fourth-order Trotter,
and longest powered time 1000. The base time is 1000 / 512 = 1.953125.
Requested dt = 0.1 gives ceil(base_time / dt) = 20 divisions and an actual
step of 0.09765625, unchanged across powers by power_strategy="repeat".
The pauli_sequence controlled mapper preserves the formula's declared disjoint layers.
Identity preparation, inverse QFT, and phase measurements are included;
there is no final spin-basis rotation.

Inputs specify H = sum J_a S_i^a S_j^a + sum h_a S_i^a with S = sigma / 2.
Evolution uses exp(-i H t) with hbar = 1; time is in inverse exchange-energy
units. The step still needs Trotter-error calibration at the longest power;
identity preparation is a resource-counting placeholder, not a ground state.
The QRE error budget does not bound Trotter error.
QRE settings are fixed to the notebook convention, including its 1% error budget.

Run with --help for all inputs. --j '{"1": 1.0, "2": 0.5}' sets the isotropic
exchange; --jx, --jy, and --jz override individual spin components. Couplings
accept JSON scalars, matrices, or shell maps, and fields accept scalars or
site arrays. The Python helpers also accept NumPy arrays directly.
--longest-time (also --total-time) specifies the longest powered evolution,
not the base time; --num-bits determines the corresponding base time.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    LatticeGeometry,
    LatticeGraph,
    QubitOperator,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian

if TYPE_CHECKING:
    from qdk.qre import EstimationTable

ShellCoupling: TypeAlias = float | np.ndarray | Mapping[int, float | np.ndarray]
DEFAULT_HEISENBERG_COUPLINGS = {1: 1.0, 2: 0.5}


def create_lattice(
    nx: int = 200, ny: int | None = None, *, shells: Sequence[int] = (1, 2)
) -> LatticeGraph:
    """Create an open nx-by-ny square graph with selected shells; ny defaults to nx.

    The default selection supports the J1-J2 model. Pass the union of active
    coupling shells when using different exchanges.
    """
    geometry = LatticeGeometry.square(
        nx,
        nx if ny is None else ny,
        periodic_x=False,
        periodic_y=False,
    )
    return LatticeGraph.from_geometry(geometry, shells=shells)


def create_hamiltonian(
    graph: LatticeGraph,
    *,
    j: ShellCoupling | None = None,
    jx: ShellCoupling | None = None,
    jy: ShellCoupling | None = None,
    jz: ShellCoupling | None = None,
    hx: float | np.ndarray = 0.0,
    hy: float | np.ndarray = 0.0,
    hz: float | np.ndarray = 0.0,
) -> QubitOperator:
    """Build a grouped physical-spin Heisenberg model without dense pair matrices.

    j supplies the isotropic exchange; omitted jx, jy, and jz inherit it.
    Scalars/arrays specify shell 1, mappings specify geometric shells, and
    empty mappings or zero disable an interaction. None for j selects the
    J1-J2 defaults. Exchange coefficients are divided by four and linear
    spin-field coefficients by two to obtain Pauli coefficients.
    The graph must already select every nonzero requested shell.
    """
    isotropic = DEFAULT_HEISENBERG_COUPLINGS if j is None else j
    couplings = [isotropic if value is None else value for value in (jx, jy, jz)]
    # Shell maps select sparse construction even for nearest-neighbor scalars.
    shells = [
        value if isinstance(value, Mapping) else {1: value} for value in couplings
    ]
    pauli_couplings = [
        {shell: coefficient / 4.0 for shell, coefficient in values.items()}
        for values in shells
    ]
    return create_heisenberg_hamiltonian(
        graph,
        jx=pauli_couplings[0],
        jy=pauli_couplings[1],
        jz=pauli_couplings[2],
        hx=hx / 2.0,
        hy=hy / 2.0,
        hz=hz / 2.0,
        include_term_groups=True,
    )


def build_qpe_circuit(
    hamiltonian: QubitOperator,
    *,
    dt: float,
    base_time: float,
    num_bits: int = 10,
    trotter_order: int = 4,
    weight_threshold: float = 1e-12,
) -> Circuit:
    """Build the notebook's standard QPE with symbolically repeated controlled steps.

    base_time is the time of U, not the largest controlled power. Divisions
    are ceil(base_time / dt), so the actual step is at most dt. Every power
    repeats that same step. Identity preparation and phase measurements
    match the notebook; no dense state vector or QIR export is needed.
    """
    if (
        not math.isfinite(dt)
        or dt <= 0
        or not math.isfinite(base_time)
        or base_time <= 0
    ):
        raise ValueError("dt and base_time must be finite and positive.")
    if not math.isfinite(base_time / dt):
        raise ValueError("base_time / dt must be finite.")
    if isinstance(num_bits, bool) or not isinstance(num_bits, int) or num_bits < 1:
        raise ValueError("num_bits must be a positive integer.")
    if trotter_order != 1 and (trotter_order < 2 or trotter_order % 2):
        raise ValueError("trotter_order must be 1 or a positive even integer.")
    if not math.isfinite(weight_threshold) or weight_threshold < 0:
        raise ValueError("weight_threshold must be finite and nonnegative.")
    num_steps = max(1, math.ceil(base_time / dt))
    circuit_builder = create(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=num_bits,
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "trotter",
            time=base_time,
            order=trotter_order,
            num_divisions=num_steps,
            weight_threshold=weight_threshold,
            power_strategy="repeat",
        ),
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper", "pauli_sequence"
        ),
    )
    return circuit_builder.run(
        state_preparation=identity_state_prep(num_qubits=hamiltonian.num_qubits),
        qubit_hamiltonian=hamiltonian,
    )[0]


def estimate_physical(circuit: Circuit, name: str) -> EstimationTable:
    """Run the notebook's fixed QRE sweep and add resource columns.

    The 1% fault-tolerance/synthesis error budget does not bound Trotter error.
    """
    from qdk.qre import PSSPC, LatticeSurgery, estimate  # noqa: PLC0415, RUF100
    from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux  # noqa: PLC0415, RUF100

    application = circuit.get_qre_application()
    architecture = Majorana(error_rate=1e-5)
    trace_query = (
        application.q()
        * PSSPC.q(num_ts_per_rotation=list(range(20, 45, 2)))
        * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 20)])
    )
    isa_query = ThreeAux.q() * RoundBasedFactory.q(code_query=ThreeAux.q())
    results = estimate(
        application, architecture, isa_query, trace_query, max_error=0.01, name=name
    )
    results.add_qubit_partition_column()
    results.add_factory_summary_column()
    return results


def _numeric_parameter(value: object) -> float | np.ndarray:
    """Validate a scalar or array read from a JSON parameter."""
    if isinstance(value, bool) or not isinstance(value, int | float | list):
        raise TypeError("Expected a number or numeric array.")
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Parameters must be finite.")
    return float(array) if array.ndim == 0 else array


def _parse_parameter(text: str) -> float | np.ndarray:
    """Parse a JSON scalar or site/bond array for the command line."""
    try:
        return _numeric_parameter(json.loads(text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_couplings(text: str) -> dict[int, float | np.ndarray]:
    """Parse exchange parameters while retaining the scalable shell-map path."""
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            return {1: _numeric_parameter(value)}
        couplings = {}
        for key, parameter in value.items():
            shell = int(key)
            if shell < 1 or str(shell) != key:
                raise ValueError("Shell keys must be positive integers.")
            couplings[shell] = _numeric_parameter(parameter)
        return couplings
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse lattice, Hamiltonian, and standard-QPE inputs."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--nx", type=int, default=200, help="Sites along x.")
    parser.add_argument("--ny", type=int, help="Sites along y; defaults to nx.")
    parser.add_argument(
        "--j",
        type=_parse_couplings,
        default=DEFAULT_HEISENBERG_COUPLINGS.copy(),
        help="Physical isotropic exchange: JSON scalar, matrix, or shell map.",
    )
    for axis in ("x", "y", "z"):
        parser.add_argument(
            f"--j{axis}",
            type=_parse_couplings,
            help=f"Physical J{axis} exchange override; inherits j when omitted.",
        )
        parser.add_argument(
            f"--h{axis}",
            type=_parse_parameter,
            default=0.0,
            help=f"Linear S{axis} field coefficient: JSON scalar or per-site array.",
        )
    parser.add_argument(
        "--longest-time",
        "--total-time",
        dest="longest_time",
        type=float,
        default=1000.0,
        help="Longest powered time; base_time = longest_time / 2**(num_bits - 1).",
    )
    parser.add_argument(
        "--num-bits", type=int, default=10, help="Number of standard-QPE phase qubits."
    )
    parser.add_argument(
        "--dt",
        "--trotter-step",
        type=float,
        default=0.1,
        help="Maximum Trotter step; base divisions = ceil(base_time / dt).",
    )
    parser.add_argument(
        "--trotter-order", type=int, default=4, help="1 or any positive even order."
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=1e-12,
        help="Trotter coefficient cutoff.",
    )
    args = parser.parse_args(argv)
    if args.ny is None:
        args.ny = args.nx
    if args.nx < 1 or args.ny < 1:
        parser.error("nx and ny must be positive.")
    if args.num_bits < 1:
        parser.error("num_bits must be a positive integer.")
    if not math.isfinite(args.longest_time) or args.longest_time <= 0:
        parser.error("longest_time must be finite and positive.")
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("dt must be finite and positive.")
    couplings = [
        args.j if value is None else value for value in (args.jx, args.jy, args.jz)
    ]
    args.shells = sorted(
        {
            shell
            for coupling in couplings
            for shell, value in coupling.items()
            if np.any(value != 0.0)
        }
    )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the notebook workflow with explicit command-line parameters."""
    args = parse_args(argv)
    Logger.set_global_level(Logger.LogLevel.off)
    max_power = 2 ** (args.num_bits - 1)
    base_time = args.longest_time / max_power
    graph = create_lattice(args.nx, args.ny, shells=args.shells)
    hamiltonian = create_hamiltonian(
        graph,
        j=args.j,
        jx=args.jx,
        jy=args.jy,
        jz=args.jz,
        hx=args.hx,
        hy=args.hy,
        hz=args.hz,
    )
    print(f"Open {args.nx} x {args.ny} Heisenberg standard-QPE benchmark", flush=True)
    print(
        f"Sites: {graph.num_sites}; Hamiltonian Pauli terms: {hamiltonian.num_terms}",
        flush=True,
    )
    circuit = build_qpe_circuit(
        hamiltonian,
        dt=args.dt,
        base_time=base_time,
        num_bits=args.num_bits,
        trotter_order=args.trotter_order,
        weight_threshold=args.weight_threshold,
    )
    num_steps = max(1, math.ceil(base_time / args.dt))
    print(
        f"QPE: bits={args.num_bits}, base_time={base_time}, max_power={max_power}, longest_time={args.longest_time}",
        flush=True,
    )
    print(
        f"Requested step={args.dt}, effective_step={base_time / num_steps}, "
        f"Trotter order={args.trotter_order}, divisions={num_steps}, "
        f"total_powered_steps={num_steps * (2**args.num_bits - 1)}",
        flush=True,
    )
    estimates = estimate_physical(circuit, f"{args.nx}x{args.ny} lattice")
    print(f"QRE statistics: {estimates.stats}")
    if not estimates:
        print("No feasible estimates for the fixed notebook QRE settings.")
    else:
        print(estimates.as_frame().to_string(index=False))


if __name__ == "__main__":
    main()
