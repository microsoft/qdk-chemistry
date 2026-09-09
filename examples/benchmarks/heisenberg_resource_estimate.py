"""Estimate resources for an open square-lattice Heisenberg model.

Defaults use a 200 x 200 J1-J2 lattice with J1 = 1 and J2 = 0.5, and the Kitaev
notebook's evolution/estimation settings: fourth-order Trotter evolution to
time 1000 with timestep 0.04 and a measurement-free Y-basis rotation.

Inputs specify H = sum J_a S_i^a S_j^a + sum h_a S_i^a with S = sigma / 2.
Evolution uses exp(-i H t) with hbar = 1; time is in inverse exchange-energy
units. The QRE error budget does not bound Trotter error.
QRE settings are fixed to the notebook convention, including its 1% error budget.

Run with --help for all inputs. --j '{"1": 1.0, "2": 0.5}' sets the isotropic
exchange; --jx, --jy, and --jz override individual spin components. Couplings
accept JSON scalars, matrices, or shell maps, and fields accept scalars or
site arrays. The Python helpers also accept NumPy arrays directly.
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
from typing import TYPE_CHECKING

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DrivenQubitHamiltonian,
    LatticeGraph,
    QubitOperator,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian

if TYPE_CHECKING:
    from qdk.qre import EstimationTable

ShellCoupling = float | np.ndarray | Mapping[int, float | np.ndarray]
DEFAULT_HEISENBERG_COUPLINGS = {1: 1.0, 2: 0.5}


def create_lattice(nx: int = 200, ny: int | None = None) -> LatticeGraph:
    """Create an open nx-by-ny square lattice; ny defaults to nx."""
    return LatticeGraph.square(
        nx,
        nx if ny is None else ny,
        periodic_x=False,
        periodic_y=False,
    )


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
    """
    isotropic = DEFAULT_HEISENBERG_COUPLINGS if j is None else j
    couplings = [isotropic if value is None else value for value in (jx, jy, jz)]
    # Shell maps select sparse construction even for nearest-neighbor scalars.
    shells = [value if isinstance(value, Mapping) else {1: value} for value in couplings]
    pauli_couplings = [{shell: coefficient / 4.0 for shell, coefficient in values.items()} for values in shells]
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


def build_time_evolution_circuit(
    hamiltonian: QubitOperator,
    *,
    dt: float,
    total_time: float,
    trotter_order: int = 4,
    weight_threshold: float = 1e-12,
) -> Circuit:
    """Build one Euler interval with a symbolically repeated packed Trotter step.

    As in the notebook, the division count is round(total_time / dt); the
    effective timestep is total_time divided by that count. No state vector,
    full-register Pauli labels, or expanded circuit is constructed.
    """
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(total_time) or total_time <= 0:
        raise ValueError("dt and total_time must be finite and positive.")
    if not math.isfinite(total_time / dt) or round(total_time / dt) < 1:
        raise ValueError("total_time / dt must round to a finite, positive division count.")
    if trotter_order != 1 and (trotter_order < 2 or trotter_order % 2):
        raise ValueError("trotter_order must be 1 or a positive even integer.")
    if not math.isfinite(weight_threshold) or weight_threshold < 0:
        raise ValueError("weight_threshold must be finite and nonnegative.")
    num_steps = round(total_time / dt)
    zero_hamiltonian = QubitOperator.from_sparse_terms(hamiltonian.num_qubits, [{}], np.array([0.0]))
    time_dependent_hamiltonian = DrivenQubitHamiltonian(
        hamiltonian,
        zero_hamiltonian,
        drive=lambda _time: 0.0,
    )
    # Multiple Euler intervals would combine and expand the repeated formulas.
    circuit_builder = create(
        "evolution_circuit_builder",
        "euler",
        evolution_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "trotter",
            order=trotter_order,
            num_divisions=num_steps,
            weight_threshold=weight_threshold,
        ),
        propagator=AlgorithmRef("propagator", "magnus", order=1),
        circuit_mapper=AlgorithmRef("circuit_mapper", "pauli_sequence"),
        total_time=total_time,
        dt=total_time,
    )
    state_prep = identity_state_prep(num_qubits=hamiltonian.num_qubits)
    return circuit_builder.run(time_dependent_hamiltonian, state_prep)


def estimate_physical(circuit: Circuit, name: str) -> EstimationTable:
    """Run the notebook's fixed QRE sweep and add resource columns.

    The 1% fault-tolerance/synthesis error budget does not bound Trotter error.
    """
    from qdk.qre import PSSPC, LatticeSurgery, estimate
    from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux

    application = circuit.get_qre_application()
    architecture = Majorana(error_rate=1e-5)
    trace_query = (
        application.q()
        * PSSPC.q(num_ts_per_rotation=list(range(20, 45, 2)))
        * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 20)])
    )
    isa_query = ThreeAux.q() * RoundBasedFactory.q(code_query=ThreeAux.q())
    results = estimate(application, architecture, isa_query, trace_query, max_error=0.01, name=name)
    results.add_qubit_partition_column()
    results.add_factory_summary_column()
    return results


def _numeric_parameter(value: object) -> float | np.ndarray:
    """Validate a scalar or array read from a JSON parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
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
    """Parse lattice, Hamiltonian, and evolution inputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        "--total-time", type=float, default=1000.0, help="Evolution time in inverse exchange-energy units."
    )
    parser.add_argument(
        "--dt", "--trotter-step", type=float, default=0.04, help="Requested step; divisions = round(total_time / dt)."
    )
    parser.add_argument("--trotter-order", type=int, default=4, help="1 or any positive even order.")
    parser.add_argument("--weight-threshold", type=float, default=1e-12, help="Trotter coefficient cutoff.")
    parser.add_argument(
        "--spin-direction",
        type=float,
        nargs=3,
        default=(0.0, 1.0, 0.0),
        help="Final measurement-free spin-basis rotation.",
    )
    args = parser.parse_args(argv)
    if args.ny is None:
        args.ny = args.nx
    if args.nx < 1 or args.ny < 1:
        parser.error("nx and ny must be positive.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the notebook workflow with explicit command-line parameters."""
    args = parse_args(argv)
    Logger.set_global_level(Logger.LogLevel.off)
    graph = create_lattice(args.nx, args.ny)
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
    print(f"Open {args.nx} x {args.ny} Heisenberg benchmark", flush=True)
    print(f"Sites: {graph.num_sites}; Hamiltonian Pauli terms: {hamiltonian.num_terms}", flush=True)
    circuit = build_time_evolution_circuit(
        hamiltonian,
        dt=args.dt,
        total_time=args.total_time,
        trotter_order=args.trotter_order,
        weight_threshold=args.weight_threshold,
    )
    circuit = circuit.with_uniform_spin_basis_rotation(args.spin_direction, num_qubits=graph.num_sites)
    num_steps = round(args.total_time / args.dt)
    print(f"Spin basis rotation direction: {args.spin_direction}")
    print(
        f"Evolution: total_time={args.total_time}, requested_step={args.dt}, "
        f"effective_step={args.total_time / num_steps}, Trotter order={args.trotter_order}, divisions={num_steps}",
        flush=True,
    )
    estimates = estimate_physical(circuit, f"Heisenberg {args.nx} x {args.ny}")
    print(f"QRE statistics: {estimates.stats}")
    if not estimates:
        print("No feasible estimates for the fixed notebook QRE settings.")
    else:
        print(estimates.as_frame().to_string(index=False))


if __name__ == "__main__":
    main()
