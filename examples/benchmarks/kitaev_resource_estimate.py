"""Estimate resources for an open, complete-plaquette Kitaev lattice.

Defaults reproduce the 200 x 200 notebook: fourth-order Trotter evolution to
time 1000 with timestep 0.04, followed by a measurement-free Y-basis rotation.
Exchange inputs are physical-spin couplings (S = sigma / 2), in meV for the
default parameters. Evolution uses exp(-i H t) with hbar = 1, so time is in
inverse exchange-energy units. The QRE error budget does not bound Trotter error.
QRE settings are fixed to the notebook convention, including its 1% error budget.

Run with --help for all inputs. Couplings accept JSON numbers or shell maps,
for example --kx '{"1": -13.3, "2": -0.67, "3": 0.1}'. JSON arrays can specify
bond-dependent couplings. The Python helpers also accept NumPy arrays directly.
Use --output-csv to save the result table as well as printing it.
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
from pathlib import Path
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
from qdk_chemistry.utils.model_hamiltonians import create_kitaev_hamiltonian

if TYPE_CHECKING:
    from qdk.qre import EstimationTable

ShellCoupling = float | np.ndarray | Mapping[int, float | np.ndarray]
DEFAULT_KITAEV_COUPLINGS = {1: -13.3, 2: -0.67, 3: 0.1}
DEFAULT_HEISENBERG_COUPLINGS = {1: -1.3, 3: 1.0}


def create_lattice(nx: int = 200, ny: int | None = None) -> LatticeGraph:
    """Create an open nx-by-ny complete-plaquette patch; ny defaults to nx."""
    return LatticeGraph.honeycomb_plaquettes(
        nx,
        nx if ny is None else ny,
        periodic_x=False,
        periodic_y=False,
    )


def create_hamiltonian(
    graph: LatticeGraph,
    *,
    kx: ShellCoupling | None = None,
    ky: ShellCoupling | None = None,
    kz: ShellCoupling | None = None,
    j: ShellCoupling | None = None,
    gamma: float | np.ndarray = 9.4,
    gamma_prime: float | np.ndarray = -2.3,
    gamma_x: float | np.ndarray | None = None,
    gamma_y: float | np.ndarray | None = None,
    gamma_z: float | np.ndarray | None = None,
    gamma_prime_x: float | np.ndarray | None = None,
    gamma_prime_y: float | np.ndarray | None = None,
    gamma_prime_z: float | np.ndarray | None = None,
    magnetic_field_abc: Sequence[float] = (0.0, 10.0, 0.0),
    g_factors_abc: Sequence[float] = (2.3, 2.3, 1.3),
    bohr_magneton: float = 5.988e-2,
    crystallographic_transform: np.ndarray | None = None,
    spin_basis_transform: np.ndarray | None = None,
) -> QubitOperator:
    """Build the grouped Kitaev-Heisenberg-Gamma model without dense pair matrices.

    Couplings follow create_kitaev_hamiltonian, including its physical-spin
    normalization. Scalars/arrays specify shell 1; mappings specify geometric
    shells. None selects the notebook defaults for kx, ky, kz, j, and the
    crystallographic transform. Empty mappings or zero explicitly disable terms.
    Gamma flavor overrides and both basis transforms are forwarded unchanged.
    """
    if crystallographic_transform is None:
        crystallographic_transform = np.array(
            [
                [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
                [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
                [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
            ]
        )
    # Shell mappings select the packed, geometry-grouped construction path,
    # including when every requested exchange is a nearest-neighbor scalar.
    couplings = [
        DEFAULT_KITAEV_COUPLINGS if kx is None else kx,
        DEFAULT_KITAEV_COUPLINGS if ky is None else ky,
        DEFAULT_KITAEV_COUPLINGS if kz is None else kz,
        DEFAULT_HEISENBERG_COUPLINGS if j is None else j,
    ]
    shells = [value if isinstance(value, Mapping) else {1: value} for value in couplings]
    return create_kitaev_hamiltonian(
        graph,
        kx=shells[0],
        ky=shells[1],
        kz=shells[2],
        j=shells[3],
        gamma=gamma,
        gamma_prime=gamma_prime,
        gamma_x=gamma_x,
        gamma_y=gamma_y,
        gamma_z=gamma_z,
        gamma_prime_x=gamma_prime_x,
        gamma_prime_y=gamma_prime_y,
        gamma_prime_z=gamma_prime_z,
        magnetic_field_abc=np.asarray(magnetic_field_abc),
        g_factors_abc=np.asarray(g_factors_abc),
        bohr_magneton=bohr_magneton,
        crystallographic_transform=crystallographic_transform,
        spin_basis_transform=spin_basis_transform,
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
    """Parse lattice, Hamiltonian, evolution, and output inputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nx", type=int, default=200, help="Plaquettes along x.")
    parser.add_argument("--ny", type=int, help="Plaquettes along y; defaults to nx.")
    for axis in ("x", "y", "z"):
        parser.add_argument(
            f"--k{axis}",
            type=_parse_couplings,
            default=DEFAULT_KITAEV_COUPLINGS.copy(),
            help=f"Physical K{axis} exchange: JSON scalar, matrix, or shell map.",
        )
    parser.add_argument(
        "--j",
        type=_parse_couplings,
        default=DEFAULT_HEISENBERG_COUPLINGS.copy(),
        help="Physical isotropic exchange: JSON scalar, matrix, or shell map.",
    )
    parser.add_argument("--gamma", type=_parse_parameter, default=9.4, help="Shared nearest-neighbor Gamma.")
    parser.add_argument(
        "--gamma-prime", type=_parse_parameter, default=-2.3, help="Shared nearest-neighbor Gamma-prime."
    )
    for prefix in ("gamma", "gamma-prime"):
        for axis in ("x", "y", "z"):
            parser.add_argument(
                f"--{prefix}-{axis}",
                type=_parse_parameter,
                help=f"Override {prefix} on {axis.upper()} bonds; JSON scalar or matrix.",
            )
    parser.add_argument(
        "--magnetic-field-abc", type=float, nargs=3, default=(0.0, 10.0, 0.0), help="Field in the abc frame (T)."
    )
    parser.add_argument("--g-factors-abc", type=float, nargs=3, default=(2.3, 2.3, 1.3), help="Diagonal abc g factors.")
    parser.add_argument(
        "--bohr-magneton", type=float, default=5.988e-2, help="Field-to-exchange conversion (meV/T by default)."
    )
    parser.add_argument(
        "--crystallographic-transform",
        type=float,
        nargs=9,
        help="Row-major 3x3 rotation; defaults to the notebook abc frame.",
    )
    parser.add_argument(
        "--spin-basis-transform", type=float, nargs=9, help="Row-major 3x3 output-spin rotation; defaults to identity."
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
    parser.add_argument(
        "--output-csv", type=Path, help="Save the result table to this CSV file; create parent directories."
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
        kx=args.kx,
        ky=args.ky,
        kz=args.kz,
        j=args.j,
        gamma=args.gamma,
        gamma_prime=args.gamma_prime,
        gamma_x=args.gamma_x,
        gamma_y=args.gamma_y,
        gamma_z=args.gamma_z,
        gamma_prime_x=args.gamma_prime_x,
        gamma_prime_y=args.gamma_prime_y,
        gamma_prime_z=args.gamma_prime_z,
        magnetic_field_abc=args.magnetic_field_abc,
        g_factors_abc=args.g_factors_abc,
        bohr_magneton=args.bohr_magneton,
        crystallographic_transform=(
            None
            if args.crystallographic_transform is None
            else np.asarray(args.crystallographic_transform).reshape(3, 3)
        ),
        spin_basis_transform=(
            None if args.spin_basis_transform is None else np.asarray(args.spin_basis_transform).reshape(3, 3)
        ),
    )
    print(f"Open {args.nx} x {args.ny} complete-plaquette Kitaev benchmark", flush=True)
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
    estimates = estimate_physical(circuit, f"Kitaev {args.nx} x {args.ny}")
    print(f"QRE statistics: {estimates.stats}")
    frame = estimates.as_frame()
    if not estimates:
        print("No feasible estimates for the fixed notebook QRE settings.")
        # QRE drops all columns for an empty table; retain readable CSV headers.
        frame = frame.reindex(
            columns=[
                "name",
                "qubits",
                "runtime",
                "error",
                "physical_compute_qubits",
                "physical_factory_qubits",
                "physical_memory_qubits",
                "factories",
            ]
        )
    else:
        print(frame.to_string(index=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
        print(f"Saved result table: {args.output_csv}")


if __name__ == "__main__":
    main()
