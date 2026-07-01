"""Command-line interface for QDK Chemistry MCP tools.

All commands live at the top level — no nested groups.  Algorithm
commands use a ``run-`` prefix.

Usage::

    qdk_chem_cli run-scf --project-name myproj ...
    qdk_chem_cli summary --project-name myproj --filename wf.wavefunction.json
    qdk_chem_cli list-projects
    qdk_chem_cli workflow --config pipeline.json --project-name myproj

Compound algorithms (run-mcscf, run-qpe, run-energy) accept a ``--config``
JSON file for nested algorithm settings, with optional ``--set key=value``
overrides.  Generate a default config template with::

    qdk_chem_cli defaults --type mcscf

Use ``--dry-run`` before any command to preview parameters without executing.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import argcomplete

from qdk_chemistry import algorithms, constants
from qdk_chemistry import data as qdk_data
from qdk_chemistry.constants import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from qdk_chemistry.utils import compute_valence_space_parameters
from qdk_chemistry.utils.phase import energy_from_phase, resolve_energy_aliases

from .config import config
from .io import load_data_object, save_data_object

# ---------------------------------------------------------------------------
# Imports from the MCP server (shared backend)
# ---------------------------------------------------------------------------
from .tools import (
    create_model_hamiltonian,
    create_spin_model_hamiltonian,
    create_structure,
    describe_backend,
    get_active_space_indices,
    get_algorithm_default_settings,
    get_algorithm_default_type,
    get_ansatz,
    get_orbitals_from_input,
    get_top_configurations,
    list_cache_backends,
    list_remote_backends,
    run_active_space_selector,
    run_circuit_executor,
    run_controlled_evolution_circuit_mapper,
    run_dynamical_correlation_calculator,
    run_energy_estimator,
    run_hamiltonian_constructor,
    run_multi_configuration_calculation,
    run_multi_configuration_scf,
    run_orbital_localization,
    run_phase_estimation,
    run_projected_multi_configuration_calculation,
    run_qubit_hamiltonian_solver,
    run_qubit_mapper,
    run_resource_estimation,
    run_scf,
    run_stability_checker,
    run_state_preparation,
    run_time_evolution_builder,
)

# Registry of all subparser commands, populated by create_parser().
# Maps command name → argparse sub-parser instance.
_SUBPARSER_REGISTRY: dict[str, Any] = {}

# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════


def parse_json_arg(value: str) -> Any:
    """Parse a JSON string argument."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}") from e


def format_output(result: Any) -> str:  # noqa: PLR0911
    """Format the output result as JSON for display."""
    # Handle structured envelope from @_structured decorator
    if isinstance(result, dict) and "status" in result:
        if result["status"] == "ok":
            return json.dumps({"success": True, "result": result.get("result")}, indent=2)
        if result["status"] == "exists":
            return json.dumps({"success": True, "exists": True, "message": result.get("message")}, indent=2)
        return json.dumps({"success": False, "error": result.get("message", "Unknown error")}, indent=2)
    if isinstance(result, tuple):
        return json.dumps({"success": True, "result": list(result)}, indent=2)
    if isinstance(result, list | dict):
        return json.dumps({"success": True, "result": result}, indent=2)
    if isinstance(result, str):
        if any(result.startswith(prefix) for prefix in ["Invalid", "Failed", "Error", "EXISTS:"]):
            return json.dumps({"success": False, "error": result}, indent=2)
        return json.dumps({"success": True, "result": result}, indent=2)
    return json.dumps({"success": True, "result": str(result)}, indent=2)


def _print_result(result: Any) -> None:
    """Print a formatted JSON result to stdout."""
    print(format_output(result))


def _print_success(**kwargs: Any) -> None:
    """Print a JSON success response with arbitrary fields."""
    print(json.dumps({"success": True, **kwargs}, indent=2))


def _print_error(message: str, exit_code: int = 1) -> None:
    """Print a JSON error response and optionally exit."""
    print(json.dumps({"success": False, "error": message}, indent=2))
    if exit_code:
        sys.exit(exit_code)


def _parse_set_overrides(set_args: list[str] | None) -> dict:
    """Parse ``--set key.path=value`` arguments into a nested dict.

    Supports dotted key paths and auto-converts JSON-parseable values::

        --set mc_calculator.settings.calculate_one_rdm=true
        --set mc_calculator.settings.calculate_two_rdm=true
        --set mc_calculator.settings.calculate_mutual_information=true
        --set settings.max_iterations=50
    """
    if not set_args:
        return {}
    overrides: dict = {}
    for item in set_args:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid --set format: '{item}'. Expected key.path=value")
        key, value = item.split("=", 1)
        # Try to parse value as JSON (handles true, false, null, numbers)
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value  # Keep as string

        # Build nested dict from dotted path
        parts = key.split(".")
        target = overrides
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = parsed_value
    return overrides


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* (mutates *base*)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_config_with_overrides(args) -> dict:
    """Load a ``--config`` file and apply ``--set`` overrides.

    Returns a dict with the merged configuration.  If neither ``--config``
    nor ``--set`` is provided, returns an empty dict.
    """
    cfg: dict = {}
    config_file = getattr(args, "config", None)
    if config_file:
        with open(config_file) as f:
            cfg = json.load(f)
    set_args = getattr(args, "set", None)
    if set_args:
        overrides = _parse_set_overrides(set_args)
        _deep_merge(cfg, overrides)
    return cfg


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` and ``--set`` arguments to a parser."""
    parser.add_argument(
        "--config",
        metavar="FILE",
        help=("JSON config file for nested algorithm settings. Generate a template with: defaults --type <command>"),
    )
    parser.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Override a config value using dotted key path. "
            "Can be specified multiple times. "
            "Example: --set mc_calculator.settings.calculate_one_rdm=true"
        ),
    )


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--cache``, ``--remote``, and ``--remote-config`` arguments."""
    parser.add_argument(
        "--cache",
        metavar="NAME_OR_PATH",
        help=(
            "Enable result caching. Pass a registered cache backend name "
            "(e.g. 'folder', 'cosmosdb') or a filesystem path for a folder cache. "
            "Use 'list-cache-backends' to see available backends."
        ),
    )
    parser.add_argument(
        "--remote",
        metavar="BACKEND",
        help=(
            "Execute on a remote backend instead of locally "
            "(e.g. 'ssh', 'local'). Requires --cache. "
            "Use 'list-remote-backends' to see available backends."
        ),
    )
    parser.add_argument(
        "--remote-config",
        type=parse_json_arg,
        metavar="JSON",
        help=(
            "Backend-specific configuration as JSON "
            '(e.g. \'{"pool": "gpu-pool", "timeout": 7200}\'). '
            "Use 'describe-backend' to see parameters for a specific backend."
        ),
    )


def _get_execution_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Extract cache/remote/remote_config kwargs from parsed args."""
    kwargs: dict[str, Any] = {}
    cache = getattr(args, "cache", None)
    remote = getattr(args, "remote", None)
    remote_config = getattr(args, "remote_config", None)
    if cache is not None:
        kwargs["cache"] = cache
    if remote is not None:
        kwargs["remote"] = remote
    if remote_config is not None:
        kwargs["remote_config"] = remote_config
    return kwargs


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM group — command handlers
# ═══════════════════════════════════════════════════════════════════════════


def cmd_scf(args):
    """Run SCF (Hartree-Fock / DFT) calculation."""
    settings = getattr(args, "settings", None) or {}
    result = run_scf(
        project_name=args.project_name,
        structure_filename=args.structure_filename,
        out_wavefunction_filename=args.out_wavefunction_filename,
        charge=args.charge,
        spin_multiplicity=args.spin_multiplicity,
        basis_set=args.basis_set,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_active_space(args):
    """Run active space selector."""
    settings = getattr(args, "settings", None) or {}
    result = run_active_space_selector(
        project_name=args.project_name,
        wavefunction_filename=args.wavefunction_filename,
        out_wavefunction_filename=args.out_wavefunction_filename,
        charge=args.charge,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_localize(args):
    """Run orbital localization."""
    settings = getattr(args, "settings", None) or {}
    result = run_orbital_localization(
        project_name=args.project_name,
        wavefunction_filename=args.wavefunction_filename,
        out_wavefunction_filename=args.out_wavefunction_filename,
        loc_indices_alpha=args.loc_indices_alpha,
        loc_indices_beta=args.loc_indices_beta,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_correlate(args):
    """Run dynamical correlation calculator (MP2, CCSD, …)."""
    settings = getattr(args, "settings", None) or {}
    result = run_dynamical_correlation_calculator(
        project_name=args.project_name,
        ansatz_filename=args.ansatz_filename,
        out_wavefunction_filename=args.out_wavefunction_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_stability(args):
    """Run wavefunction stability checker."""
    settings = getattr(args, "settings", None) or {}
    result = run_stability_checker(
        project_name=args.project_name,
        wavefunction_filename=args.wavefunction_filename,
        out_stability_result_filename=args.out_stability_result_filename,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_hamiltonian(args):
    """Construct fermionic Hamiltonian from orbitals."""
    result = run_hamiltonian_constructor(
        project_name=args.project_name,
        orbitals_filename=args.orbitals_filename,
        out_hamiltonian_filename=args.out_hamiltonian_filename,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_model_hamiltonian(args):
    """Create a model Hamiltonian on a lattice."""
    result = create_model_hamiltonian(
        project_name=args.project_name,
        model=args.model,
        out_hamiltonian_filename=args.out_hamiltonian_filename,
        lattice_type=args.lattice_type,
        lattice_params=parse_json_arg(args.lattice_params),
        epsilon=parse_json_arg(args.epsilon) if args.epsilon else 0.0,
        t=parse_json_arg(args.t) if args.t else 1.0,
        U=parse_json_arg(args.U) if args.U else 0.0,
        V=parse_json_arg(args.V) if args.V else None,
        z=parse_json_arg(args.z) if args.z else 1.0,
        potential=args.potential,
        potential_params=parse_json_arg(args.potential_params) if args.potential_params else None,
        overwrite=getattr(args, "overwrite", False),
    )
    _print_result(result)


def cmd_spin_model(args):
    """Create a spin model Hamiltonian on a lattice."""
    result = create_spin_model_hamiltonian(
        project_name=args.project_name,
        model=args.model,
        out_qubit_hamiltonian_filename=args.out_qubit_hamiltonian_filename,
        lattice_type=args.lattice_type,
        lattice_params=parse_json_arg(args.lattice_params),
        jx=parse_json_arg(args.jx) if args.jx else 0.0,
        jy=parse_json_arg(args.jy) if args.jy else 0.0,
        jz=parse_json_arg(args.jz) if args.jz else 0.0,
        hx=parse_json_arg(args.hx) if args.hx else 0.0,
        hy=parse_json_arg(args.hy) if args.hy else 0.0,
        hz=parse_json_arg(args.hz) if args.hz else 0.0,
        j=parse_json_arg(args.j) if args.j else None,
        h=parse_json_arg(args.h) if args.h else None,
        overwrite=getattr(args, "overwrite", False),
    )
    _print_result(result)


def cmd_casci(args):
    """Run multi-configuration (CASCI / selected-CI) calculation."""
    settings = getattr(args, "settings", None) or {}
    result = run_multi_configuration_calculation(
        project_name=args.project_name,
        hamiltonian_filename=args.hamiltonian_filename,
        out_wavefunction_filename=args.out_wavefunction_filename,
        n_active_alpha_electrons=args.n_active_alpha_electrons,
        n_active_beta_electrons=args.n_active_beta_electrons,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_mcscf(args):
    """Run multi-configuration SCF (MCSCF / CASSCF).

    This is a compound algorithm with nested sub-algorithms for the
    Hamiltonian constructor and multi-configuration calculator.
    Use ``--config`` for full control over sub-algorithm settings,
    or ``--set`` for quick overrides.
    """
    cfg = _load_config_with_overrides(args)
    ham_cfg = cfg.get("ham_constructor", {})
    mc_cfg = cfg.get("mc_calculator", {})
    mcscf_cfg = cfg.get("mcscf", {})

    result = run_multi_configuration_scf(
        project_name=args.project_name,
        orbitals_filename=args.orbitals_filename,
        out_wavefunction_filename=args.out_wavefunction_filename,
        n_active_alpha_electrons=args.n_active_alpha_electrons,
        n_active_beta_electrons=args.n_active_beta_electrons,
        ham_constructor_algorithm_name=ham_cfg.get("algorithm_name"),
        ham_constructor_settings=ham_cfg.get("settings", {}),
        mc_calculator_algorithm_name=mc_cfg.get("algorithm_name"),
        mc_calculator_settings=mc_cfg.get("settings", {}),
        settings=mcscf_cfg.get("settings", {}),
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_sparse_ci(args):
    """Run projected multi-configuration (sparse CI) calculation."""
    settings = getattr(args, "settings", None) or {}
    result = run_projected_multi_configuration_calculation(
        project_name=args.project_name,
        hamiltonian_filename=args.hamiltonian_filename,
        configurations_json=args.configurations_json,
        out_wavefunction_filename=args.out_wavefunction_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_qubit_map(args):
    """Map fermionic Hamiltonian to qubit Hamiltonian."""
    settings = getattr(args, "settings", None) or {}
    result = run_qubit_mapper(
        project_name=args.project_name,
        hamiltonian_filename=args.hamiltonian_filename,
        out_qubit_hamiltonian_filename=args.out_qubit_hamiltonian_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_state_prep(args):
    """Generate state preparation quantum circuit."""
    settings = getattr(args, "settings", None) or {}
    result = run_state_preparation(
        project_name=args.project_name,
        wavefunction_filename=args.wavefunction_filename,
        out_circuit_filename=args.out_circuit_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_qubit_solve(args):
    """Exact diagonalization of qubit Hamiltonian."""
    settings = getattr(args, "settings", None) or {}
    result = run_qubit_hamiltonian_solver(
        project_name=args.project_name,
        qubit_hamiltonian_filename=args.qubit_hamiltonian_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_resource_estimation(args):
    """Estimate quantum resources required for a circuit."""
    settings = getattr(args, "settings", None) or {}
    result = run_resource_estimation(
        project_name=args.project_name,
        circuit_filename=args.circuit_filename,
        out_resource_estimator_data_filename=args.out_resource_estimator_data_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_energy(args):
    """Estimate energy from circuit measurements.

    This is a compound algorithm with optional noise model.
    Use ``--config`` for full control, or ``--set`` for quick overrides.
    """
    cfg = _load_config_with_overrides(args)
    energy_cfg = cfg.get("energy_estimator", {})

    result = run_energy_estimator(
        project_name=args.project_name,
        circuit_filename=args.circuit_filename,
        qubit_hamiltonian_filename=args.qubit_hamiltonian_filename,
        out_energy_result_filename=args.out_energy_result_filename,
        out_measurement_data_filename=args.out_measurement_data_filename,
        total_shots=args.total_shots,
        noise_model=energy_cfg.get("noise_model"),
        algorithm_name=energy_cfg.get("algorithm_name"),
        settings=energy_cfg.get("settings", {}),
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_qpe_build_evolution(args):
    """Build time evolution unitary U = exp(-iHt)."""
    settings = getattr(args, "settings", None) or {}
    result = run_time_evolution_builder(
        project_name=args.project_name,
        qubit_hamiltonian_filename=args.qubit_hamiltonian_filename,
        evolution_time=args.evolution_time,
        out_time_evolution_unitary_filename=args.out_time_evolution_unitary_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_qpe_map_circuit(args):
    """Map time evolution unitary to controlled circuit."""
    settings = getattr(args, "settings", None) or {}
    result = run_controlled_evolution_circuit_mapper(
        project_name=args.project_name,
        time_evolution_unitary_filename=args.time_evolution_unitary_filename,
        out_circuit_filename=args.out_circuit_filename,
        control_indices=args.control_indices if args.control_indices else [0],
        power=args.power,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_qpe_execute(args):
    """Execute a quantum circuit with shots."""
    settings = getattr(args, "settings", None) or {}
    result = run_circuit_executor(
        project_name=args.project_name,
        circuit_filename=args.circuit_filename,
        shots=args.shots,
        out_executor_data_filename=args.out_executor_data_filename,
        algorithm_name=args.algorithm_name,
        settings=settings,
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_qpe(args):
    """Run quantum phase estimation (full pipeline).

    Sub-algorithms are configured inline in the settings dict::

        {"qpe": {"settings": {
            "num_bits": 10, "evolution_time": 1.0,
            "evolution_builder": {"algorithm_name": "trotter", "order": 2},
            "circuit_mapper": {"algorithm_name": "pauli_sequence"},
            "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"}
        }}}
    """
    cfg = _load_config_with_overrides(args)
    qpe_cfg = cfg.get("qpe", {})

    result = run_phase_estimation(
        project_name=args.project_name,
        state_prep_circuit_filename=args.state_prep_circuit_filename,
        qubit_hamiltonian_filename=args.qubit_hamiltonian_filename,
        out_qpe_result_filename=args.out_qpe_result_filename,
        algorithm_name=qpe_cfg.get("algorithm_name", args.algorithm_name),
        settings=qpe_cfg.get("settings", {}),
        **_get_execution_kwargs(args),
    )
    _print_result(result)


def cmd_defaults(args):
    """Print default algorithm settings or a config template.

    With ``--type``, generates a compound config template (mcscf, qpe, energy).
    With ``--algorithm-type``, shows the default algorithm name.
    With ``--algorithm-type --algorithm-name``, shows settings for that algorithm.
    """
    alg_type = getattr(args, "algorithm_type", None)
    config_type = getattr(args, "type", None)

    if config_type:
        templates = {
            "mcscf": {
                "ham_constructor": {
                    "algorithm_name": None,
                    "settings": algorithms.create("hamiltonian_constructor").settings().to_dict(),
                },
                "mc_calculator": {
                    "algorithm_name": None,
                    "settings": algorithms.create("multi_configuration_calculator").settings().to_dict(),
                },
                "mcscf": {
                    "settings": {},
                },
            },
            "qpe": {
                "qpe": {
                    "algorithm_name": None,
                    "settings": {
                        "num_bits": 10,
                        "evolution_time": 1.0,
                        "evolution_builder": {"algorithm_name": "trotter"},
                        "circuit_mapper": {"algorithm_name": "pauli_sequence"},
                        "circuit_executor": {"algorithm_name": "qdk_sparse_state_simulator"},
                    },
                },
            },
            "energy": {
                "energy_estimator": {
                    "algorithm_name": None,
                    "settings": {},
                    "noise_model": None,
                },
            },
        }
        if config_type not in templates:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Unknown config type '{config_type}'. Available: {', '.join(templates)}",
                    },
                    indent=2,
                )
            )
            sys.exit(1)
        _print_success(result=templates[config_type])
        return

    if alg_type:
        alg_name = getattr(args, "algorithm_name", None)
        if alg_name:
            result = get_algorithm_default_settings(
                algorithm_type=alg_type,
                algorithm_name=alg_name,
            )
        else:
            result = get_algorithm_default_type(algorithm_type=alg_type)
        _print_result(result)
        return

    # Neither --type nor --algorithm-type: show help
    print(
        json.dumps(
            {
                "success": False,
                "error": "Provide --type (mcscf|qpe|energy) for a config template, "
                "or --algorithm-type <type> [--algorithm-name <name>] for algorithm defaults.",
            },
            indent=2,
        )
    )
    sys.exit(1)


def cmd_list_algorithms(args):
    """List all available algorithm types and implementations."""
    alg_type = getattr(args, "algorithm_type", None)
    if alg_type:
        result = algorithms.available(alg_type)
        _print_result(result)
    else:
        result = algorithms.available()
        _print_result(result)


def cmd_list_cache_backends(_args):
    """List available cache backend names."""
    result = list_cache_backends()
    _print_result(result)


def cmd_list_remote_backends(_args):
    """List available remote execution backend names."""
    result = list_remote_backends()
    _print_result(result)


def cmd_describe_backend(args):
    """Describe configuration parameters for a cache or remote backend."""
    result = describe_backend(
        backend_type=args.backend_type,
        name=args.name,
    )
    _print_result(result)


# ═══════════════════════════════════════════════════════════════════════════
# DATA group — command handlers
# ═══════════════════════════════════════════════════════════════════════════


def _get_data_classes():
    """Return the list of known data classes for auto-detection."""
    return [
        qdk_data.Structure,
        qdk_data.Wavefunction,
        qdk_data.Hamiltonian,
        qdk_data.Orbitals,
        qdk_data.Ansatz,
        qdk_data.ConfigurationSet,
        qdk_data.QubitHamiltonian,
        qdk_data.Circuit,
        qdk_data.StabilityResult,
        qdk_data.QpeResult,
        qdk_data.EnergyExpectationResult,
        qdk_data.MeasurementData,
    ]


def cmd_data_summary(args):
    """Print a human-readable summary of any data file."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    for cls in _get_data_classes():
        try:
            obj = load_data_object(filename, cls)
            summary = obj.get_summary() if hasattr(obj, "get_summary") else str(obj)
            _print_success(type=cls.__name__, summary=summary)
            return
        except (RuntimeError, ValueError, FileNotFoundError, OSError):
            continue

    _print_error(f"Could not load '{filename}' as any known data type.")


def cmd_data_convert(args):
    """Convert a data file between JSON and HDF5 formats."""
    filename = args.filename.split("/")[-1]
    out_filename = args.out_filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    for cls in _get_data_classes():
        try:
            obj = load_data_object(filename, cls)
            save_data_object(obj, out_filename)
            _print_result(out_filename)
            return
        except (RuntimeError, ValueError, FileNotFoundError, OSError):
            continue

    _print_error(f"Could not load '{filename}' as any known data type.")


def cmd_data_get_orbitals(args):
    """Extract and save orbitals from a Wavefunction/Hamiltonian/Ansatz/ConfigurationSet."""
    result = get_orbitals_from_input(
        project_name=args.project_name,
        input_filename=args.input_filename,
        out_orbitals_filename=args.out_orbitals_filename,
    )
    _print_result(result)


def cmd_data_get_active_space_indices(args):
    """Get active, inactive, and virtual orbital space indices."""
    result = get_active_space_indices(
        project_name=args.project_name,
        input_filename=args.input_filename,
    )
    _print_result(result)


def cmd_data_get_ansatz(args):
    """Build and save an Ansatz from wavefunction + Hamiltonian."""
    result = get_ansatz(
        project_name=args.project_name,
        wavefunction_filename=args.wavefunction_filename,
        hamiltonian_filename=args.hamiltonian_filename,
        out_ansatz_filename=args.out_ansatz_filename,
    )
    _print_result(result)


def cmd_data_get_top_configurations(args):
    """Get top CI determinants ranked by coefficient magnitude."""
    result = get_top_configurations(
        project_name=args.project_name,
        wavefunction_filename=args.wavefunction_filename,
        max_determinants=args.max_determinants,
    )
    _print_result(result)


def cmd_data_create_structure(args):
    """Create a molecular structure in a project."""
    result = create_structure(
        project_name=args.project_name,
        coordinates_json=args.coordinates_json,
        symbols=args.symbols,
        nuclear_charges=args.nuclear_charges,
        masses=args.masses,
        filename_to_save=args.filename_to_save,
    )
    _print_result(result)


def cmd_data_get_energy(args):
    """Get energy value from a Wavefunction, QpeResult, or EnergyExpectationResult."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    # Try Wavefunction — energy is in the summary
    try:
        wf = load_data_object(filename, qdk_data.Wavefunction)
        summary = wf.get_summary()
        _print_success(summary=summary, source="Wavefunction")
        return
    except (RuntimeError, ValueError, FileNotFoundError, OSError):
        pass

    # Try QpeResult
    try:
        qpe = load_data_object(filename, qdk_data.QpeResult)
        summary = qpe.get_summary() if hasattr(qpe, "get_summary") else str(qpe)
        result = {"success": True, "summary": summary, "source": "QpeResult"}
        if hasattr(qpe, "to_dict"):
            result["data"] = qpe.to_dict()
        print(json.dumps(result, indent=2))
        return
    except (RuntimeError, ValueError, FileNotFoundError, OSError):
        pass

    # Try EnergyExpectationResult
    try:
        eer = load_data_object(filename, qdk_data.EnergyExpectationResult)
        summary = eer.get_summary() if hasattr(eer, "get_summary") else str(eer)
        result = {"success": True, "summary": summary, "source": "EnergyExpectationResult"}
        if hasattr(eer, "to_dict"):
            result["data"] = eer.to_dict()
        print(json.dumps(result, indent=2))
        return
    except (RuntimeError, ValueError, FileNotFoundError, OSError):
        pass

    _print_error(f"Could not extract energy from '{filename}'.")


def cmd_data_get_structure_xyz(args):
    """Export structure as XYZ format string."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        structure = load_data_object(filename, qdk_data.Structure)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    xyz_str = structure.to_xyz()

    out_file = getattr(args, "out_file", None)
    if out_file:
        Path(out_file).write_text(xyz_str)
        _print_result(out_file)
    else:
        _print_success(xyz=xyz_str)


def cmd_data_get_circuit_qasm(args):
    """Export circuit as OpenQASM string."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        circuit = load_data_object(filename, qdk_data.Circuit)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    qasm_str = circuit.get_qasm()
    _print_success(qasm=qasm_str)


def cmd_data_get_circuit_stats(args):
    """Analyze circuit resource profile: gates, depth, qubits."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        circuit = load_data_object(filename, qdk_data.Circuit)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    try:
        from qdk_chemistry.plugins.qiskit._interop.circuit import CircuitInfo  # noqa: PLC0415

        qiskit_circuit = circuit.get_qiskit_circuit()
        info = CircuitInfo(circuit=qiskit_circuit)
        stats = info.summary()
        stats["gate_counts"] = dict(info.gate_counts)
        _print_success(**stats)
    except Exception as e:  # noqa: BLE001
        _print_error(f"Failed to analyze circuit: {e!s}")


def cmd_data_get_qubit_hamiltonian_info(args):
    """Inspect qubit Hamiltonian: qubits, terms, norm, hermiticity."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        qh = load_data_object(filename, qdk_data.QubitHamiltonian)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    info: dict[str, object] = {"success": True}
    # QubitHamiltonian is a Python class with direct attributes
    if hasattr(qh, "num_qubits"):
        info["num_qubits"] = qh.num_qubits
    if hasattr(qh, "pauli_strings"):
        info["num_terms"] = len(qh.pauli_strings)
    if hasattr(qh, "schatten_norm"):
        info["schatten_norm"] = qh.schatten_norm
    if hasattr(qh, "is_hermitian"):
        info["is_hermitian"] = bool(qh.is_hermitian())
    info["summary"] = qh.get_summary() if hasattr(qh, "get_summary") else str(qh)
    print(json.dumps(info, indent=2))


def cmd_data_get_stability_result(args):
    """Inspect wavefunction stability result."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        sr = load_data_object(filename, qdk_data.StabilityResult)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    info: dict[str, object] = {"success": True}
    for method in ["is_stable", "is_internal_stable", "is_external_stable"]:
        if hasattr(sr, method):
            info[method] = bool(getattr(sr, method)())
    info["summary"] = sr.get_summary() if hasattr(sr, "get_summary") else str(sr)
    print(json.dumps(info, indent=2))


def cmd_data_get_qpe_result(args):
    """Inspect QPE result: energies, phase, bits."""
    filename = args.filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        qpe = load_data_object(filename, qdk_data.QpeResult)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    result: dict[str, object] = {"success": True}
    if hasattr(qpe, "to_dict"):
        result.update(qpe.to_dict())
    result["summary"] = qpe.get_summary() if hasattr(qpe, "get_summary") else str(qpe)
    print(json.dumps(result, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# UTILS group — command handlers
# ═══════════════════════════════════════════════════════════════════════════


def cmd_utils_list_projects(_args):
    """List all projects in the scratch directory."""
    projects_dir = config.projects_dir
    if not projects_dir.exists():
        _print_result([])
        return
    projects = sorted(d.name for d in projects_dir.iterdir() if d.is_dir())
    _print_result(projects)


def cmd_utils_create_project(args):
    """Create a new project directory."""
    project_dir = config.projects_dir / args.project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    _print_result(str(project_dir))


def cmd_utils_list_files(args):
    """List data files in a project, with inferred types."""
    project_dir = config.projects_dir / args.project_name
    if not project_dir.exists():
        _print_error(f"Project '{args.project_name}' not found.")

    files = []
    for f in sorted(project_dir.iterdir()):
        if f.is_file():
            entry = {"filename": f.name, "size_bytes": f.stat().st_size}
            # Infer type from filename marker
            for ext in [".json", ".hdf5", ".h5"]:
                if f.name.endswith(ext):
                    base = f.name[: -len(ext)]
                    parts = base.rsplit(".", 1)
                    if len(parts) == 2:
                        entry["data_type"] = parts[1]
                    break
            files.append(entry)
    _print_result(files)


def cmd_utils_convert_coordinates(args):
    """Convert coordinates between Bohr and Angstrom."""
    coords = args.coordinates
    if args.to_angstrom:
        converted = [[c * BOHR_TO_ANGSTROM for c in atom] for atom in coords]
        unit = "angstrom"
    else:
        converted = [[c * ANGSTROM_TO_BOHR for c in atom] for atom in coords]
        unit = "bohr"
    _print_success(coordinates=converted, unit=unit)


def cmd_utils_convert_energy(args):
    """Convert energy between Hartree, eV, and kcal/mol."""
    value = args.value
    from_unit = args.from_unit.lower()
    to_unit = args.to_unit.lower()

    # Normalize to Hartree first
    to_hartree = {
        "hartree": 1.0,
        "ev": constants.EV_TO_HARTREE,
        "kcal/mol": constants.KCAL_PER_MOL_TO_HARTREE,
        "kj/mol": constants.KJ_PER_MOL_TO_HARTREE,
    }
    from_hartree = {
        "hartree": 1.0,
        "ev": constants.HARTREE_TO_EV,
        "kcal/mol": constants.HARTREE_TO_KCAL_PER_MOL,
        "kj/mol": constants.HARTREE_TO_KJ_PER_MOL,
    }

    if from_unit not in to_hartree:
        print(
            json.dumps(
                {"success": False, "error": f"Unknown unit '{from_unit}'. Use: hartree, ev, kcal/mol, kj/mol"}, indent=2
            )
        )
        sys.exit(1)
    if to_unit not in from_hartree:
        print(
            json.dumps(
                {"success": False, "error": f"Unknown unit '{to_unit}'. Use: hartree, ev, kcal/mol, kj/mol"}, indent=2
            )
        )
        sys.exit(1)

    hartree_value = value * to_hartree[from_unit]
    converted = hartree_value * from_hartree[to_unit]
    print(
        json.dumps(
            {
                "success": True,
                "input": {"value": value, "unit": from_unit},
                "output": {"value": converted, "unit": to_unit},
            },
            indent=2,
        )
    )


def cmd_utils_compute_valence_params(args):
    """Compute valence space parameters (active electrons & orbitals)."""
    filename = args.wavefunction_filename.split("/")[-1]
    project_dir = config.projects_dir / args.project_name
    os.chdir(project_dir)

    try:
        wf = load_data_object(filename, qdk_data.Wavefunction)
    except (RuntimeError, ValueError) as e:
        _print_error(str(e))

    n_electrons, n_orbitals = compute_valence_space_parameters(wf, args.charge)
    print(
        json.dumps(
            {
                "success": True,
                "n_active_electrons": n_electrons,
                "n_active_orbitals": n_orbitals,
            },
            indent=2,
        )
    )


def cmd_utils_resolve_phase_energy(args):
    """Resolve QPE phase to energy with alias handling."""
    raw_energy = energy_from_phase(args.phase_fraction, evolution_time=args.evolution_time)
    resolved = resolve_energy_aliases(
        raw_energy,
        evolution_time=args.evolution_time,
        reference_energy=args.reference_energy,
    )
    print(
        json.dumps(
            {
                "success": True,
                "phase_fraction": args.phase_fraction,
                "raw_energy": raw_energy,
                "resolved_energy": resolved,
            },
            indent=2,
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parser construction
# ═══════════════════════════════════════════════════════════════════════════


def _add_simple_algorithm_args(parser, algorithm_name_help=None):
    """Add the standard --algorithm-name and --settings args."""
    parser.add_argument("--algorithm-name", help=algorithm_name_help or "Algorithm implementation name")
    parser.add_argument("--settings", type=parse_json_arg, help="Algorithm settings (JSON dict)")
    _add_execution_args(parser)


def _create_algorithm_parsers(subparsers):
    """Register all algorithm subcommands under the ``run`` group."""
    # scf
    p = subparsers.add_parser(
        "scf",
        help="Run SCF (Hartree-Fock / DFT) calculation",
        description="Run a self-consistent field calculation to produce an initial wavefunction.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--structure-filename", required=True, help="Input structure filename (e.g. mol.structure.json)")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    p.add_argument("--charge", type=int, required=True, help="System charge")
    p.add_argument("--spin-multiplicity", type=int, required=True, help="Spin multiplicity (1=singlet, 2=doublet, …)")
    p.add_argument("--basis-set", required=True, help="Basis set name (e.g. sto-3g, cc-pvdz)")
    _add_simple_algorithm_args(p, "SCF solver name (default: pyscf)")
    p.set_defaults(func=cmd_scf)

    # active-space
    p = subparsers.add_parser(
        "active-space",
        help="Select active orbital space",
        description="Select which orbitals are active for multi-reference calculations. "
        "Use qdk_valence for an initial selection based on atomic valence. "
        "Use qdk_autocas_eos to automatically determine the active space from orbital "
        "entanglement entropies (requires a prior SCI calculation computing RDMs "
        "and mutual information).",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Input wavefunction filename")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    p.add_argument("--charge", type=int, help="System charge (required for qdk_valence)")
    _add_simple_algorithm_args(
        p,
        "Selector algorithm: qdk_valence (initial, needs charge), "
        "qdk_autocas / qdk_autocas_eos (automatic from orbital entropies, needs RDMs), "
        "qdk_occupation (occupation-based)",
    )
    p.set_defaults(func=cmd_active_space)

    # localize
    p = subparsers.add_parser(
        "localize",
        help="Localize orbitals",
        description="Apply orbital localization (Pipek-Mezey, MP2 natural orbitals, …).",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Input wavefunction filename")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    p.add_argument("--loc-indices-alpha", type=parse_json_arg, required=True, help="Alpha orbital indices (JSON list)")
    p.add_argument("--loc-indices-beta", type=parse_json_arg, help="Beta orbital indices (JSON list)")
    _add_simple_algorithm_args(p, "Localizer: qdk_pipek_mezey, qdk_mp2_natural_orbitals, qdk_vvhv")
    p.set_defaults(func=cmd_localize)

    # correlate
    p = subparsers.add_parser(
        "correlate",
        help="Add dynamical correlation (MP2, CCSD, …)",
        description="Run dynamical correlation on top of an ansatz.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--ansatz-filename", required=True, help="Ansatz filename")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    _add_simple_algorithm_args(p)
    p.set_defaults(func=cmd_correlate)

    # stability
    p = subparsers.add_parser(
        "stability",
        help="Check wavefunction stability",
        description="Analyze internal/external wavefunction stability.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Wavefunction filename")
    p.add_argument("--out-stability-result-filename", required=True, help="Output stability result filename")
    p.add_argument("--settings", type=parse_json_arg, help="Settings (JSON dict)")
    _add_execution_args(p)
    p.set_defaults(func=cmd_stability)

    # hamiltonian
    p = subparsers.add_parser(
        "hamiltonian",
        help="Construct fermionic Hamiltonian from orbitals",
        description="Build the electronic Hamiltonian in the active orbital space.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--orbitals-filename", required=True, help="Orbitals filename")
    p.add_argument("--out-hamiltonian-filename", required=True, help="Output Hamiltonian filename")
    _add_execution_args(p)
    p.set_defaults(func=cmd_hamiltonian)

    # model-hamiltonian
    p = subparsers.add_parser(
        "model-hamiltonian",
        help="Create a model Hamiltonian on a lattice",
        description=(
            "Build a fermionic Hamiltonian for lattice models (Hückel, Hubbard, PPP)\n"
            "without molecular structure input.\n\n"
            "Examples:\n"
            "  model-hamiltonian --project-name h --model hubbard\n"
            '    --lattice-type chain --lattice-params \'{"n": 6, "periodic": true}\'\n'
            "    --epsilon 0 --t 1.0 --U 4.0 --out-hamiltonian-filename hubbard.hamiltonian.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--model", required=True, help="Model type: huckel, hubbard, ppp")
    p.add_argument("--out-hamiltonian-filename", required=True, help="Output Hamiltonian filename")
    p.add_argument(
        "--lattice-type", required=True, help="Lattice: chain, square, triangular, honeycomb, kagome, custom"
    )
    p.add_argument(
        "--lattice-params", required=True, help='Lattice parameters (JSON), e.g. \'{"n": 6, "periodic": true}\''
    )
    p.add_argument("--epsilon", help="On-site energy (float or JSON list)")
    p.add_argument("--t", help="Hopping integral (float or JSON 2D list)")
    p.add_argument("--U", help="On-site Coulomb repulsion (float or JSON list)")
    p.add_argument("--V", help="Intersite Coulomb matrix (float or JSON 2D list, PPP)")
    p.add_argument("--z", help="Effective core charges (float or JSON list, PPP)")
    p.add_argument("--potential", help="Auto-compute V: ohno or mataga_nishimoto (PPP)")
    p.add_argument("--potential-params", help='Potential params (JSON), e.g. \'{"R": 2.5, "epsilon_r": 1.0}\'')
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    p.set_defaults(func=cmd_model_hamiltonian)

    # spin-model
    p = subparsers.add_parser(
        "spin-model",
        help="Create a spin model Hamiltonian on a lattice",
        description=(
            "Build a qubit Hamiltonian for spin models (Heisenberg, Ising)\n"
            "directly — no qubit mapping needed.\n\n"
            "Examples:\n"
            "  spin-model --project-name ising --model ising\n"
            '    --lattice-type square --lattice-params \'{"nx": 3, "ny": 3}\'\n'
            "    --j 1.0 --h 0.5 --out-qubit-hamiltonian-filename ising.qubit_hamiltonian.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--model", required=True, help="Model type: heisenberg, ising")
    p.add_argument("--out-qubit-hamiltonian-filename", required=True, help="Output QubitHamiltonian filename")
    p.add_argument(
        "--lattice-type", required=True, help="Lattice: chain, square, triangular, honeycomb, kagome, custom"
    )
    p.add_argument("--lattice-params", required=True, help="Lattice parameters (JSON)")
    p.add_argument("--jx", help="XX coupling (float or JSON 2D list, Heisenberg)")
    p.add_argument("--jy", help="YY coupling (float or JSON 2D list, Heisenberg)")
    p.add_argument("--jz", help="ZZ coupling (float or JSON 2D list, Heisenberg)")
    p.add_argument("--hx", help="External field X (float or JSON list, Heisenberg)")
    p.add_argument("--hy", help="External field Y (float or JSON list, Heisenberg)")
    p.add_argument("--hz", help="External field Z (float or JSON list, Heisenberg)")
    p.add_argument("--j", help="ZZ coupling (float or JSON 2D list, Ising)")
    p.add_argument("--h", help="Transverse field X (float or JSON list, Ising)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    p.set_defaults(func=cmd_spin_model)

    # casci
    p = subparsers.add_parser(
        "casci",
        help="Run CASCI / selected-CI calculation",
        description="Multi-configuration calculation (full CI in active space or selected CI).",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--hamiltonian-filename", required=True, help="Hamiltonian filename")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    p.add_argument("--n-active-alpha-electrons", type=int, required=True, help="Number of active alpha electrons")
    p.add_argument("--n-active-beta-electrons", type=int, help="Number of active beta electrons")
    _add_simple_algorithm_args(p, "CI solver: macis_cas, macis_asci")
    p.set_defaults(func=cmd_casci)

    # mcscf (compound — uses --config / --set)
    p = subparsers.add_parser(
        "mcscf",
        help="Run MCSCF / CASSCF calculation",
        description=(
            "Multi-configuration self-consistent field calculation.\n\n"
            "This is a compound algorithm with nested sub-algorithms:\n"
            "  - ham_constructor: builds the Hamiltonian each iteration\n"
            "  - mc_calculator: solves the CI problem each iteration\n\n"
            "Use --config FILE to provide nested settings, and --set to override:\n\n"
            "  Generate a template:  defaults --type mcscf\n\n"
            "Config file structure:\n"
            '  {"ham_constructor": {"algorithm_name": "...", "settings": {...}},\n'
            '   "mc_calculator":   {"algorithm_name": "...", "settings": {...}},\n'
            '   "mcscf":           {"settings": {...}}}\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--orbitals-filename", required=True, help="Orbitals filename")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    p.add_argument("--n-active-alpha-electrons", type=int, required=True, help="Number of active alpha electrons")
    p.add_argument("--n-active-beta-electrons", type=int, help="Number of active beta electrons")
    _add_config_args(p)
    _add_execution_args(p)
    p.set_defaults(func=cmd_mcscf)

    # sparse-ci
    p = subparsers.add_parser(
        "sparse-ci",
        help="Run projected (sparse) multi-configuration CI",
        description="Evaluate energy for a given set of determinants.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--hamiltonian-filename", required=True, help="Hamiltonian filename")
    p.add_argument("--configurations-json", required=True, help="JSON array of configuration strings")
    p.add_argument("--out-wavefunction-filename", required=True, help="Output wavefunction filename")
    _add_simple_algorithm_args(p, "Solver: macis_pmc")
    p.set_defaults(func=cmd_sparse_ci)

    # qubit-map
    p = subparsers.add_parser(
        "qubit-map",
        help="Map fermionic Hamiltonian to qubit Hamiltonian",
        description="Apply Jordan-Wigner or other fermion-to-qubit mapping.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--hamiltonian-filename", required=True, help="Fermionic Hamiltonian filename")
    p.add_argument("--out-qubit-hamiltonian-filename", required=True, help="Output qubit Hamiltonian filename")
    _add_simple_algorithm_args(p)
    p.set_defaults(func=cmd_qubit_map)

    # state-prep
    p = subparsers.add_parser(
        "state-prep",
        help="Generate state preparation circuit",
        description="Build a quantum circuit that prepares the wavefunction state.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Wavefunction filename")
    p.add_argument("--out-circuit-filename", required=True, help="Output circuit filename")
    _add_simple_algorithm_args(p, "State prep method (e.g. qiskit_regular_isometry)")
    p.set_defaults(func=cmd_state_prep)

    # qubit-solve
    p = subparsers.add_parser(
        "qubit-solve",
        help="Exact diagonalization of qubit Hamiltonian",
        description="Compute exact eigenvalues of a qubit Hamiltonian (for small systems).",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--qubit-hamiltonian-filename", required=True, help="Qubit Hamiltonian filename")
    _add_simple_algorithm_args(p)
    p.set_defaults(func=cmd_qubit_solve)

    # resource-estimation
    p = subparsers.add_parser(
        "resource-estimation",
        help="Estimate quantum resources for a circuit",
        description="Run quantum resource estimation on a circuit to get logical and physical resource profiles.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--circuit-filename", required=True, help="Circuit filename")
    p.add_argument(
        "--out-resource-estimator-data-filename", required=True, help="Output resource estimator data filename"
    )
    _add_simple_algorithm_args(p, "Resource estimator algorithm (default: qdk_qre_v3)")
    p.set_defaults(func=cmd_resource_estimation)

    # energy (compound — uses --config / --set)
    p = subparsers.add_parser(
        "energy",
        help="Estimate energy from circuit measurements",
        description=(
            "Run energy estimation using quantum circuit measurements.\n\n"
            "This is a compound algorithm with optional noise model.\n"
            "Use --config FILE for full settings, --set for overrides:\n\n"
            "  Generate a template:  defaults --type energy\n\n"
            "Config file structure:\n"
            '  {"energy_estimator": {"algorithm_name": "...", "settings": {...},\n'
            '   "noise_model": {...}}}\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--circuit-filename", required=True, help="Circuit filename")
    p.add_argument("--qubit-hamiltonian-filename", required=True, help="Qubit Hamiltonian filename")
    p.add_argument("--out-energy-result-filename", required=True, help="Output energy result filename")
    p.add_argument("--out-measurement-data-filename", required=True, help="Output measurement data filename")
    p.add_argument("--total-shots", type=int, required=True, help="Total measurement shots")
    _add_config_args(p)
    _add_execution_args(p)
    p.set_defaults(func=cmd_energy)

    # --- QPE step commands ---

    # qpe-build-evolution
    p = subparsers.add_parser(
        "qpe-build-evolution",
        help="Build time evolution unitary U = exp(-iHt)",
        description="Construct the time evolution unitary from a qubit Hamiltonian.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--qubit-hamiltonian-filename", required=True, help="Qubit Hamiltonian filename")
    p.add_argument("--evolution-time", type=float, required=True, help="Evolution time t for U = exp(-iHt)")
    p.add_argument("--out-time-evolution-unitary-filename", required=True, help="Output unitary filename")
    _add_simple_algorithm_args(p)
    p.set_defaults(func=cmd_qpe_build_evolution)

    # qpe-map-circuit
    p = subparsers.add_parser(
        "qpe-map-circuit",
        help="Map time evolution unitary to controlled circuit",
        description="Apply controlled unitary mapping for phase kickback.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--time-evolution-unitary-filename", required=True, help="Time evolution unitary filename")
    p.add_argument("--out-circuit-filename", required=True, help="Output circuit filename")
    p.add_argument("--control-indices", type=parse_json_arg, help="Control qubit indices (JSON list, default: [0])")
    p.add_argument("--power", type=int, default=1, help="Power for controlled unitary (default: 1)")
    _add_simple_algorithm_args(p)
    p.set_defaults(func=cmd_qpe_map_circuit)

    # qpe-execute
    p = subparsers.add_parser(
        "qpe-execute",
        help="Execute a quantum circuit with shots",
        description="Run a circuit on a simulator or backend and collect measurement results.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--circuit-filename", required=True, help="Circuit filename")
    p.add_argument("--shots", type=int, required=True, help="Number of measurement shots")
    p.add_argument("--out-executor-data-filename", required=True, help="Output executor data filename")
    _add_simple_algorithm_args(p)
    p.set_defaults(func=cmd_qpe_execute)

    # qpe (compound — full pipeline, uses --config / --set)
    p = subparsers.add_parser(
        "qpe",
        help="Run quantum phase estimation (full pipeline)",
        description=(
            "Run the full QPE workflow.\n\n"
            "Sub-algorithms (evolution builder, circuit mapper, circuit executor)\n"
            "are configured inline in the settings dict.\n\n"
            "Use --config FILE for QPE-level settings, --set for overrides:\n\n"
            "  Generate a template:  defaults --type qpe\n\n"
            "Config file structure:\n"
            '  {"qpe": {"settings": {"num_bits": 10, "evolution_time": 1.0,\n'
            '    "evolution_builder": {"algorithm_name": "trotter", "order": 2}}}}\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--state-prep-circuit-filename", required=True, help="State preparation circuit filename")
    p.add_argument("--qubit-hamiltonian-filename", required=True, help="Qubit Hamiltonian filename")
    p.add_argument("--out-qpe-result-filename", required=True, help="Output QPE result filename")
    p.add_argument("--algorithm-name", help="QPE algorithm (iterative, qiskit_standard)")
    _add_config_args(p)
    _add_execution_args(p)
    p.set_defaults(func=cmd_qpe)


def _create_config_parsers(subparsers):
    """Register config/discovery subcommands under the ``config`` group."""
    # defaults
    p = subparsers.add_parser(
        "defaults",
        help="Show algorithm defaults or generate config template",
        description=(
            "Query algorithm defaults or generate compound config templates.\n\n"
            "  Config template:      qc config defaults --type mcscf|qpe|energy\n"
            "  Default algorithm:    qc config defaults --algorithm-type scf_solver\n"
            "  Algorithm settings:   qc config defaults --algorithm-type scf_solver --algorithm-name pyscf\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--type", help="Generate config template for compound algorithm (mcscf, qpe, energy)")
    p.add_argument("--algorithm-type", help="Query default for this algorithm type")
    p.add_argument("--algorithm-name", help="Query settings for this specific algorithm")
    p.set_defaults(func=cmd_defaults)

    # algorithms
    p = subparsers.add_parser(
        "algorithms",
        help="List available algorithm types and implementations",
        description="Show all registered algorithm factories.",
    )
    p.add_argument("--algorithm-type", help="Filter to a specific algorithm type")
    p.set_defaults(func=cmd_list_algorithms)

    # cache-backends
    p = subparsers.add_parser(
        "cache-backends",
        help="List available cache backend names",
        description="Show all registered cache backends (folder, cosmosdb, etc.).",
    )
    p.set_defaults(func=cmd_list_cache_backends)

    # remote-backends
    p = subparsers.add_parser(
        "remote-backends",
        help="List available remote execution backend names",
        description="Show all registered remote backends (local, ssh, etc.).",
    )
    p.set_defaults(func=cmd_list_remote_backends)

    # describe-backend
    p = subparsers.add_parser(
        "describe-backend",
        help="Describe configuration parameters for a backend",
        description="Show __init__ parameters for a cache or remote backend.",
    )
    p.add_argument("--backend-type", required=True, choices=["cache", "remote"], help="Backend category")
    p.add_argument("--name", required=True, help="Registered backend name (e.g. 'folder', 'ssh')")
    p.set_defaults(func=cmd_describe_backend)


def _create_data_parsers(subparsers):
    """Register all data subcommands."""
    # summary
    p = subparsers.add_parser(
        "summary",
        help="Print human-readable summary of any data file",
        description="Load a data file and display its get_summary() output.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Data filename (e.g. mol.wavefunction.json)")
    p.set_defaults(func=cmd_data_summary)

    # convert
    p = subparsers.add_parser(
        "convert",
        help="Convert data file between JSON and HDF5",
        description="Load a data file and save in a different format.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Input filename")
    p.add_argument("--out-filename", required=True, help="Output filename (use .json or .hdf5 extension)")
    p.set_defaults(func=cmd_data_convert)

    # get-orbitals
    p = subparsers.add_parser(
        "get-orbitals",
        help="Extract orbitals from Wavefunction/Hamiltonian/Ansatz",
        description="Save the orbital data from any object that has get_orbitals().",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--input-filename", required=True, help="Input filename")
    p.add_argument("--out-orbitals-filename", required=True, help="Output orbitals filename")
    p.set_defaults(func=cmd_data_get_orbitals)

    # get-active-space-indices
    p = subparsers.add_parser(
        "get-active-space-indices",
        help="Get active/inactive/virtual orbital indices",
        description="Show orbital space partitioning after active space selection.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--input-filename", required=True, help="Filename with orbital info (wavefunction, ansatz, …)")
    p.set_defaults(func=cmd_data_get_active_space_indices)

    # get-ansatz
    p = subparsers.add_parser(
        "get-ansatz",
        help="Build Ansatz from wavefunction + Hamiltonian",
        description="Combine wavefunction and Hamiltonian into an Ansatz object.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Wavefunction filename")
    p.add_argument("--hamiltonian-filename", required=True, help="Hamiltonian filename")
    p.add_argument("--out-ansatz-filename", required=True, help="Output ansatz filename")
    p.set_defaults(func=cmd_data_get_ansatz)

    # get-top-configurations
    p = subparsers.add_parser(
        "get-top-configurations",
        help="Get top CI determinants by coefficient",
        description="Extract the largest-weight determinants from a multi-config wavefunction.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Wavefunction filename")
    p.add_argument("--max-determinants", type=int, help="Maximum number of determinants to return")
    p.set_defaults(func=cmd_data_get_top_configurations)

    # upload-structure
    p = subparsers.add_parser(
        "upload-structure",
        help="Upload a molecular structure to a project",
        description="Create a Structure data object from coordinates and symbols.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--coordinates-json", required=True, help="JSON 2D array of coordinates in Bohr: '[[x,y,z],...]'")
    p.add_argument("--symbols", nargs="+", required=True, help="Element symbols (e.g. H H or O H H)")
    p.add_argument("--nuclear-charges", type=parse_json_arg, help="Nuclear charges (JSON list)")
    p.add_argument("--masses", type=parse_json_arg, help="Atomic masses (JSON list)")
    p.add_argument("--filename-to-save", default="structure.structure.json", help="Output filename")
    p.set_defaults(func=cmd_data_create_structure)

    # get-energy
    p = subparsers.add_parser(
        "get-energy",
        help="Get energy from Wavefunction, QpeResult, or EnergyResult",
        description="Extract the energy value from a results file.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Data filename")
    p.set_defaults(func=cmd_data_get_energy)

    # get-structure-xyz
    p = subparsers.add_parser(
        "get-structure-xyz",
        help="Export structure as XYZ format",
        description="Convert a Structure to the standard XYZ text format.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Structure filename")
    p.add_argument("--out-file", help="Write XYZ to file instead of stdout")
    p.set_defaults(func=cmd_data_get_structure_xyz)

    # get-circuit-qasm
    p = subparsers.add_parser(
        "get-circuit-qasm",
        help="Export circuit as OpenQASM string",
        description="Convert a Circuit to OpenQASM format.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Circuit filename")
    p.set_defaults(func=cmd_data_get_circuit_qasm)

    # get-circuit-stats
    p = subparsers.add_parser(
        "get-circuit-stats",
        help="Analyze circuit resource profile",
        description="Show gate counts, depth, qubit count, and Clifford/non-Clifford breakdown.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Circuit filename")
    p.set_defaults(func=cmd_data_get_circuit_stats)

    # get-qubit-hamiltonian-info
    p = subparsers.add_parser(
        "get-qubit-hamiltonian-info",
        help="Inspect qubit Hamiltonian properties",
        description="Show qubit count, term count, Schatten norm, and Hermiticity.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Qubit Hamiltonian filename")
    p.set_defaults(func=cmd_data_get_qubit_hamiltonian_info)

    # get-stability-result
    p = subparsers.add_parser(
        "get-stability-result",
        help="Inspect stability analysis result",
        description="Check whether a wavefunction is stable (internal and external).",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="Stability result filename")
    p.set_defaults(func=cmd_data_get_stability_result)

    # get-qpe-result
    p = subparsers.add_parser(
        "get-qpe-result",
        help="Inspect QPE result",
        description="Show energies, phase, bits, and branch candidates from QPE.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--filename", required=True, help="QPE result filename")
    p.set_defaults(func=cmd_data_get_qpe_result)


def _create_project_parsers(subparsers):
    """Register project management subcommands under the ``project`` group."""
    # list
    p = subparsers.add_parser("list", help="List all projects in the scratch directory")
    p.set_defaults(func=cmd_utils_list_projects)

    # create
    p = subparsers.add_parser("create", help="Create a new project directory")
    p.add_argument("--project-name", required=True, help="Project name to create")
    p.set_defaults(func=cmd_utils_create_project)

    # files
    p = subparsers.add_parser(
        "files", help="List data files in a project", description="Show all files with inferred data types."
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.set_defaults(func=cmd_utils_list_files)


def _create_utils_parsers(subparsers):
    """Register utility subcommands under the ``util`` group."""
    # convert-coordinates
    p = subparsers.add_parser("convert-coordinates", help="Convert coordinates between Bohr and Angstrom")
    p.add_argument(
        "--coordinates", type=parse_json_arg, required=True, help="JSON 2D array of coordinates: '[[x,y,z],...]'"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--to-angstrom", action="store_true", help="Convert from Bohr to Angstrom")
    group.add_argument("--to-bohr", action="store_true", help="Convert from Angstrom to Bohr")
    p.set_defaults(func=cmd_utils_convert_coordinates)

    # convert-energy
    p = subparsers.add_parser(
        "convert-energy",
        help="Convert energy between units",
        description="Supported units: hartree, ev, kcal/mol, kj/mol",
    )
    p.add_argument("--value", type=float, required=True, help="Energy value to convert")
    p.add_argument("--from-unit", required=True, help="Source unit (hartree, ev, kcal/mol, kj/mol)")
    p.add_argument("--to-unit", required=True, help="Target unit (hartree, ev, kcal/mol, kj/mol)")
    p.set_defaults(func=cmd_utils_convert_energy)

    # compute-valence-params
    p = subparsers.add_parser(
        "compute-valence-params",
        help="Compute valence space parameters",
        description="Determine active electrons and orbitals for the valence space.",
    )
    p.add_argument("--project-name", required=True, help="Project name")
    p.add_argument("--wavefunction-filename", required=True, help="Wavefunction filename")
    p.add_argument("--charge", type=int, required=True, help="System charge")
    p.set_defaults(func=cmd_utils_compute_valence_params)

    # resolve-phase-energy
    p = subparsers.add_parser(
        "resolve-phase-energy",
        help="Resolve QPE phase to energy with alias handling",
        description="Convert a measured phase fraction to energy, resolving 2pi aliases.",
    )
    p.add_argument("--phase-fraction", type=float, required=True, help="Measured phase fraction from QPE")
    p.add_argument("--evolution-time", type=float, required=True, help="Evolution time used in QPE")
    p.add_argument("--reference-energy", type=float, required=True, help="Reference energy for alias resolution")
    p.set_defaults(func=cmd_utils_resolve_phase_energy)


# ═══════════════════════════════════════════════════════════════════════════
# Workflow command — run multi-step pipelines from a JSON config
# ═══════════════════════════════════════════════════════════════════════════

_WORKFLOW_COMMANDS = {
    "upload-structure": create_structure,
    "scf": run_scf,
    "active-space": run_active_space_selector,
    "localize": run_orbital_localization,
    "correlate": run_dynamical_correlation_calculator,
    "stability": run_stability_checker,
    "hamiltonian": run_hamiltonian_constructor,
    "model-hamiltonian": create_model_hamiltonian,
    "spin-model": create_spin_model_hamiltonian,
    "casci": run_multi_configuration_calculation,
    "mcscf": run_multi_configuration_scf,
    "sparse-ci": run_projected_multi_configuration_calculation,
    "qubit-map": run_qubit_mapper,
    "state-prep": run_state_preparation,
    "qubit-solve": run_qubit_hamiltonian_solver,
    "resource-estimation": run_resource_estimation,
    "energy": run_energy_estimator,
    "get-orbitals": get_orbitals_from_input,
    "get-active-space-indices": get_active_space_indices,
    "get-ansatz": get_ansatz,
    "get-top-configurations": get_top_configurations,
    "qpe-build-evolution": run_time_evolution_builder,
    "qpe-map-circuit": run_controlled_evolution_circuit_mapper,
    "qpe-execute": run_circuit_executor,
    "qpe": run_phase_estimation,
}


def _get_output_params(step_args: dict) -> list[str]:
    """Extract output filenames from step arguments."""
    outputs = []
    for k, v in step_args.items():
        if isinstance(v, str) and (k.startswith("out_") or k == "filename_to_save"):
            outputs.append(v)
    return outputs


def _get_input_params(step_args: dict) -> dict[str, str]:
    """Extract input filename parameters (param_name → filename or list of filenames)."""
    inputs: dict[str, Any] = {}
    for k, v in step_args.items():
        if k.startswith("out_") or k == "filename_to_save":
            continue
        if (isinstance(v, str) and k.endswith("_filename")) or (isinstance(v, list) and k.endswith("_filenames")):
            inputs[k] = v
    return inputs


def _resolve_step_refs(step_args: dict, step_outputs: list[dict]) -> dict:
    """Resolve ``$prev``, ``$prev.N``, and ``$step.N`` variable references in step args.

    Reference syntax (used in workflow JSON ``args`` values):
      - ``$prev``    — first output filename of the immediately preceding step
      - ``$prev.0``, ``$prev.1`` — Nth output of the previous step
      - ``$step.3``  — first output of step 3
      - ``$step.3.1`` — second output of step 3
    """
    ref_pattern = re.compile(r"^\$(?:prev|step\.(\d+))(?:\.(\d+))?$")

    def _resolve_one(val: str, current_step: int) -> str:
        m = ref_pattern.match(val)
        if not m:
            return val
        step_ref = m.group(1)
        idx = int(m.group(2)) if m.group(2) is not None else 0

        target_step = (current_step - 2) if step_ref is None else (int(step_ref) - 1)

        if target_step < 0 or target_step >= len(step_outputs):
            return val  # can't resolve — flagged by dry-run validation

        outputs = step_outputs[target_step].get("produces", [])
        if idx < len(outputs):
            return outputs[idx]
        return val  # index out of range — flagged by dry-run validation

    resolved: dict = {}
    for k, v in step_args.items():
        if isinstance(v, str):
            resolved[k] = _resolve_one(v, len(step_outputs) + 1)
        elif isinstance(v, list):
            resolved[k] = [_resolve_one(item, len(step_outputs) + 1) if isinstance(item, str) else item for item in v]
        else:
            resolved[k] = v
    return resolved


def _validate_required_params(cmd: str, step_args: dict, step_num: int) -> list[str]:
    """Check that required parameters for a command are present."""
    fn = _WORKFLOW_COMMANDS[cmd]
    sig = inspect.signature(fn)
    errors = []
    for name, param in sig.parameters.items():
        if name == "project_name":
            continue
        if param.default is inspect.Parameter.empty and name not in step_args:
            errors.append(f"Step {step_num} ({cmd}): missing required parameter '{name}'")
    return errors


def _dry_run_workflow(config_path: str, project_name: str):
    """Validate a workflow config without executing: check commands, params, and filename chaining."""
    try:
        with open(config_path) as f:
            workflow = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"dry_run": True, "valid": False, "error": f"Failed to load config: {e}"}, indent=2))
        sys.exit(1)

    steps = workflow.get("steps", [])
    if not steps:
        print(json.dumps({"dry_run": True, "valid": False, "error": "No 'steps' in workflow config."}, indent=2))
        sys.exit(1)

    produced_files: set = set()  # filenames produced by earlier steps
    plan: list = []
    errors: list = []
    warnings: list = []

    for i, step in enumerate(steps, 1):
        cmd = step.get("command", "")
        step_args = dict(step.get("args", {}))
        step_args.setdefault("project_name", project_name)

        # Check command exists
        if cmd not in _WORKFLOW_COMMANDS:
            errors.append(f"Step {i}: unknown command '{cmd}'")
            plan.append({"step": i, "command": cmd, "status": "error"})
            continue

        # Resolve $prev / $step.N references
        step_args = _resolve_step_refs(step_args, plan)

        # Flag unresolved references as errors
        for param, value in step_args.items():
            vals = value if isinstance(value, list) else [value]
            for v in vals:
                if isinstance(v, str) and v.startswith("$"):
                    errors.append(f"Step {i} ({cmd}): unresolved reference '{v}' in parameter '{param}'")

        # Check required params
        param_errors = _validate_required_params(cmd, step_args, i)
        errors.extend(param_errors)

        # Check input filename chaining (strings and lists)
        inputs = _get_input_params(step_args)
        chained_from = {}
        for param, value in inputs.items():
            filenames = value if isinstance(value, list) else [value]
            statuses = []
            for filename in filenames:
                if filename in produced_files:
                    statuses.append(f"{filename}: chained")
                else:
                    statuses.append(f"{filename}: must exist on disk")
                    warnings.append(
                        f"Step {i} ({cmd}): input '{param}={filename}' "
                        f"not produced by a previous step — must already exist on disk"
                    )
            chained_from[param] = statuses if isinstance(value, list) else statuses[0]

        # Track outputs
        outputs = _get_output_params(step_args)
        for out in outputs:
            produced_files.add(out)

        step_info: dict[str, Any] = {
            "step": i,
            "command": cmd,
            "args": {k: v for k, v in step_args.items() if k != "project_name"},
        }
        if chained_from:
            step_info["inputs"] = chained_from
        if outputs:
            step_info["produces"] = outputs
        plan.append(step_info)

    valid = len(errors) == 0
    result: dict[str, Any] = {
        "dry_run": True,
        "valid": valid,
        "project_name": project_name,
        "total_steps": len(steps),
        "plan": plan,
    }
    if errors:
        result["errors"] = errors
    if warnings:
        result["warnings"] = warnings

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if valid else 1)


def cmd_workflow(args):
    """Execute a multi-step workflow from a JSON configuration file."""
    # Handle --dry-run at workflow level (overrides the global --dry-run handler)
    if getattr(args, "dry_run", False):
        _dry_run_workflow(args.config, args.project_name)
        return

    try:
        with open(args.config) as f:
            workflow = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _print_error(f"Failed to load config: {e}")

    steps = workflow.get("steps", [])
    if not steps:
        _print_error("No 'steps' in workflow config.")

    results: list[dict[str, object]] = []
    for i, step in enumerate(steps, 1):
        cmd = step.get("command", "")
        step_args = dict(step.get("args", {}))

        if cmd not in _WORKFLOW_COMMANDS:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Step {i}: unknown command '{cmd}'",
                        "available_commands": sorted(_WORKFLOW_COMMANDS),
                        "completed": results,
                    },
                    indent=2,
                )
            )
            sys.exit(1)

        step_args.setdefault("project_name", args.project_name)
        # Resolve $prev / $step.N references using outputs from prior steps
        step_outputs = [{"produces": _get_output_params(dict(s.get("args", {})))} for s in steps[: i - 1]]
        step_args = _resolve_step_refs(step_args, step_outputs)
        result = _WORKFLOW_COMMANDS[cmd](**step_args)
        entry = {"step": i, "command": cmd, "result": result}
        results.append(entry)

        # Stop on error
        if isinstance(result, dict) and result.get("status") == "error":
            print(json.dumps({"success": False, "failed_step": i, "steps": results}, indent=2, default=str))
            sys.exit(1)

    print(json.dumps({"success": True, "steps": results}, indent=2, default=str))


def _create_workflow_parser(subparsers):
    """Register the workflow subcommand."""
    p = subparsers.add_parser(
        "workflow",
        help="Run a multi-step workflow from a JSON config",
        description=(
            "Execute a sequence of QDK Chemistry commands defined in a JSON file.\n\n"
            "Config format:\n"
            '  {"steps": [\n'
            '    {"command": "upload-structure", "args": {"coordinates_json": "...", ...}},\n'
            '    {"command": "scf", "args": {"structure_filename": "$prev", ...}}\n'
            "  ]}\n\n"
            "The --project-name is injected into each step automatically.\n"
            "Use $prev to reference the output of the previous step, or\n"
            "$step.N to reference the output of step N (1-indexed).\n"
            "Use --dry-run to preview the workflow without executing it.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, metavar="FILE", help="JSON workflow config file")
    p.add_argument("--project-name", required=True, help="Project name (applied to all steps)")
    p.set_defaults(func=cmd_workflow)


# ═══════════════════════════════════════════════════════════════════════════
# Describe command — machine-readable command introspection
# ═══════════════════════════════════════════════════════════════════════════


def cmd_describe(args):
    """Output a JSON schema for a CLI or workflow command."""
    target = args.target_command

    # Check workflow commands first (subset), then CLI parser commands
    all_commands: dict[str, Any] = {}

    # Workflow-available commands (backed by server functions)
    for name, fn in _WORKFLOW_COMMANDS.items():
        all_commands[name] = fn

    # Also allow describing CLI-only commands via parser introspection
    create_parser()  # ensure registry is populated
    parser_commands = _SUBPARSER_REGISTRY

    if target not in all_commands and target not in parser_commands:
        available = sorted(set(list(all_commands.keys()) + list(parser_commands.keys())))
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"Unknown command '{target}'",
                    "available_commands": available,
                },
                indent=2,
            )
        )
        sys.exit(1)

    # For workflow commands: introspect the backing function
    if target in all_commands:
        fn = all_commands[target]
        sig = inspect.signature(fn)
        params = {}
        for name, param in sig.parameters.items():
            if name == "project_name":
                continue
            info: dict[str, Any] = {}
            if param.annotation != inspect.Parameter.empty:
                info["type"] = str(param.annotation).replace("typing.", "")
            if param.default is not inspect.Parameter.empty:
                info["default"] = param.default
                info["required"] = False
            else:
                info["required"] = True
            # Classify as input or output
            if name.startswith("out_") or name == "filename_to_save":
                info["role"] = "output"
            elif name.endswith(("_filename", "_filenames")):
                info["role"] = "input"
            params[name] = info

        doc = (fn.__doc__ or "").split("\n")[0].strip()
        schema = {
            "command": target,
            "description": doc,
            "parameters": params,
            "workflow_compatible": True,
        }
        print(json.dumps(schema, indent=2, default=str))
        return

    # For CLI-only commands: introspect the argparse parser
    sub_parser = parser_commands[target]
    params = {}
    for action in sub_parser._actions:  # noqa: SLF001  # argparse has no public action iterator
        if action.dest in ("help", "func", "command"):
            continue
        info = {}
        if action.type:
            info["type"] = action.type.__name__ if hasattr(action.type, "__name__") else str(action.type)
        if action.default is not None:
            info["default"] = action.default
        info["required"] = action.required if hasattr(action, "required") else False
        if action.help:
            info["help"] = action.help
        # Classify as input or output based on naming conventions
        dest = action.dest
        if dest.startswith("out_") or dest in ("filename_to_save", "out_html", "out_file", "out_filename"):
            info["role"] = "output"
        elif dest.endswith(("_filename", "_filenames")) or dest == "filename":
            info["role"] = "input"
        params[dest] = info

    schema = {
        "command": target,
        "description": sub_parser.description or (sub_parser.format_usage().strip()),
        "parameters": params,
        "workflow_compatible": target in all_commands,
    }
    print(json.dumps(schema, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════════
# List-commands — machine-readable command listing
# ═══════════════════════════════════════════════════════════════════════════


def cmd_list_commands(_args):
    """List all available commands in JSON format for agent discovery."""
    create_parser()  # ensure registry is populated
    parser_commands = _SUBPARSER_REGISTRY

    commands = []
    for name, sub_parser in sorted(parser_commands.items()):
        entry: dict[str, Any] = {
            "name": name,
            "help": sub_parser.description or "",
            "workflow_compatible": name in _WORKFLOW_COMMANDS,
        }
        # Categorise by group
        if name in ("run", "data", "config", "project", "util", "setup"):
            entry["category"] = "group"
        elif name in ("workflow", "describe", "list-commands"):
            entry["category"] = "meta"
        elif name == "remote-run":
            entry["category"] = "internal"
        commands.append(entry)

    print(json.dumps({"commands": commands, "total": len(commands)}, indent=2))


def _create_list_commands_parser(subparsers):
    """Register the list-commands subcommand."""
    p = subparsers.add_parser(
        "list-commands",
        help="List all available commands in JSON format",
        description="Output a JSON array of all commands with names, "
        "help text, categories, and workflow compatibility.",
    )
    p.set_defaults(func=cmd_list_commands)


def _create_describe_parser(subparsers):
    """Register the describe subcommand."""
    p = subparsers.add_parser(
        "describe",
        help="Show machine-readable JSON schema for a command",
        description=(
            "Output a JSON description of any command's parameters,\n"
            "types, defaults, and input/output roles.  Useful for agents\n"
            "and programmatic tool discovery.\n\n"
            "Examples:\n"
            "  qdk_chem_cli describe run-scf\n"
            "  qdk_chem_cli describe workflow\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("target_command", metavar="COMMAND", help="Command name to describe")
    p.set_defaults(func=cmd_describe)


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# Agent config deployment
# ═══════════════════════════════════════════════════════════════════════════

_AGENT_CONFIGS = {
    "vscode": "Agent configs for VS Code / GitHub Copilot Chat (skills, agents, MCP, instructions)",
    "copilot": "Copilot CLI instructions (copilot-instructions.md)",
    "claude": "Claude Code instructions (CLAUDE.md + skills)",
}

# Map from flavor to the target subdirectory for shared skills
_SKILLS_DEPLOY_TARGETS = {
    "vscode": ".github/skills",
    "claude": ".claude/skills",
}


def _get_agent_configs_dir() -> Path:
    """Return the path to the bundled agent_configs directory."""
    return Path(__file__).parent / "agent_configs"


_VERSION_PLACEHOLDER = "{{QDK_CHEMISTRY_VERSION}}"


def _get_package_version() -> str:
    """Return the qdk_chemistry package version string (e.g. 'v1.1.0')."""
    from qdk_chemistry import __version__  # noqa: PLC0415

    return f"v{__version__}"


def _copy_with_version(src: Path, dest: Path) -> None:
    """Copy a file, replacing the ``{{QDK_CHEMISTRY_VERSION}}`` placeholder with the real version."""
    if src.suffix in (".md", ".json", ".yaml", ".yml", ".jsonc"):
        text = src.read_text(encoding="utf-8")
        if _VERSION_PLACEHOLDER in text:
            text = text.replace(_VERSION_PLACEHOLDER, _get_package_version())
        dest.write_text(text, encoding="utf-8")
        shutil.copystat(src, dest)
    else:
        shutil.copy2(src, dest)


def _copytree_with_version(src: Path, dest: Path, force: bool = False) -> tuple[list[str], list[str]]:
    """Recursively copy a directory tree, injecting the package version into text files.

    Returns (deployed_files, skipped_files) — only individual file paths, no dirs.
    """
    deployed: list[str] = []
    skipped: list[str] = []
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        if item.is_dir():
            (dest / item.relative_to(src)).mkdir(parents=True, exist_ok=True)
            continue
        rel = item.relative_to(src)
        target = dest / rel
        if target.exists() and not force:
            skipped.append(str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_with_version(item, target)
        deployed.append(str(target))
    return deployed, skipped


def _deploy_shared_skills(
    configs_dir: Path, target: Path, skills_subdir: str, force: bool = False
) -> tuple[list[str], list[str]]:
    """Copy shared skills into the target's skills directory.

    Returns (deployed_files, skipped_files).
    """
    shared_skills = configs_dir / "shared" / "skills"
    if not shared_skills.is_dir():
        return [], []

    dest_skills = target / skills_subdir
    return _copytree_with_version(shared_skills, dest_skills, force=force)


def _generate_mcp_json(target: Path) -> str:
    """Generate .vscode/mcp.json with the correct qdk_chem_mcp command for the current environment."""
    mcp_bin = shutil.which("qdk_chem_mcp")

    # Detect if we're inside a virtual environment
    venv = os.environ.get("VIRTUAL_ENV")

    server_entry: dict[str, Any] = {}
    if mcp_bin:
        # qdk_chem_mcp is on PATH — use it directly
        server_entry = {"command": mcp_bin}
    elif venv:
        # Not on PATH but we know the venv — activate and run
        activate = os.path.join(venv, "bin", "activate")
        server_entry = {"command": "bash", "args": ["-c", f"source {activate} && qdk_chem_mcp"]}
    else:
        # Fallback: assume it'll be on PATH at runtime
        server_entry = {"command": "qdk_chem_mcp"}

    mcp_config = {
        "servers": {
            "qdk_chemistry": server_entry,
        }
    }

    vscode_dir = target / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    mcp_path = vscode_dir / "mcp.json"
    mcp_path.write_text(json.dumps(mcp_config, indent=4) + "\n")
    return str(mcp_path)


def cmd_setup_agents(args):
    """Deploy agent configuration files to a target directory."""
    target = Path(args.target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    configs_dir = _get_agent_configs_dir()
    flavor = args.flavor
    force = args.force
    no_mcp = args.no_mcp
    component = getattr(args, "component", None)

    if not configs_dir.is_dir():
        print(json.dumps({"success": False, "error": "Bundled agent_configs not found. Package may be incomplete."}))
        sys.exit(1)

    flavors = [flavor] if flavor != "all" else list(_AGENT_CONFIGS.keys())
    deployed = []
    skipped = []

    for f in flavors:
        src = configs_dir / f
        if not src.is_dir():
            print(json.dumps({"success": False, "error": f"Config flavor '{f}' not found in bundled data."}))
            sys.exit(1)

        if f == "vscode":
            # Deploy components based on --component filter
            deploy_instructions = component in (None, "instructions")
            deploy_agents = component in (None, "agents")
            deploy_skills = component in (None, "skills")
            deploy_mcp = component is None and not no_mcp

            for child in src.iterdir():
                # .github/ directory contains agents + copilot-instructions.md
                if child.is_dir() and child.name == ".github":
                    for gh_child in child.iterdir():
                        if gh_child.name == "agents" and not deploy_agents:
                            continue
                        if gh_child.name == "copilot-instructions.md" and not deploy_instructions:
                            continue
                        dest = target / child.name / gh_child.name
                        if gh_child.is_dir():
                            d, s = _copytree_with_version(gh_child, dest, force=force)
                            deployed.extend(d)
                            skipped.extend(s)
                        else:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            if dest.exists() and not force:
                                skipped.append(str(dest))
                            else:
                                _copy_with_version(gh_child, dest)
                                deployed.append(str(dest))
                elif child.is_file():
                    dest = target / child.name
                    if dest.exists() and not force:
                        skipped.append(str(dest))
                    else:
                        _copy_with_version(child, dest)
                        deployed.append(str(dest))

            if deploy_skills:
                d, s = _deploy_shared_skills(configs_dir, target, _SKILLS_DEPLOY_TARGETS["vscode"], force=force)
                deployed.extend(d)
                skipped.extend(s)

            if deploy_mcp:
                mcp_path = target / ".vscode" / "mcp.json"
                if mcp_path.exists() and not force:
                    skipped.append(str(mcp_path))
                else:
                    deployed.append(_generate_mcp_json(target))

        elif f == "claude":
            for child in src.iterdir():
                dest = target / child.name
                if dest.exists() and not force:
                    skipped.append(str(dest))
                else:
                    _copy_with_version(child, dest)
                    deployed.append(str(dest))
            d, s = _deploy_shared_skills(configs_dir, target, _SKILLS_DEPLOY_TARGETS["claude"], force=force)
            deployed.extend(d)
            skipped.extend(s)

        elif f == "copilot":
            for child in src.iterdir():
                dest = target / child.name
                if dest.exists() and not force:
                    skipped.append(str(dest))
                else:
                    _copy_with_version(child, dest)
                    deployed.append(str(dest))

    result: dict[str, Any] = {"success": True, "flavor": flavor, "target": str(target), "deployed": deployed}
    if skipped:
        result["skipped"] = skipped
        result["hint"] = "Some files already exist. Use --force to overwrite."
    print(json.dumps(result, indent=2))


def cmd_setup_mcp(args):
    """Generate .vscode/mcp.json with the correct paths for the current environment."""
    target = Path(args.target_dir).resolve()
    mcp_path = _generate_mcp_json(target)
    print(json.dumps({"success": True, "target": str(target), "mcp_json": mcp_path}, indent=2))


def _create_setup_parsers(subparsers):
    """Register setup subcommands under the ``setup`` group."""
    flavor_list = ", ".join(_AGENT_CONFIGS.keys())
    p = subparsers.add_parser(
        "agents",
        help="Deploy agent configuration files (skills, agents, instructions) to a directory",
        description=(
            "Copy bundled agent configuration files into a target directory.\n"
            "This sets up the .github/ folder with skills, agents, and\n"
            "copilot-instructions.md for VS Code / Copilot Chat, or deploys\n"
            "CLAUDE.md / copilot-instructions.md for other AI tool integrations.\n\n"
            f"Available flavors: {flavor_list}, all\n\n"
            "By default, existing files are NOT overwritten. Use --force to update.\n\n"
            "Examples:\n"
            "  qc setup agents --target-dir . --flavor vscode\n"
            "  qc setup agents --target-dir . --flavor vscode --component instructions\n"
            "  qc setup agents --target-dir . --flavor vscode --force\n"
            "  qc setup agents --target-dir . --flavor vscode --no-mcp\n"
            "  qc setup agents --target-dir /my/project --flavor all --force\n"
            "  qc setup agents --target-dir . --flavor claude\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--target-dir",
        required=True,
        help="Directory to deploy agent config files into",
    )
    p.add_argument(
        "--flavor",
        default="all",
        choices=[*_AGENT_CONFIGS, "all"],
        help="Which config set to deploy (default: all)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing files (default: skip existing files)",
    )
    p.add_argument(
        "--no-mcp",
        action="store_true",
        default=False,
        help="Skip generating .vscode/mcp.json (vscode flavor only)",
    )
    p.add_argument(
        "--component",
        choices=["instructions", "agents", "skills"],
        default=None,
        help="Deploy only a specific component (vscode flavor only). "
        "instructions = .github/copilot-instructions.md, "
        "agents = .github/agents/, skills = .github/skills/",
    )
    p.set_defaults(func=cmd_setup_agents)

    # mcp
    p = subparsers.add_parser(
        "mcp",
        help="Generate .vscode/mcp.json for the QDK Chemistry MCP server",
        description=(
            "Generate a .vscode/mcp.json file that points to the QDK Chemistry\n"
            "MCP server using the correct paths for your current environment.\n\n"
            "The command auto-detects whether qdk_chem_mcp is on PATH (uses the\n"
            "absolute path) or inside a virtual environment (wraps with activate).\n\n"
            "Examples:\n"
            "  qdk_chem_cli setup mcp --target-dir .\n"
            "  qdk_chem_cli setup mcp --target-dir /path/to/my/project\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--target-dir",
        required=True,
        help="Directory to generate .vscode/mcp.json in",
    )
    p.set_defaults(func=cmd_setup_mcp)


# ═══════════════════════════════════════════════════════════════════════════
# remote-run — single CLI command executed on remote compute nodes
# ═══════════════════════════════════════════════════════════════════════════


def cmd_remote_run(args: argparse.Namespace) -> None:
    """Execute a serialized algorithm job from an input directory.

    This command is invoked by the remote execution backend on compute
    nodes.  It replaces the generated Python script with a stable,
    single-command interface.

    Steps:
        1. Optionally connect to a remote cache (from the manifest).
        2. Check the cache for a full result hit → write outputs and exit.
        3. Deserialize inputs (resolving ``"cached"`` entries from the cache).
        4. Reconstruct and execute the algorithm.
        5. Store results in the remote cache (best-effort).
        6. Serialize outputs to the output directory.
    """
    from qdk_chemistry.algorithms import create as create_algorithm  # noqa: PLC0415
    from qdk_chemistry.remote.serialization import (  # noqa: PLC0415
        deserialize_inputs,
        serialize_outputs,
    )

    input_dir = args.input_dir
    output_dir = args.output_dir

    # 1) Connect to remote cache if the manifest includes one
    cache = None
    run_hash = None
    try:
        manifest_path = Path(input_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            run_hash = manifest.get("run_hash")
            cache_info = manifest.get("remote_cache")
            if cache_info and cache_info.get("name"):
                from qdk_chemistry.remote.cache import get_cache  # noqa: PLC0415

                name = cache_info["name"]
                cache_config = {k: v for k, v in cache_info.items() if k != "name"}
                cache = get_cache(name, **cache_config)
    except Exception:  # noqa: BLE001
        pass  # best-effort; proceed without cache

    # 2) Check cache for a full result hit
    result: Any = None
    if cache is not None and run_hash is not None:
        try:
            job = cache.get_job(run_hash)
            if job is not None and getattr(job, "output_hashes", None):
                status = getattr(job, "status", "")
                if status in ("retrieved", "Succeeded"):
                    items: list[Any] = []
                    hit = True
                    for entry in job.output_hashes:
                        if "value" in entry:
                            items.append(entry["value"])
                        else:
                            data = cache.get_data(entry["hash"])
                            if data is None:
                                hit = False
                                break
                            items.append(data)
                    if hit:
                        result = items[0] if len(items) == 1 else tuple(items)
                        print("CACHE HIT: Results loaded from remote cache")
        except Exception:  # noqa: BLE001
            pass  # cache miss — compute below

    # 3) Deserialize inputs and run the algorithm
    if result is None:
        inputs = deserialize_inputs(input_dir, cache=cache)

        algorithm = create_algorithm(
            inputs["algorithm_type"],
            inputs["algorithm_name"],
        )
        for key, value in inputs["settings"].items():
            algorithm.settings().set(key, value)

        result = algorithm.run(*inputs["args"], **inputs["kwargs"])

        # 4) Store in remote cache (best-effort)
        if cache is not None and run_hash is not None:
            try:
                import datetime  # noqa: PLC0415

                from qdk_chemistry.data._hashing import collect_content_hashes  # noqa: PLC0415
                from qdk_chemistry.remote.job import Job  # noqa: PLC0415

                output_hashes = collect_content_hashes(result)
                result_items = result if isinstance(result, tuple) else (result,)
                for entry, item in zip(output_hashes, result_items, strict=False):
                    if "value" not in entry:
                        cache.put_data(entry["hash"], item)

                job_obj = Job(
                    job_id=run_hash[:12],
                    backend="remote",
                    backend_config={},
                    backend_state={},
                    algorithm_info={
                        "type": inputs.get("algorithm_type"),
                        "name": inputs.get("algorithm_name"),
                        "settings": inputs.get("settings", {}),
                    },
                    status="retrieved",
                    submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    run_hash=run_hash,
                    output_hashes=output_hashes,
                )
                cache.put_job(run_hash, job_obj)
            except Exception:  # noqa: BLE001
                pass

    # 5) Serialize outputs
    serialize_outputs(output_dir, result)
    print(json.dumps({"success": True, "output_dir": str(output_dir)}))


def _create_remote_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``remote-run`` subcommand."""
    p = subparsers.add_parser(
        "remote-run",
        help="(internal) Execute a serialized job on a compute node",
        description=(
            "Run a pre-serialized algorithm job.  Reads inputs from\n"
            "--input-dir, executes the algorithm, and writes outputs to\n"
            "--output-dir.  If the input manifest contains remote_cache\n"
            "coordinates, the cache is checked first and results are\n"
            "stored on completion.\n\n"
            "This command is not intended for direct use — it is invoked\n"
            "by remote execution backends on compute nodes."
        ),
    )
    p.add_argument("--input-dir", required=True, help="Directory containing serialized inputs and manifest.json")
    p.add_argument("--output-dir", required=True, help="Directory to write serialized outputs to")
    p.set_defaults(func=cmd_remote_run)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with grouped subcommands.

    Top-level groups::

        qc run scf ...          # algorithm execution
        qc data summary ...     # data inspection / extraction
        qc config algorithms    # configuration & discovery
        qc project list         # project management
        qc util convert-energy  # unit conversion helpers
        qc setup agents ...     # agent/MCP deployment
        qc workflow ...         # multi-step pipelines
        qc describe ...         # command introspection

    """
    parser = argparse.ArgumentParser(
        prog="qc",
        description=(
            "QDK Chemistry CLI — quantum chemistry from the command line.\n\n"
            "Commands are organised into groups:\n\n"
            "  qc run scf --project-name h2 --structure-filename h2.structure.json ...\n"
            "  qc data summary --project-name h2 --filename h2.wavefunction.json\n"
            "  qc config algorithms\n"
            "  qc project list\n"
            "  qc workflow --config pipeline.json --project-name h2\n\n"
            "Use ``qc <group> -h`` to see subcommands within a group.\n\n"
            "If you are an agent:\n"
            "  Use ``qc setup agents`` to deploy skills and instructions\n"
            "  to your workspace so your AI assistant understands QDK Chemistry.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    from importlib.metadata import version as _pkg_version  # noqa: PLC0415

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_pkg_version('qdk-chemistry')}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what the command would do without executing it",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command groups")

    # ── run ──────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run algorithms (SCF, CASCI, QPE, …)")
    run_sub = run_parser.add_subparsers(dest="subcommand")
    _create_algorithm_parsers(run_sub)

    # ── data ─────────────────────────────────────────────────────────────
    data_parser = subparsers.add_parser("data", help="Inspect, extract, and convert data files")
    data_sub = data_parser.add_subparsers(dest="subcommand")
    _create_data_parsers(data_sub)

    # ── config ───────────────────────────────────────────────────────────
    config_parser = subparsers.add_parser("config", help="List algorithms, backends, and defaults")
    config_sub = config_parser.add_subparsers(dest="subcommand")
    _create_config_parsers(config_sub)

    # ── project ──────────────────────────────────────────────────────────
    project_parser = subparsers.add_parser("project", help="Manage projects")
    project_sub = project_parser.add_subparsers(dest="subcommand")
    _create_project_parsers(project_sub)

    # ── util ─────────────────────────────────────────────────────────────
    util_parser = subparsers.add_parser("util", help="Unit conversion and helpers")
    util_sub = util_parser.add_subparsers(dest="subcommand")
    _create_utils_parsers(util_sub)

    # ── setup ────────────────────────────────────────────────────────────
    setup_parser = subparsers.add_parser("setup", help="Deploy agent configs and MCP server")
    setup_sub = setup_parser.add_subparsers(dest="subcommand")
    _create_setup_parsers(setup_sub)

    # ── top-level commands ───────────────────────────────────────────────
    _create_workflow_parser(subparsers)
    _create_describe_parser(subparsers)
    _create_list_commands_parser(subparsers)
    _create_remote_run_parser(subparsers)

    # Populate the module-level registry for use by describe/list-commands
    _SUBPARSER_REGISTRY.clear()
    _SUBPARSER_REGISTRY.update(subparsers.choices)

    argcomplete.autocomplete(parser)
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Groups require a subcommand
    if args.command in ("run", "data", "config", "project", "util", "setup") and not getattr(args, "subcommand", None):
        # Print help for the group parser
        parser.parse_args([args.command, "-h"])
        sys.exit(1)

    # --dry-run: show parameters without executing
    # Workflow handles its own dry-run with step validation
    if getattr(args, "dry_run", False) and args.command != "workflow":
        params = {
            k: v
            for k, v in vars(args).items()
            if k not in ("func", "command", "subcommand", "dry_run") and v is not None
        }
        plan = {
            "dry_run": True,
            "command": args.command,
            "subcommand": getattr(args, "subcommand", None),
            "parameters": params,
        }
        print(json.dumps(plan, indent=2, default=str))
        sys.exit(0)

    try:
        args.func(args)
    except Exception as e:  # noqa: BLE001
        error_output = {
            "success": False,
            "error": str(e),
            "type": type(e).__name__,
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
