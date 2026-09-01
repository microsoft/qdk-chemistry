"""Unit tests for qdk_chemistry.ui CLI functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer

# Import CLI functions for direct testing
from qdk_chemistry.ui.cli import (
    _deep_merge,
    _parse_set_overrides,
    create_parser,
    format_output,
    main,
    parse_json_arg,
)
from qdk_chemistry.ui.config import config

# ==================== Test Fixtures ====================


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory."""
    original_projects_dir = config.projects_dir
    original_cwd = Path.cwd()
    config.projects_dir = tmp_path
    try:
        yield tmp_path
    finally:
        config.projects_dir = original_projects_dir
        os.chdir(original_cwd)


@pytest.fixture
def h2_structure_file(temp_project_dir):
    """Create an H2 structure file in a temporary project."""
    project_path = temp_project_dir / "test_project"
    project_path.mkdir(exist_ok=True)

    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    symbols = ["H", "H"]
    structure = data.Structure(coordinates=coords, symbols=symbols)
    structure_file = project_path / "h2.structure.json"
    structure.to_json_file(str(structure_file))
    return "h2.structure.json"


@pytest.fixture
def h2_wavefunction_file(temp_project_dir, h2_structure_file):
    """Create an H2 wavefunction file."""
    project_path = temp_project_dir / "test_project"

    # Load structure
    structure = data.Structure.from_json_file(str(project_path / h2_structure_file))

    # Run SCF
    scf_solver = algorithms.create("scf_solver")
    _, wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")

    wf_file = project_path / "h2.wavefunction.json"
    wavefunction.to_json_file(str(wf_file))
    return "h2.wavefunction.json"


# ==================== Shared Helper Tests ====================


def test_parse_json_arg():
    """Test JSON argument parsing."""
    # Valid JSON
    assert parse_json_arg('{"key": "value"}') == {"key": "value"}
    assert parse_json_arg("[1, 2, 3]") == [1, 2, 3]
    assert parse_json_arg("null") is None
    assert parse_json_arg("123") == 123

    # Invalid JSON should raise ArgumentTypeError
    with pytest.raises(argparse.ArgumentTypeError):
        parse_json_arg("invalid json")


def test_format_output():
    """Test output formatting."""
    # Test tuple
    result = format_output((1.5, "file.json"))
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["result"] == [1.5, "file.json"]

    # Test list
    result = format_output([1, 2, 3])
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["result"] == [1, 2, 3]

    # Test dict
    result = format_output({"key": "value"})
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["result"] == {"key": "value"}

    # Test success string
    result = format_output("output.json")
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["result"] == "output.json"

    # Test error string
    result = format_output("Failed to load file")
    parsed = json.loads(result)
    assert parsed["success"] is False
    assert "Failed to load file" in parsed["error"]

    # Test structured envelope (from @_structured decorator)
    result = format_output({"status": "ok", "result": "output.json"})
    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["result"] == "output.json"

    result = format_output({"status": "error", "message": "something went wrong"})
    parsed = json.loads(result)
    assert parsed["success"] is False
    assert "something went wrong" in parsed["error"]


def test_parse_set_overrides():
    """Test --set key=value parsing."""
    result = _parse_set_overrides(
        [
            "mc_calculator.settings.calculate_one_rdm=true",
            "mc_calculator.settings.calculate_two_rdm=true",
            "mc_calculator.settings.calculate_mutual_information=true",
            "settings.max_iterations=50",
        ]
    )
    assert result == {
        "mc_calculator": {
            "settings": {
                "calculate_one_rdm": True,
                "calculate_two_rdm": True,
                "calculate_mutual_information": True,
            }
        },
        "settings": {"max_iterations": 50},
    }

    # String values
    result = _parse_set_overrides(["algorithm_name=pyscf"])
    assert result == {"algorithm_name": "pyscf"}

    # Empty
    assert _parse_set_overrides(None) == {}
    assert _parse_set_overrides([]) == {}


def test_deep_merge():
    """Test recursive dict merging."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overrides = {"a": {"b": 10, "e": 5}, "f": 6}
    _deep_merge(base, overrides)
    assert base == {"a": {"b": 10, "c": 2, "e": 5}, "d": 3, "f": 6}


def test_create_parser():
    """Test parser creation."""
    parser = create_parser()
    assert parser.prog == "qc"
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    assert "setup" not in subparsers.choices

    args = parser.parse_args(["config", "defaults", "--algorithm-type", "scf_solver"])
    assert args.command == "config"
    assert args.subcommand == "defaults"
    assert args.algorithm_type == "scf_solver"


def test_resolve_phase_energy_uses_product_formula_mapping(temp_project_dir, capsys, monkeypatch):
    """Resolve aliases using the sign and scale stored by a product-formula unitary."""
    project_path = temp_project_dir / "test_project"
    project_path.mkdir()
    unitary = data.UnitaryRepresentation(
        PauliProductFormulaContainer(
            step_terms=[ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.5)],
            step_reps=1,
            num_qubits=1,
            scale=2.0,
        )
    )
    filename = "evolution.unitary_representation.json"
    unitary.to_json_file(str(project_path / filename))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qc",
            "util",
            "resolve-phase-energy",
            "--project-name",
            "test_project",
            "--unitary-representation-filename",
            filename,
            "--phase-fraction",
            "0.25",
            "--reference-energy",
            str(3.0 * np.pi / 4.0),
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["container_type"] == "pauli_product_formula"
    assert result["raw_energy"] == pytest.approx(-np.pi / 4.0)
    assert result["resolved_energy"] == pytest.approx(3.0 * np.pi / 4.0)


def test_resolve_phase_energy_uses_quantum_walk_mapping(temp_project_dir, capsys, monkeypatch):
    """Use the cosine inversion supplied by a quantum-walk container without periodic energy aliases."""
    project_path = temp_project_dir / "test_project"
    project_path.mkdir()
    unitary = data.UnitaryRepresentation(LCUWalkContainer(block_encoding=None, scale=4.0))
    monkeypatch.setattr("qdk_chemistry.ui.cli.load_data_object", lambda *_args: unitary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qc",
            "util",
            "resolve-phase-energy",
            "--project-name",
            "test_project",
            "--unitary-representation-filename",
            "walk.unitary_representation.json",
            "--phase-fraction",
            "0.25",
            "--reference-energy",
            "3.0",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["container_type"] == "lcu_walk"
    assert result["raw_energy"] == pytest.approx(0.0, abs=1e-12)
    assert result["resolved_energy"] == pytest.approx(0.0, abs=1e-12)


def test_resolve_phase_energy_rejects_raw_block_encoding(temp_project_dir, capsys, monkeypatch):
    """Preserve the container's refusal to infer a phase mapping for a raw block encoding."""
    project_path = temp_project_dir / "test_project"
    project_path.mkdir()
    unitary = data.UnitaryRepresentation(LCUContainer(prepare=None, select=None))
    monkeypatch.setattr("qdk_chemistry.ui.cli.load_data_object", lambda *_args: unitary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qc",
            "util",
            "resolve-phase-energy",
            "--project-name",
            "test_project",
            "--unitary-representation-filename",
            "block.unitary_representation.json",
            "--phase-fraction",
            "0.25",
            "--reference-energy",
            "0.0",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        main()

    result = json.loads(capsys.readouterr().out)
    assert result["success"] is False
    assert result["type"] == "NotImplementedError"
    assert "Wrap it in an LCUWalkContainer" in result["error"]


# ==================== Algorithm Group Tests ====================


def test_cli_algorithm_defaults(capsys):
    """Test algorithm defaults command."""
    sys.argv = ["qc", "config", "defaults", "--algorithm-type", "scf_solver"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert isinstance(result["result"], str)


def test_cli_algorithm_defaults_settings(capsys):
    """Test algorithm defaults with settings."""
    sys.argv = ["qc", "config", "defaults", "--algorithm-type", "scf_solver"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True


def test_cli_algorithm_defaults_config_template(capsys):
    """Test generating a config template for compound algorithms."""
    sys.argv = ["qc", "config", "defaults", "--type", "mcscf"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    template = json.loads(captured.out)
    assert template["success"] is True
    result = template["result"]
    assert "ham_constructor" in result
    assert "mc_calculator" in result
    assert "mcscf" in result


def test_cli_algorithm_list(capsys):
    """Test algorithm list command."""
    sys.argv = ["qc", "config", "algorithms"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True


def test_cli_data_create_structure(temp_project_dir, capsys):
    """Test data upload-structure command."""
    project_path = temp_project_dir / "test_project"
    project_path.mkdir(exist_ok=True)

    coords_json = "[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]"

    sys.argv = [
        "qc",
        "data",
        "upload-structure",
        "--project-name",
        "test_project",
        "--coordinates-json",
        coords_json,
        "--symbols",
        "H",
        "H",
        "--filename-to-save",
        "test.structure.json",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True

    # Verify file was created
    assert (project_path / "test.structure.json").exists()


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_algorithm_error_handling(capsys):
    """Test error handling for invalid inputs."""
    sys.argv = [
        "qc",
        "run",
        "scf",
        "--project-name",
        "test_project",
        "--structure-filename",
        "nonexistent.structure.json",
        "--out-wavefunction-filename",
        "out.wavefunction.json",
        "--charge",
        "0",
        "--spin-multiplicity",
        "1",
        "--basis-set",
        "sto-3g",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is False
    assert "error" in result


def test_cli_no_command(capsys):
    """Test CLI with no command shows help."""
    sys.argv = ["qc"]

    with pytest.raises(SystemExit):
        main()

    captured = capsys.readouterr()
    # Should show help message
    assert "usage:" in captured.out.lower() or captured.err


def _extract_json(stdout: str):
    """Extract the JSON object from stdout, skipping log lines."""
    # Find first '{' which starts the JSON output
    idx = stdout.find("{")
    if idx == -1:
        return None
    return json.loads(stdout[idx:])


@pytest.mark.usefixtures("temp_project_dir", "h2_structure_file")
def test_cli_subprocess_invocation():
    """Test CLI invocation via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "qdk_chemistry.ui.cli", "config", "defaults", "--algorithm-type", "scf_solver"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    output = _extract_json(result.stdout)
    assert output is not None
    assert output["success"] is True


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_module_invocation():
    """Test CLI can be invoked as a module."""
    result = subprocess.run(
        [sys.executable, "-m", "qdk_chemistry.ui.cli", "config", "defaults", "--algorithm-type", "scf_solver"],
        check=False,
        capture_output=True,
        text=True,
    )

    output = _extract_json(result.stdout)
    assert output is not None
    assert output["success"] is True


def test_cli_algorithm_defaults_orbital_localizer(capsys):
    """Test algorithm defaults for orbital_localizer."""
    sys.argv = [
        "qc",
        "config",
        "defaults",
        "--algorithm-type",
        "orbital_localizer",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True


# ==================== Parser Structure Tests ====================


def test_cli_algorithm_energy_parser():
    """Test algorithm energy parser with multiple filenames."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "energy",
            "--project-name",
            "test_project",
            "--circuit-filename",
            "circuit.circuit.json",
            "--qubit-hamiltonian-filename",
            "qh1.qubit_ham.h5",
            "--out-energy-result-filename",
            "energy.result.json",
            "--out-measurement-data-filename",
            "measurement.data.json",
            "--total-shots",
            "1000",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "energy"
    assert args.qubit_hamiltonian_filename == "qh1.qubit_ham.h5"


def test_cli_algorithm_qpe_build_evolution_parser():
    """Test qpe-build-evolution parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "qpe-build-evolution",
            "--project-name",
            "test_project",
            "--qubit-hamiltonian-filename",
            "qh.qubit_hamiltonian.h5",
            "--evolution-time",
            "0.1",
            "--out-time-evolution-unitary-filename",
            "teu.time_evolution_unitary.json",
            "--algorithm-name",
            "trotter",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "qpe-build-evolution"
    assert args.project_name == "test_project"
    assert args.evolution_time == 0.1
    assert args.algorithm_name == "trotter"


def test_cli_algorithm_qpe_map_circuit_parser():
    """Test qpe-map-circuit parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "qpe-map-circuit",
            "--project-name",
            "test_project",
            "--time-evolution-unitary-filename",
            "teu.time_evolution_unitary.json",
            "--out-circuit-filename",
            "ctrl.circuit.json",
            "--power",
            "4",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "qpe-map-circuit"
    assert args.project_name == "test_project"
    assert args.power == 4


def test_cli_algorithm_qpe_execute_parser():
    """Test qpe-execute parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "qpe-execute",
            "--project-name",
            "test_project",
            "--circuit-filename",
            "circuit.circuit.json",
            "--shots",
            "1000",
            "--out-executor-data-filename",
            "exec.circuit_executor_data.json",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "qpe-execute"
    assert args.project_name == "test_project"
    assert args.shots == 1000


def test_cli_algorithm_qpe_parser():
    """Test qpe (full pipeline) parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "qpe",
            "--project-name",
            "test_project",
            "--state-prep-circuit-filename",
            "circuit.circuit.json",
            "--qubit-hamiltonian-filename",
            "ham.qubithamiltonian.h5",
            "--out-qpe-result-filename",
            "qpe.qperesult.json",
            "--algorithm-name",
            "iterative",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "qpe"
    assert args.project_name == "test_project"
    assert args.algorithm_name == "iterative"


def test_cli_data_get_top_configurations_parser():
    """Test data get-top-configurations parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "get-top-configurations",
            "--project-name",
            "test_project",
            "--wavefunction-filename",
            "wf.wavefunction.json",
            "--max-determinants",
            "10",
        ]
    )

    assert args.command == "data"
    assert args.subcommand == "get-top-configurations"
    assert args.project_name == "test_project"
    assert args.wavefunction_filename == "wf.wavefunction.json"
    assert args.max_determinants == 10


def test_cli_algorithm_sparse_ci_parser():
    """Test algorithm sparse-ci parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "sparse-ci",
            "--project-name",
            "test_project",
            "--hamiltonian-filename",
            "ham.hamiltonian.json",
            "--configurations-json",
            '["22000000", "20200000"]',
            "--out-wavefunction-filename",
            "sparse.wavefunction.json",
            "--algorithm-name",
            "macis_pmc",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "sparse-ci"
    assert args.configurations_json == '["22000000", "20200000"]'
    assert args.algorithm_name == "macis_pmc"


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_algorithm_mcscf_with_config(tmp_path):
    """Test mcscf command with --config file."""
    parser = create_parser()

    # Create config file
    config_data = {
        "ham_constructor": {"algorithm_name": None, "settings": {}},
        "mc_calculator": {"algorithm_name": None, "settings": {}},
        "mcscf": {"settings": {}},
    }
    config_file = tmp_path / "mcscf.json"
    config_file.write_text(json.dumps(config_data))

    args = parser.parse_args(
        [
            "run",
            "mcscf",
            "--project-name",
            "test_project",
            "--orbitals-filename",
            "h2.orbitals.json",
            "--out-wavefunction-filename",
            "h2.wavefunction.json",
            "--n-active-alpha-electrons",
            "1",
            "--config",
            str(config_file),
            "--set",
            "mc_calculator.settings.calculate_one_rdm=true",
            "--set",
            "mc_calculator.settings.calculate_two_rdm=true",
            "--set",
            "mc_calculator.settings.calculate_mutual_information=true",
        ]
    )

    assert args.command == "run"
    assert args.subcommand == "mcscf"
    assert args.config == str(config_file)
    assert args.set == [
        "mc_calculator.settings.calculate_one_rdm=true",
        "mc_calculator.settings.calculate_two_rdm=true",
        "mc_calculator.settings.calculate_mutual_information=true",
    ]


# ==================== Utils Group Tests ====================


def test_cli_utils_list_projects(temp_project_dir, capsys):
    """Test utils list-projects command."""
    # Create some project dirs
    (temp_project_dir / "proj_a").mkdir()
    (temp_project_dir / "proj_b").mkdir()

    sys.argv = ["qc", "project", "list"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert "proj_a" in result["result"]
    assert "proj_b" in result["result"]


def test_cli_utils_create_project(temp_project_dir, capsys):
    """Test utils create-project command."""
    sys.argv = ["qc", "project", "create", "--project-name", "new_project"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert (temp_project_dir / "new_project").exists()


@pytest.mark.parametrize("project_name", ["../outside", r"..\outside", r"C:\outside"])
@pytest.mark.parametrize("command", ["create", "files"])
def test_cli_utils_rejects_non_component_project_names(temp_project_dir, capsys, command, project_name):
    sys.argv = ["qc", "project", command, "--project-name", project_name]

    with contextlib.suppress(SystemExit):
        main()

    result = json.loads(capsys.readouterr().out)
    assert result["success"] is False
    assert "single path component" in result["error"]
    assert not (temp_project_dir.parent / "outside").exists()


@pytest.mark.usefixtures("temp_project_dir", "h2_structure_file")
def test_cli_utils_list_files(capsys):
    """Test utils list-files command."""
    sys.argv = ["qc", "project", "files", "--project-name", "test_project"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert len(result["result"]) >= 1
    assert any(f["filename"] == "h2.structure.json" for f in result["result"])


def test_cli_utils_convert_energy(capsys):
    """Test utils convert-energy command."""
    sys.argv = [
        "qc",
        "util",
        "convert-energy",
        "--value",
        "1.0",
        "--from-unit",
        "hartree",
        "--to-unit",
        "ev",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert result["output"]["unit"] == "ev"
    assert abs(result["output"]["value"] - 27.2114) < 0.01


def test_cli_utils_convert_coordinates(capsys):
    """Test utils convert-coordinates command."""
    sys.argv = [
        "qc",
        "util",
        "convert-coordinates",
        "--coordinates",
        "[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]",
        "--to-angstrom",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert result["unit"] == "angstrom"


# ==================== Data Group Tests ====================


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_data_summary(h2_structure_file, capsys):
    """Test data summary command."""
    sys.argv = [
        "qc",
        "data",
        "summary",
        "--project-name",
        "test_project",
        "--filename",
        h2_structure_file,
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert result["type"] == "Structure"


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_data_get_energy(h2_wavefunction_file, capsys):
    """Test data get-energy command."""
    sys.argv = [
        "qc",
        "data",
        "get-energy",
        "--project-name",
        "test_project",
        "--filename",
        h2_wavefunction_file,
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert result["source"] == "Wavefunction"


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_data_get_structure_xyz(h2_structure_file, capsys):
    """Test data get-structure-xyz command."""
    sys.argv = [
        "qc",
        "data",
        "get-structure-xyz",
        "--project-name",
        "test_project",
        "--filename",
        h2_structure_file,
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert "xyz" in result
    assert "H" in result["xyz"]


def test_cli_model_hamiltonian_parser():
    """Test model-hamiltonian parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "model-hamiltonian",
            "--project-name",
            "test_project",
            "--model",
            "hubbard",
            "--lattice-type",
            "chain",
            "--lattice-params",
            '{"n": 4, "periodic": false}',
            "--out-hamiltonian-filename",
            "hubbard.hamiltonian.json",
            "--epsilon",
            "0",
            "--t",
            "1.0",
            "--U",
            "4.0",
        ]
    )

    assert args.subcommand == "model-hamiltonian"
    assert args.model == "hubbard"
    assert args.lattice_type == "chain"
    assert args.t == "1.0"
    assert args.U == "4.0"


def test_cli_spin_model_parser():
    """Test spin-model parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "spin-model",
            "--project-name",
            "test_project",
            "--model",
            "ising",
            "--lattice-type",
            "square",
            "--lattice-params",
            '{"nx": 2, "ny": 2}',
            "--out-qubit-hamiltonian-filename",
            "ising.qubit_hamiltonian.json",
            "--j",
            "1.0",
            "--h",
            "0.5",
        ]
    )

    assert args.subcommand == "spin-model"
    assert args.model == "ising"
    assert args.lattice_type == "square"
    assert args.j == "1.0"
    assert args.h == "0.5"
