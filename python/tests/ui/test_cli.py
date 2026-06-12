"""Unit tests for qdk_chemistry.ui CLI functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import contextlib
import json
import subprocess
import sys

import pytest

from qdk_chemistry import algorithms, data

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
    config.projects_dir = tmp_path
    yield tmp_path
    config.projects_dir = original_projects_dir


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
    assert parser.prog == "qdk_chem_cli"

    # Test that flat command parsing works
    args = parser.parse_args(["defaults", "--algorithm-type", "scf_solver"])
    assert args.command == "defaults"
    assert args.algorithm_type == "scf_solver"


# ==================== Algorithm Group Tests ====================


def test_cli_algorithm_defaults(capsys):
    """Test algorithm defaults command."""
    sys.argv = ["qdk_chem_cli", "defaults", "--algorithm-type", "scf_solver"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert isinstance(result["result"], str)


def test_cli_algorithm_defaults_settings(capsys):
    """Test algorithm defaults with settings."""
    sys.argv = ["qdk_chem_cli", "defaults", "--algorithm-type", "scf_solver"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True


def test_cli_algorithm_defaults_config_template(capsys):
    """Test generating a config template for compound algorithms."""
    sys.argv = ["qdk_chem_cli", "defaults", "--type", "mcscf"]

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
    sys.argv = ["qdk_chem_cli", "list-algorithms"]

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
        "qdk_chem_cli",
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


def test_cli_algorithm_scf(temp_project_dir, h2_structure_file, capsys):
    """Test algorithm scf command."""
    sys.argv = [
        "qdk_chem_cli",
        "run-scf",
        "--project-name",
        "test_project",
        "--structure-filename",
        h2_structure_file,
        "--out-wavefunction-filename",
        "scf_output.wavefunction.json",
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
    assert result["success"] is True
    assert isinstance(result["result"], list)
    assert len(result["result"]) == 2  # (energy, filename)

    # Verify file was created
    project_path = temp_project_dir / "test_project"
    assert (project_path / "scf_output.wavefunction.json").exists()


def test_cli_algorithm_hamiltonian(temp_project_dir, h2_wavefunction_file, capsys):
    """Test algorithm hamiltonian command."""
    project_path = temp_project_dir / "test_project"

    # First, extract orbitals from wavefunction
    wf = data.Wavefunction.from_json_file(str(project_path / h2_wavefunction_file))
    orbitals = wf.get_orbitals()
    orbitals_file = project_path / "h2.orbitals.json"
    orbitals.to_json_file(str(orbitals_file))

    sys.argv = [
        "qdk_chem_cli",
        "run-hamiltonian",
        "--project-name",
        "test_project",
        "--orbitals-filename",
        "h2.orbitals.json",
        "--out-hamiltonian-filename",
        "h2.hamiltonian.json",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True

    # Verify file was created
    assert (project_path / "h2.hamiltonian.json").exists()


def test_cli_algorithm_active_space(temp_project_dir, h2_wavefunction_file, capsys):
    """Test algorithm active-space command."""
    sys.argv = [
        "qdk_chem_cli",
        "run-active-space",
        "--project-name",
        "test_project",
        "--wavefunction-filename",
        h2_wavefunction_file,
        "--out-wavefunction-filename",
        "h2_active.wavefunction.json",
        "--charge",
        "0",
        "--algorithm-name",
        "qdk_valence",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True

    # Verify file was created
    project_path = temp_project_dir / "test_project"
    assert (project_path / "h2_active.wavefunction.json").exists()


def test_cli_data_get_orbitals(temp_project_dir, h2_wavefunction_file, capsys):
    """Test data get-orbitals command."""
    sys.argv = [
        "qdk_chem_cli",
        "get-orbitals",
        "--project-name",
        "test_project",
        "--input-filename",
        h2_wavefunction_file,
        "--out-orbitals-filename",
        "extracted.orbitals.json",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True

    # Verify file was created
    project_path = temp_project_dir / "test_project"
    assert (project_path / "extracted.orbitals.json").exists()


def test_cli_data_get_ansatz(temp_project_dir, h2_wavefunction_file, capsys):
    """Test data get-ansatz command."""
    project_path = temp_project_dir / "test_project"

    # Create hamiltonian file
    wf = data.Wavefunction.from_json_file(str(project_path / h2_wavefunction_file))
    orbitals = wf.get_orbitals()
    ham_constructor = algorithms.create("hamiltonian_constructor")
    hamiltonian = ham_constructor.run(orbitals)
    ham_file = project_path / "h2.hamiltonian.json"
    hamiltonian.to_json_file(str(ham_file))

    sys.argv = [
        "qdk_chem_cli",
        "get-ansatz",
        "--project-name",
        "test_project",
        "--wavefunction-filename",
        h2_wavefunction_file,
        "--hamiltonian-filename",
        "h2.hamiltonian.json",
        "--out-ansatz-filename",
        "h2.ansatz.json",
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True

    # Verify file was created
    assert (project_path / "h2.ansatz.json").exists()


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_algorithm_scf_with_settings(h2_structure_file, capsys):
    """Test algorithm scf with JSON settings."""
    settings_json = '{"max_iterations": 100, "convergence_threshold": 1e-8}'

    sys.argv = [
        "qdk_chem_cli",
        "run-scf",
        "--project-name",
        "test_project",
        "--structure-filename",
        h2_structure_file,
        "--out-wavefunction-filename",
        "scf_custom.wavefunction.json",
        "--charge",
        "0",
        "--spin-multiplicity",
        "1",
        "--basis-set",
        "sto-3g",
        "--settings",
        settings_json,
    ]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True


@pytest.mark.usefixtures("temp_project_dir")
def test_cli_algorithm_error_handling(capsys):
    """Test error handling for invalid inputs."""
    sys.argv = [
        "qdk_chem_cli",
        "run-scf",
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
    sys.argv = ["qdk_chem_cli"]

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
        [sys.executable, "-m", "qdk_chemistry.ui.cli", "defaults", "--algorithm-type", "scf_solver"],
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
        [sys.executable, "-m", "qdk_chemistry.ui.cli", "defaults", "--algorithm-type", "scf_solver"],
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
        "qdk_chem_cli",
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
            "run-energy",
            "--project-name",
            "test_project",
            "--circuit-filename",
            "circuit.circuit.json",
            "--qubit-hamiltonian-filenames",
            "qh1.qubit_ham.h5",
            "qh2.qubit_ham.h5",
            "--out-energy-result-filename",
            "energy.result.json",
            "--out-measurement-data-filename",
            "measurement.data.json",
            "--total-shots",
            "1000",
        ]
    )

    assert args.command == "run-energy"
    assert args.qubit_hamiltonian_filenames == ["qh1.qubit_ham.h5", "qh2.qubit_ham.h5"]


def test_cli_algorithm_qpe_config_evolution_parser():
    """Test qpe-config-evolution parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe-config-evolution",
            "--project-name",
            "test_project",
            "--out-config-filename",
            "teb.config.json",
            "--algorithm-name",
            "default",
            "--settings",
            '{"key": "value"}',
        ]
    )

    assert args.command == "run-qpe-config-evolution"
    assert args.project_name == "test_project"
    assert args.out_config_filename == "teb.config.json"
    assert args.algorithm_name == "default"
    assert args.settings == {"key": "value"}


def test_cli_algorithm_qpe_config_mapper_parser():
    """Test qpe-config-mapper parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe-config-mapper",
            "--project-name",
            "test_project",
        ]
    )

    assert args.command == "run-qpe-config-mapper"
    assert args.project_name == "test_project"


def test_cli_algorithm_qpe_config_executor_parser():
    """Test qpe-config-executor parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe-config-executor",
            "--project-name",
            "test_project",
        ]
    )

    assert args.command == "run-qpe-config-executor"
    assert args.project_name == "test_project"


def test_cli_algorithm_qpe_build_evolution_parser():
    """Test qpe-build-evolution parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe-build-evolution",
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

    assert args.command == "run-qpe-build-evolution"
    assert args.project_name == "test_project"
    assert args.evolution_time == 0.1
    assert args.algorithm_name == "trotter"


def test_cli_algorithm_qpe_map_circuit_parser():
    """Test qpe-map-circuit parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe-map-circuit",
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

    assert args.command == "run-qpe-map-circuit"
    assert args.project_name == "test_project"
    assert args.power == 4


def test_cli_algorithm_qpe_execute_parser():
    """Test qpe-execute parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe-execute",
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

    assert args.command == "run-qpe-execute"
    assert args.project_name == "test_project"
    assert args.shots == 1000


def test_cli_algorithm_qpe_parser():
    """Test qpe (full pipeline) parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-qpe",
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

    assert args.command == "run-qpe"
    assert args.project_name == "test_project"
    assert args.algorithm_name == "iterative"


def test_cli_data_get_top_configurations_parser():
    """Test data get-top-configurations parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "get-top-configurations",
            "--project-name",
            "test_project",
            "--wavefunction-filename",
            "wf.wavefunction.json",
            "--max-determinants",
            "10",
        ]
    )

    assert args.command == "get-top-configurations"
    assert args.project_name == "test_project"
    assert args.wavefunction_filename == "wf.wavefunction.json"
    assert args.max_determinants == 10


def test_cli_algorithm_sparse_ci_parser():
    """Test algorithm sparse-ci parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-sparse-ci",
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

    assert args.command == "run-sparse-ci"
    assert args.configurations_json == '["22000000", "20200000"]'
    assert args.algorithm_name == "macis_pmc"


def test_cli_algorithm_filter_pauli_parser():
    """Test algorithm filter-pauli parser."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run-filter-pauli",
            "--project-name",
            "test_project",
            "--qubit-hamiltonian-filename",
            "ham.qubithamiltonian.h5",
            "--wavefunction-filename",
            "wf.wavefunction.json",
            "--out-qubit-hamiltonians-prefix",
            "grouped_ham",
            "--trimming-tolerance",
            "1e-6",
        ]
    )

    assert args.command == "run-filter-pauli"
    assert args.project_name == "test_project"
    assert args.qubit_hamiltonian_filename == "ham.qubithamiltonian.h5"
    assert args.wavefunction_filename == "wf.wavefunction.json"
    assert args.out_qubit_hamiltonians_prefix == "grouped_ham"
    assert args.trimming_tolerance == 1e-6


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
            "run-mcscf",
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

    assert args.command == "run-mcscf"
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

    sys.argv = ["qdk_chem_cli", "list-projects"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert "proj_a" in result["result"]
    assert "proj_b" in result["result"]


def test_cli_utils_create_project(temp_project_dir, capsys):
    """Test utils create-project command."""
    sys.argv = ["qdk_chem_cli", "create-project", "--project-name", "new_project"]

    with contextlib.suppress(SystemExit):
        main()

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert (temp_project_dir / "new_project").exists()


@pytest.mark.usefixtures("temp_project_dir", "h2_structure_file")
def test_cli_utils_list_files(capsys):
    """Test utils list-files command."""
    sys.argv = ["qdk_chem_cli", "list-files", "--project-name", "test_project"]

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
        "qdk_chem_cli",
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
        "qdk_chem_cli",
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
        "qdk_chem_cli",
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
        "qdk_chem_cli",
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
        "qdk_chem_cli",
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
