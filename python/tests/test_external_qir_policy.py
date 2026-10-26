"""Execution policy for directly supplied and restored QIR."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import pytest

import qdk_chemistry.algorithms.circuit_executor.qdk as executor_module
from qdk_chemistry.algorithms.circuit_executor.qdk import QdkFullStateSimulator
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData


@pytest.fixture(params=["direct", "json", "hdf5"])
def external_circuit(request, tmp_path):
    """External execution cannot be authorized by fields in serialized data."""
    circuit = Circuit(qir="external QIR")
    if request.param == "direct":
        return circuit
    if request.param == "json":
        payload = circuit.to_json()
        payload["_qir_requires_opt_in"] = False
        return Circuit.from_json(payload)
    with h5py.File(tmp_path / "external.circuit.h5", "w") as file:
        circuit.to_hdf5(file)
        file.attrs["_qir_requires_opt_in"] = False
        return Circuit.from_hdf5(file)


def test_external_qir_is_rejected_before_execution(external_circuit, monkeypatch):
    """The default policy rejects external QIR before calling the native engine."""
    run_qir = Mock()
    monkeypatch.setattr(executor_module, "run_qir", run_qir)

    with pytest.raises(ValueError, match="allow_external_qir=True"):
        QdkFullStateSimulator().run(external_circuit, shots=1)

    run_qir.assert_not_called()


def test_external_qir_accepts_explicit_opt_in(external_circuit, monkeypatch):
    """A trusted caller can explicitly authorize external QIR execution."""
    run_qir = Mock(return_value=[["Zero"]])
    monkeypatch.setattr(executor_module, "run_qir", run_qir)
    executor = QdkFullStateSimulator()
    executor.settings().set("allow_external_qir", True)

    result = executor.run(external_circuit, shots=1)

    run_qir.assert_called_once_with("external QIR", shots=1, noise=None, seed=42, type="cpu")
    assert result.bitstring_counts == {"0": 1}


def test_cached_factory_compilation_does_not_require_opt_in(monkeypatch):
    """Compiling a factory must not make subsequent executions external."""
    compile_qir = Mock(return_value="compiler-generated QIR")
    program = Mock()
    program._qdk_context = SimpleNamespace(compile=compile_qir)
    circuit = Circuit(qsharp_factory=QsharpFactoryData(program=program, parameter={}))
    run_qir = Mock(return_value=[["Zero"]])
    monkeypatch.setattr(executor_module, "run_qir", run_qir)
    executor = QdkFullStateSimulator()

    assert executor.run(circuit, shots=1).bitstring_counts == {"0": 1}
    assert executor.run(circuit, shots=1).bitstring_counts == {"0": 1}

    compile_qir.assert_called_once_with(program)
    assert run_qir.call_count == 2
