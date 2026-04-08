"""Qiskit Aer Simulator circuit executor for QDK/Chemistry.

This module provides a CircuitExecutor implementation that uses Qiskit Aer Simulator
to execute quantum circuits. It accepts QDK/Chemistry Circuit and QuantumErrorProfile
data classes and returns measurement bitstring results via CircuitExecutorData.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qiskit import transpile
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import qdk_chemistry.plugins.qiskit as qiskit_plugin
from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor

if qiskit_plugin.QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME:
    import qiskit_ibm_runtime.fake_provider

import qdk_chemistry.plugins.qiskit._interop.transpiler as _transpiler_module
from qdk_chemistry.data import (
    Circuit,
    CircuitExecutorData,
    QuantumErrorProfile,
    Settings,
)
from qdk_chemistry.plugins.qiskit._interop.noise_model import (
    get_noise_model_from_profile,
)
from qdk_chemistry.plugins.qiskit._interop.transpiler import *  # noqa: F403
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitAerSimulator", "QiskitAerSimulatorSettings"]


def _resolve_passes(names: list[str]) -> list:
    """Resolve pass names to instances using the transpiler module's __all__."""
    passes = []
    for name in names:
        if name not in _transpiler_module.__all__:
            raise ValueError(f"Unknown pass '{name}'. Available passes: {_transpiler_module.__all__}")
        passes.append(getattr(_transpiler_module, name)())
    return passes


class QiskitAerSimulatorSettings(Settings):
    """Settings for the Qiskit Aer Simulator circuit executor."""

    def __init__(self) -> None:
        """Initialize Qiskit Aer Simulator settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("seed", "int", 42)
        self._set_default("method", "string", "statevector")
        self._set_default("transpile_optimization_level", "int", 0)


class QiskitAerSimulator(CircuitExecutor):
    """Qiskit Aer Simulator circuit executor implementation."""

    def __init__(
        self,
        method: str = "statevector",
        seed: int = 42,
        transpile_optimization_level: int = 0,
    ) -> None:
        """Initialize the Qiskit Aer Simulator circuit executor.

        Args:
            method: The simulation method to use.
            seed: The random seed for simulation reproducibility.
            transpile_optimization_level: The optimization level for transpilation.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QiskitAerSimulatorSettings()
        self._settings.set("seed", seed)
        self._settings.set("method", method)
        self._settings.set("transpile_optimization_level", transpile_optimization_level)

    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
        device_backend_name: str | None = None,
        pre_transpilation_passes: list[str] | None = None,
        post_transpilation_passes: list[str] | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit using the Qiskit Aer Simulator.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: Optional noise profile to apply during execution.
            device_backend_name: Optional name of a fake device backend to use for noise modeling.
            pre_transpilation_passes: Optional list of pass names to apply before transpilation.
            post_transpilation_passes: Optional list of pass names to apply after transpilation.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """
        Logger.trace_entering()
        meas_circuit = circuit.get_qiskit_circuit()
        Logger.debug("Qiskit QuantumCircuit loaded.")

        if noise is not None and device_backend_name is not None:
            raise ValueError("Cannot specify both a noise model and a device backend. Please choose one or the other.")

        opt_level = self._settings.get("transpile_optimization_level")

        if pre_transpilation_passes:
            meas_circuit = PassManager(
                _resolve_passes(pre_transpilation_passes),
            ).run(meas_circuit)

        if device_backend_name is not None:
            if not qiskit_plugin.QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME:
                raise ImportError(
                    "The fake_provider module from qiskit_ibm_runtime is required for device backend simulation. "
                    "Install it with: pip install qiskit-ibm-runtime"
                )

            provider = qiskit_ibm_runtime.fake_provider.FakeProviderForBackendV2()
            try:
                device_backend = provider.backend(device_backend_name)
            except QiskitBackendNotFoundError:
                available = sorted(b.name for b in provider.backends())
                available_backends = ", ".join(available)
                raise ValueError(
                    f"Unknown device backend '{device_backend_name}'. Available backends: {available_backends}"
                ) from None

            backend = AerSimulator.from_backend(device_backend)
            backend.set_options(
                method=self._settings.get("method"),
                seed_simulator=self._settings.get("seed"),
            )

            transpiled_circuit = transpile(
                meas_circuit,
                backend=device_backend,
                optimization_level=opt_level,
            )

        else:
            noise_model = get_noise_model_from_profile(noise) if noise else None
            backend = AerSimulator(
                method=self._settings.get("method"),
                seed_simulator=self._settings.get("seed"),
                noise_model=noise_model,
            )
            if noise_model:
                transpiled_circuit = transpile(
                    meas_circuit,
                    basis_gates=noise_model.basis_gates,
                    optimization_level=opt_level,
                )
            else:
                # Use qiskit_aer NoiseModel() default basis gates if no noise model is provided
                transpiled_circuit = transpile(
                    meas_circuit,
                    basis_gates=NoiseModel().basis_gates,
                    optimization_level=opt_level,
                )

        if post_transpilation_passes:
            transpiled_circuit = PassManager(
                _resolve_passes(post_transpilation_passes),
            ).run(transpiled_circuit)

        raw_results = backend.run(transpiled_circuit, shots=shots).result()
        counts = raw_results.get_counts()
        Logger.debug(f"Measurement results obtained: {counts}")
        return CircuitExecutorData(
            bitstring_counts=counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
        )

    def name(self) -> str:
        """Return the algorithm name as qiskit_aer_simulator."""
        return "qiskit_aer_simulator"
