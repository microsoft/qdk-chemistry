"""geomeTRIC-backed geometry optimizer."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from geometric.engine import Engine
from geometric.nifty import bohr2ang

from qdk_chemistry.algorithms import GeometryOptimizer, GeometryOptimizerSettings
from qdk_chemistry.data import BasisSet, Orbitals, Structure, Wavefunction
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from geometric.molecule import Molecule

    from qdk_chemistry.algorithms import NuclearDerivativeCalculator
    from qdk_chemistry.data import NuclearHessian

__all__ = [
    "GEOMETRIC_OPTIMIZER_ALGORITHMS",
    "GeometricOptimizer",
    "GeometricOptimizerSettings",
]


GEOMETRIC_OPTIMIZER_ALGORITHMS = ("tric", "tric-p", "dlc", "hdlc", "prim", "cart")

_GeometryOptimizationInput: TypeAlias = Orbitals | BasisSet | Wavefunction | str
_GEOMETRIC_LOGGING_LOCK = RLock()


class GeometricOptimizerSettings(GeometryOptimizerSettings):
    """Settings for the geomeTRIC geometry optimizer.

    By default, geomeTRIC requires the energy, RMS and maximum gradient,
    and RMS and maximum displacement criteria to be satisfied together.
    """

    def __init__(self):
        """Initialize geomeTRIC optimizer defaults."""
        super().__init__()
        self._set_default(
            "transition_state", "bool", False, "Run transition-state optimization instead of minimization."
        )
        self._set_default(
            "algorithm",
            "string",
            "tric",
            "geomeTRIC optimizer algorithm.",
            limit=list(GEOMETRIC_OPTIMIZER_ALGORITHMS),
        )
        self._set_default(
            "convergence_energy",
            "double",
            1.0e-6,
            "Energy-change threshold in Hartree required for convergence.",
            limit=(0.0, 1.0),
        )
        self._set_default(
            "convergence_rms_gradient",
            "double",
            3.0e-4,
            "RMS gradient threshold in Hartree/Bohr required for convergence.",
            limit=(0.0, 1.0),
        )
        self._set_default(
            "convergence_max_gradient",
            "double",
            4.5e-4,
            "Maximum gradient threshold in Hartree/Bohr required for convergence.",
            limit=(0.0, 1.0),
        )
        self._set_default(
            "convergence_rms_displacement",
            "double",
            1.2e-3,
            "RMS displacement threshold in Angstrom required for convergence.",
            limit=(0.0, 1.0),
        )
        self._set_default(
            "convergence_max_displacement",
            "double",
            1.8e-3,
            "Maximum displacement threshold in Angstrom required for convergence.",
            limit=(0.0, 1.0),
        )
        self._set_default("print_level", "int", 0, "geomeTRIC output verbosity level.")


class _QdkDerivativeEngine(Engine):
    """geomeTRIC engine that evaluates QDK/Chemistry nuclear derivatives."""

    def __init__(
        self,
        structure: Structure,
        charge: int,
        spin_multiplicity: int,
        optimizer_input: _GeometryOptimizationInput,
        n_inactive_orbitals: int,
        derivative_calculator: NuclearDerivativeCalculator | None,
        molecule: Molecule,
    ):
        super().__init__(molecule)
        self._structure = structure
        self._charge = charge
        self._spin_multiplicity = spin_multiplicity
        self._optimizer_input = optimizer_input
        self._n_inactive_orbitals = n_inactive_orbitals
        self._derivative_calculator = derivative_calculator
        self._last_energy: float | None = None
        self._last_structure = structure
        self._last_wavefunction: Wavefunction | None = None

    def _structure_from_bohr_coordinates(self, coordinates: np.ndarray) -> Structure:
        """Create a QDK/Chemistry structure from Bohr coordinates."""
        matrix = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        return Structure(
            matrix, self._structure.get_elements(), self._structure.get_masses(), self._structure.get_nuclear_charges()
        )

    def last_coordinates(self) -> np.ndarray:
        """Return the most recently evaluated coordinates."""
        return np.asarray(self._last_structure.get_coordinates(), dtype=float)

    def cached_result(self, coordinates: np.ndarray) -> tuple[float, Wavefunction | None] | None:
        """Return the cached energy and wavefunction when coordinates match."""
        matrix = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        if self._last_energy is None or not np.allclose(matrix, self.last_coordinates(), rtol=0.0, atol=1.0e-12):
            return None
        return self._last_energy, self._last_wavefunction

    def calc_new(self, coordinates: np.ndarray, dirname: str) -> dict[str, np.ndarray | float]:  # noqa: ARG002
        """Evaluate energy and gradients for geomeTRIC."""
        if self._derivative_calculator is None:
            raise RuntimeError("A nuclear derivative calculator is required to evaluate coordinates")
        structure = self._structure_from_bohr_coordinates(coordinates)
        energy, gradients, _hessian, wavefunction = self._derivative_calculator.run(
            structure,
            self._charge,
            self._spin_multiplicity,
            self._optimizer_input,
            self._n_inactive_orbitals,
        )
        self._last_energy = energy
        self._last_structure = structure
        self._last_wavefunction = wavefunction
        gradient = np.asarray(gradients.get_values(), dtype=float)
        return {"energy": energy, "gradient": gradient}


class GeometricOptimizer(GeometryOptimizer):
    """Geometry optimizer implemented with the geomeTRIC Python library."""

    def __init__(self):
        """Initialize the geomeTRIC optimizer."""
        Logger.trace_entering()
        super().__init__()
        self._settings = GeometricOptimizerSettings()

    def name(self) -> str:
        """Return the implementation name."""
        return "geometric"

    def _run_impl(
        self,
        structure: Structure,
        charge: int,
        spin_multiplicity: int,
        optimizer_input: _GeometryOptimizationInput,
        n_inactive_orbitals: int = 0,
    ) -> tuple[float, Structure, NuclearHessian | None, Wavefunction | None]:
        """Optimize a molecular structure using geomeTRIC."""
        Logger.trace_entering()
        from geometric.molecule import Molecule  # noqa: PLC0415
        from geometric.optimize import run_optimizer  # noqa: PLC0415

        molecule = Molecule()
        molecule.elem = structure.get_atomic_symbols()
        molecule.xyzs = [np.asarray(structure.get_coordinates(), dtype=float) * bohr2ang]

        derivative_calculator = self._create_nested("derivative_calculator")
        derivative_calculator.settings().set("compute_hessian", False)
        engine = _QdkDerivativeEngine(
            structure,
            charge,
            spin_multiplicity,
            optimizer_input,
            n_inactive_orbitals,
            derivative_calculator,
            molecule,
        )

        params = self._geometric_options()
        params.update({"customengine": engine, "input": None})

        with TemporaryDirectory(prefix="qdk-chemistry-geometric-") as tmpdir:
            prefix = Path(tmpdir) / "qdk-chemistry"
            with _preserve_root_logging():
                result = run_optimizer(**params, prefix=str(prefix), dirname=tmpdir)

        optimized_coordinates = np.asarray(_extract_coordinates(result, engine), dtype=float).reshape((-1, 3))
        optimized_structure = Structure(
            optimized_coordinates, structure.get_elements(), structure.get_masses(), structure.get_nuclear_charges()
        )

        cached_result = engine.cached_result(optimized_coordinates)
        if not self._settings["compute_hessian"] and cached_result is not None:
            final_energy, wavefunction = cached_result
            return final_energy, optimized_structure, None, wavefunction

        final_calculator = self._create_nested("derivative_calculator")
        final_calculator.settings().set("compute_hessian", self._settings["compute_hessian"])
        final_energy, _gradients, hessian, wavefunction = final_calculator.run(
            optimized_structure,
            charge,
            spin_multiplicity,
            optimizer_input,
            n_inactive_orbitals,
        )
        if not self._settings["compute_hessian"]:
            hessian = None

        return final_energy, optimized_structure, hessian, wavefunction

    def _geometric_options(self) -> dict[str, Any]:
        return {
            "transition": self._settings["transition_state"],
            "coordsys": self._settings["algorithm"],
            "maxiter": self._settings["max_iterations"],
            "convergence_energy": self._settings["convergence_energy"],
            "convergence_grms": self._settings["convergence_rms_gradient"],
            "convergence_gmax": self._settings["convergence_max_gradient"],
            "convergence_drms": self._settings["convergence_rms_displacement"],
            "convergence_dmax": self._settings["convergence_max_displacement"],
            "verbose": self._settings["print_level"],
        }


@contextmanager
def _preserve_root_logging() -> Iterator[None]:
    """Restore root logging after geomeTRIC replaces its configuration."""
    with _GEOMETRIC_LOGGING_LOCK:
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        # fileConfig closes every handler registered with logging, including
        # handlers that belong to the host application.
        handler_list = logging._handlerList  # type: ignore[attr-defined]  # noqa: SLF001
        logging_lock = logging._lock  # type: ignore[attr-defined]  # noqa: SLF001
        original_handler_ids = {id(handler) for handler in original_handlers}
        with logging_lock:
            original_handler_refs = [
                handler_ref
                for handler_ref in handler_list
                if handler_ref() is not None and id(handler_ref()) in original_handler_ids
            ]
            handler_list[:] = [
                handler_ref
                for handler_ref in handler_list
                if handler_ref() is None or id(handler_ref()) not in original_handler_ids
            ]

        try:
            yield
        finally:
            geometric_handlers = root_logger.handlers[:]
            for handler in geometric_handlers:
                root_logger.removeHandler(handler)
                if handler not in original_handlers:
                    handler.close()
            for handler in original_handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(original_level)

            with logging_lock:
                registered_handler_ids = {
                    id(handler_ref()) for handler_ref in handler_list if handler_ref() is not None
                }
                handler_list.extend(
                    handler_ref
                    for handler_ref in original_handler_refs
                    if handler_ref() is not None and id(handler_ref()) not in registered_handler_ids
                )
                for handler in original_handlers:
                    if handler.name is not None:
                        # Trigger re-registration of named handlers in logging's global handler registry.
                        handler.name = handler.name


def _extract_coordinates(result: Any, engine: _QdkDerivativeEngine | None) -> np.ndarray:
    """Extract final coordinates from geomeTRIC in Bohr."""
    if isinstance(result, np.ndarray):
        return result / bohr2ang
    if hasattr(result, "xyzs") and result.xyzs:
        return np.asarray(result.xyzs[-1], dtype=float) / bohr2ang
    if isinstance(result, dict):
        for key in ("coords", "coordinates", "xyz", "xyzs"):
            if key in result:
                value = result[key]
                if key == "xyzs" and value:
                    value = value[-1]
                return np.asarray(value, dtype=float) / bohr2ang
    if engine is None:
        raise ValueError("geomeTRIC result did not contain final coordinates, and no engine was provided")
    return engine.last_coordinates()
