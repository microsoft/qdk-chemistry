"""geomeTRIC-backed geometry optimizer."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from geometric.engine import Engine

from qdk_chemistry.algorithms import GeometryOptimizer
from qdk_chemistry.data import AlgorithmRef, NuclearHessian, Settings, Structure
from qdk_chemistry.utils import Logger

__all__ = [
    "GEOMETRIC_OPTIMIZER_ALGORITHMS",
    "GeometricOptimizer",
    "GeometricOptimizerSettings",
]


GEOMETRIC_OPTIMIZER_ALGORITHMS = {
    "tric": "tric",
    "tric_p": "tric-p",
    "dlc": "dlc",
    "hdlc": "hdlc",
    "prim": "prim",
    "cartesian": "cart",
}


class GeometricOptimizerSettings(Settings):
    """Settings for the geomeTRIC geometry optimizer."""

    def __init__(self, *, transition_state: bool = False, coordinate_system: str = "tric"):
        """Initialize geomeTRIC optimizer defaults."""
        super().__init__()
        self._set_default(
            "derivative_calculator",
            "algorithm_ref",
            AlgorithmRef("nuclear_derivative_calculator", "finite_difference"),
            "Nuclear derivative calculator used to evaluate energies and gradients.",
        )
        self._set_default(
            "transition_state", "bool", transition_state, "Run transition-state optimization instead of minimization."
        )
        self._set_default(
            "coordinate_system",
            "string",
            coordinate_system,
            "geomeTRIC coordinate-system optimizer algorithm.",
            limit=list(GEOMETRIC_OPTIMIZER_ALGORITHMS.values()),
        )
        self._set_default("max_iterations", "int", 300, "Maximum number of geometry optimization steps.")
        self._set_default("convergence_energy", "double", 1.0e-6, "Energy convergence threshold.")
        self._set_default("convergence_gradient", "double", 3.0e-4, "Gradient convergence threshold.")
        self._set_default("convergence_displacement", "double", 1.2e-3, "Displacement convergence threshold.")
        self._set_default("compute_hessian", "bool", False, "Compute a Hessian at the optimized geometry.")
        self._set_default("print_level", "int", 0, "geomeTRIC output verbosity level.")


class _QdkDerivativeEngine(Engine):
    """geomeTRIC engine that evaluates QDK/Chemistry nuclear derivatives."""

    def __init__(
        self,
        structure: Structure,
        charge: int,
        spin_multiplicity: int,
        seed_or_basis: Any,
        derivative_calculator: Any,
        molecule: Any,
    ):
        super().__init__(molecule)
        self._structure = structure
        self._charge = charge
        self._spin_multiplicity = spin_multiplicity
        self._seed_or_basis = seed_or_basis
        self._derivative_calculator = derivative_calculator
        self._last_energy = None
        self._last_structure = structure
        self._last_wavefunction = None

    def structure_from_coordinates(self, coordinates: np.ndarray) -> Structure:
        """Create a QDK/Chemistry structure with updated coordinates."""
        matrix = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        return Structure(
            matrix, self._structure.get_elements(), self._structure.get_masses(), self._structure.get_nuclear_charges()
        )

    def last_coordinates(self) -> np.ndarray:
        """Return the most recently evaluated coordinates."""
        return np.asarray(self._last_structure.get_coordinates(), dtype=float)

    def calc_new(self, coordinates: np.ndarray, dirname: str) -> dict[str, np.ndarray | float]:  # noqa: ARG002
        """Evaluate energy and gradients for geomeTRIC."""
        structure = self.structure_from_coordinates(coordinates)
        energy, gradients, _hessian, wavefunction = self._derivative_calculator.run(
            structure, self._charge, self._spin_multiplicity, self._seed_or_basis
        )
        self._last_energy = energy
        self._last_structure = structure
        self._last_wavefunction = wavefunction
        return {"energy": energy, "gradient": np.asarray(gradients.get_values(), dtype=float)}


class GeometricOptimizer(GeometryOptimizer):
    """Geometry optimizer implemented with the geomeTRIC Python library."""

    def __init__(self, *, algorithm: str = "tric", transition_state: bool = False, name: str | None = None):
        """Initialize the geomeTRIC optimizer."""
        Logger.trace_entering()
        super().__init__()
        if algorithm not in GEOMETRIC_OPTIMIZER_ALGORITHMS:
            raise ValueError(f"Unknown geomeTRIC optimizer algorithm: {algorithm}")
        self._algorithm = algorithm
        self._coordinate_system = GEOMETRIC_OPTIMIZER_ALGORITHMS[algorithm]
        self._transition_state = transition_state
        self._name = name
        self._settings = GeometricOptimizerSettings(
            transition_state=transition_state,
            coordinate_system=self._coordinate_system,
        )

    def name(self) -> str:
        """Return the implementation name."""
        if self._name is not None:
            return self._name
        mode = "tsopt" if self._transition_state else "geoopt"
        return f"geometric_{mode}_{self._algorithm}"

    def aliases(self) -> list[str]:
        """Return accepted factory aliases."""
        return [self.name()]

    def _run_impl(
        self, structure: Structure, charge: int, spin_multiplicity: int, seed_or_basis: Any
    ) -> tuple[float, Structure, Any | None, NuclearHessian | None]:
        """Optimize a molecular structure using geomeTRIC."""
        Logger.trace_entering()
        from geometric.molecule import Molecule  # noqa: PLC0415
        from geometric.optimize import run_optimizer  # noqa: PLC0415

        molecule = Molecule()
        molecule.elem = structure.get_atomic_symbols()
        molecule.xyzs = [np.asarray(structure.get_coordinates(), dtype=float)]

        derivative_calculator = self._create_nested("derivative_calculator")
        derivative_calculator.settings().set("compute_hessian", False)
        engine = _QdkDerivativeEngine(
            structure, charge, spin_multiplicity, seed_or_basis, derivative_calculator, molecule
        )

        params = self._geometric_options()
        params.update({"customengine": engine, "input": None})

        with TemporaryDirectory(prefix="qdk-chemistry-geometric-") as tmpdir:
            result = run_optimizer(**params, prefix=f"{tmpdir}/qdk-chemistry", dirname=tmpdir)

        optimized_coordinates = _extract_coordinates(result, engine)
        optimized_structure = engine.structure_from_coordinates(optimized_coordinates)

        final_calculator = self._create_nested("derivative_calculator")
        final_calculator.settings().set("compute_hessian", self._settings["compute_hessian"])
        final_energy, _gradients, hessian, wavefunction = final_calculator.run(
            optimized_structure, charge, spin_multiplicity, seed_or_basis
        )
        if not self._settings["compute_hessian"]:
            hessian = None

        return final_energy, optimized_structure, wavefunction, hessian

    def _geometric_options(self) -> dict[str, Any]:
        return {
            "transition": self._settings["transition_state"],
            "coordsys": self._settings["coordinate_system"],
            "maxiter": self._settings["max_iterations"],
            "convergence_energy": self._settings["convergence_energy"],
            "convergence_grms": self._settings["convergence_gradient"],
            "convergence_drms": self._settings["convergence_displacement"],
            "verbose": self._settings["print_level"],
        }


def _extract_coordinates(result: Any, engine: _QdkDerivativeEngine) -> np.ndarray:
    """Extract final coordinates from geomeTRIC's return value."""
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "xyzs") and result.xyzs:
        return np.asarray(result.xyzs[-1], dtype=float)
    if isinstance(result, dict):
        for key in ("coords", "coordinates", "xyz", "xyzs"):
            if key in result:
                value = result[key]
                if key == "xyzs" and value:
                    value = value[-1]
                return np.asarray(value, dtype=float)
    return engine.last_coordinates()
