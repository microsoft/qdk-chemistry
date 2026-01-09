"""Telemetry event logging for QDK Chemistry module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import time
from functools import wraps
from typing import Any

from .telemetry import log_telemetry


def get_basis_functions_bucket(basis_functions: str | int) -> str:
    """Categorize the number of basis functions into buckets for telemetry aggregation.

    This function groups number of basis functions into discrete buckets to enable
    meaningful aggregation and analysis in telemetry data. Rather than tracking
    exact counts (which would result in too many unique values), this bucketing
    approach provides useful ranges for performance and usage analysis.

    Args:
        basis_functions: The number of basis functions.

    Returns:
        The bucket that the basis function count falls into.

    Examples:
        >>> get_basis_functions_bucket(7)
        "10"
        >>> get_basis_functions_bucket(23)
        "30"
        >>> get_basis_functions_bucket(150)
        "150"
        >>> get_basis_functions_bucket(750)
        "800"
        >>> get_basis_functions_bucket(1500)
        "1500+"
        >>> get_basis_functions_bucket("unknown")
        "unknown"

    """
    if basis_functions == "unknown":
        return "unknown"

    basis_functions = int(basis_functions)

    if basis_functions < 50:
        # Buckets of 10 for basis functions 1-49
        return str(((basis_functions - 1) // 10 + 1) * 10)
    if basis_functions <= 500:
        # Buckets of 50 for basis functions 50-500
        return str(((basis_functions - 1) // 50 + 1) * 50)
    if basis_functions < 1500:
        # Buckets of 100 for basis functions 501-1499
        return str(((basis_functions - 1) // 100 + 1) * 100)
    # Returns "1500+" for basis functions >= 1500
    return "1500+"


def extract_data(result: Any) -> str:
    """Extract number of basis functions from algorithm result.

    This function handles both single qdk_data objects and tuple results
    (e.g., (energy, qdk_data) pairs) returned by QDK chemistry algorithms.
    It extracts the number of basis functions from the qdk_data's orbital data for telemetry tracking.

    Args:
        result: Algorithm result, either a qdk_data object or a tuple containing
            a qdk_data (typically at index 1 for (energy, wavefunction) pairs).

    Returns:
        Bucket of basis functions (e.g., "10", "50", "100")
        or "unknown" if no qdk_data is available.

    Examples:
        >>> # Single qdk_data result
        >>> n_basis = extract_data(qdk_data)
        '50'

        >>> # Tuple result (energy, qdk_data)
        >>> n_basis = extract_data((energy, qdk_data))
        '100'

        >>> # No qdk_data
        >>> n_basis = extract_data(some_other_result)
        'unknown'

    """
    qdk_data = None
    if isinstance(result, tuple) and len(result) > 1 and hasattr(result[1], "orbitals"):
        qdk_data = result[1]
    elif hasattr(result, "orbitals"):
        qdk_data = result

    if qdk_data:
        try:
            orbitals = qdk_data.orbitals
            # Check if get_basis_set method exists
            if hasattr(orbitals, "get_basis_set"):
                n_basis = get_basis_functions_bucket(orbitals.get_basis_set().get_num_atomic_orbitals())
            else:
                n_basis = "unknown"

            return n_basis
        except (AttributeError, TypeError, RuntimeError):
            # Silently handle missing attributes
            pass
    return "unknown"


def on_qdk_chemistry_import() -> None:
    """Logs a telemetry event indicating that the QDK Chemistry module has been imported."""
    log_telemetry("qdk_chemistry.import", 1)


def on_algorithm(algorithm_type: str, algorithm_name: str) -> None:
    """Logs a telemetry event for the execution of a quantum chemistry algorithm.

    Args:
        algorithm_type: The type or category of the algorithm being executed.
        algorithm_name: The specific name of the algorithm.

    """
    log_telemetry(
        "qdk_chemistry.algorithm",
        1,
        properties={"algorithm_type": algorithm_type, "algorithm_name": algorithm_name},
    )


def on_algorithm_end(
    algorithm_type: str,
    duration_sec: float,
    status: str,
    algorithm_name: str,
    error_type: str | None = None,
    **properties,
) -> None:
    """Logs the execution duration and outcome of a chemistry algorithm.

    Logs relevant metadata about algorithm execution including timing,
    success/failure status, and additional contextual information.

    Args:
        algorithm_type: The category of algorithm executed (e.g.,
            'scf_solver', 'active_space_selector').
        duration_sec: The time taken to execute the algorithm,
            in seconds.
        status: The result of the execution, typically 'success'
            or 'failed'.
        algorithm_name: The specific implementation or backend
            used (e.g., 'qdk', 'pyscf').
        error_type: The type of error encountered, if
            any. Defaults to None.
        properties: Additional contextual information about the
            execution (e.g., 'num_basis_functions').

    """
    telemetry_properties = {
        "algorithm_type": algorithm_type,
        "algorithm_name": algorithm_name,
        "status": status,
        "error_type": error_type,
        **properties,
    }

    log_telemetry(
        "qdk_chemistry.algorithm.durationSec",
        duration_sec,
        properties=telemetry_properties,
        type="histogram",
    )


def telemetry_tracker():
    """Decorator to track telemetry for algorithm run execution."""

    def decorator(run_method):
        @wraps(run_method)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = run_method(self, *args, **kwargs)
                duration = time.perf_counter() - start_time
                try:
                    n_basis = extract_data(result)
                    on_algorithm_end(
                        algorithm_type=self.type_name(),
                        algorithm_name=self.name(),
                        duration_sec=duration,
                        status="success",
                        num_basis_functions=n_basis,
                    )
                except (AttributeError, TypeError, IndexError):
                    on_algorithm_end(
                        algorithm_type=self.type_name(),
                        algorithm_name=self.name(),
                        duration_sec=duration,
                        status="success",
                    )
                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                on_algorithm_end(
                    algorithm_type=self.type_name(),
                    algorithm_name=self.name(),
                    duration_sec=duration,
                    status="failed",
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator
