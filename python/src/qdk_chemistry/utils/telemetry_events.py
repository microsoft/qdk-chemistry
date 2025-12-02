"""Telemetry event logging for QDK Chemistry module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from .telemetry import log_telemetry


def get_basis_functions_bucket(basis_functions: str | int) -> str:
    """Categorize the number of basis functions into buckets for telemetry aggregation.

    This function groups basis function counts into discrete buckets to enable
    meaningful aggregation and analysis in telemetry data. Rather than tracking
    exact counts (which would result in too many unique values), this bucketing
    approach provides useful ranges for performance and usage analysis.

    Args:
        basis_functions (Union[str, int]): The number of basis functions in a
        calculation. Can be an integer count or the string "unknown"
        if the count is not available.

    Returns:
        str: A string representation of the bucket that the basis function count
             falls into.

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

    Notes:
        The bucketing scheme is designed for typical quantum chemistry calculations
        with varying granularity based on system size:

        * Fine granularity (10s) for small molecules (<50 basis functions)
        * Medium granularity (50s) for medium molecules (50-500 basis functions)
        * Coarse granularity (100s) for large molecules (>500 basis functions)

    """
    if basis_functions == "unknown":
        return "unknown"

    basis_functions = int(basis_functions)

    if basis_functions < 50:
        # Intervals of 10 for < 50
        return str(((basis_functions - 1) // 10 + 1) * 10)
    if basis_functions <= 500:
        # Intervals of 50 for 50-500
        return str(((basis_functions - 1) // 50 + 1) * 50)
    if basis_functions < 1500:
        # Intervals of 100 for 500-1000
        return str(((basis_functions - 1) // 100 + 1) * 100)
    # 1000+ for anything >= 1000
    return "1500+"


def extract_data(result):
    """Extract molecular formula and basis function count from algorithm result.

    This function handles both single qdk_data objects and tuple results
    (e.g., (energy, qdk_data) pairs) returned by QDK chemistry algorithms.
    It extracts the molecular formula and number of basis functions from the
    qdk_data's orbital data for telemetry tracking.

    Args:
        result: Algorithm result, either a qdk_data object or a tuple containing
            a qdk_data (typically at index 1 for (energy, wavefunction) pairs).

    Returns:
        tuple[str, str]: A tuple containing:
            - Molecular formula (str): Chemical formula with element counts (e.g., "H2O", "CH4")
              or "unknown" if no qdk_data data is available.
            - Basis functions bucket (str): Bucketed count of basis functions (e.g., "10", "50", "100")
              or "unknown" if no qdk_data data is available.

    Examples:
        >>> # Single qdk_data result
        >>> formula, n_basis = extract_qdk_data_data(qdk_data)
        ('H2O', '50')

        >>> # Tuple result (energy, qdk_data)
        >>> formula, n_basis = extract_qdk_data_data((energy, qdk_data))
        ('CH4', '100')

        >>> # No qdk_data data
        >>> formula, n_basis = extract_qdk_data_data(some_other_result)
        ('unknown', 'unknown')

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
                n_basis = get_basis_functions_bucket(orbitals.get_basis_set().get_num_basis_functions())
            else:
                n_basis = "unknown"

            return n_basis
        except (AttributeError, TypeError, RuntimeError):
            # Silently handle missing attributes
            pass
    return "unknown"


def on_qdk_chemistry_import() -> None:
    """Logs a telemetry event indicating that the QDK Chemistry module has been imported.

    This function should be called when the QDK Chemistry package has been imported,
    to track usage statistics for analytics and improvement purposes.
    """
    log_telemetry("qdk_chemistry.import", 1)


def on_algorithm(algorithm_type: str, algorithm_name: str) -> None:
    """Logs a telemetry event for the execution of a quantum chemistry algorithm.

    Args:
        algorithm_type (str): The type or category of the algorithm being executed.
        algorithm_name (str): The specific name of the algorithm.

    Returns:
        None

    This will run whenever an algorithm is created to track initialization statistics.

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
        algorithm_type (str): The category of algorithm executed (e.g.,
            'scf_solver', 'active_space_selector').
        duration_sec (float): The time taken to execute the algorithm,
            in seconds.
        status (str): The result of the execution, typically 'success'
            or 'failed'.
        algorithm_name (str): The specific implementation or backend
            used (e.g., 'qdk', 'pyscf').
        error_type (str | None): The type of error encountered, if
            any. Defaults to None.
        **properties: Additional contextual information about the
            execution (e.g., 'num_basis_functions', 'molecular_formula').

    Returns:
        None

    Notes:
        This function emits a telemetry event recording the algorithm's
        execution time and associated metadata. If an error occurred,
        the error type is included in the telemetry properties.

    """
    telemetry_properties = {
        "algorithm_type": algorithm_type,
        "algorithm_name": algorithm_name,
        "status": status,
        "error_type": error_type,
        **properties,
    }
    if error_type is not None:
        telemetry_properties["error_type"] = error_type

    log_telemetry(
        "qdk_chemistry.algorithm.durationSec",
        duration_sec,
        properties=telemetry_properties,
        type="histogram",
    )


def on_test_call(name: str) -> None:
    """Logs a telemetry event for a test call.

    This function records a telemetry event with the name "qdk_chemistry.
    test" and includes the provided test name as a property. The event is
    logged as a counter type.

    Args:
        name (str): The name of the test to be logged in the telemetry event.

    Returns:
        None

    Used for testing purposes.

    """
    log_telemetry(
        "qdk_chemistry.test",
        1,
        properties={"test_name": name},
        type="counter",
    )
