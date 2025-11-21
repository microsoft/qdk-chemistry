# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .telemetry import log_telemetry
import math
from typing import Union

def on_qdk_chemistry_import() -> None:
    """
    Logs a telemetry event indicating that the QDK Chemistry module has been imported.
    
    This function should be called when the QDK Chemistry package has been imported,
    to track usage statistics for analytics and improvement purposes.
    """
    log_telemetry("qdk_chemistry.import", 1)


def on_algorithm(
    algorithm_type: str, 
    algorithm_name: str
) -> None:
    """
    Logs a telemetry event for the execution of a quantum chemistry algorithm.
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
        properties={
            "algorithm_type": algorithm_type,
            "algorithm_name": algorithm_name
            },
    )


def on_algorithm_end(
    algorithm_type: str,
    duration_sec: float,
    status: str,
    algorithm_name: str,
    error_type: Union[str, None] = None,
    **properties
) -> None:
    """
    Logs the execution duration and outcome of a chemistry algorithm, along with relevant metadata.
    Args:
        - algorithm_type (str): The category of algorithm executed (e.g., 'scf_solver', 'active_space_selector').
        - duration_sec (float): The time taken to execute the algorithm, in seconds.
        - status (str): The result of the execution, typically 'success' or 'failed'.
        - algorithm_name (str): The specific implementation or backend used (e.g., 'qdk', 'pyscf').
        - error_type (Optional[str]): The type of error encountered, if any. Defaults to None.
        - **properties: Additional contextual information about the execution (e.g., 'num_atoms', 'basis_set').
    Returns:
        None
    Notes:
        This function emits a telemetry event recording the algorithm's execution time and associated metadata.
        If an error occurred, the error type is included in the telemetry properties.
    """
    telemetry_properties = {
        "algorithm_type": algorithm_type,
        "algorithm_name": algorithm_name,
        "status": status,
        "error_type": error_type,
        **properties
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
    """
    Logs a telemetry event for a test call.
    This function records a telemetry event with the name "qdk_chemistry.test" and includes the provided test name as a property. The event is logged as a counter type.
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