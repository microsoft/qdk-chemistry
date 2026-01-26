"""Telemetry module for QDK Chemistry.

Module sends telemetry directly to Azure Monitor using a similar mechanism and
format to the Azure Monitor OpenTelemetry Python SDK. It only supports custom metrics of
type "counter" and "histogram" for now. Its goal is to be minimal in size and dependencies,
and easy to read to understand exactly what data is being sent.

To use this API, simply call `log_telemetry` with the metric name, value, and any other
optional properties. The telemetry will be batched and sent at a regular intervals (60 sec),
and when the process is about to exit.

Disable qdk_chemistry Python telemetry by setting the environment variable
`QSHARP_PYTHON_TELEMETRY` to one of the following: `none`, `disabled`, `false`, or `0`.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import atexit
import json
import locale
import logging
import os
import platform
import sys
import time
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from queue import Empty, SimpleQueue
from threading import Thread
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)

try:
    # Define the package version
    QDK_CHEMISTRY_VERSION = version("qdk-chemistry")
except PackageNotFoundError:
    # Fallback if package not installed
    QDK_CHEMISTRY_VERSION = "0.0.0.dev0"

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    # Backport for Python 3.10
    UTC = timezone.utc

# Application Insights configuration
AIKEY = os.environ.get("QSHARP_PYTHON_AI_KEY") or "95d25b22-8b6d-448e-9677-78ad4047a95a"
AIURL = os.environ.get("QSHARP_PYTHON_AI_URL") or "https://westus2-2.in.applicationinsights.azure.com/v2.1/track"

# Environment variables take precedence, else disable telemetry for non 'stable' builds
QSHARP_PYTHON_TELEMETRY = (os.environ.get("QSHARP_PYTHON_TELEMETRY") or "").lower()
TELEMETRY_ENABLED = (
    True
    if QSHARP_PYTHON_TELEMETRY in ["1", "true", "enabled"]
    else (
        False
        if QSHARP_PYTHON_TELEMETRY in ["0", "false", "disabled", "none"]
        else ("dev" not in QDK_CHEMISTRY_VERSION)  # Auto-disable for dev builds
    )
)

BATCH_INTERVAL_SEC = int(os.environ.get("QSHARP_PYTHON_TELEMETRY_INTERVAL") or 60)


# The below is taken from the Azure Monitor Python SDK
def _getlocale() -> str:
    try:
        with warnings.catch_warnings():
            # Workaround for https://github.com/python/cpython/issues/82986
            # by continuing to use getdefaultlocale() even though it has been deprecated.
            # Ignore the deprecation warnings to reduce noise
            warnings.simplefilter("ignore", category=DeprecationWarning)
            return locale.getdefaultlocale()[0] or ""
    except AttributeError:
        # Use this as a fallback if locale.getdefaultlocale() doesn't exist (>Py3.13)
        return locale.getlocale()[0] or ""


# Minimal device information to include with telemetry
AI_DEVICE_LOCALE = _getlocale()
AI_DEVICE_OS_VERSION = platform.version()


class Metric(TypedDict):
    """Used internally for objects in the telemetry queue."""

    name: str
    value: float
    count: int
    properties: dict[str, Any]
    type: str


class PendingMetric(Metric):
    """Used internally to aggregate metrics before sending."""

    min: float
    max: float


# Maintain a collection of custom metrics to log, stored by metric name with a list entry
# for each unique set of properties per metric name
pending_metrics: dict[str, list[PendingMetric]] = {}

# The telemetry queue is used to send telemetry from the main thread to the telemetry thread
# This simplifies any thread-safety concerns, and avoids the need for locks, etc.
telemetry_queue: Any = SimpleQueue()  # type 'Any' until we get off Python 3.8 builds


def log_telemetry(
    name: str,
    value: float,
    count: int = 1,
    properties: dict[str, Any] | None = None,
    type: Literal["counter", "histogram"] = "counter",  # noqa: A002
) -> None:
    """Log a custom telemetry metric.

    Logs a custom metric with the name provided. Properties are optional and can be used to
    capture additional context about the metric (but should be a relatively static set of
    values, as each unique set of properties will be sent as a separate metric and creates
    a separate 'dimension' in the backend telemetry store).

    The type can be either 'counter' or 'histogram'. A 'counter' is a simple value
    that is summed over time, such as how many times an event occurs, while a
    'histogram' is used to track 'quantitative' values, such as the distribution of values
    over time, e.g., the duration of an operation.
    """
    if not TELEMETRY_ENABLED:
        return

    if properties is None:
        properties = {}

    obj: Metric = {
        "name": name,
        "value": value,
        "count": count,
        "properties": {**properties, "qdk_chemistry.version": QDK_CHEMISTRY_VERSION},
        "type": type,
    }

    logger.debug("Queuing telemetry: %s", obj)
    telemetry_queue.put(obj)


def _add_to_pending(metric: Metric):
    """Used by the telemetry thread to aggregate metrics before sending."""
    if metric["type"] not in ["counter", "histogram"]:
        raise Exception("Metric must be of type counter or histogram")

    # Get or create the entry list for this name
    name_entries = pending_metrics.setdefault(metric["name"], [])

    # Try to find the entry with matching properties
    # This relies on the fact dicts with matching keys/values compare equal in Python
    prop_entry = next(
        (entry for entry in name_entries if entry["properties"] == metric["properties"]),
        None,
    )
    if prop_entry is None:
        new_entry: PendingMetric = {
            **metric,
            "min": metric["value"],
            "max": metric["value"],
        }
        name_entries.append(new_entry)
    else:
        if prop_entry["type"] != metric["type"]:
            raise Exception("Cannot mix counter and histogram for the same metric name")
        prop_entry["value"] += metric["value"]
        prop_entry["count"] += metric["count"]
        prop_entry["min"] = min(prop_entry["min"], metric["value"])
        prop_entry["max"] = max(prop_entry["max"], metric["value"])


def _pending_to_payload() -> list[dict[str, Any]]:
    """Converts the pending metrics to the JSON payload for Azure Monitor."""
    result_array: list[dict[str, Any]] = []
    formatted_time = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    for metric_list in pending_metrics.values():
        for unique_props in metric_list:
            # The below matches the entry format for Azure Monitor REST API
            entry: dict[str, Any] = {
                "ver": 1,
                "name": "Microsoft.ApplicationInsights.Metric",
                "time": formatted_time,
                "sampleRate": 100.0,
                "iKey": AIKEY,
                "tags": {
                    "ai.device.locale": AI_DEVICE_LOCALE,
                    "ai.device.osVersion": AI_DEVICE_OS_VERSION,
                },
                "data": {
                    "baseType": "MetricData",
                    "baseData": {
                        "ver": 2,
                        "metrics": [
                            {
                                "name": unique_props["name"],
                                "value": unique_props["value"],
                                "count": unique_props["count"],
                            }
                        ],
                        "properties": unique_props["properties"],
                    },
                },
            }

            if unique_props["type"] == "histogram":
                # Histogram values differ only in that they have min/max values also
                entry["data"]["baseData"]["metrics"][0]["min"] = unique_props["min"]
                entry["data"]["baseData"]["metrics"][0]["max"] = unique_props["max"]

            result_array.append(entry)

    return result_array


def _post_telemetry() -> bool:
    """Posts the pending telemetry to Azure Monitor."""
    if len(pending_metrics) == 0:
        return True

    payload = json.dumps(_pending_to_payload()).encode("utf-8")
    logger.debug("Sending telemetry request: %s", payload)
    try:
        request = urllib.request.Request(AIURL, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=10) as response:
            logger.debug("Telemetry response: %s", response.status)
            # On a successful post, clear the pending list. (Else they will be included on the next retry)
            pending_metrics.clear()
            return True

    except urllib.error.HTTPError as e:
        logger.debug("HTTP error posting telemetry (status %d): %s", e.code, e.reason)
        return False
    except urllib.error.URLError as e:
        logger.debug("URL error posting telemetry: %s", e.reason)
        return False
    except OSError as e:
        logger.debug("Network/system error posting telemetry: %s", e)
        return False
    except (ValueError, TypeError) as e:
        logger.debug("Data serialization error in telemetry: %s", e)
        return False


def _telemetry_thread_start():
    """Starts the telemetry background thread that processes and posts telemetry metrics in batches."""
    next_post_sec: float | None = None

    def on_metric(msg: Metric):
        """Handles a new metric message by adding & scheduling it to the pending batch."""
        nonlocal next_post_sec

        # Add to the pending batch to send next
        _add_to_pending(msg)

        # Schedule the next post if one is not scheduled
        if next_post_sec is None:
            next_post_sec = time.monotonic() + BATCH_INTERVAL_SEC

    while True:
        try:
            # Block if no timeout, else wait a maximum of time until the next post is due
            timeout: float | None = None
            if next_post_sec:
                timeout = max(next_post_sec - time.monotonic(), 0)
            msg = telemetry_queue.get(timeout=timeout)

            if msg == "exit":
                logger.debug("Exiting telemetry thread")
                if not _post_telemetry():
                    logger.debug("Failed to post telemetry on exit")
                return
            on_metric(msg)
            # Loop until the queue has been drained. This will cause the 'Empty' exception
            # below once the queue is empty and it's time to post
            continue
        except Empty:
            # No more telemetry within timeout, so write what we have pending
            _ = _post_telemetry()

        # If we get here, it's after a post attempt. Pending will still have items if the attempt
        # failed, so update the time for the next attempt in that case.
        next_post_sec = None if not pending_metrics else time.monotonic() + BATCH_INTERVAL_SEC


def _on_exit():
    """On exit handler to flush telemetry before process exits."""
    logger.debug("In on_exit handler")
    telemetry_queue.put("exit")
    # Wait at most 3 seconds for the telemetry thread to flush and exit
    telemetry_thread.join(timeout=3)


# Mark the telemetry thread as a daemon thread, else it will keep the process alive when the main thread exits
if TELEMETRY_ENABLED:
    telemetry_thread = Thread(target=_telemetry_thread_start, daemon=True)
    telemetry_thread.start()
    atexit.register(_on_exit)
