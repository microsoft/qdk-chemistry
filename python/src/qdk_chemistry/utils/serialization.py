"""Serialization helpers shared across QDK Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import numpy as np


def real_array_to_json(values: np.ndarray) -> dict[str, Any]:
    """Serialize a real array as nested lists plus its shape."""
    array = np.asarray(values, dtype=float)
    return {"values": array.tolist(), "shape": list(array.shape)}


def real_array_from_json(data: Any) -> np.ndarray:
    """Rebuild an array written by ``real_array_to_json`` or an older bare list."""
    if isinstance(data, dict):
        return np.asarray(data["values"], dtype=float).reshape(data["shape"])
    return np.asarray(data, dtype=float)


def complex_array_to_json(values: np.ndarray) -> dict[str, Any]:
    """Serialize an array as split real/imaginary lists with its shape and dtype."""
    array = np.asarray(values)
    return {
        "real": array.real.tolist(),
        "imag": array.imag.tolist(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def complex_array_from_json(data: dict[str, Any]) -> np.ndarray:
    """Rebuild an array written by ``complex_array_to_json`` or an older codec."""
    dtype = np.dtype(data["dtype"]) if "dtype" in data else np.dtype(complex)
    real = np.asarray(data["real"])
    if np.issubdtype(dtype, np.complexfloating):
        array = (real + 1j * np.asarray(data["imag"])).astype(dtype)
    else:
        array = real.astype(dtype)
    shape = data.get("shape")
    return array.reshape(shape) if shape is not None else array
