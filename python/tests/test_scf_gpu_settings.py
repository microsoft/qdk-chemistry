"""Test independent GPU in-core cutoffs without requiring GPU hardware."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms import create


@pytest.mark.parametrize(
    ("algorithm", "backend", "default"),
    [
        ("scf_solver", "qdk", 160),
        ("scf_solver", "qdk_stabilized", 160),
        ("stability_checker", "qdk", 250),
    ],
)
def test_gpu_incore_cutoff(algorithm, backend, default):
    """Cutoffs retain their defaults and reject negative basis counts."""
    settings = create(algorithm, backend).settings()
    assert settings.get("gpu_incore_max_basis_functions") == default
    for value in (0, 300):
        settings.set("gpu_incore_max_basis_functions", value)
        assert settings.get("gpu_incore_max_basis_functions") == value
    with pytest.raises(ValueError, match="out of allowed range"):
        settings.set("gpu_incore_max_basis_functions", -1)


def test_scf_and_stability_cutoffs_are_independent():
    """Tuning SCF must not change the stability checker's cutoff."""
    solver = create("scf_solver", "qdk")
    checker = create("stability_checker", "qdk")
    solver.settings().set("gpu_incore_max_basis_functions", 300)
    assert checker.settings().get("gpu_incore_max_basis_functions") == 250
    assert solver.settings().get("eri_method") == "auto"
    assert solver.settings().get("device") == "auto"
