"""Tests for warm-up-aware Q# repetition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import pytest
from qdk import TargetProfile
from qdk.qsharp import QSharpError, Result

from qdk_chemistry.utils.qsharp import create_qsharp_context

_VARIANTS = {"LoopA": 0, "LoopCA": 1}


@pytest.mark.parametrize("variant", _VARIANTS.values(), ids=_VARIANTS.keys())
@pytest.mark.parametrize(
    ("num_iterations", "num_warmup_iterations", "expected"),
    [(0, 3, 0), (5, 0, 0), (5, 2, 7), (5, 9, 10), (1_000_000, 2, 1_999_997)],
)
def test_logical_counts(qsharp_test_context, variant, num_iterations, num_warmup_iterations, expected):
    """Count warm-up iterations exactly and fold the remaining cost without unrolling."""
    counts = qsharp_test_context.logical_counts(
        qsharp_test_context.code.QDKChemistry.TestUtils.LoopTests.TestLoopTCount,
        num_iterations,
        num_warmup_iterations,
        variant,
    )
    assert counts.get("tCount", 0) == expected


@pytest.mark.parametrize("variant", _VARIANTS.values(), ids=_VARIANTS.keys())
@pytest.mark.parametrize(("num_iterations", "num_warmup_iterations"), [(-1, 0), (3, -1)])
def test_negative_arguments_are_rejected(qsharp_test_context, variant, num_iterations, num_warmup_iterations):
    """Reject negative iteration or warm-up counts."""
    with pytest.raises(QSharpError, match="must be non-negative"):
        qsharp_test_context.logical_counts(
            qsharp_test_context.code.QDKChemistry.TestUtils.LoopTests.TestLoopTCount,
            num_iterations,
            num_warmup_iterations,
            variant,
        )


@pytest.mark.parametrize(("num_iterations", "num_warmup_iterations"), [(0, 0), (3, 2)])
def test_a_captured_operation_is_applied_every_iteration(qsharp_test_utils, num_iterations, num_warmup_iterations):
    """Each iteration flips the register, so its parity reveals the number of applications."""
    flip_all = qsharp_test_utils.LoopTests.TestMakeFlipAllOp()
    results = qsharp_test_utils.LoopTests.TestLoopAppliesCapturedOp(flip_all, 2, num_iterations, num_warmup_iterations)

    assert results == [Result.One if num_iterations % 2 else Result.Zero] * 2


@pytest.mark.parametrize("profile", [TargetProfile.Base, TargetProfile.Adaptive_RIF])
def test_loops_compile_to_qir(profile):
    """Both variants must simulate every iteration and compile captured callbacks to QIR."""
    context = create_qsharp_context(profile)
    context.eval((Path(__file__).parent / "qsharp" / "TestLoop.qs").read_text(encoding="utf-8"))
    loop = context.code.QDKChemistry.TestUtils.LoopTests
    for variant in _VARIANTS.values():
        assert loop.TestLoopVisitsEveryIteration(3, 5, variant) == [Result.One] * 3
        assert "__quantum__qis__x__body" in str(context.compile(loop.TestLoopVisitsEveryIteration, 3, 1, variant))
    assert "__quantum__qis__x__body" in str(
        context.compile(loop.TestLoopAppliesCapturedOp, loop.TestMakeFlipAllOp(), 2, 3, 1)
    )
