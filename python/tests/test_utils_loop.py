"""Tests for the Q# warm-up-aware ``Loop`` primitives."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import pytest
from qdk import TargetProfile
from qdk.qsharp import QSharpError, Result

from qdk_chemistry.utils.qsharp import create_qsharp_context

_VARIANTS = {"Loop": 0, "LoopA": 1, "LoopCA": 2}


def _t_count(context, num_iterations: int, num_warmup_iterations: int, variant: int) -> int:
    """T count of a loop whose iteration ``i`` applies ``i`` T gates."""
    counts = context.logical_counts(
        context.code.QDKChemistry.TestUtils.LoopTests.TestLoopTCount, num_iterations, num_warmup_iterations, variant
    )
    return counts.get("tCount", 0)


@pytest.mark.parametrize("variant", _VARIANTS.values(), ids=_VARIANTS.keys())
class TestLoopResourceEstimation:
    """Under resource estimation only the warm-up iterations and one representative are counted."""

    @pytest.mark.parametrize(
        ("num_iterations", "num_warmup_iterations", "expected"),
        [
            (5, 0, 0),
            (5, 1, 4),
            (5, 2, 7),
            (5, 4, 10),
            (5, 5, 10),
            (5, 9, 10),
            (1, 0, 0),
            (0, 0, 0),
            (0, 3, 0),
        ],
    )
    def test_iteration_w_stands_in_for_the_rest(
        self, qsharp_test_context, variant, num_iterations, num_warmup_iterations, expected
    ):
        """The representative is iteration ``w``, counted ``n - w`` times after ``w`` exact ones."""
        assert _t_count(qsharp_test_context, num_iterations, num_warmup_iterations, variant) == expected

    def test_a_long_loop_is_not_unrolled(self, qsharp_test_context, variant):
        """A repeated iteration must scale its cost without visiting every index."""
        assert _t_count(qsharp_test_context, 1_000_000, 2, variant) == 1 + 2 * (1_000_000 - 2)

    @pytest.mark.parametrize(("num_iterations", "num_warmup_iterations"), [(-1, 0), (3, -1)])
    def test_negative_arguments_are_rejected(self, qsharp_test_context, variant, num_iterations, num_warmup_iterations):
        """Negative iteration or warm-up counts have no meaning."""
        with pytest.raises(QSharpError, match="must be non-negative"):
            _t_count(qsharp_test_context, num_iterations, num_warmup_iterations, variant)


@pytest.mark.parametrize("variant", _VARIANTS.values(), ids=_VARIANTS.keys())
@pytest.mark.parametrize("num_warmup_iterations", [0, 1, 3, 7])
def test_simulation_runs_every_iteration(qsharp_test_utils, variant, num_warmup_iterations):
    """Outside resource estimation the warm-up count must not change which iterations run."""
    num_iterations = 4
    results = qsharp_test_utils.LoopTests.TestLoopVisitsEveryIteration(num_iterations, num_warmup_iterations, variant)

    assert results == [Result.One] * num_iterations


@pytest.mark.parametrize("num_warmup_iterations", [0, 2])
@pytest.mark.parametrize("num_iterations", [0, 1, 2, 3])
def test_a_captured_operation_is_applied_every_iteration(qsharp_test_utils, num_iterations, num_warmup_iterations):
    """Each iteration flips the register, so its parity reveals the number of applications."""
    flip_all = qsharp_test_utils.LoopTests.MakeTestFlipAllOp()
    results = qsharp_test_utils.LoopTests.TestLoopAppliesCapturedOp(flip_all, 2, num_iterations, num_warmup_iterations)

    assert results == [Result.One if num_iterations % 2 else Result.Zero] * 2


@pytest.mark.parametrize("profile", [TargetProfile.Base, TargetProfile.Adaptive_RIF])
def test_loops_compile_to_qir(profile):
    """Iterations capturing data or a caller-supplied operation must compile to QIR."""
    context = create_qsharp_context(profile)
    context.eval((Path(__file__).parent / "qsharp" / "TestLoop.qs").read_text(encoding="utf-8"))
    loop = context.code.QDKChemistry.TestUtils.LoopTests
    programs = [str(context.compile(loop.TestLoopVisitsEveryIteration, 3, 1, v)) for v in _VARIANTS.values()]
    programs.append(str(context.compile(loop.TestLoopAppliesCapturedOp, loop.MakeTestFlipAllOp(), 2, 3, 1)))

    assert all("__quantum__qis__x__body" in program for program in programs)
