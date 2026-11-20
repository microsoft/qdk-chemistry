"""Tests for the Q# box-restricted skew-tree table lookup."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import random

import pytest
from qdk.test_utils import ArithmeticOpTester

# (fields, rows, word width): two padded fields, a padded high field, a ragged 1D table, an unpadded low field.
_SHAPES = [
    ([(2, 3), (2, 3)], 11, 4),
    ([(3, 5), (2, 3)], 21, 3),
    ([], 6, 5),
    ([(1, 2), (3, 6)], 16, 2),
]
_SHAPE_IDS = ["box3x3", "box5x3", "ragged1d", "box2x6"]


def _xor_rows(rows, width):
    """The bitwise XOR of *rows*."""
    out = [False] * width
    for row in rows:
        out = [a != b for a, b in zip(out, row, strict=True)]
    return out


def _shape(fields, num_rows, width, seed):
    """Random table with garbage outside the box, plus its address width and the box addresses."""
    rng = random.Random(seed)
    num_bits = sum(bits for bits, _ in fields) or math.ceil(math.log2(num_rows))
    data = [[rng.random() < 0.5 for _ in range(width)] for _ in range(num_rows)]
    offsets = [sum(bits for bits, _ in fields[:j]) for j in range(len(fields))]
    box = [
        x
        for x in range(num_rows)
        if all(
            (x >> offset) & ((1 << bits) - 1) < extent for offset, (bits, extent) in zip(offsets, fields, strict=True)
        )
    ]
    return data, num_bits, box


@pytest.mark.parametrize("num_controls", [0, 1, 2])
@pytest.mark.parametrize(("fields", "num_rows", "width"), _SHAPES, ids=_SHAPE_IDS)
def test_skew_tree_loads_every_in_box_address(
    qsharp_test_context, qsharp_test_utils, fields, num_rows, width, num_controls
):
    """Every in-box address loads its row when all controls are on and nothing when one is off, leaving inputs clean."""
    data, num_bits, box = _shape(fields, num_rows, width, seed=num_rows)
    op = qsharp_test_utils.SkewTreeTests.TestMakeSkewTreeCopyOp(data, fields, num_controls)
    tester = ArithmeticOpTester(op, [max(num_controls, 1), num_bits, width, width], context=qsharp_test_context)
    on = (1 << num_controls) - 1
    # ArithmeticOpTester reads registers back as little-endian integers.
    failures = [
        x for x in box if tester.run([on, x, 0, 0]) != [on, x, 0, sum(1 << i for i, bit in enumerate(data[x]) if bit)]
    ]
    if num_controls:
        failures += [("off", x) for x in box if tester.run([0, x, 0, 0]) != [0, x, 0, 0]]
    assert not failures


@pytest.mark.parametrize("num_controls", [0, 1, 2])
@pytest.mark.parametrize(("fields", "num_rows", "width"), _SHAPES, ids=_SHAPE_IDS)
def test_skew_tree_phase_matches_an_independent_reference(qsharp_test_utils, fields, num_rows, width, num_controls):
    """On superposed controls and addresses the lookup matches the reference phase and catches a one-bit mutation."""
    data, num_bits, box = _shape(fields, num_rows, width, seed=num_rows + 1)
    # Outside the box the lookup writes the XOR of the in-box subset-XOR coefficients below the address.
    coefficients = {p: _xor_rows((data[q] for q in range(p + 1) if q & p == q), width) for p in box}
    reference = [_xor_rows((c for p, c in coefficients.items() if p & x == p), width) for x in range(1 << num_bits)]
    assert all(reference[x] == data[x] for x in box)
    tests = qsharp_test_utils.SkewTreeTests

    assert tests.TestSkewTreePhaseAgreement(data, reference, fields, num_bits, num_controls)
    mutated = [list(row) for row in data]
    mutated[box[-1]][0] = not mutated[box[-1]][0]
    assert not tests.TestSkewTreePhaseAgreement(mutated, reference, fields, num_bits, num_controls)


def test_skew_tree_costs_one_toffoli_per_box_node_and_undercuts_a_padded_select(qsharp_test_context, qsharp_test_utils):
    """Singly controlled, the lookup costs box size minus one Toffolis, below controlled ``Select`` on the table."""
    fields, num_rows, width = _SHAPES[0]
    data, num_bits, box = _shape(fields, num_rows, width, seed=0)

    def toffolis(*args):
        """Toffoli count of tracing *args*."""
        counts = qsharp_test_context.logical_counts(*args)
        return counts["cczCount"] + counts["ccixCount"]

    skew = toffolis(qsharp_test_utils.SkewTreeTests.TestSkewTreeResourceProbe, data, fields, num_bits, 1)
    padded = toffolis(qsharp_test_utils.SOSSAWalkTests.TestLoadRotationWordResourceProbe, data, 0)
    assert skew == len(box) - 1
    assert padded == num_rows - 1
    assert skew < padded
