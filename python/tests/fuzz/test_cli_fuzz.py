"""Property-based tests for untrusted CLI input."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import copy
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qdk_chemistry.ui.cli import _deep_merge, _parse_set_overrides, create_parser, parse_json_arg

_JSON_SCALARS = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text()
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=20), children, max_size=5),
    max_leaves=20,
)
_ARGV_TOKEN = st.text(max_size=80)


@given(st.lists(_ARGV_TOKEN, max_size=20))
def test_cli_parser_rejects_or_parses_arbitrary_argv(argv: list[str]) -> None:
    """Arbitrary argument vectors only parse or produce an argparse exit."""
    parser = create_parser()
    exit_code = None

    try:
        parser.parse_args(argv)
    except SystemExit as error:
        exit_code = error.code

    assert exit_code is None or isinstance(exit_code, int)


@given(_JSON_VALUES)
def test_json_argument_parser_round_trips_json_values(value: object) -> None:
    """Every bounded JSON value survives CLI JSON parsing."""
    encoded = json.dumps(value)

    assert parse_json_arg(encoded) == value


@given(st.text(max_size=200))
def test_json_argument_parser_reports_invalid_text(value: str) -> None:
    """Text that is not JSON produces the documented argparse error."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_json_arg(value)
    else:
        assert json.dumps(parse_json_arg(value), sort_keys=True) == json.dumps(decoded, sort_keys=True)


@given(
    st.lists(
        st.tuples(st.lists(st.text(max_size=12), min_size=1, max_size=8), _JSON_VALUES).map(
            lambda item: f"{'.'.join(item[0])}={json.dumps(item[1])}"
        ),
        max_size=12,
    )
)
def test_set_overrides_handle_bounded_json_values(overrides: list[str]) -> None:
    """Dotted overrides produce a dictionary or a documented parse error."""
    original = list(overrides)

    try:
        result = _parse_set_overrides(overrides)
    except argparse.ArgumentTypeError:
        result = None

    assert result is None or isinstance(result, dict)
    assert overrides == original


@given(
    st.dictionaries(st.text(max_size=20), _JSON_VALUES, max_size=8),
    st.dictionaries(st.text(max_size=20), _JSON_VALUES, max_size=8),
)
def test_deep_merge_is_deterministic(base: dict, overrides: dict) -> None:
    """Merging bounded JSON mappings is deterministic and leaves overrides unchanged."""
    first_base = copy.deepcopy(base)
    second_base = copy.deepcopy(base)
    original_overrides = copy.deepcopy(overrides)

    assert _deep_merge(first_base, overrides) == _deep_merge(second_base, overrides)
    assert overrides == original_overrides
