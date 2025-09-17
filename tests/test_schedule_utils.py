from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from schedule_utils import format_time_range


@pytest.mark.parametrize(
    "start, block, expected",
    [
        (7, 4, "0700-1100"),
        (23, 4, "2300-0300"),
        (27, 4, "0300-0700"),
        (-1, 4, "2300-0300"),
    ],
)
def test_format_time_range_wraps_correctly(start, block, expected):
    assert format_time_range(start, block) == expected


@pytest.mark.parametrize(
    "start, block, expected",
    [
        (3, None, "0300-0400"),
        (3, 0, "0300-0400"),
        ("3", "1", "0300-0400"),
    ],
)
def test_format_time_range_defaults_to_one_hour(start, block, expected):
    assert format_time_range(start, block) == expected
