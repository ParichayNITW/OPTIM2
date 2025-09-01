"""Tests for loop case generation helpers."""

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so ``pipeline_model`` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pipeline_model import (  # type: ignore
    _generate_loop_cases_by_diameter,
    _generate_loop_cases_by_flags,
)


def test_generate_loop_cases_by_diameter_includes_bypass_then_loop_only():
    """Case-2 with two loops should include [2, 3]."""
    cases = _generate_loop_cases_by_diameter(2, False)
    assert [2, 3] in cases


def test_generate_loop_cases_by_flags_includes_bypass_then_loop_only():
    """Mixed-flag generator should also include [2, 3]."""
    cases = _generate_loop_cases_by_flags([False, False])
    assert [2, 3] in cases

