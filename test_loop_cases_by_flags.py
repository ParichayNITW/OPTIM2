import pipeline_model as pm


def test_equal_diameter_options_include_loop_only():
    cases = pm._generate_loop_cases_by_flags([True], [False])
    assert cases == [[0], [1], [3]]


def test_different_diameter_mixing_allowed_all_modes():
    cases = pm._generate_loop_cases_by_flags([False], [True])
    assert cases == [[0], [1], [2], [3]]


def test_different_diameter_no_mixing_excludes_parallel():
    cases = pm._generate_loop_cases_by_flags([False], [False])
    assert cases == [[0], [2], [3]]


def test_combinations_are_unique():
    # A mix of equal and unequal diameters should not yield duplicate combos
    cases = pm._generate_loop_cases_by_flags([True, False], [False, False])
    assert len(cases) == len({tuple(c) for c in cases})
