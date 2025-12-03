import pytest

from pipeline_model import _normalise_station_profile, _profile_ppm_metrics


def test_profile_ppm_metrics_prefers_injected_value_when_profile_zero():
    profile_entries = _normalise_station_profile(((1.0, 0.0),))
    inlet, outlet, treated_len = _profile_ppm_metrics(profile_entries, inj_ppm_main=6.5)

    assert inlet == pytest.approx(6.5)
    assert outlet == pytest.approx(0.0)
    assert treated_len == pytest.approx(0.0)


def test_profile_ppm_metrics_uses_profile_when_no_injection():
    profile_entries = _normalise_station_profile(((2.0, 4.0), (3.0, 0.0)))
    inlet, outlet, treated_len = _profile_ppm_metrics(profile_entries, inj_ppm_main=0.0)

    assert inlet == pytest.approx(4.0)
    assert outlet == pytest.approx(4.0)
    assert treated_len == pytest.approx(2.0)
