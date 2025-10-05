from types import SimpleNamespace

import pandas as pd

import pipeline_optimization_app as app


def test_collect_search_depth_kwargs_handles_missing_pipeline_constants(monkeypatch):
    """UI defaults should survive when backend constants are absent."""

    stub_model = SimpleNamespace()
    monkeypatch.setattr(app, "pipeline_model", stub_model, raising=False)

    session = app.st.session_state
    tracked_keys = [
        "search_rpm_step",
        "search_dra_step",
        "search_coarse_multiplier",
        "search_state_top_k",
        "search_state_cost_margin",
    ]
    sentinel = object()
    previous_values = {}
    for key in tracked_keys:
        previous_values[key] = session.get(key, sentinel)
        if key in session:
            del session[key]

    try:
        defaults = app._collect_search_depth_kwargs()
    finally:
        for key, value in previous_values.items():
            if value is sentinel:
                if key in session:
                    del session[key]
            else:
                session[key] = value

    assert defaults == {
        "rpm_step": 25,
        "dra_step": 2,
        "coarse_multiplier": 5.0,
        "state_top_k": 50,
        "state_cost_margin": 5000.0,
    }


def test_collect_segment_floors_includes_zero_floor_segments() -> None:
    """Zero-lacing segments should be preserved for display."""

    baseline = {
        "segments": [
            {"station_idx": 0, "length_km": 6.0, "dra_ppm": 0.0, "dra_perc": 0.0},
            {"station_idx": 1, "length_km": 4.0, "dra_ppm": 5.0},
            {"station_idx": 2, "length_km": 3.5},
        ]
    }

    result = app._collect_segment_floors(baseline)

    assert len(result) == 3
    assert result[0]["dra_ppm"] == 0.0
    assert result[0]["dra_perc"] == 0.0
    assert result[1]["dra_ppm"] == 5.0
    assert result[1]["dra_perc"] == 0.0
    assert result[2]["dra_ppm"] == 0.0
    assert result[2]["dra_perc"] == 0.0


def test_collect_segment_floors_backfills_missing_segments() -> None:
    """Station pairs without explicit requirements should surface as zero floors."""

    baseline = {
        "segments": [
            {"station_idx": 1, "length_km": 4.0, "dra_ppm": 5.0, "suction_head": 0.7},
        ],
        "segment_lengths": [6.0, 4.0, 3.5],
        "suction_heads": [0.5, 0.7, 0.9],
    }

    result = app._collect_segment_floors(baseline)

    assert [entry["station_idx"] for entry in result] == [0, 1, 2]
    assert [entry["length_km"] for entry in result] == [6.0, 4.0, 3.5]
    assert [entry["dra_ppm"] for entry in result] == [0.0, 5.0, 0.0]
    assert [entry["suction_head"] for entry in result] == [0.5, 0.7, 0.9]


def test_segment_floor_dataframe_handles_zero_floor_rows() -> None:
    """Rendering helper should build a table even when floors are zero."""

    stations = [
        {"name": "Alpha"},
        {"name": "Bravo"},
    ]
    baseline_segments = [
        {"station_idx": 0, "length_km": 10.0, "dra_ppm": 0.0, "dra_perc": 0.0},
        {"station_idx": 1, "length_km": 8.0, "dra_ppm": 4.5, "dra_perc": 9.0, "suction_head": 1.2},
    ]

    seg_df = app._build_segment_floor_dataframe(
        baseline_segments,
        stations,
        terminal_name="Terminal",
        default_suction=0.5,
    )

    assert isinstance(seg_df, pd.DataFrame)
    assert list(seg_df["Segment"]) == ["Alpha → Bravo", "Bravo → Terminal"]
    assert list(seg_df["Floor PPM"]) == [0.0, 4.5]
    assert list(seg_df["Floor %DR"]) == [0.0, 9.0]
    assert list(seg_df["Suction head (m)"]) == [0.5, 1.2]
