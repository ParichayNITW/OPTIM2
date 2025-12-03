from types import SimpleNamespace

import pipeline_optimization_app as app


def test_collect_search_depth_kwargs_handles_missing_pipeline_constants(monkeypatch):
    """UI defaults should survive when backend constants are absent."""

    stub_model = SimpleNamespace()
    monkeypatch.setattr(app, "pipeline_model", stub_model, raising=False)

    session = app.st.session_state
    tracked_keys = [
        "search_rpm_step",
        "search_dra_step",
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
        "collect_state_audit": True,
    }
