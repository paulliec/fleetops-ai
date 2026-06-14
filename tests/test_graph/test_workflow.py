"""Orchestrator graph tests — fan-out/fan-in and failure isolation.

Hermetic: the agents are stubbed, so this runs without Snowflake or the
Open-Meteo API. We only care about the graph wiring here, not the agents'
internals (those have their own coverage). Run with pytest or directly:

    py -m tests.test_graph.test_workflow
"""

from graph import workflow


def _stub(agent, attr, fn):
    """Swap an agent entry point, return a restore callable."""
    orig = getattr(agent, attr)
    setattr(agent, attr, fn)
    return lambda: setattr(agent, attr, orig)


def test_both_agents_present():
    """Fan-out runs both; fan-in merges both outputs into one state."""
    restore = [
        _stub(workflow.maintenance, "forecast", lambda: ["m1", "m2", "m3"]),
        _stub(workflow.weather, "assess", lambda: ["w1", "w2"]),
    ]
    try:
        final = workflow.run()
    finally:
        for r in restore:
            r()

    assert final["maintenance"] == ["m1", "m2", "m3"]
    assert final["weather"] == ["w1", "w2"]
    assert final["errors"] == []


def test_single_failure_isolated():
    """One node raising doesn't sink the run — other output survives,
    failure is captured in state."""
    def boom():
        raise RuntimeError("snowflake timeout")

    restore = [
        _stub(workflow.maintenance, "forecast", boom),
        _stub(workflow.weather, "assess", lambda: ["w1", "w2"]),
    ]
    try:
        final = workflow.run()
    finally:
        for r in restore:
            r()

    assert final["maintenance"] is None          # failed agent left unset
    assert final["weather"] == ["w1", "w2"]       # healthy agent intact
    assert len(final["errors"]) == 1
    assert final["errors"][0]["agent"] == "maintenance"
    assert "snowflake timeout" in final["errors"][0]["error"]


if __name__ == "__main__":
    test_both_agents_present()
    print("PASS: both agents present, outputs merged")
    test_single_failure_isolated()
    print("PASS: single agent failure isolated, error captured in state")
