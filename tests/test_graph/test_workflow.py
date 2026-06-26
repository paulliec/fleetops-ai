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


def _stub_all_ok():
    """Stub all four agent entry points plus the orchestrator's synthesis, so
    the wiring test runs without Snowflake or the LLM."""
    return [
        _stub(workflow.maintenance, "forecast", lambda: ["m1", "m2", "m3"]),
        _stub(workflow.weather, "assess", lambda: ["w1", "w2"]),
        _stub(workflow.demand, "forecast", lambda: ["d1", "d2"]),
        _stub(workflow.staffing, "assess", lambda: ["s1", "s2"]),
        _stub(workflow.orch, "synthesize", lambda state: ["pkg"]),
        _stub(workflow.orch, "to_dict", lambda pkgs: [{"region": "X"}]),
    ]


def test_all_agents_present():
    """Fan-out runs all four; fan-in merges every output, then the orchestrator
    synthesizes recommendations into state."""
    restore = _stub_all_ok()
    try:
        final = workflow.run()
    finally:
        for r in restore:
            r()

    assert final["maintenance"] == ["m1", "m2", "m3"]
    assert final["weather"] == ["w1", "w2"]
    assert final["demand"] == ["d1", "d2"]
    assert final["staffing"] == ["s1", "s2"]
    assert final["recommendations"] == [{"region": "X"}]
    assert final["errors"] == []


def test_single_failure_isolated():
    """One node raising (demand) doesn't sink the run — the other three
    outputs survive and the failure is captured in state."""
    def boom():
        raise RuntimeError("cortex timeout")

    restore = _stub_all_ok()
    restore.append(_stub(workflow.demand, "forecast", boom))
    try:
        final = workflow.run()
    finally:
        for r in restore:
            r()

    assert final["demand"] is None               # failed agent left unset
    assert final["maintenance"] == ["m1", "m2", "m3"]   # other three intact
    assert final["weather"] == ["w1", "w2"]
    assert final["staffing"] == ["s1", "s2"]
    assert len(final["errors"]) == 1
    assert final["errors"][0]["agent"] == "demand"
    assert "cortex timeout" in final["errors"][0]["error"]


if __name__ == "__main__":
    test_all_agents_present()
    print("PASS: all four agents present, outputs merged")
    test_single_failure_isolated()
    print("PASS: single agent failure isolated, other three intact")
