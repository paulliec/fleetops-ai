"""LangGraph orchestrator — fan-out to the specialist agents, fan-in to a
combined state.

Topology: a virtual START fans out to all four specialist nodes (Maintenance,
Weather, Demand, Staffing), which run in parallel; each edges into the
orchestrator node, which acts as the fan-in barrier (LangGraph won't run it
until all four upstream nodes finish).

The agents stay deterministic/ML and untouched — the nodes are thin wrappers
around their existing entry points (maintenance.forecast / weather.assess /
demand.forecast / staffing.assess), each with try/except so one agent failing
can't take down the run. This mirrors the per-base isolation already in
weather.py.

No LLM reasoning here yet. The orchestrator node is a pass-through join;
ranked, conflict-aware synthesis lands in step 5.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from agents import maintenance, weather, demand, staffing
from agents.maintenance import AircraftForecast
from agents.weather import BaseForecast
from agents.demand import BaseDemandForecast
from agents.staffing import BaseCoverage


class FleetState(TypedDict):
    # one named field per specialist, matching their actual return types.
    # None = the agent hasn't run (or failed) — distinct from an empty list,
    # which means "ran, found nothing".
    maintenance: list[AircraftForecast] | None
    weather: list[BaseForecast] | None
    demand: list[BaseDemandForecast] | None
    staffing: list[BaseCoverage] | None
    # parallel nodes can write this in the same superstep, so it needs a
    # reducer to merge concurrent appends instead of clobbering.
    errors: Annotated[list[dict], operator.add]
    # placeholder for step 5 — LLM ranks/reconciles agent outputs into this.
    recommendations: list | None


def _initial_state() -> FleetState:
    return {
        "maintenance": None, "weather": None, "demand": None, "staffing": None,
        "errors": [], "recommendations": None,
    }


# -- nodes -------------------------------------------------------------------
# Each node opens its own Snowflake connection (agent default) — safer than
# sharing a cursor across parallel branches.

def maintenance_node(state: FleetState) -> dict:
    try:
        return {"maintenance": maintenance.forecast()}
    except Exception as e:
        return {"errors": [{"agent": "maintenance", "error": str(e)}]}


def weather_node(state: FleetState) -> dict:
    try:
        return {"weather": weather.assess()}
    except Exception as e:
        return {"errors": [{"agent": "weather", "error": str(e)}]}


def demand_node(state: FleetState) -> dict:
    try:
        return {"demand": demand.forecast()}
    except Exception as e:
        return {"errors": [{"agent": "demand", "error": str(e)}]}


def staffing_node(state: FleetState) -> dict:
    try:
        return {"staffing": staffing.assess()}
    except Exception as e:
        return {"errors": [{"agent": "staffing", "error": str(e)}]}


def orchestrator_node(state: FleetState) -> dict:
    # Fan-in join. All four agents' structured outputs are already merged into
    # state by the time this runs; nothing to collect.
    # TODO: step 5 - LLM ranks aircraft and reconciles conflicts across the
    # four agents' outputs into `recommendations`.
    return {}


# -- graph -------------------------------------------------------------------

def build_graph():
    g = StateGraph(FleetState)
    g.add_node("maintenance", maintenance_node)
    g.add_node("weather", weather_node)
    g.add_node("demand", demand_node)
    g.add_node("staffing", staffing_node)
    g.add_node("orchestrator", orchestrator_node)

    # fan-out: all four specialists hang off START -> scheduled in one superstep
    g.add_edge(START, "maintenance")
    g.add_edge(START, "weather")
    g.add_edge(START, "demand")
    g.add_edge(START, "staffing")
    # fan-in: orchestrator waits for all four before running
    g.add_edge("maintenance", "orchestrator")
    g.add_edge("weather", "orchestrator")
    g.add_edge("demand", "orchestrator")
    g.add_edge("staffing", "orchestrator")
    g.add_edge("orchestrator", END)

    return g.compile()


def run(graph=None) -> FleetState:
    """Invoke the graph and return the combined final state."""
    graph = graph or build_graph()
    return graph.invoke(_initial_state())


if __name__ == "__main__":
    final = run()

    def _count(v):
        return len(v) if v is not None else "FAILED"

    print(f"maintenance: {_count(final['maintenance'])} aircraft")
    print(f"weather:     {_count(final['weather'])} bases")
    print(f"demand:      {_count(final['demand'])} bases")
    print(f"staffing:    {_count(final['staffing'])} bases")
    print(f"errors:      {final['errors']}")
