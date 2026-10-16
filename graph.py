"""LangGraph supervisor assembly for G-EDCM."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (crowd_kg_analysis_node, deep_synthesizer_node, feedback_node, profiler_node,
                    rag_search_discovery_node)
from state import SystemState


def route_entry(state: SystemState) -> str:
    """Supervisor entry: post-visit feedback goes to the feedback node, everything else is profiled."""
    return "feedback" if state.get("last_feedback") else "profiler"


def route_after_feedback(state: SystemState) -> str:
    """Severe crowding / disappointment triggers a full dynamic re-route; otherwise just re-synthesize."""
    return "crowd_kg_analysis" if state.get("reroute_needed") else "deep_synthesizer"


def build_graph():
    g = StateGraph(SystemState)
    g.add_node("profiler", profiler_node)
    g.add_node("crowd_kg_analysis", crowd_kg_analysis_node)
    g.add_node("rag_search_discovery", rag_search_discovery_node)
    g.add_node("deep_synthesizer", deep_synthesizer_node)
    g.add_node("feedback", feedback_node)

    g.add_conditional_edges(START, route_entry, {"feedback": "feedback", "profiler": "profiler"})
    g.add_edge("profiler", "crowd_kg_analysis")
    g.add_edge("crowd_kg_analysis", "rag_search_discovery")
    g.add_edge("rag_search_discovery", "deep_synthesizer")
    g.add_conditional_edges("feedback", route_after_feedback,
                            {"crowd_kg_analysis": "crowd_kg_analysis", "deep_synthesizer": "deep_synthesizer"})
    g.add_edge("deep_synthesizer", END)
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()
