"""LangChain tools: crowd analysis, knowledge graph, hinterland RAG and live web search."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from langchain_core.tools import tool

from kg_engine import KG, get_nearby_offbeat_locations, get_neighbors, get_offbeat_within, resolve_location
from rag_engine import search_hinterland_docs

IST = timezone(timedelta(hours=5, minutes=30))


def get_crowd_snapshot(location: str) -> Optional[Dict]:
    """Mock real-time crowd model: baseline adjusted by time of day and weekend."""
    node = resolve_location(location)
    if not node:
        return None
    data = KG.nodes[node]
    base, category = data["crowd_baseline"], data["category"]
    now = datetime.now(IST)
    hour, weekend = now.hour, now.weekday() >= 5
    adj = 0
    if category == "Beach":
        if 11 <= hour <= 13 or 16 <= hour <= 19:
            adj += 1
        elif hour < 8 or hour >= 21:
            adj -= 2
        if weekend:
            adj += 1
    else:
        if 10 <= hour <= 14:
            adj += 1
        elif hour < 8 or hour >= 18:
            adj -= 1
        if weekend and base >= 5:
            adj += 1
    level = max(1, min(10, base + adj))
    if level <= 3:
        status = "Low"
    elif level <= 5:
        status = "Moderate"
    elif level <= 7:
        status = "High"
    else:
        status = "Severe - near/over capacity"
    return {"location": node, "category": category, "level": level, "status": status,
            "occupancy_pct": min(100, level * 10 + 5), "as_of": now.strftime("%a %H:%M IST")}


@tool
def get_crowd_density(location: str) -> str:
    """Get the current estimated crowd level (1-10), occupancy and advisory for a Goan location."""
    snap = get_crowd_snapshot(location)
    if not snap:
        return f"No crowd data for '{location}'. Known places: {', '.join(KG.nodes)}."
    advice = ("Avoid now - consider an offbeat alternative." if snap["level"] >= 8
              else "Expect crowds; go early morning or late evening." if snap["level"] >= 6
              else "Comfortable to visit.")
    return (f"{snap['location']} ({snap['category']}): crowd level {snap['level']}/10, "
            f"{snap['status']}, ~{snap['occupancy_pct']}% capacity as of {snap['as_of']}. {advice}")


@tool
def query_knowledge_graph(location: str) -> str:
    """Look up connected Goan places, travel times in minutes, and low-crowd alternatives near a location."""
    node = resolve_location(location)
    if not node:
        return f"Unknown location '{location}'. Known places: {', '.join(KG.nodes)}."
    d = KG.nodes[node]
    lines = [f"{node}: {d['category']}, taluka {d['taluka']}, crowd baseline {d['crowd_baseline']}/10.",
             "Directly connected:"]
    for n in get_neighbors(node):
        lines.append(f"- {n['name']} ({n['category']}, crowd {n['crowd_baseline']}/10): {n['minutes']} min")
    near = get_nearby_offbeat_locations(node, max_crowd=5)
    lines.append("Adjacent low-crowd options: " +
                 (", ".join(f"{n['name']} ({n['minutes']} min)" for n in near) or "none"))
    wider = get_offbeat_within(node, max_minutes=90, max_crowd=4)[:5]
    lines.append("Offbeat within 90 min: " +
                 (", ".join(f"{n['name']} ({n['minutes']} min)" for n in wider) or "none"))
    return "\n".join(lines)


@tool
def query_hinterland_rag(query: str) -> str:
    """Search curated details on offbeat Goan hinterland, heritage, eco spots, homestays and food."""
    return search_hinterland_docs(query, k=3)


@tool
def live_web_search(query: str) -> str:
    """Search the live web (Tavily if TAVILY_API_KEY is set, else DuckDuckGo) for weather and local events."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            from langchain_tavily import TavilySearch
            
            # Execute search
            response = TavilySearch(max_results=3).invoke({"query": query})
            
            # Safely parse results list from dictionary response
            if isinstance(response, dict):
                results_list = response.get("results", [])
            elif isinstance(response, list):
                results_list = response
            else:
                results_list = []

            if not results_list:
                return f"No live search results found for query: {query}"

            formatted = []
            for r in results_list:
                if isinstance(r, dict):
                    content = r.get("content", "")[:300]
                    url = r.get("url", "")
                    title = r.get("title", "Result")
                    formatted.append(f"- **{title}**: {content} ({url})")
            
            return "\n".join(formatted) if formatted else str(response)

        # Fallback to DuckDuckGo
        from langchain_community.tools import DuckDuckGoSearchRun
        return str(DuckDuckGoSearchRun().invoke(query))[:1200]

    except Exception as exc:  # Network or package issues must not crash the graph
        return f"Live search unavailable right now ({type(exc).__name__}: {str(exc)}). Check local weather/event listings."

ALL_TOOLS = [get_crowd_density, query_knowledge_graph, query_hinterland_rag, live_web_search]
