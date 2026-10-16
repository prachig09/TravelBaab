"""Worker nodes and sub-agents for the G-EDCM supervisor graph.

Every node works with or without an OpenAI key: tools always ground the data
deterministically, and the LLM (when available) refines and narrates.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from kg_engine import KG, find_locations_in_text, get_offbeat_within, resolve_location, travel_time
from rag_engine import get_place_info
from state import SystemState
from tools import (get_crowd_density, get_crowd_snapshot, live_web_search, query_hinterland_rag,
                   query_knowledge_graph)

PERSONAS = [
    "Beach & Adventure",
    "Heritage, Museums & Architecture",
    "Orthodox, Spiritual & Pilgrimage",
    "Eco & Hinterland Explorers",
]
PERSONA_CATEGORIES = {
    PERSONAS[0]: ["Beach", "Hinterland"],
    PERSONAS[1]: ["Heritage"],
    PERSONAS[2]: ["Spiritual", "Heritage"],
    PERSONAS[3]: ["Hinterland"],
}
DEFAULT_TOLERANCE = {PERSONAS[0]: 7, PERSONAS[1]: 5, PERSONAS[2]: 4, PERSONAS[3]: 3}
PERSONA_KEYWORDS = {
    PERSONAS[0]: ["beach", "surf", "water sport", "adventure", "party", "kayak", "scuba", "paragliding", "swim"],
    PERSONAS[1]: ["heritage", "museum", "architecture", "portuguese", "fort", "history", "colonial", "church"],
    PERSONAS[2]: ["spiritual", "temple", "pilgrim", "orthodox", "cathedral", "basilica", "shrine", "meditat"],
    PERSONAS[3]: ["eco", "nature", "forest", "hinterland", "wildlife", "bird", "trek", "waterfall", "spice", "offbeat"],
}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
STOPS_PER_DAY = 2


# --------------------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def get_llm():
    """ChatOpenAI if a key is configured, otherwise None (deterministic offline mode)."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("GEDCM_MODEL", "gpt-4o-mini"), temperature=0.3)


def _latest_human_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def infer_persona(text: str) -> Optional[str]:
    low = text.lower()
    scores = {p: sum(k in low for k in kws) for p, kws in PERSONA_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def _parse_days(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*[- ]?\s*day", text.lower())
    if m:
        return max(1, min(7, int(m.group(1))))
    for word, n in NUM_WORDS.items():
        if re.search(rf"\b{word}[- ]?day", text.lower()):
            return n
    return None


def crowd_level_from_text(text: str) -> Optional[int]:
    """Map a free-text crowd comment to a 1-5 level (5 = severely crowded)."""
    low = text.lower()
    if re.search(r"packed|jammed|overcrowd|chaos|swarm|stampede|unbearabl|too many people|shoulder to shoulder", low):
        return 5
    if re.search(r"crowded|very busy|lots of (people|tourists)|touristy|queue|long line", low):
        return 4
    if re.search(r"moderate|bit busy|somewhat busy|okay crowd|fairly busy", low):
        return 3
    if re.search(r"quiet|empty|peaceful|few people|deserted|serene|uncrowded|calm", low):
        return 1
    return None


def extract_feedback(text: str) -> Optional[Dict]:
    """Detect post-visit feedback in a chat message (location + rating/visit wording)."""
    locs = find_locations_in_text(text)
    if not locs:
        return None
    low = text.lower()
    rating = None
    for pat in (r"\b([1-5])\s*(?:/|out of)\s*5", r"\b([1-5])\s*stars?", r"rat(?:ed|ing|e)[^0-9]{0,15}([1-5])\b"):
        m = re.search(pat, low)
        if m:
            rating = int(m.group(1))
            break
    visited = re.search(r"\b(visited|went to|been to|just left|back from|reached|was at|checked out|explored)\b", low)
    if rating is None and not visited:
        return None
    return {
        "location": locs[0],
        "rating": rating or 3,
        "crowd_comment": text.strip(),
        "crowd_level": crowd_level_from_text(text),
    }


def _compute_candidates(state: SystemState) -> Tuple[str, List[Dict]]:
    loc = resolve_location(state.get("current_location") or "") or "Calangute"
    tol = int(state.get("crowd_tolerance") or 5)
    persona = state.get("user_persona") or PERSONAS[3]
    exclude = set(state.get("visited_places", [])) | set(state.get("avoid_places", []))
    max_crowd = max(tol, 2)
    cands = get_offbeat_within(loc, 180, max_crowd, PERSONA_CATEGORIES.get(persona), exclude)
    if len(cands) < 3:  # widen categories before giving up
        cands = get_offbeat_within(loc, 240, max_crowd, None, exclude)
    for c in cands:
        snap = get_crowd_snapshot(c["name"]) or {}
        c["live_level"] = snap.get("level", c["crowd_baseline"])
        c["live_status"] = snap.get("status", "Unknown")
    return loc, cands[:12]


# --------------------------------------------------------------------------- 1. Profiler
def profiler_node(state: SystemState) -> Dict:
    text = _latest_human_text(state.get("messages", []))
    persona = state.get("user_persona") or ""
    inferred = infer_persona(text)
    if inferred:
        persona = inferred
    if not persona:
        persona = PERSONAS[3]

    tol = state.get("crowd_tolerance") or DEFAULT_TOLERANCE[persona]
    if inferred and not state.get("crowd_tolerance"):
        tol = DEFAULT_TOLERANCE[persona]
    low = text.lower()
    if re.search(r"avoid(ing)? crowds?|no crowds?|offbeat|peaceful|quiet|away from (the )?crowds?", low):
        tol = min(tol, 3)
    elif re.search(r"don'?t mind crowds?|lively|party|nightlife|vibrant", low):
        tol = max(tol, 7)

    update: Dict = {"user_persona": persona, "crowd_tolerance": int(tol), "reroute_needed": False}
    days = _parse_days(text)
    if days:
        update["trip_days"] = days
    elif not state.get("trip_days"):
        update["trip_days"] = 3
    mentioned = find_locations_in_text(text)
    if mentioned:
        update["current_location"] = mentioned[0]
    elif not state.get("current_location"):
        update["current_location"] = "Calangute"
    return update


# --------------------------------------------------------------------------- 2. Crowd & KG analysis
def crowd_kg_analysis_node(state: SystemState) -> Dict:
    loc, cands = _compute_candidates(state)
    text = _latest_human_text(state.get("messages", []))
    hubs = [loc] + [h for h in find_locations_in_text(text) if h != loc]
    hubs += [h for h in ("Baga", "Calangute") if h not in hubs]
    crowd_report = "\n".join(get_crowd_density.invoke({"location": h}) for h in hubs[:4])
    kg_summary = query_knowledge_graph.invoke({"location": loc})
    return {
        "current_location": loc,
        "candidate_locations": cands,
        "crowd_report": crowd_report,
        "kg_summary": kg_summary,
    }


# --------------------------------------------------------------------------- 3. RAG & search discovery
@lru_cache(maxsize=1)
def _discovery_agent():
    llm = get_llm()
    if llm is None:
        return None
    return create_react_agent(
        llm,
        tools=[query_hinterland_rag, live_web_search],
        prompt=(
            "You are the Discovery sub-agent for Goa eco-tourism. Use the tools to find offbeat places "
            "matching the persona and crowd tolerance, and any weather/event risks. Reply in <=120 words."
        ),
    )


def rag_search_discovery_node(state: SystemState) -> Dict:
    persona = state.get("user_persona", PERSONAS[3])
    names = [c["name"] for c in state.get("candidate_locations", [])[:6]]
    rag = query_hinterland_rag.invoke({"query": f"{persona} offbeat low crowd {' '.join(names)}"})
    web = live_web_search.invoke({"query": "Goa weather forecast and local events this week"})
    notes = f"### Hinterland knowledge\n{rag}\n\n### Live updates\n{web}"
    agent = _discovery_agent()
    if agent is not None:
        try:
            res = agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=f"Persona: {persona}. Crowd tolerance {state.get('crowd_tolerance')}/10. "
                            f"Candidates: {', '.join(names)}. Summarize best fits and any weather/event warnings."
                        )
                    ]
                }
            )
            notes += f"\n\n### Discovery agent summary\n{res['messages'][-1].content}"
        except Exception:
            pass
    return {"discovery_notes": notes}


# --------------------------------------------------------------------------- 4. Deep synthesizer
def _advisory(level: int, tol: int) -> str:
    if level >= 8:
        return "⚠️ **Severe crowding expected** - visit very early or skip."
    if level > tol:
        return "⚠️ Busier than your comfort level - go early morning."
    if level >= 5:
        return "Moderate crowds - plan around midday peaks."
    return "✅ Low crowds."


def _build_draft(state: SystemState) -> str:
    persona = state.get("user_persona", PERSONAS[3])
    tol = int(state.get("crowd_tolerance") or 5)
    visited = set(state.get("visited_places", []))
    loc = state.get("current_location") or "Calangute"
    cands = [c for c in state.get("candidate_locations", []) if c["name"] not in visited]
    if not cands:
        loc, cands = _compute_candidates(state)

    days = int(state.get("trip_days") or 3)
    wanted = min(len(cands), days * STOPS_PER_DAY)
    remaining, route, here = list(cands), [], loc
    for _ in range(wanted):  # nearest-neighbour ordering over the KG
        nxt = min(remaining, key=lambda c: (travel_time(here, c["name"]) or 999, c["crowd_baseline"]))
        route.append((here, nxt))
        remaining.remove(nxt)
        here = nxt["name"]

    out = [
        f"# 🌿 Goa EcoRoute Itinerary - {persona}",
        f"*Starting from **{loc}** · crowd tolerance **{tol}/10** · rebalanced away from Baga/Calangute*",
        "",
    ]
    if state.get("feedback_note"):
        out += [f"> 🔄 {state['feedback_note']}", ""]
    if visited:
        out += [f"**Already visited:** {', '.join(sorted(visited))}", ""]
    if not route:
        out += [
            "No further low-crowd stops match your profile right now. Raise your crowd tolerance or "
            "change persona to see more options."
        ]
    else:
        for i in range(0, len(route), STOPS_PER_DAY):
            out.append(f"## Day {i // STOPS_PER_DAY + 1}")
            for prev, c in route[i : i + STOPS_PER_DAY]:
                mins = travel_time(prev, c["name"])
                info = get_place_info(c["name"])
                out += [
                    f"### 📍 {c['name']} ({c['category']})",
                    f"- 🚗 **Travel:** {mins} min from {prev}",
                    f"- 👥 **Crowd now:** {c['live_level']}/10 ({c['live_status']}) · baseline {c['crowd_baseline']}/10",
                    f"- {_advisory(c['live_level'], tol)}",
                ]
                if info:
                    out.append(f"- ℹ️ {info['text']}")
                out.append("")
    if state.get("crowd_report"):
        out += ["## 🚦 Hub Crowd Advisory", *[f"- {l}" for l in state["crowd_report"].splitlines()], ""]
    web = state.get("discovery_notes", "").split("### Live updates")[-1].strip()
    if web:
        out += ["## 🌦️ Live Updates", web[:600], ""]
    out.append("*After each stop, tell me e.g. “Visited Savoi Plantation, 4/5, it was quiet” and I'll adapt the rest.*")
    return "\n".join(out)


def deep_synthesizer_node(state: SystemState) -> Dict:
    draft = _build_draft(state)
    final = draft
    llm = get_llm()
    if llm is not None:
        try:
            res = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are the Goa EcoRoute synthesizer. Polish this itinerary into friendly, "
                            "well-structured Markdown. Keep every travel time, crowd number and warning "
                            "exactly as given; do not add places or facts not present."
                        )
                    ),
                    HumanMessage(content=draft),
                ]
            )
            final = res.content if isinstance(res.content, str) and res.content.strip() else draft
        except Exception:
            final = draft
    return {"messages": [AIMessage(content=final)], "current_itinerary": final, "feedback_note": ""}


# --------------------------------------------------------------------------- 5. Feedback processing
def feedback_node(state: SystemState) -> Dict:
    fb = state.get("last_feedback") or {}
    loc = resolve_location(fb.get("location", ""))
    base: Dict = {"last_feedback": {}, "reroute_needed": False}
    if not loc:
        base["feedback_note"] = "I couldn't match that feedback to a known place, so your plan is unchanged."
        return base

    rating = int(fb.get("rating") or 3)
    level = fb.get("crowd_level") or crowd_level_from_text(fb.get("crowd_comment", ""))
    tol = int(state.get("crowd_tolerance") or 5)
    visited = list(state.get("visited_places", []))
    if loc not in visited:
        visited.append(loc)
    avoid = list(state.get("avoid_places", []))
    notes = [f"Thanks for rating **{loc}** {rating}/5."]
    reroute = False

    if level and level >= 4:
        drop = 2 if level >= 5 else 1
        tol = max(1, tol - drop)
        reroute = True
        notes.append(
            f"You found it crowded, so I lowered your crowd tolerance to **{tol}/10** "
            f"and re-routed the remaining stops toward quieter places near {loc}."
        )
    elif level and level <= 2 and rating >= 4:
        notes.append("Quiet and enjoyable - I'll keep prioritising similarly calm spots.")
    if rating <= 2:
        if loc not in avoid:
            avoid.append(loc)
        notes.append("Since it disappointed you, I'll steer away from similar picks.")
        if not level or level < 4:
            reroute = True

    history = list(state.get("feedback_history", [])) + [{"location": loc, "rating": rating, "crowd_level": level}]
    return {
        "visited_places": visited,
        "avoid_places": avoid,
        "crowd_tolerance": tol,
        "current_location": loc,
        "reroute_needed": reroute,
        "last_feedback": {},
        "feedback_note": " ".join(notes),
        "feedback_history": history,
    }