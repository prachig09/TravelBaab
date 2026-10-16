"""Shared LangGraph state for G-EDCM."""
from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SystemState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_persona: str            # one of the four persona names
    crowd_tolerance: int         # 1 (hates crowds) .. 10 (fine with crowds)
    visited_places: List[str]
    last_feedback: Dict          # {location, rating, crowd_comment, crowd_level}
    current_itinerary: str

    # Working fields shared between nodes
    current_location: str
    trip_days: int
    avoid_places: List[str]
    candidate_locations: List[Dict]
    crowd_report: str
    kg_summary: str
    discovery_notes: str
    reroute_needed: bool
    feedback_note: str
    feedback_history: List[Dict]
