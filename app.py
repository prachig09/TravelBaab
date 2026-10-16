"""Chainlit UI for the Goa EcoRoute & Dynamic Capacity Manager (G-EDCM)."""
from __future__ import annotations

import os
import uuid
from typing import Dict

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from agents import DEFAULT_TOLERANCE, PERSONAS, extract_feedback
from graph import graph

NODE_LABELS = {
    "profiler": " Persona Profiler",
    "crowd_kg_analysis": " Crowd & Knowledge Graph Analysis",
    "rag_search_discovery": " RAG & Live Search Discovery",
    "deep_synthesizer": " Deep Itinerary Synthesizer",
    "feedback": " Feedback Processor",
}


def _summarize(node: str, delta: Dict) -> str:
    if node == "profiler":
        return (
            f"Persona: **{delta.get('user_persona')}** · crowd tolerance {delta.get('crowd_tolerance')}/10 · "
            f"{delta.get('trip_days', '?')} day(s) · start: {delta.get('current_location', '?')}"
        )
    if node == "crowd_kg_analysis":
        names = ", ".join(c["name"] for c in delta.get("candidate_locations", [])[:6]) or "none"
        return f"Low-crowd candidates: {names}\n\n{delta.get('crowd_report', '')}"
    if node == "rag_search_discovery":
        return (delta.get("discovery_notes", "") or "")[:700]
    if node == "feedback":
        return delta.get("feedback_note", "") + (
            "\n\n⚠️ Severe crowding detected - triggering dynamic re-route." if delta.get("reroute_needed") else ""
        )
    if node == "deep_synthesizer":
        return "Itinerary assembled successfully."
    return "Done"


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    cl.user_session.set("pending", {})

    api_active = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    mode = "AI-enhanced mode" if api_active else "offline (rule-based) mode"

    actions = [cl.Action(name="select_persona", payload={"persona": p}, label=p) for p in PERSONAS]
    await cl.Message(
        content=(
            "# Welcome to TravelBaab: Goa EcoRoute & Dynamic Capacity Expert\n"
            "I help you enjoy Goa **without** the Baga-Calangute crush, steering you to heritage, "
            "spiritual, and hinterland gems.\n\n"
            f"*Running in **{mode}**.*\n\n"
            "**First, pick your traveller profile:**"
        ),
        actions=actions,
    ).send()


@cl.action_callback("select_persona")
async def on_persona(action: cl.Action) -> None:
    persona = action.payload["persona"]
    pending = cl.user_session.get("pending") or {}
    pending["user_persona"] = persona
    pending["crowd_tolerance"] = DEFAULT_TOLERANCE[persona]
    cl.user_session.set("pending", pending)

    await cl.Message(
        content=(
            f"Great — profile set to **{persona}**.\n\nNow tell me about your trip, e.g.:\n"
            "- *“Plan a 3-day trip from Calangute, I want to avoid crowds”*\n\n"
            "After visiting a place, tell me: *“Visited Savoi Plantation, 4/5, it was quiet”* "
            "and I'll adapt the rest of your route."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content
    pending = cl.user_session.get("pending") or {}
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}

    payload: Dict = {"messages": [HumanMessage(content=text)], **pending}
    cl.user_session.set("pending", {})

    feedback = extract_feedback(text)
    payload["last_feedback"] = feedback if feedback else {}

    final_text = None
    try:
        async for update in graph.astream(payload, config, stream_mode="updates"):
            for node, delta in update.items():
                if not delta:
                    continue
                async with cl.Step(name=NODE_LABELS.get(node, node), type="tool") as step:
                    step.output = _summarize(node, delta)
                for m in delta.get("messages", []) or []:
                    if isinstance(m, AIMessage):
                        final_text = m.content
    except Exception as exc:
        await cl.Message(content=f"⚠️ Something went wrong while planning: `{exc}`").send()
        return

    await cl.Message(
        content=final_text
        or "I couldn't build an itinerary — try telling me your trip length and start point."
    ).send()