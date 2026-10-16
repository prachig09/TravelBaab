"""NetworkX Knowledge Graph for Goa: locations, crowd baselines and travel times."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import networkx as nx

import re

# (name, category, crowd_baseline 1-10, taluka)
LOCATIONS = [
    # --- BEACHES (6) ---
    ("Baga", "Beach", 10, "Bardez"),
    ("Calangute", "Beach", 9, "Bardez"),
    ("Candolim", "Beach", 8, "Bardez"),
    ("Anjuna", "Beach", 8, "Bardez"),
    ("Vagator", "Beach", 7, "Bardez"),
    ("Palolem", "Beach", 7, "Canacona"),
    ("Agonda", "Beach", 4, "Canacona"),
    ("Kakolem Beach", "Beach", 2, "Canacona"),
    ("Hollant Beach", "Beach", 3, "Mormugao"),

    # --- HERITAGE (6) ---
    ("Mapusa", "Heritage", 6, "Bardez"),
    ("Panjim", "Heritage", 6, "Tiswadi"),
    ("Fontainhas", "Heritage", 5, "Tiswadi"),
    ("Divar Island", "Heritage", 3, "Tiswadi"),
    ("Chandor", "Heritage", 2, "Salcete"),
    ("Margao", "Heritage", 5, "Salcete"),
    ("Cabo de Rama", "Heritage", 3, "Canacona"),
    ("Rachol Seminary", "Heritage", 2, "Salcete"),

    # --- SPIRITUAL / TEMPLES (6) ---
    ("Old Goa", "Spiritual", 7, "Tiswadi"),
    ("Ponda", "Spiritual", 5, "Ponda"),
    ("Tambdi Surla", "Spiritual", 2, "Sanguem"),
    ("Mangueshi Temple", "Spiritual", 6, "Ponda"),
    ("Chandreshwar Bhootnath", "Spiritual", 2, "Quepem"),
    ("Brahma Temple Carambolim", "Spiritual", 1, "Sattari"),

    # --- HINTERLAND / MOUNTAINS / NATURE (6) ---
    ("Chorao Island", "Hinterland", 2, "Tiswadi"),
    ("Savoi Plantation", "Hinterland", 3, "Ponda"),
    ("Bondla", "Hinterland", 3, "Ponda"),
    ("Mayem Lake", "Hinterland", 3, "Bicholim"),
    ("Dudhsagar Falls", "Hinterland", 8, "Sanguem"),
    ("Netravali", "Hinterland", 2, "Sanguem"),
    ("Cotigao", "Hinterland", 2, "Canacona"),
    ("Chorla Ghats", "Hinterland", 2, "Sattari"),
]

# Weighted Edges: (origin, destination, travel_minutes)
EDGES = [
    # Bardez & Beach Circuit
    ("Calangute", "Baga", 10),
    ("Calangute", "Candolim", 8),
    ("Calangute", "Anjuna", 20),
    ("Calangute", "Mapusa", 25),
    ("Calangute", "Divar Island", 45),
    ("Baga", "Anjuna", 18),
    ("Anjuna", "Vagator", 8),
    ("Anjuna", "Mapusa", 20),
    ("Vagator", "Mapusa", 25),

    # Tiswadi & Central Connections
    ("Mapusa", "Panjim", 25),
    ("Mapusa", "Mayem Lake", 35),
    ("Mayem Lake", "Panjim", 30),
    ("Candolim", "Panjim", 25),
    ("Panjim", "Fontainhas", 5),
    ("Panjim", "Old Goa", 20),
    ("Panjim", "Chorao Island", 20),
    ("Panjim", "Margao", 40),
    ("Divar Island", "Old Goa", 15),
    ("Divar Island", "Panjim", 25),
    ("Divar Island", "Netravali", 70),
    ("Chorao Island", "Old Goa", 15),

    # Ponda & Spiritual Belt
    ("Old Goa", "Ponda", 25),
    ("Ponda", "Mangueshi Temple", 10),
    ("Mangueshi Temple", "Panjim", 30),
    ("Ponda", "Savoi Plantation", 15),
    ("Ponda", "Bondla", 35),
    ("Ponda", "Margao", 30),
    ("Savoi Plantation", "Tambdi Surla", 40),
    ("Savoi Plantation", "Bondla", 30),

    # Sattari & Hinterland Mountains
    ("Mayem Lake", "Brahma Temple Carambolim", 30),
    ("Brahma Temple Carambolim", "Chorla Ghats", 40),
    ("Tambdi Surla", "Chorla Ghats", 50),
    ("Tambdi Surla", "Dudhsagar Falls", 25),
    ("Tambdi Surla", "Bondla", 50),

    # Mormugao & Salcete
    ("Panjim", "Hollant Beach", 35),
    ("Hollant Beach", "Margao", 30),
    ("Margao", "Rachol Seminary", 15),
    ("Rachol Seminary", "Chandor", 15),
    ("Margao", "Chandor", 20),
    ("Chandor", "Chandreshwar Bhootnath", 25),

    # South Goa (Canacona & Sanguem)
    ("Margao", "Palolem", 55),
    ("Margao", "Agonda", 50),
    ("Margao", "Cabo de Rama", 45),
    ("Margao", "Netravali", 55),
    ("Chandor", "Netravali", 45),
    ("Chandreshwar Bhootnath", "Netravali", 35),
    ("Agonda", "Palolem", 20),
    ("Agonda", "Cabo de Rama", 25),
    ("Cabo de Rama", "Kakolem Beach", 15),
    ("Kakolem Beach", "Agonda", 20),
    ("Palolem", "Cotigao", 20),
    ("Netravali", "Cotigao", 30),
]


def build_graph() -> nx.Graph:
    g = nx.Graph()
    for name, category, crowd, taluka in LOCATIONS:
        g.add_node(name, name=name, category=category, crowd_baseline=crowd, taluka=taluka)
    for a, b, minutes in EDGES:
        g.add_edge(a, b, weight=minutes, travel_minutes=minutes)
    return g


KG: nx.Graph = build_graph()


def resolve_location(name: str) -> Optional[str]:
    """Map user free-text (e.g. 'divar') to a canonical node name ('Divar Island')."""
    if not name:
        return None
    q = name.strip().lower()
    for node in KG.nodes:
        if node.lower() == q:
            return node
    for node in KG.nodes:
        n = node.lower()
        if q in n or n in q:
            return node
    return None


def find_locations_in_text(text: str) -> List[str]:
    """
    Extract all canonical graph location names mentioned in an arbitrary user prompt.
    Sorted by length in descending order to avoid partial matches 
    (e.g., matching 'Divar' instead of 'Divar Island').
    """
    if not text:
        return []

    text_lower = text.lower()
    found_locations = []

    # Sort nodes by longest string length first to give priority to full names
    sorted_nodes = sorted(KG.nodes, key=len, reverse=True)

    for node in sorted_nodes:
        # Check canonical full name match
        node_lower = node.lower()
        if re.search(r'\b' + re.escape(node_lower) + r'\b', text_lower):
            if node not in found_locations:
                found_locations.append(node)
            continue

        # Check keyword aliases (e.g., 'divar' matching 'Divar Island', 'kakolem' matching 'Kakolem Beach')
        first_word = node_lower.split()[0]
        if len(first_word) > 3 and re.search(r'\b' + re.escape(first_word) + r'\b', text_lower):
            if node not in found_locations:
                found_locations.append(node)

    return found_locations


def travel_time(a: str, b: str) -> Optional[int]:
    """Shortest travel time in minutes between two places (None if unknown/unreachable)."""
    na, nb = resolve_location(a), resolve_location(b)
    if not na or not nb:
        return None
    if na == nb:
        return 0
    try:
        return int(nx.shortest_path_length(KG, na, nb, weight="weight"))
    except nx.NetworkXNoPath:
        return None


def get_neighbors(location: str) -> List[Dict]:
    node = resolve_location(location)
    if not node:
        return []
    out = []
    for nbr in KG.neighbors(node):
        d = KG.nodes[nbr]
        out.append({"name": nbr, "category": d["category"], "crowd_baseline": d["crowd_baseline"],
                    "minutes": KG[node][nbr]["weight"]})
    return sorted(out, key=lambda x: x["minutes"])


def get_nearby_offbeat_locations(current_loc: str, max_crowd: int = 5) -> List[Dict]:
    """Adjacent nodes whose crowd baseline is <= max_crowd, nearest first."""
    return [n for n in get_neighbors(current_loc) if n["crowd_baseline"] <= max_crowd]


def get_offbeat_within(
    current_loc: str,
    max_minutes: int = 150,
    max_crowd: int = 5,
    categories: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[Dict]:
    """Multi-hop version: every low-crowd node reachable within max_minutes (Dijkstra)."""
    origin = resolve_location(current_loc)
    if not origin:
        return []
    cats = set(categories) if categories else None
    excl = set(exclude or [])
    dist = nx.single_source_dijkstra_path_length(KG, origin, cutoff=max_minutes, weight="weight")
    out = []
    for node, minutes in dist.items():
        if node == origin or node in excl:
            continue
        d = KG.nodes[node]
        if d["crowd_baseline"] > max_crowd:
            continue
        if cats and d["category"] not in cats:
            continue
        out.append({"name": node, "category": d["category"], "crowd_baseline": d["crowd_baseline"],
                    "minutes": int(minutes), "taluka": d["taluka"]})
    return sorted(out, key=lambda x: (x["minutes"], x["crowd_baseline"]))
