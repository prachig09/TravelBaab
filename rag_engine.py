"""In-memory ChromaDB RAG over offbeat Goan places, GTDC-style homestays and food spots."""
from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from functools import lru_cache
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings



try:  # preferred package
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

HINTERLAND_DOCS: List[Dict] = [
    # --- BEACHES ---
    {
        "title": "Baga Beach",
        "location": "Baga",
        "type": "beach",
        "text": (
            "A popular, energetic beach in Bardez known for lively shacks, water sports, and nightlife along "
            "Tito's Lane. Very crowded throughout the year. Tip: visit early morning for a calmer beach walk."
        ),
    },
    {
        "title": "Calangute Beach",
        "location": "Calangute",
        "type": "beach",
        "text": (
            "One of North Goa's largest and busiest beaches in Bardez. Offers extensive water sports, street markets, "
            "and tourist hubs. High crowd levels year-round. Tip: keep belongings secure in heavy crowds."
        ),
    },
    {
        "title": "Candolim Beach",
        "location": "Candolim",
        "type": "beach",
        "text": (
            "A long beach strip starting near Fort Aguada in Bardez, slightly quieter than Baga but well-developed "
            "with beach resorts, shacks, and water activities."
        ),
    },
    {
        "title": "Anjuna Beach",
        "location": "Anjuna",
        "type": "beach",
        "text": (
            "Famous for its rocky shores, bohemian red-clay coastal cliffs, and Wednesday flea market in Bardez. "
            "Popular spot for sunset views and beach music venues."
        ),
    },
    {
        "title": "Vagator Beach",
        "location": "Vagator",
        "type": "beach",
        "text": (
            "Dramatic red cliffs overlooking Big Vagator and Little Vagator (Ozran) beaches in Bardez, shadowed by "
            "Chapora Fort. Popular for sunset gatherings and cliffside cafes."
        ),
    },
    {
        "title": "Palolem Beach",
        "location": "Palolem",
        "type": "beach",
        "text": (
            "A crescent-shaped beach in Canacona with calm waters suitable for kayaking, dolphin trips, and "
            "seasonal wooden beach huts. Steady crowd but relaxed atmosphere."
        ),
    },
    {
        "title": "Agonda Beach",
        "location": "Agonda",
        "type": "beach",
        "text": (
            "A long, relatively calm beach in Canacona with turtle nesting season, yoga stays and beach "
            "shacks. A quieter alternative to Baga and Calangute for people who still want sand and sea."
        ),
    },
    {
        "title": "Kakolem Beach (Tiger Beach)",
        "location": "Kakolem Beach",
        "type": "beach",
        "text": (
            "A secluded, hidden cove beach in Canacona tucked beneath steep cliffs with a small natural stream "
            "flowing onto the sand. Extremely low crowds. Tip: requires climbing down a steep path; wear good shoes."
        ),
    },
    {
        "title": "Hollant Beach",
        "location": "Hollant Beach",
        "type": "beach",
        "text": (
            "A peaceful, shallow bay near Vasco in Mormugao with scenic sunrise views and rocky patches. Ideal for "
            "a quiet wading experience without heavy surf."
        ),
    },

    # --- HERITAGE ---
    {
        "title": "Mapusa Market & Town",
        "location": "Mapusa",
        "type": "heritage",
        "text": (
            "The commercial hub of North Goa in Bardez, famed for its Friday Market selling spice mixes, local sausages "
            "(choris), pottery, and fresh produce. Bustling local heritage market experience."
        ),
    },
    {
        "title": "Panjim City & Mandovi Promenade",
        "location": "Panjim",
        "type": "heritage",
        "text": (
            "The capital city of Goa in Tiswadi along the Mandovi River featuring colonial architecture, modern "
            "promenades, river cruises, art galleries, and historic churches like Our Lady of the Immaculate Conception."
        ),
    },
    {
        "title": "Fontainhas Latin Quarter Walking Tour",
        "location": "Fontainhas",
        "type": "heritage",
        "text": (
            "Panjim's Portuguese-era Latin Quarter with pastel houses, azulejo tiles, balcaos and the Chapel of "
            "St Sebastian. Small art galleries and heritage cafes. Tip: walk at sunrise or late afternoon for "
            "light and fewer people; look out for the Sao Tome area lanes."
        ),
    },
    {
        "title": "Divar Island Heritage Trail",
        "location": "Divar Island",
        "type": "heritage",
        "text": (
            "Reached by a short ferry from Old Goa or Ribandar. Quiet Portuguese-era villages, the Piedade church "
            "on the hill, colourful houses and paddy fields. Best explored by bicycle or on foot. Tip: ferries are "
            "frequent but stop at night; the Bonderam festival is held here in August."
        ),
    },
    {
        "title": "Braganza Pereira House, Chandor",
        "location": "Chandor",
        "type": "heritage",
        "text": (
            "A grand Indo-Portuguese mansion in the Salcete village of Chandor, still home to descendants of "
            "the family. Ornate salons, chandeliers and old-world furniture, with guided visits. Tip: "
            "Chandor also has village homestays; verify availability and GTDC listings directly."
        ),
    },
    {
        "title": "Margao Cultural & Commercial Center",
        "location": "Margao",
        "type": "heritage",
        "text": (
            "South Goa's main cultural capital in Salcete, featuring historic colonial mansions, the municipal market, "
            "Holy Spirit Church square, and transport links across South Goa."
        ),
    },
    {
        "title": "Cabo de Rama Fort",
        "location": "Cabo de Rama",
        "type": "heritage",
        "text": (
            "A hilltop fort on the south coast in Canacona with sweeping Arabian Sea views, a small chapel and ruined "
            "ramparts. Close to Agonda. Tip: sunset visits are rewarding but bring water."
        ),
    },
    {
        "title": "Rachol Seminary & Fortress Gate",
        "location": "Rachol Seminary",
        "type": "heritage",
        "text": (
            "A historic 16th-century Jesuit seminary set atop a hill in Salcete along the Zuari River. Features "
            "a historic museum, library, and quiet, serene architectural grounds. Modest dress required."
        ),
    },

    # --- SPIRITUAL / TEMPLES ---
    {
        "title": "Old Goa Churches & Basilica of Bom Jesus",
        "location": "Old Goa",
        "type": "spiritual",
        "text": (
            "UNESCO World Heritage site in Tiswadi housing Se Cathedral and Basilica of Bom Jesus (storing Saint "
            "Francis Xavier's relics). Grand Portuguese-Manueline and Baroque architectural complex."
        ),
    },
    {
        "title": "Ponda Temple Circuit",
        "location": "Ponda",
        "type": "spiritual",
        "text": (
            "A cluster of temples around Ponda taluka including Shri Mangueshi and Shri Shantadurga, known for "
            "Goan temple architecture and lamp towers (deepstambhas). Busy during festivals. Tip: visit on "
            "weekday mornings and follow temple dress codes."
        ),
    },
    {
        "title": "Mangueshi Temple",
        "location": "Mangueshi Temple",
        "type": "spiritual",
        "text": (
            "A prominent 18th-century Goan Hindu temple dedicated to Lord Manguesh (Shiva) in Priol, Ponda. Known "
            "for its seven-storey octagonal lamp tower (deepstambha) and scenic temple water tank."
        ),
    },
    {
        "title": "Tambdi Surla Mahadev Temple",
        "location": "Tambdi Surla",
        "type": "spiritual",
        "text": (
            "A 12th-13th century Kadamba-era basalt temple dedicated to Shiva, among the oldest surviving temples "
            "in Goa, set at the edge of the Western Ghats forest. Intricate stone carving and a peaceful setting. "
            "Tip: pair with a short forest walk to the nearby waterfall; dress modestly."
        ),
    },
    {
        "title": "Chandreshwar Bhootnath Temple",
        "location": "Chandreshwar Bhootnath",
        "type": "spiritual",
        "text": (
            "Ancient Shiva temple situated atop Chandranath Hill in Quepem. Offers panoramic views of South Goa's "
            "verdant valleys and forests. Accessible via driveable hill road or stone stairs."
        ),
    },
    {
        "title": "Brahma Temple Carambolim",
        "location": "Brahma Temple Carambolim",
        "type": "spiritual",
        "text": (
            "A rare shrine dedicated to Lord Brahma located in Satpal village, Sattari (near Valpoi/Carambolim belt). "
            "Houses an exquisitely carved 12th-century Kadamba period deity idol. Extremely peaceful and offbeat."
        ),
    },

    # --- HINTERLAND / MOUNTAINS / NATURE ---
    {
        "title": "Salim Ali Bird Sanctuary, Chorao Island",
        "location": "Chorao Island",
        "type": "eco",
        "text": (
            "Mangrove sanctuary on the Mandovi river reached by a short ferry from Ribandar. Boat and boardwalk "
            "visits for kingfishers, herons and other waterbirds, plus occasional mudskippers. Tip: go at low "
            "tide near early morning for best birding."
        ),
    },
    {
        "title": "Savoi Spice Plantation",
        "location": "Savoi Plantation",
        "type": "eco",
        "text": (
            "Family-run spice farm near Ponda with guided walks among pepper, cardamom, vanilla, nutmeg and "
            "cashew. Tours usually end with a traditional Goan lunch served on banana leaf. Tip: book a morning "
            "slot and ask for the guided walk."
        ),
    },
    {
        "title": "Bondla Wildlife Sanctuary",
        "location": "Bondla",
        "type": "eco",
        "text": (
            "Goa's smallest wildlife sanctuary in Ponda with a small zoo, deer park, botanical garden and "
            "forest trails. Good for bird watching and families. Tip: closed on certain weekdays, so check "
            "current opening days before you go."
        ),
    },
    {
        "title": "Mayem Lake",
        "location": "Mayem Lake",
        "type": "eco",
        "text": (
            "A calm freshwater lake near Bicholim with boating, walking paths and a small park. A good "
            "low-crowd evening stop on the way between the north coast and Panjim."
        ),
    },
    {
        "title": "Dudhsagar Waterfalls",
        "location": "Dudhsagar Falls",
        "type": "eco",
        "text": (
            "A four-tiered waterfall on the Mandovi River inside Bhagwan Mahaveer Sanctuary near Sanguem. High cascades "
            "resembling a sea of milk. Requires forest jeep safari or railway trek permit. Very popular in season."
        ),
    },
    {
        "title": "Netravali Bubble Lake & Wildlife Sanctuary",
        "location": "Netravali",
        "type": "eco",
        "text": (
            "A small natural spring inside Netravali Wildlife Sanctuary in Sanguem where bubbles rise from the "
            "lake bed when you clap. Surrounded by forest, with a short trail and hanging bridge nearby. Quiet on "
            "weekdays. Tip: go early morning and carry insect repellent."
        ),
    },
    {
        "title": "Cotigao Wildlife Sanctuary",
        "location": "Cotigao",
        "type": "eco",
        "text": (
            "Dense forest sanctuary in Canacona with watchtowers, a nature trail and rich birdlife. Quiet and "
            "uncrowded even in peak season. Tip: ask for the watchtower route and visit in the morning."
        ),
    },
    {
        "title": "Chorla Ghats Eco Trails",
        "location": "Chorla Ghats",
        "type": "eco",
        "text": (
            "Misty Western Ghats mountain range in Sattari, border of Goa, Karnataka, and Maharashtra. Known for "
            "sub-tropical rainforests, endangered wildlife, waterfalls, eco-resorts, and trekking trails."
        ),
    },

    # --- ACCOMMODATIONS & CULINARY ---
    {
        "title": "Homestays and Village Stays in South Goa",
        "location": "Chandor",
        "type": "stay",
        "text": (
            "Village homestays in Chandor, Netravali and Cotigao areas offer home-cooked Goan meals and quiet "
            "surroundings. GTDC also lists accommodation across the state; confirm current availability and "
            "rates on the official GTDC website."
        ),
    },
    {
        "title": "Local Goan Food Stops",
        "location": "Panjim",
        "type": "food",
        "text": (
            "Look for small Goan eateries near Fontainhas and Panjim for fish curry rice, xacuti, sorpotel "
            "and bebinca. Ponda-side spice farms often serve traditional lunches. Check current opening "
            "hours locally."
        ),
    },
]


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings so RAG works without any API key."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class SafeGoogleEmbeddings(Embeddings):
    """Wraps GoogleGenerativeAIEmbeddings and falls back to HashingEmbeddings on API failure."""

    def __init__(self, google_embeddings: Embeddings) -> None:
        self.google_embeddings = google_embeddings
        self.fallback = HashingEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.google_embeddings.embed_documents(texts)
        except Exception as e:
            print(f"[RAG] Google Embeddings API call failed ({e}). Falling back to HashingEmbeddings.")
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.google_embeddings.embed_query(text)
        except Exception as e:
            return self.fallback.embed_query(text)


def _get_embeddings() -> Embeddings:
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            # Clean API key in case quotes were passed in shell exports
            cleaned_key = gemini_key.strip("'\"")

            google_emb = GoogleGenerativeAIEmbeddings(
                model="text-embedding-001",
                google_api_key=cleaned_key,
                task_type="retrieval_document",
            )
            return SafeGoogleEmbeddings(google_emb)
        except Exception:
            pass

    return HashingEmbeddings()

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Initializes and caches an in-memory ChromaDB vector store."""
    store = Chroma(
        collection_name=f"goa_hinterland_{uuid.uuid4().hex[:8]}",
        embedding_function=_get_embeddings(),
    )
    docs = [
        Document(
            page_content=f"{d['title']}. {d['text']}",
            metadata={"title": d["title"], "location": d["location"], "type": d["type"]},
        )
        for d in HINTERLAND_DOCS
    ]
    store.add_documents(docs)
    return store

def get_place_info(location: str) -> Optional[Dict]:
    """Return the first curated entry (non-stay/food preferred) for a KG location."""
    matches = [d for d in HINTERLAND_DOCS if d["location"].lower() == location.lower()]
    for d in matches:
        if d["type"] not in ("stay", "food"):
            return d
    return matches[0] if matches else None


def search_hinterland_docs(query: str, k: int = 3) -> str:
    """Top-k semantic matches, formatted as a context block."""
    results = get_vectorstore().similarity_search(query, k=k)
    if not results:
        return "No matching hinterland entries found."
    blocks = []
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        blocks.append(f"[{i}] {m['title']} (near {m['location']}, {m['type']})\n{doc.page_content}")
    return "\n\n".join(blocks)
