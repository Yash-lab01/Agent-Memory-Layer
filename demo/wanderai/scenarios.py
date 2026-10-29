"""Predefined showcase scenarios and seed knowledge triplets for WanderAI demo.

Provides realistic multi-turn, cross-trip dialogue examples demonstrating:
1. Long-term preference persistence across different trips.
2. Contradiction resolution and non-destructive supersession (e.g. airline loyalty and budget).
3. Knowledge graph multi-hop itinerary reasoning.
"""

from typing import Any

# Domain knowledge triplets connecting destinations, districts, attractions, and amenities
SEED_TRAVEL_TRIPLETS: list[tuple[str, str, str]] = [
    # Paris Destination Topology
    ("Paris", "HAS_DISTRICT", "Le Marais"),
    ("Paris", "HAS_DISTRICT", "Montmartre"),
    ("Paris", "HAS_ATTRACTION", "Centre Pompidou"),
    ("Paris", "HAS_ATTRACTION", "Louvre Museum"),
    ("Centre Pompidou", "CATEGORY", "Contemporary Art"),
    ("Centre Pompidou", "LOCATION", "Le Marais"),
    ("Le Marais", "KNOWN_FOR", "Plant-Based Bakeries"),
    ("Le Marais", "KNOWN_FOR", "Contemporary Art Galleries"),
    ("Le Marais", "HAS_HOTEL", "CitizenM Boutique Hotel"),
    ("CitizenM Boutique Hotel", "PRICE_TIER", "Budget Under $140"),
    ("CitizenM Boutique Hotel", "STYLE", "Boutique Design"),
    ("Wild & The Moon", "CUISINE", "Organic Vegan"),
    ("Wild & The Moon", "LOCATION", "Le Marais"),
    # Tokyo Destination Topology
    ("Tokyo", "HAS_DISTRICT", "Roppongi"),
    ("Tokyo", "HAS_DISTRICT", "Shinjuku"),
    ("Tokyo", "HAS_ATTRACTION", "Mori Art Museum"),
    ("Mori Art Museum", "CATEGORY", "Contemporary Art"),
    ("Mori Art Museum", "LOCATION", "Roppongi"),
    ("Tokyo", "HAS_HOTEL", "Hotel Gracery Shinjuku"),
    ("Hotel Gracery Shinjuku", "PRICE_TIER", "Mid-Range"),
    ("Ain Soph Journey", "CUISINE", "Gourmet Vegan"),
    ("Ain Soph Journey", "LOCATION", "Shinjuku"),
]

# Canonical 4-turn demonstration scenario
WANDERAI_SCENARIOS: list[dict[str, Any]] = [
    {
        "turn": 1,
        "title": "Trip 1 Setup (Tokyo) - Core Traveler Preferences Established",
        "session_id": "trip_tokyo_spring",
        "user_message": (
            "I'm planning a 7-day trip to Tokyo next spring. "
            "I am strictly vegan, hold Star Alliance Gold status, "
            "and love contemporary art museums."
        ),
        "expected_learnings": [
            "Dietary: Strictly vegan",
            "Loyalty: Star Alliance Gold",
            "Interests: Contemporary art museums",
            "Destination: Tokyo (Spring)",
        ],
    },
    {
        "turn": 2,
        "title": "Trip 1 Update - Preference Contradiction & Budget Constraint",
        "session_id": "trip_tokyo_spring",
        "user_message": (
            "Actually, I switched my primary airline loyalty program to OneWorld Sapphire, "
            "and my accommodation budget is strictly under $150 per night."
        ),
        "expected_learnings": [
            "Loyalty update: OneWorld Sapphire (Star Alliance Gold marked SUPERSEDED)",
            "Budget constraint: Under $150 per night",
        ],
    },
    {
        "turn": 3,
        "title": "Trip 2 Planning (Paris) - Cross-Trip Recall Without Re-prompting",
        "session_id": "trip_paris_autumn",
        "user_message": (
            "Help me plan a 4-day autumn trip to Paris. "
            "Recommend a neighborhood to stay in and where I should grab dinner."
        ),
        "expected_learnings": [
            "Agent automatically retrieves vegan dining and <$150 boutique hotel preferences",
            "Zero amnesia across sessions; no need to repeat dietary or budget constraints",
        ],
    },
    {
        "turn": 4,
        "title": "Itinerary Synthesis - Multi-Hop Destination Reasoning",
        "session_id": "trip_paris_autumn",
        "user_message": (
            "What contemporary art museum is located near my hotel in Le Marais, "
            "and where can I get breakfast nearby?"
        ),
        "expected_learnings": [
            "Graph traversal: Le Marais -> Centre Pompidou (Contemporary Art)",
            "Graph traversal: Le Marais -> Wild & The Moon (Vegan Bakery)",
        ],
    },
]
