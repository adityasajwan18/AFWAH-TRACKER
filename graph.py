# ============================================================
# backend/api/routes/graph.py
#
# GET /api/graph-data         — D3.js nodes + links
# GET /api/patient-zero/{id}  — Trace origin of a story
# GET /api/spreaders/{id}     — Who spread a story
# POST /api/seed-graph        — Load mock data into Neo4j
# ============================================================

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.db import neo4j_client

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/graph-data",
    summary="Get graph data for D3.js force-directed visualization",
    description="Returns nodes (users) and links (retweet edges) in D3-compatible JSON.",
)
async def get_graph_data(
    story_id: Optional[str] = Query(
        None,
        description="Filter to a specific misinformation cluster. E.g. STORY_001",
    )
) -> dict:
    """Export graph data for D3.js. Works with or without Neo4j."""
    data = neo4j_client.get_graph_data(story_id=story_id)
    return data


@router.get(
    "/patient-zero/{story_id}",
    summary="Trace the origin (Patient Zero) of a misinformation cluster",
)
async def get_patient_zero(story_id: str) -> dict:
    """
    Identify who originally posted the fake news and how it propagated.
    story_id options: STORY_001, STORY_002, STORY_003
    """
    valid_stories = ["STORY_001", "STORY_002", "STORY_003"]
    if story_id not in valid_stories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid story_id. Choose from: {valid_stories}",
        )

    result = neo4j_client.trace_patient_zero(story_id=story_id)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@router.get(
    "/spreaders/{story_id}",
    summary="Get all users who spread a specific misinformation story",
)
async def get_spreaders(story_id: str) -> dict:
    """Returns the list of users who retweeted/shared a fake news cluster."""
    spreaders = neo4j_client.find_spreaders(story_id=story_id)
    return {
        "story_id": story_id,
        "total_spreaders": len(spreaders),
        "spreaders": spreaders,
        "neo4j_active": neo4j_client.is_available(),
    }


@router.post(
    "/seed-graph",
    summary="Seed mock data into Neo4j (run once after startup)",
    description="Loads users, posts, and retweet edges from mock JSON into Neo4j.",
)
async def seed_graph() -> dict:
    """
    One-click Neo4j seeding for the demo.
    Call this once after starting Neo4j and the API server.
    """
    import json, os

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

    def load(f):
        with open(os.path.join(DATA_DIR, f)) as fp:
            return json.load(fp)

    try:
        users = load("mock_users.json")
        posts = load("mock_posts.json")
        edges = load("mock_retweet_edges.json")
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data files not found. Run data_generator.py first. Error: {e}",
        )

    if not neo4j_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="Neo4j is not reachable. Check your connection settings in .env",
        )

    success = neo4j_client.seed_graph(users, posts, edges)
    return {
        "success": success,
        "seeded": {
            "users": len(users),
            "posts": len(posts),
            "edges": len(edges),
        },
        "message": "Graph seeded successfully. Visit /api/graph-data to verify.",
    }
