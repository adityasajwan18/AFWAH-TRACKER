# ============================================================
# backend/api/routes/posts.py
#
# GET /api/posts              — List all mock posts
# GET /api/viral-score/{id}   — Get viral score for a post
# GET /api/stats              — Dashboard summary stats
# ============================================================

import json
import os
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from backend.core.models import PostsResponse, PostViralScoreResponse
from backend.ml.classifier import classify_text
from backend.utils.viral_score import calculate_viral_score

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Data Loading ──────────────────────────────────────────────
# Load mock data once at module import time.
# In Phase 3 this is replaced by MongoDB queries.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

def _load_json(filename: str) -> list:
    """Load a JSON file from the /data directory."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}. Run data_generator.py first.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Eager-load data into memory (fast reads, fine for MVP scale)
_POSTS: list[dict] = _load_json("mock_posts.json")
_USERS: list[dict] = _load_json("mock_users.json")
_EDGES: list[dict] = _load_json("mock_retweet_edges.json")

# Build lookup maps for O(1) access
_POST_MAP: dict[str, dict] = {p["post_id"]: p for p in _POSTS}
_USER_MAP: dict[str, dict] = {u["user_id"]: u for u in _USERS}

logger.info(f"Loaded {len(_POSTS)} posts, {len(_USERS)} users, {len(_EDGES)} edges.")


# ── Routes ────────────────────────────────────────────────────

@router.get(
    "/posts",
    response_model=PostsResponse,
    summary="Get all mock social media posts",
    description="Returns all 100 generated posts. Filter by label or story_id.",
)
async def get_posts(
    label: Optional[str] = Query(
        None,
        description="Filter by label: misinformation | panic-inducing | safe",
        examples=["misinformation"],
    ),
    story_id: Optional[str] = Query(
        None,
        description="Filter by story cluster: STORY_001 | STORY_002 | STORY_003",
    ),
    limit: int = Query(100, ge=1, le=100, description="Max results to return"),
) -> PostsResponse:
    """Serve mock posts with optional filtering."""

    posts = _POSTS

    if label:
        posts = [p for p in posts if p.get("label") == label]

    if story_id:
        posts = [p for p in posts if p.get("story_id") == story_id]

    posts = posts[:limit]

    # Compute label distribution for dashboard charts
    label_counts: dict[str, int] = {}
    for p in _POSTS:  # Always count across ALL posts
        lbl = p.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    return PostsResponse(
        total=len(posts),
        posts=posts,
        label_counts=label_counts,
    )


@router.get(
    "/viral-score/{post_id}",
    response_model=PostViralScoreResponse,
    summary="Get the viral potential score for a specific post",
)
async def get_viral_score(post_id: str) -> PostViralScoreResponse:
    """
    Retrieve a post by ID and calculate its Viral Potential Score.
    Also runs NLP classification if the stored label is missing.
    """
    post = _POST_MAP.get(post_id)
    if not post:
        raise HTTPException(
            status_code=404,
            detail=f"Post '{post_id}' not found. Valid example: 'fn_primary_001'",
        )

    author = _USER_MAP.get(post["author_id"], {})

    # Collect follower counts of all users who retweeted this post
    sharers = [
        e["from_user"] for e in _EDGES
        if e.get("original_post_id") == post_id or e.get("post_id") == post_id
    ]
    sharer_follower_counts = [
        _USER_MAP[uid]["follower_count"]
        for uid in sharers
        if uid in _USER_MAP
    ]

    # Use stored label or run NLP if missing
    nlp_label = post.get("label", "safe")
    if nlp_label == "unknown":
        nlp_result = classify_text(post["content"])
        nlp_label = nlp_result["label"]

    # Parse timestamp
    try:
        posted_at = datetime.fromisoformat(post["timestamp"]).replace(tzinfo=timezone.utc)
    except Exception:
        posted_at = datetime.now(timezone.utc)

    eng = post.get("engagement", {})
    viral = calculate_viral_score(
        retweet_count=eng.get("retweet_count", 0),
        like_count=eng.get("like_count", 0),
        reply_count=eng.get("reply_count", 0),
        posted_at=posted_at,
        author_follower_count=author.get("follower_count", 100),
        content=post["content"],
        nlp_label=nlp_label,
        sharer_follower_counts=sharer_follower_counts,
    )

    return PostViralScoreResponse(
        post_id=post_id,
        content_preview=post["content"][:120] + "...",
        nlp_label=nlp_label,
        viral=viral,
    )


@router.get(
    "/stats",
    summary="Dashboard summary statistics",
    description="Returns aggregate stats for the dashboard header cards.",
)
async def get_stats() -> dict:
    """Quick summary stats for the dashboard KPI cards."""

    total = len(_POSTS)
    label_counts: dict[str, int] = {}
    for p in _POSTS:
        lbl = p.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    misinfo_posts = [p for p in _POSTS if p.get("label") == "misinformation"]
    total_rts_on_misinfo = sum(
        p.get("engagement", {}).get("retweet_count", 0) for p in misinfo_posts
    )

    patient_zero = next(
        (p for p in _POSTS if p.get("is_patient_zero") and p.get("story_id") == "STORY_001"),
        None,
    )

    return {
        "total_posts_analyzed": total,
        "misinformation_count": label_counts.get("misinformation", 0),
        "panic_inducing_count": label_counts.get("panic-inducing", 0),
        "safe_count": label_counts.get("safe", 0),
        "total_retweets_on_misinfo": total_rts_on_misinfo,
        "active_clusters": 2,   # STORY_001 + STORY_002
        "graph_edges": len(_EDGES),
        "patient_zero_post_id": patient_zero["post_id"] if patient_zero else None,
        "patient_zero_author": patient_zero["author_id"] if patient_zero else None,
    }
