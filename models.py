# ============================================================
# backend/core/models.py
#
# Pydantic Models — Request & Response Schemas
# ─────────────────────────────────────────────
# These define the exact shape of data flowing in and out of
# our API endpoints. FastAPI auto-generates Swagger docs from
# these — great for the demo's /docs page.
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Analyze Endpoint ──────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Request body for POST /api/analyze"""

    text: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The social media post content to analyze.",
        examples=["BREAKING: Government secretly poisoning water supply. RT to warn everyone! #Exposed"],
    )
    post_id: Optional[str] = Field(
        None,
        description="Optional post ID for cross-referencing with stored data.",
        examples=["fn_primary_001"],
    )
    author_id: Optional[str] = Field(
        None,
        description="Optional author ID for viral score context.",
    )
    retweet_count: int = Field(default=0, ge=0)
    like_count: int = Field(default=0, ge=0)
    reply_count: int = Field(default=0, ge=0)
    author_follower_count: int = Field(default=100, ge=0)
    posted_at: Optional[datetime] = Field(
        None,
        description="Post timestamp (ISO 8601). Defaults to now if not provided.",
    )


class NLPResult(BaseModel):
    """NLP classification result sub-object."""
    label: str
    scores: dict[str, float]
    confidence: float
    model_used: str
    is_flagged: bool
    latency_ms: float


class ViralScoreResult(BaseModel):
    """Viral score sub-object."""
    score: int
    tier: str
    tier_color: str
    breakdown: dict
    recommendation: str


class AnalyzeResponse(BaseModel):
    """Full response for POST /api/analyze"""
    post_id: Optional[str]
    text_preview: str         # First 100 chars of the text
    nlp: NLPResult
    viral: ViralScoreResult
    analyzed_at: datetime


# ── Posts Endpoint ────────────────────────────────────────────

class PostEngagement(BaseModel):
    retweet_count: int
    like_count: int
    reply_count: int


class Post(BaseModel):
    """A single social media post."""
    post_id: str
    author_id: str
    content: str
    timestamp: str
    platform: str
    label: str
    story_id: Optional[str]
    is_retweet_of_id: Optional[str]
    is_patient_zero: bool
    engagement: PostEngagement


class PostsResponse(BaseModel):
    """Response for GET /api/posts"""
    total: int
    posts: list[Post]
    label_counts: dict[str, int]


# ── Viral Score Endpoint ──────────────────────────────────────

class PostViralScoreResponse(BaseModel):
    """Response for GET /api/viral-score/{post_id}"""
    post_id: str
    content_preview: str
    nlp_label: str
    viral: ViralScoreResult
