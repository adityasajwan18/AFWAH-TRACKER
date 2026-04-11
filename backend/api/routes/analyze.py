# ============================================================
# backend/api/routes/analyze.py
#
# POST /api/analyze  — NLP Classification + Viral Score
# POST /api/fact-check — Rumor Credibility Analysis
# ─────────────────────────────────────────────────────
# The core endpoint. Takes post text + metadata,
# runs NLP classification and viral score, returns results.
# ============================================================

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.models import AnalyzeRequest, AnalyzeResponse
from backend.ml.classifier import classify_text
from backend.ml.rumor_analyzer import analyze_rumor
from backend.utils.viral_score import calculate_viral_score

logger = logging.getLogger(__name__)
router = APIRouter()


class FactCheckRequest(BaseModel):
    claim: str


class FactCheckResponse(BaseModel):
    credibility_score: int
    confidence: float
    sentiment: str
    markers: list
    details: str
    similar_claims: list = []  # NEW: Similar claims from history
    analyzed_at: datetime


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a post for misinformation",
    description="""
    Takes a social media post's text and metadata, then returns:
    - **NLP Classification**: misinformation | panic-inducing | safe
    - **Confidence scores** for each label
    - **Viral Potential Score** (0–100) with tier and recommendation
    """,
)
async def analyze_post(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Main analysis endpoint. Runs synchronously since HuggingFace
    inference is CPU-bound. In production, offload to a task queue.
    """
    logger.info(f"Analyzing post: {request.text[:60]}...")

    # ── Step 1: NLP Classification ────────────────────────────
    try:
        nlp_result = classify_text(request.text)
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"NLP classification error: {str(e)}",
        )

    # ── Step 2: Viral Potential Score ─────────────────────────
    posted_at = request.posted_at or datetime.now(timezone.utc)

    try:
        viral_result = calculate_viral_score(
            retweet_count=request.retweet_count,
            like_count=request.like_count,
            reply_count=request.reply_count,
            posted_at=posted_at,
            author_follower_count=request.author_follower_count,
            content=request.text,
            nlp_label=nlp_result["label"],
            sharer_follower_counts=[],  # Populated in Phase 3 from Neo4j
        )
    except Exception as e:
        logger.error(f"Viral score failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Viral score calculation error: {str(e)}",
        )

    # ── Step 3: Build response ────────────────────────────────
    return AnalyzeResponse(
        post_id=request.post_id,
        text_preview=request.text[:100] + ("..." if len(request.text) > 100 else ""),
        nlp=nlp_result,
        viral=viral_result,
        analyzed_at=datetime.now(timezone.utc),
    )


@router.post(
    "/fact-check",
    response_model=FactCheckResponse,
    summary="Fact-check a rumor or claim",
    description="""
    Analyzes a rumor or claim for credibility,sentiment, and misinformation markers.
    
    Returns:
    - **credibility_score**: 0-100 (higher = more credible)
    - **confidence**: 0-1 (analysis confidence)
    - **sentiment**: positive, negative, neutral
    - **markers**: List of detected patterns
    - **details**: Detailed analysis explanation
    """,
)
async def fact_check_claim(request: FactCheckRequest) -> FactCheckResponse:
    """
    Analyze a claim/rumor for credibility and misinformation.
    """
    logger.info(f"Fact-checking claim: {request.claim[:60]}...")
    
    if not request.claim or len(request.claim.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Claim must be at least 5 characters long"
        )
    
    try:
        result = analyze_rumor(request.claim)
        
        return FactCheckResponse(
            credibility_score=result["credibility_score"],
            confidence=result["confidence"],
            sentiment=result["sentiment"],
            markers=result["markers"],
            details=result["details"],
            similar_claims=result.get("similar_claims", []),  # NEW: Include similar claims
            analyzed_at=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Fact-check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fact-check analysis error: {str(e)}",
        )

