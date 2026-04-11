# ============================================================
# backend/api/routes/model_insights.py
#
# Model Diagnostics & Insights Endpoints
# ──────────────────────────────────────
# Endpoints for:
# - Getting detailed analysis explanations
# - Model performance metrics
# - Cache statistics
# - ML model debugging
# ============================================================

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from backend.ml.classifier import classify_text
from backend.ml.stance_detector import detect_stance, analyze_claim_confidence
from backend.ml.sarcasm_detector import detect_sarcasm, estimate_literal_meaning
from backend.ml.model_optimizer import (
    get_cache_stats, get_performance_stats, reset_performance_stats, clear_cache
)
from backend.ml.model_explainability import explain_classification

logger = logging.getLogger(__name__)
router = APIRouter()


class DetailedAnalysisRequest(BaseModel):
    """Request for detailed analysis with all sub-components."""
    text: str
    include_cache_stats: bool = False
    include_performance_stats: bool = False


class DetailedAnalysisResponse(BaseModel):
    """Comprehensive analysis response with all details."""
    classification: Dict[str, Any]
    stance: Dict[str, Any]
    sarcasm: Dict[str, Any]
    claim_confidence: Dict[str, Any]
    literal_meaning: Optional[Dict[str, Any]]
    explanation: Optional[Dict[str, Any]]
    cache_stats: Optional[Dict[str, Any]]
    performance_stats: Optional[Dict[str, Any]]
    analyzed_at: datetime


@router.post(
    "/detailed-analysis",
    response_model=DetailedAnalysisResponse,
    summary="Get detailed analysis with all sub-components",
    description="""
    Performs comprehensive analysis including:
    - NLP classification (misinformation, panic-inducing, safe)
    - Stance detection (opinion vs. factual)
    - Sarcasm/irony detection
    - Claim confidence analysis
    - Model explainability breakdown
    - System performance metrics
    """,
)
async def detailed_analysis(request: DetailedAnalysisRequest) -> DetailedAnalysisResponse:
    """
    Run all analysis components on a piece of text.
    """
    try:
        text = request.text.strip()
        
        if len(text) < 5:
            raise HTTPException(status_code=400, detail="Text must be at least 5 characters")
        
        # ── Main classification ───────────────────────────────
        classification = classify_text(text, include_explanations=True)
        
        # ── Stance & confidence ───────────────────────────────
        stance = detect_stance(text)
        claim_confidence = analyze_claim_confidence(text)
        
        # ── Sarcasm analysis ───────────────────────────────────
        sarcasm = detect_sarcasm(text)
        literal_meaning = None
        if sarcasm["combined_score"] > 0.4:
            literal_meaning = estimate_literal_meaning(text)
        
        # ── Optional metrics ──────────────────────────────────
        cache_stats = None
        if request.include_cache_stats:
            cache_stats = get_cache_stats()
        
        performance_stats = None
        if request.include_performance_stats:
            performance_stats = get_performance_stats()
        
        return DetailedAnalysisResponse(
            classification=classification,
            stance=stance,
            sarcasm=sarcasm,
            claim_confidence=claim_confidence,
            literal_meaning=literal_meaning,
            explanation=classification.get("explanation"),
            cache_stats=cache_stats,
            performance_stats=performance_stats,
            analyzed_at=datetime.now(timezone.utc),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detailed analysis error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@router.get(
    "/model-health",
    summary="Get model health and performance metrics",
    description="Returns information about model status, caching, and performance.",
)
async def model_health() -> Dict[str, Any]:
    """
    Get the health status of all ML models.
    """
    try:
        from backend.ml.classifier import get_model_status
        
        return {
            "status": "healthy",
            "model_status": get_model_status(),
            "cache_stats": get_cache_stats(),
            "performance_stats": get_performance_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post(
    "/clear-cache",
    summary="Clear prediction cache",
    description="Clears the in-memory prediction cache. Useful for testing.",
)
async def clear_prediction_cache() -> Dict[str, str]:
    """
    Clear all cached predictions.
    """
    try:
        clear_cache()
        return {"status": "success", "message": "Prediction cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.post(
    "/reset-metrics",
    summary="Reset performance metrics",
    description="Resets performance tracking metrics.",
)
async def reset_metrics() -> Dict[str, str]:
    """
    Reset performance statistics.
    """
    try:
        reset_performance_stats()
        return {"status": "success", "message": "Performance metrics reset"}
    except Exception as e:
        logger.error(f"Metrics reset error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset metrics")


@router.get(
    "/model-comparison",
    summary="Compare single-model vs ensemble classification",
    description="For testing: shows how different models classify the same text.",
)
async def model_comparison(text: str) -> Dict[str, Any]:
    """
    Compare classification results from single model vs. ensemble.
    Useful for understanding model accuracy improvements.
    """
    if not text or len(text) < 5:
        raise HTTPException(status_code=400, detail="Text must be at least 5 characters")
    
    try:
        from backend.ml.ensemble_classifier import classify_with_ensemble
        
        # Single model classification
        single_model = classify_text(text, include_explanations=False)
        
        # Ensemble classification
        ensemble = classify_with_ensemble(text)
        
        return {
            "text_preview": text[:100],
            "single_model": single_model,
            "ensemble": ensemble,
            "agreement": single_model["label"] == ensemble["label"],
            "latency_comparison": {
                "single_model_ms": single_model.get("latency_ms", 0),
                "ensemble_ms": ensemble.get("latency_ms", 0),
            },
        }
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail="Comparison failed")


# ── Debugging endpoints (use carefully in production) ────

@router.get(
    "/debug/cache-contents",
    summary="View cache contents (debug only)",
    description="Returns current cached predictions (limited to 10 for privacy).",
    tags=["Debug"],
)
async def debug_cache_contents() -> Dict[str, Any]:
    """
    Get cache contents for debugging purposes.
    Limited to 10 most recent entries to protect privacy.
    """
    from backend.ml.model_optimizer import _prediction_cache
    
    # Return only the keys and metadata, not predictions
    cache_info = {
        "total_entries": len(_prediction_cache),
        "entries": []
    }
    
    # Show last 10 entries
    for i, (key, value) in enumerate(list(_prediction_cache.items())[-10:]):
        cache_info["entries"].append({
            "cache_key": key[:8] + "...",  # Don't expose full hash
            "timestamp": value.get("timestamp"),
            "prediction_label": value.get("prediction", {}).get("label"),
        })
    
    return cache_info
