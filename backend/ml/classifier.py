# ============================================================
# backend/ml/classifier.py
#
# NLP Misinformation Classifier
# ─────────────────────────────
# Wraps a HuggingFace zero-shot classification model.
# Uses facebook/bart-large-mnli by default — no training needed,
# works out of the box on any text.
#
# Design decisions for the hackathon:
#  1. LAZY LOADING — model loads on first request, not at startup.
#     This prevents a 30-second startup delay during the demo.
#  2. SINGLETON — one model instance shared across all requests.
#  3. FALLBACK — if the model can't load (no internet, OOM),
#     we fall back to a keyword-based heuristic so the demo
#     never crashes. Hackathon survival rule #1.
# ============================================================

import time
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Label Configuration ───────────────────────────────────────
# These are the candidate labels passed to zero-shot classification.
# The model scores how well each label fits the input text.
CANDIDATE_LABELS = ["misinformation", "panic-inducing", "safe"]

# Confidence thresholds — tune these for demo quality
MISINFO_THRESHOLD = 0.45   # Score > this → flag as misinformation
PANIC_THRESHOLD = 0.40    # Score > this → flag as panic-inducing


# ── Keyword Fallback ──────────────────────────────────────────
# If the HuggingFace model fails to load, these keyword patterns
# act as a fast heuristic classifier. Rough but demo-safe.
MISINFO_KEYWORDS = [
    r"government.{0,20}secret", r"they.{0,10}don.t want you",
    r"mainstream media.{0,15}silent", r"cover.?up", r"wake up",
    r"share before.{0,15}deleted", r"exposed", r"they.re hiding",
    r"suppressed", r"must share", r"rt to warn", r"chemical",
    r"experimental", r"leaked", r"confirmed.{0,20}source",
]

PANIC_KEYWORDS = [
    r"urgent", r"breaking", r"hospital", r"unknown symptoms",
    r"health alert", r"stay alert", r"avoid", r"outbreak",
    r"spreading fast", r"authorities.{0,15}alerted",
]


def _keyword_classify(text: str) -> dict:
    """
    Rule-based fallback classifier using regex keyword patterns.
    Returns the same schema as the HuggingFace classifier.
    """
    text_lower = text.lower()

    misinfo_hits = sum(1 for p in MISINFO_KEYWORDS if re.search(p, text_lower))
    panic_hits = sum(1 for p in PANIC_KEYWORDS if re.search(p, text_lower))

    # Simple scoring: each hit contributes a fixed weight
    misinfo_score = min(0.95, misinfo_hits * 0.18)
    panic_score = min(0.90, panic_hits * 0.15)
    safe_score = max(0.05, 1.0 - misinfo_score - panic_score)

    scores = {
        "misinformation": round(misinfo_score, 4),
        "panic-inducing": round(panic_score, 4),
        "safe": round(safe_score, 4),
    }

    top_label = max(scores, key=scores.__getitem__)
    return {
        "label": top_label,
        "scores": scores,
        "model_used": "keyword_fallback",
    }


# ── Singleton State ───────────────────────────────────────────
_classifier = None          # The HuggingFace pipeline object
_model_load_failed = False  # If True, skip future load attempts
_model_name_loaded = None   # Track which model is loaded


def get_classifier():
    """
    Lazily load and return the HuggingFace zero-shot pipeline.
    Thread-safe for FastAPI's async workers (single-threaded event loop).
    Returns None if the model failed to load — triggers fallback.
    """
    global _classifier, _model_load_failed, _model_name_loaded

    if _classifier is not None:
        return _classifier  # Already loaded — fast path

    if _model_load_failed:
        return None         # Don't retry after a failure

    try:
        from backend.core.config import settings
        from transformers import pipeline

        logger.info(f"⏳ Loading NLP model: {settings.HF_MODEL_NAME} ...")
        t0 = time.time()

        _classifier = pipeline(
            task="zero-shot-classification",
            model=settings.HF_MODEL_NAME,
            model_kwargs={"cache_dir": settings.HF_CACHE_DIR},
            # Use CPU — remove device_map for hackathon portability
        )

        elapsed = time.time() - t0
        _model_name_loaded = settings.HF_MODEL_NAME
        logger.info(f"✅ Model loaded in {elapsed:.1f}s: {settings.HF_MODEL_NAME}")
        return _classifier

    except Exception as e:
        logger.warning(f"⚠️  HuggingFace model failed to load: {e}")
        logger.warning("⚠️  Falling back to keyword-based classifier.")
        _model_load_failed = True
        return None


# ── Main Classification Function ──────────────────────────────

def classify_text(text: str, include_explanations: bool = True) -> dict:
    """
    Classify a social media post as misinformation, panic-inducing, or safe.
    Enhanced with stance detection, sarcasm analysis, and explainability.

    Args:
        text: The raw post content to analyze.
        include_explanations: Whether to include model explanations (adds latency).

    Returns:
        {
            "label":            str,   # Top predicted label
            "scores":           dict,  # Score per label (0.0 – 1.0)
            "confidence":       float, # Score of the top label
            "model_used":       str,   # Which model produced this
            "is_flagged":       bool,  # True if misinfo or panic
            "latency_ms":       float, # Inference time
            "stance":           dict,  # NEW: Stance analysis
            "sarcasm":          dict,  # NEW: Sarcasm detection
            "explanation":      dict,  # NEW: Model explanation
            "recommendation":   str,   # NEW: Action recommendation
        }
    """
    from backend.ml.model_optimizer import (
        get_cached_prediction, cache_prediction, 
        record_prediction_time
    )
    from backend.ml.stance_detector import detect_stance
    from backend.ml.sarcasm_detector import detect_sarcasm
    from backend.ml.model_explainability import explain_classification
    
    t0 = time.time()
    text = text.strip()

    # ── Check cache first ──────────────────────────────────────
    cached_result = get_cached_prediction(text)
    if cached_result:
        latency = time.time() - t0
        record_prediction_time(latency * 1000, cache_hit=True)
        cached_result["latency_ms"] = round(latency * 1000, 2)
        cached_result["from_cache"] = True
        return cached_result

    if not text:
        result = {
            "label": "safe",
            "scores": {"misinformation": 0.0, "panic-inducing": 0.0, "safe": 1.0},
            "confidence": 1.0,
            "model_used": "empty_input_guard",
            "is_flagged": False,
            "latency_ms": 0.0,
            "from_cache": False,
        }
        cache_prediction(text, result)
        return result

    # ── Analyze stance & sarcasm first ─────────────────────
    stance_info = detect_stance(text)
    sarcasm_info = detect_sarcasm(text)
    
    # If highly sarcastic, adjust interpretation
    is_highly_sarcastic = sarcasm_info["combined_score"] > 0.6

    clf = get_classifier()

    if clf is None:
        # ── Fallback path ─────────────────────────────────────
        result = _keyword_classify(text)
    else:
        # ── HuggingFace path ──────────────────────────────────
        try:
            raw = clf(
                sequences=text,
                candidate_labels=CANDIDATE_LABELS,
                multi_label=False,   # Mutually exclusive labels
            )
            # HuggingFace returns parallel lists — zip into a dict
            scores = dict(zip(raw["labels"], raw["scores"]))
            top_label = raw["labels"][0]   # Already sorted by score desc

            result = {
                "label": top_label,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "model_used": _model_name_loaded,
            }

        except Exception as e:
            logger.error(f"Inference error: {e} — using keyword fallback")
            result = _keyword_classify(text)

    # ── Confidence adjustment based on sarcasm ─────────────────
    top_score = result["scores"][result["label"]]
    
    if is_highly_sarcastic:
        # Reduce confidence if sarcasm detected (inherent uncertainty)
        top_score = top_score * 0.7
        logger.info(f"Confidence reduced due to high sarcasm ({sarcasm_info['combined_score']:.2f})")

    is_flagged = (
        result["label"] == "misinformation" and top_score >= MISINFO_THRESHOLD
    ) or (
        result["label"] == "panic-inducing" and top_score >= PANIC_THRESHOLD
    )

    # ── Generate recommendation ────────────────────────────────
    recommendation = _generate_recommendation(
        result["label"], top_score, stance_info, sarcasm_info
    )

    # ── Build enriched result ─────────────────────────────────
    result["confidence"] = round(top_score, 4)
    result["is_flagged"] = is_flagged
    result["stance"] = stance_info
    result["sarcasm"] = sarcasm_info
    result["recommendation"] = recommendation
    result["from_cache"] = False

    # ── Add explanations if requested ──────────────────────────
    if include_explanations:
        result["explanation"] = explain_classification(
            text, result["label"], result["confidence"]
        )

    latency_ms = round((time.time() - t0) * 1000, 2)
    result["latency_ms"] = latency_ms
    
    # ── Cache result ───────────────────────────────────────────
    cache_prediction(text, result)
    record_prediction_time(latency_ms, cache_hit=False)

    return result


def _generate_recommendation(label: str, confidence: float, 
                           stance_info: dict, sarcasm_info: dict) -> str:
    """
    Generate an action recommendation based on analysis.
    """
    if sarcasm_info["combined_score"] > 0.6:
        return "⚠️ MANUAL REVIEW: High sarcasm detected. Verify actual intent before flagging."
    
    if label == "misinformation":
        if confidence > 0.85:
            return "🚩 HIGH RISK: Likely misinformation. Recommend fact-checking and possible removal."
        elif confidence > 0.65:
            return "⚠️ MEDIUM RISK: Possible misinformation. Monitor and verify before action."
        else:
            return "❓ LOW CONFIDENCE: Insufficient evidence. Manual review recommended."
    
    elif label == "panic-inducing":
        if confidence > 0.8:
            return "⚠️ HIGH PANIC RISK: May cause unnecessary alarm. Consider flagging pending verification."
        else:
            return "ℹ️ LOW PANIC RISK: Unlikely to cause public concern."
    
    else:  # safe
        return "✅ SAFE: Appears to be legitimate information."


def get_model_status() -> dict:
    """Return the current status of the ML model (for /health endpoint)."""
    if _classifier is not None:
        return {"status": "loaded", "model": _model_name_loaded}
    elif _model_load_failed:
        return {"status": "fallback_active", "model": "keyword_classifier"}
    else:
        return {"status": "not_loaded_yet", "model": None}
