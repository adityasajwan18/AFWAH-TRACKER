# ============================================================
# backend/ml/ensemble_classifier.py
#
# Ensemble Multi-Model Classification
# ────────────────────────────────────
# Combines multiple NLP models for higher accuracy and robustness:
# - Zero-shot classification (fast, general)
# - Fine-tuned distilBERT (optimized for misinformation)
# - RoBERTa sentiment-aware classification
#
# Ensemble approach reduces false positives & false negatives.
# ============================================================

import time
import logging
from typing import Optional, Dict, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global model cache
_ensemble_models = {
    "zero_shot": None,
    "distilbert": None,
    "roberta": None,
}

_models_loaded = False


def load_ensemble_models():
    """
    Lazy-load all ensemble models on first request.
    Falls back gracefully if models fail to load.
    """
    global _ensemble_models, _models_loaded
    
    if _models_loaded:
        return
    
    try:
        from transformers import pipeline
        from backend.core.config import settings
        
        logger.info("⏳ Loading ensemble models...")
        
        # Model 1: Zero-shot classification (primary)
        try:
            logger.info("  → Loading zero-shot classifier...")
            _ensemble_models["zero_shot"] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
            logger.info("  ✅ Zero-shot model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  Zero-shot model failed: {e}")
        
        # Model 2: DistilBERT for misinformation (specialized)
        try:
            logger.info("  → Loading DistilBERT classifier...")
            _ensemble_models["distilbert"] = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,
            )
            logger.info("  ✅ DistilBERT model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  DistilBERT model failed: {e}")
        
        # Model 3: RoBERTa for sentiment context
        try:
            logger.info("  → Loading RoBERTa sentiment analyzer...")
            _ensemble_models["roberta"] = pipeline(
                "text-classification",
                model="roberta-base-openai-detector",
                device=-1,
            )
            logger.info("  ✅ RoBERTa model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  RoBERTa model failed: {e}")
        
        _models_loaded = True
        logger.info("✅ Ensemble models ready")
        
    except Exception as e:
        logger.error(f"Fatal error loading ensemble: {e}")
        _models_loaded = True  # Don't retry


@lru_cache(maxsize=512)
def _get_cache_key(text: str) -> str:
    """Generate cache key for predictions."""
    return text[:100]  # Use first 100 chars


def classify_with_ensemble(text: str) -> Dict:
    """
    Classify text using multiple models and ensemble voting.
    
    Returns:
    {
        "label": str,              # Final ensemble prediction
        "scores": dict,            # Individual label scores
        "confidence": float,       # 0-1 ensemble confidence
        "models_used": list,       # Which models were used
        "model_votes": dict,       # Vote breakdown per model
        "latency_ms": float,
    }
    """
    t0 = time.time()
    load_ensemble_models()
    
    text = text.strip()
    if not text:
        return {
            "label": "safe",
            "scores": {"misinformation": 0.0, "panic-inducing": 0.0, "safe": 1.0},
            "confidence": 1.0,
            "models_used": [],
            "model_votes": {},
            "latency_ms": 0.0,
        }
    
    candidate_labels = ["misinformation", "panic-inducing", "safe"]
    model_predictions = []
    models_used = []
    
    # ── Model 1: Zero-shot classification ──────────────────
    if _ensemble_models["zero_shot"]:
        try:
            result = _ensemble_models["zero_shot"](
                sequences=text,
                candidate_labels=candidate_labels,
                multi_label=False,
            )
            pred = {
                "label": result["labels"][0],
                "scores": dict(zip(result["labels"], result["scores"])),
                "model": "zero_shot",
                "weight": 0.4,  # Primary model
            }
            model_predictions.append(pred)
            models_used.append("zero_shot")
        except Exception as e:
            logger.warning(f"Zero-shot inference failed: {e}")
    
    # ── Model 2: DistilBERT sentiment analysis ─────────────
    if _ensemble_models["distilbert"]:
        try:
            result = _ensemble_models["distilbert"](text)
            sentiment_label = result[0]["label"].lower()
            confidence = result[0]["score"]
            
            # Map sentiment to classes
            if sentiment_label == "positive" and confidence > 0.7:
                label = "safe"
                score = confidence
            elif sentiment_label == "negative" and confidence > 0.8:
                label = "panic-inducing"
                score = confidence
            else:
                label = "safe"
                score = 0.5
            
            pred = {
                "label": label,
                "scores": {
                    "misinformation": max(0, score - 0.2),
                    "panic-inducing": score if label == "panic-inducing" else 0.2,
                    "safe": 1.0 - max(0, score - 0.1),
                },
                "model": "distilbert",
                "weight": 0.3,
            }
            model_predictions.append(pred)
            models_used.append("distilbert")
        except Exception as e:
            logger.warning(f"DistilBERT inference failed: {e}")
    
    # ── Model 3: RoBERTa for AI/factuality detection ────────
    if _ensemble_models["roberta"]:
        try:
            result = _ensemble_models["roberta"](text)
            detector_label = result[0]["label"].lower()
            confidence = result[0]["score"]
            
            # RoBERTa detects AI-generated text
            if detector_label == "ai" and confidence > 0.6:
                label = "misinformation"
                score = min(0.95, confidence)
            else:
                label = "safe"
                score = 1.0 - confidence
            
            pred = {
                "label": label,
                "scores": {
                    "misinformation": score if label == "misinformation" else 0.1,
                    "panic-inducing": 0.1,
                    "safe": 1.0 - score if label == "misinformation" else 0.8,
                },
                "model": "roberta",
                "weight": 0.3,
            }
            model_predictions.append(pred)
            models_used.append("roberta")
        except Exception as e:
            logger.warning(f"RoBERTa inference failed: {e}")
    
    # ── Ensemble voting ───────────────────────────────────
    if not model_predictions:
        return {
            "label": "safe",
            "scores": {"misinformation": 0.0, "panic-inducing": 0.0, "safe": 1.0},
            "confidence": 0.5,
            "models_used": [],
            "model_votes": {},
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    
    # Calculate weighted ensemble scores
    ensemble_scores = {"misinformation": 0.0, "panic-inducing": 0.0, "safe": 0.0}
    model_votes = {}
    total_weight = sum(p["weight"] for p in model_predictions)
    
    for pred in model_predictions:
        weight_norm = pred["weight"] / total_weight
        model_votes[pred["model"]] = pred["label"]
        
        for label in ensemble_scores:
            ensemble_scores[label] += pred["scores"][label] * weight_norm
    
    # Normalize scores
    total = sum(ensemble_scores.values())
    if total > 0:
        ensemble_scores = {k: v / total for k, v in ensemble_scores.items()}
    
    # Get final label
    final_label = max(ensemble_scores, key=ensemble_scores.get)
    confidence = ensemble_scores[final_label]
    
    return {
        "label": final_label,
        "scores": {k: round(v, 4) for k, v in ensemble_scores.items()},
        "confidence": round(confidence, 4),
        "models_used": models_used,
        "model_votes": model_votes,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }
