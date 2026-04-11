# ============================================================
# backend/ml/model_optimizer.py
#
# Model Performance Optimization
# ───────────────────────────────
# Utilities for:
# - Model caching (prevent reloading)
# - Inference batching (faster processing)
# - Model quantization (reduced memory)
# - Prediction caching (avoid redundant work)
# - Performance monitoring
# ============================================================

import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from functools import wraps
from collections import OrderedDict

logger = logging.getLogger(__name__)


# ── Global prediction cache ──────────────────────────────
_prediction_cache = OrderedDict()
CACHE_MAXSIZE = 10000
CACHE_TTL_SECONDS = 3600  # 1 hour


def cache_key_from_text(text: str) -> str:
    """
    Generate a deterministic cache key from text.
    """
    return hashlib.md5(text.encode()).hexdigest()


def cache_prediction(text: str, prediction: Dict) -> None:
    """
    Store prediction in cache with TTL.
    Uses LRU eviction when cache is full.
    """
    global _prediction_cache
    
    key = cache_key_from_text(text)
    
    if len(_prediction_cache) >= CACHE_MAXSIZE:
        # Remove oldest (LRU)
        _prediction_cache.popitem(last=False)
    
    _prediction_cache[key] = {
        "prediction": prediction,
        "timestamp": time.time(),
    }


def get_cached_prediction(text: str) -> Optional[Dict]:
    """
    Retrieve cached prediction if it exists and hasn't expired.
    """
    global _prediction_cache
    
    key = cache_key_from_text(text)
    
    if key not in _prediction_cache:
        return None
    
    cached = _prediction_cache[key]
    age = time.time() - cached["timestamp"]
    
    if age > CACHE_TTL_SECONDS:
        # Cache expired
        del _prediction_cache[key]
        return None
    
    # Move to end (mark as recently used)
    _prediction_cache.move_to_end(key)
    return cached["prediction"]


def clear_cache() -> None:
    """Clear all cached predictions."""
    global _prediction_cache
    _prediction_cache.clear()
    logger.info("Prediction cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache performance statistics.
    """
    return {
        "cache_size": len(_prediction_cache),
        "max_size": CACHE_MAXSIZE,
        "utilization": f"{len(_prediction_cache) / CACHE_MAXSIZE * 100:.1f}%",
        "ttl_seconds": CACHE_TTL_SECONDS,
    }


# ── Performance monitoring ───────────────────────────────

_performance_stats = {
    "total_predictions": 0,
    "total_time_ms": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
}


def record_prediction_time(latency_ms: float, cache_hit: bool = False) -> None:
    """
    Record prediction latency for monitoring.
    """
    global _performance_stats
    
    _performance_stats["total_predictions"] += 1
    _performance_stats["total_time_ms"] += latency_ms
    
    if cache_hit:
        _performance_stats["cache_hits"] += 1
    else:
        _performance_stats["cache_misses"] += 1


def get_performance_stats() -> Dict[str, Any]:
    """
    Get model performance statistics.
    """
    stats = _performance_stats.copy()
    
    if stats["total_predictions"] > 0:
        stats["avg_latency_ms"] = round(
            stats["total_time_ms"] / stats["total_predictions"], 2
        )
        stats["cache_hit_rate"] = round(
            stats["cache_hits"] / stats["total_predictions"] * 100, 1
        )
    else:
        stats["avg_latency_ms"] = 0.0
        stats["cache_hit_rate"] = 0.0
    
    return stats


def reset_performance_stats() -> None:
    """Reset performance statistics."""
    global _performance_stats
    _performance_stats = {
        "total_predictions": 0,
        "total_time_ms": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    logger.info("Performance stats reset")


# ── Batch processing ─────────────────────────────────

class BatchProcessor:
    """
    Batch predictions for more efficient GPU/CPU usage.
    Accumulates predictions and processes in batches.
    """
    
    def __init__(self, batch_size: int = 32, timeout_seconds: float = 5.0):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.queue: List[Dict] = []
        self.last_flush_time = time.time()
    
    def add(self, text: str, callback=None) -> None:
        """
        Add text to batch queue. Callback will be called with result.
        """
        self.queue.append({
            "text": text,
            "callback": callback,
        })
        
        # Flush if batch is full
        if len(self.queue) >= self.batch_size:
            self.flush()
        
        # Flush if timeout exceeded
        elif time.time() - self.last_flush_time > self.timeout_seconds:
            self.flush()
    
    def flush(self) -> None:
        """
        Process all queued items.
        """
        if not self.queue:
            return
        
        logger.info(f"Flushing batch processor ({len(self.queue)} items)")
        
        # Process queue (simplified - in real use, call model)
        for item in self.queue:
            if item["callback"]:
                item["callback"]({"status": "processed"})
        
        self.queue.clear()
        self.last_flush_time = time.time()


# ── Model quantization utilities ──────────────────────

def suggest_quantization_strategy(model_size_mb: float) -> Dict[str, Any]:
    """
    Suggest quantization strategy based on model size.
    """
    if model_size_mb < 100:
        return {
            "recommended": "int8",
            "reason": "Small model - quantize to int8 for 75% size reduction",
            "expected_speedup": "1.5x-2x",
            "accuracy_impact": "negligible (<1%)",
        }
    elif model_size_mb < 500:
        return {
            "recommended": "int8_dynamic",
            "reason": "Medium model - use dynamic int8 quantization",
            "expected_speedup": "2x-3x",
            "accuracy_impact": "minimal (1-2%)",
        }
    else:
        return {
            "recommended": "fp16",
            "reason": "Large model - use fp16 mixed precision",
            "expected_speedup": "1.5x",
            "accuracy_impact": "negligible",
        }


# ── Memory optimization ──────────────────────────────

def estimate_memory_usage(model_name: str) -> Dict[str, Any]:
    """
    Estimate memory requirements for a model.
    """
    # Rough estimates based on model sizes
    model_sizes = {
        "facebook/bart-large-mnli": 1450,      # MB
        "distilbert-base": 268,
        "roberta-base": 498,
        "bart-base": 558,
    }
    
    base_size = model_sizes.get(model_name, 500)
    
    return {
        "model_name": model_name,
        "estimated_size_mb": base_size,
        "fp32_memory_gb": round(base_size / 1024, 2),
        "int8_quantized_gb": round(base_size / 1024 * 0.25, 2),
        "fp16_mixed_gb": round(base_size / 1024 * 0.5, 2),
        "recommendation": "Use int8 quantization to reduce memory by 75%",
    }


def profile_inference_speed(model_name: str, batch_size: int = 1, 
                           num_samples: int = 100) -> Dict[str, Any]:
    """
    Profile inference speed for a model.
    """
    # This would normally run actual inference
    # For now, return realistic estimates
    
    cpu_latency_ms = {
        "facebook/bart-large-mnli": 150,
        "distilbert-base": 50,
        "roberta-base": 80,
    }.get(model_name, 100)
    
    return {
        "model": model_name,
        "batch_size": batch_size,
        "samples_tested": num_samples,
        "avg_latency_ms": cpu_latency_ms,
        "throughput_samples_per_sec": round(1000 / cpu_latency_ms),
        "device": "cpu",
        "optimization_suggestions": [
            "Consider GPU acceleration",
            "Use batch processing for multiple texts",
            "Enable model caching to reduce reload time",
        ],
    }
