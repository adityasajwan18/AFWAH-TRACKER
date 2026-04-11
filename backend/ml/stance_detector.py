# ============================================================
# backend/ml/stance_detector.py
#
# Stance Detection — Opinion vs. Factual Claims
# ──────────────────────────────────────────────
# Detects whether text expresses:
# - Support/opposition to a claim (OPINION)
# - Statement of fact (FACTUAL)
# - Neutral observation (NEUTRAL)
#
# Useful for distinguishing opinion-based misinformation
# from factual misinformation.
# ============================================================

import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)


# Patterns for stance indicators
OPINION_PHRASES = [
    r"i (think|believe|feel|consider|reckon|assume)",
    r"in my (opinion|view|judgment)",
    r"(i argue|i contend|i claim)",
    r"(it seems|it appears|it looks like)",
    r"(potentially|possibly|arguably|seemingly)",
    r"(should|must|ought to|shouldn't|mustn't)",
    r"^(not sure|probably|probably not)",
    r"(agree|disagree|support|oppose)",
]

FACTUAL_PHRASES = [
    r"(study|research|data|evidence|statistics|report)",
    r"(according to|based on|shows that|proves)",
    r"(scientist|expert|doctor|authority)",
    r"(fact|proven|demonstrated|confirmed)",
    r"(found that|discovered|identified|revealed)",
    r"^(the|a|this|that) .{0,50} (is|are|was|were)",
]

SPECULATIVE_WORDS = [
    "might", "could", "possibly", "perhaps", "maybe",
    "allegedly", "reportedly", "supposedly", "claims",
]

CERTAIN_WORDS = [
    "definitely", "certainly", "absolutely", "undoubtedly",
    "proven", "confirmed", "verified", "factually"
]


def detect_stance(text: str) -> Dict[str, Any]:
    """
    Detect the stance/type of a statement.
    
    Returns:
    {
        "stance": "opinion" | "factual" | "neutral",
        "opinion_score": float (0-1),
        "factual_score": float (0-1),
        "certainty": float (0-1),  # How certain is the author?
        "markers": list,
        "details": str,
    }
    """
    text_lower = text.lower()
    opinion_hits = 0
    factual_hits = 0
    
    # Count pattern matches
    for pattern in OPINION_PHRASES:
        if re.search(pattern, text_lower):
            opinion_hits += 1
    
    for pattern in FACTUAL_PHRASES:
        if re.search(pattern, text_lower):
            factual_hits += 1
    
    # Count special words
    speculative_count = sum(1 for w in SPECULATIVE_WORDS if w in text_lower)
    certain_count = sum(1 for w in CERTAIN_WORDS if w in text_lower)
    
    # Normalize to 0-1 scale
    opinion_score = min(1.0, opinion_hits / 5.0)
    factual_score = min(1.0, factual_hits / 4.0)
    
    # Calculate certainty (0 = uncertain, 1 = very certain)
    certainty = min(1.0, (certain_count * 0.3 - speculative_count * 0.2))
    certainty = max(0.0, min(1.0, certainty))
    
    # Determine primary stance
    markers = []
    
    if opinion_score > factual_score and opinion_score > 0.3:
        stance = "opinion"
        markers.append(f"Opinion indicators detected ({opinion_hits} phrases)")
    elif factual_score > opinion_score and factual_score > 0.2:
        stance = "factual"
        markers.append(f"Factual indicators detected ({factual_hits} phrases)")
    else:
        stance = "neutral"
        markers.append("No strong stance indicators")
    
    if speculative_count > certain_count:
        markers.append(f"Speculative language ({speculative_count} words)")
    elif certain_count > speculative_count:
        markers.append(f"Highly certain language ({certain_count} words)")
    
    # Generate detail explanation
    if stance == "opinion":
        detail = "This statement appears to be an opinion or belief rather than a factual claim. " \
                 "Opinions cannot be factually false, though they may be based on false premises."
    elif stance == "factual":
        detail = "This statement appears to be a claim of fact. " \
                 "Use fact-checking resources to verify the accuracy of this claim."
    else:
        detail = "The stance of this statement is unclear. It may contain both factual and opinion elements."
    
    return {
        "stance": stance,
        "opinion_score": round(opinion_score, 3),
        "factual_score": round(factual_score, 3),
        "certainty": round(certainty, 3),
        "markers": markers,
        "details": detail,
    }


def analyze_claim_confidence(text: str) -> Dict[str, Any]:
    """
    Analyze how confident/assertive is the author about their claim.
    High confidence with no evidence = red flag for misinformation.
    
    Returns:
    {
        "confidence_level": "low" | "medium" | "high",
        "confidence_score": float (0-1),
        "has_evidence_modifiers": bool,
        "is_hedged": bool,  # Includes hedging language
    }
    """
    text_lower = text.lower()
    
    # Evidence qualifiers (reduce confidence)
    evidence_modifiers = [
        "according to", "based on", "studies show", "research indicates",
        "experts say", "data shows", "statistics reveal", "tests confirm"
    ]
    
    # Hedging language
    hedging = [
        "might", "could", "may", "perhaps", "possibly", "somewhat",
        "relatively", "fairly", "somewhat", "tends to", "suggests"
    ]
    
    has_evidence = any(mod in text_lower for mod in evidence_modifiers)
    has_hedging = any(h in text_lower for h in hedging)
    
    # Count certainty markers
    certain_count = sum(1 for w in CERTAIN_WORDS if w in text_lower)
    capital_letters = sum(1 for c in text if c.isupper())
    exclamation_marks = text.count("!")
    question_marks = text.count("?")
    
    # Calculate confidence score
    confidence_score = 0.5  # Start neutral
    
    # Increase for certainty markers
    confidence_score += certain_count * 0.1
    
    # Decrease for hedging
    confidence_score -= (sum(1 for h in hedging if h in text_lower) * 0.08)
    
    # Decrease for questions/uncertainty
    confidence_score -= (question_marks * 0.1)
    
    # Increase for aggressive style (caps, exclamation)
    confidence_score += min(0.2, capital_letters / len(text))
    confidence_score += min(0.15, exclamation_marks * 0.1)
    
    confidence_score = max(0.0, min(1.0, confidence_score))
    
    # Determine level
    if confidence_score < 0.35:
        level = "low"
    elif confidence_score > 0.65:
        level = "high"
    else:
        level = "medium"
    
    return {
        "confidence_level": level,
        "confidence_score": round(confidence_score, 3),
        "has_evidence_modifiers": has_evidence,
        "is_hedged": has_hedging,
    }
