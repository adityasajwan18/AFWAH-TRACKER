# ============================================================
# backend/ml/model_explainability.py
#
# Model Explainability & Interpretability
# ──────────────────────────────────────
# Explains why a model made a specific prediction:
# - Feature importance scores
# - Contributing tokens/phrases
# - Confidence calibration
# - Similar historical predictions
# ============================================================

import logging
import re
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)


# High-priority words that strongly indicate misinformation
MISINFORMATION_KEYWORDS = {
    "conspiracy": 0.95,
    "coverup": 0.92,
    "exposed": 0.90,
    "secret": 0.88,
    "leaked": 0.87,
    "suppressed": 0.86,
    "hidden": 0.82,
    "warned": 0.80,
    "evidence": 0.78,
    "proof": 0.75,
    "government": 0.70,
    "dangerous": 0.68,
    "toxic": 0.65,
    "experimental": 0.63,
    "unknown": 0.60,
}

PANIC_KEYWORDS = {
    "urgent": 0.90,
    "breaking": 0.85,
    "immediately": 0.82,
    "critical": 0.80,
    "emergency": 0.78,
    "outbreak": 0.76,
    "alert": 0.75,
    "spread": 0.70,
    "hospital": 0.68,
    "death": 0.65,
    "sick": 0.60,
    "symptoms": 0.58,
}

SAFE_KEYWORDS = {
    "according": -0.70,
    "study": -0.65,
    "research": -0.60,
    "confirmed": -0.55,
    "verified": -0.50,
    "official": -0.48,
}


def explain_classification(text: str, prediction_label: str, 
                          prediction_score: float) -> Dict[str, Any]:
    """
    Explain why a classification was made.
    
    Returns:
    {
        "contributing_factors": [
            {"factor": str, "impact": float, "evidence": str},
            ...
        ],
        "most_important_phrases": list,
        "keyword_matches": list,
        "calibration_info": dict,
        "similar_past_predictions": list,
        "explanation": str,
    }
    """
    text_lower = text.lower()
    words = text_lower.split()
    
    contributing_factors = []
    keyword_matches = []
    
    # ── Analyze keyword contributions ──────────────────────
    for keyword, impact_score in MISINFORMATION_KEYWORDS.items():
        if keyword in text_lower:
            keyword_matches.append({
                "keyword": keyword,
                "category": "misinformation",
                "impact": impact_score,
                "position": text_lower.find(keyword),
            })
            contributing_factors.append({
                "factor": f"Keyword: '{keyword}'",
                "impact": impact_score if prediction_label == "misinformation" else -impact_score,
                "evidence": f"Found in text at position {text_lower.find(keyword)}",
            })
    
    for keyword, impact_score in PANIC_KEYWORDS.items():
        if keyword in text_lower:
            keyword_matches.append({
                "keyword": keyword,
                "category": "panic-inducing",
                "impact": impact_score,
                "position": text_lower.find(keyword),
            })
            contributing_factors.append({
                "factor": f"Keyword: '{keyword}'",
                "impact": impact_score if prediction_label == "panic-inducing" else -impact_score,
                "evidence": f"Found in text at position {text_lower.find(keyword)}",
            })
    
    for keyword, impact_score in SAFE_KEYWORDS.items():
        if keyword in text_lower:
            keyword_matches.append({
                "keyword": keyword,
                "category": "safe",
                "impact": -impact_score,  # These reduce concern
                "position": text_lower.find(keyword),
            })
            contributing_factors.append({
                "factor": f"Credibility indicator: '{keyword}'",
                "impact": -impact_score,  # Reduces concern
                "evidence": f"Found in text - suggests sourced information",
            })
    
    # ── Analyze linguistic features ───────────────────────
    exclamation_count = text.count("!")
    question_count = text.count("?")
    caps_words = len([w for w in words if w.isupper() and len(w) > 1])
    
    if exclamation_count > 2:
        contributing_factors.append({
            "factor": f"High exclamation marks ({exclamation_count})",
            "impact": 0.15 if prediction_label in ["panic-inducing", "misinformation"] else -0.15,
            "evidence": "Emotional intensity may indicate sensationalism",
        })
    
    if caps_words > len(text.split()) * 0.1:  # >10% all caps
        contributing_factors.append({
            "factor": f"Excessive capitalization ({caps_words} words)",
            "impact": 0.12,
            "evidence": "All-caps words suggest emotional intensity or urgency",
        })
    
    # ── Extract most important phrases ──────────────────
    important_phrases = extract_important_phrases(text, prediction_label)
    
    # ── Calibration information ───────────────────────────
    calibration = {
        "raw_score": round(prediction_score, 4),
        "confidence_level": "HIGH" if prediction_score > 0.85 else "MEDIUM" if prediction_score > 0.60 else "LOW",
        "recommendations": get_calibration_recommendations(prediction_score, prediction_label),
    }
    
    # ── Sort contributing factors by impact ────────────
    contributing_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    
    # ── Generate explanation ──────────────────────────
    explanation = generate_explanation(
        prediction_label, 
        prediction_score, 
        contributing_factors[:5],  # Top 5 factors
        important_phrases
    )
    
    return {
        "contributing_factors": contributing_factors[:10],  # Top 10
        "most_important_phrases": important_phrases,
        "keyword_matches": keyword_matches,
        "calibration_info": calibration,
        "explanation": explanation,
    }


def extract_important_phrases(text: str, label: str) -> List[str]:
    """
    Extract phrases that are most relevant to the prediction.
    """
    text_lower = text.lower()
    
    # Split into sentences
    sentences = re.split(r'[.!?]\s+', text)
    
    important_phrases = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # Check if sentence contains key patterns
        if label == "misinformation":
            if any(kw in sentence for kw in MISINFORMATION_KEYWORDS.keys()):
                important_phrases.append(sentence.strip())
        elif label == "panic-inducing":
            if any(kw in sentence for kw in PANIC_KEYWORDS.keys()):
                important_phrases.append(sentence.strip())
    
    return important_phrases[:3]  # Top 3 phrases


def get_calibration_recommendations(score: float, label: str) -> List[str]:
    """
    Recommendations based on prediction confidence.
    """
    recommendations = []
    
    if score > 0.9:
        recommendations.append("Very high confidence - consider immediate action if flagged")
    elif score < 0.5:
        recommendations.append("Low confidence - manual review recommended before action")
    
    if label == "misinformation" and score > 0.75:
        recommendations.append("Consider fact-checking with external sources")
    elif label == "misinformation":
        recommendations.append("May need additional context analysis")
    
    return recommendations


def generate_explanation(label: str, score: float, factors: List[Dict], 
                        phrases: List[str]) -> str:
    """
    Generate human-readable explanation of the prediction.
    """
    explanation = f"**Classification: {label.upper()}** (Confidence: {score*100:.1f}%)\n\n"
    
    if factors:
        explanation += "**Contributing Factors:**\n"
        for i, factor in enumerate(factors[:3], 1):
            explanation += f"{i}. {factor['factor']}\n"
            explanation += f"   Impact: {factor['impact']:+.2f}\n"
    
    if phrases:
        explanation += "\n**Key Phrases:**\n"
        for phrase in phrases:
            explanation += f"- \"{phrase}\"\n"
    
    if label == "misinformation":
        explanation += "\n**Note:** Verify claims with reliable sources before sharing."
    elif label == "panic-inducing":
        explanation += "\n**Note:** May cause unnecessary alarm. Verify facts from official sources."
    else:
        explanation += "\n**Note:** Appears to be reliable information."
    
    return explanation


def compare_to_similar_predictions(current_prediction: Dict, 
                                   prediction_history: List[Dict]) -> List[Dict]:
    """
    Compare current prediction to similar past predictions.
    
    Helps identify patterns in model behavior.
    """
    current_label = current_prediction["label"]
    current_score = current_prediction["confidence"]
    
    similar_predictions = []
    
    for past_pred in prediction_history:
        if past_pred.get("label") == current_label:
            # Calculate similarity (simplified)
            score_diff = abs(past_pred.get("confidence", 0) - current_score)
            if score_diff < 0.2:  # Within 20% confidence
                similar_predictions.append({
                    "past_prediction": past_pred.get("label"),
                    "confidence_diff": round(score_diff, 3),
                    "timestamp": past_pred.get("timestamp"),
                })
    
    return similar_predictions[:5]  # Return top 5 similar
